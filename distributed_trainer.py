import json
import time
import numpy as np
import ray
import wandb
from distributed_actor import create_actor_and_learner
from tqdm import tqdm
from transformers import GenerationConfig
import os
from vllm import SamplingParams


class Trainer:
    def __init__(self, dataset, test_dataset, reward_function, config):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.reward_function = reward_function
        self.config = config

        # create generation config for more default params check: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L98
        # we can probably improve generation further by tuning params
        self.gen_config = GenerationConfig(
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            num_return_sequences=config["num_candidates"],
            do_sample=True,
            use_cache=True,
        )

        # create actors and learner
        self.actors, self.learner = create_actor_and_learner(
            config["number_of_actors"], config["model"], self.gen_config, config
        )
        assert len(self.actors) == config["number_of_actors"]
        self.num_actors = config["number_of_actors"]

        # set params
        self.episodes = config["episodes"]
        self.batch_size = config["batch_size"]
        self.learner_chunk_size = config["learner_chunk_size"]
        self.num_candidates = config["num_candidates"]
        self.max_monkey_rounds = config["max_monkey_rounds"]
        self.save_every = config["save_every"]
        self.eval_every = config["eval_every"]
        self.topk = config["topk"]
        self.run_name = config["run_name"]
        self.project_name = config["project_name"]
        self.run_directory = f"run_{self.run_name}"


        self.eval_sampling_params = SamplingParams(
            temperature=0.6, # we want deterministic behavior at eval, do we?
            max_tokens=config["max_new_tokens"],
            top_p=0.95,
        )

    def save_solutions(
        self, problems, solutions, answers, rewards, reward_threshold=0.5
    ):
        with open("solutions.json", "a") as f:
            for s, p, a, r in zip(solutions, problems, answers, rewards):
                if r > reward_threshold:
                    record = {"problem": p, "solution": s, "answer": a, "reward": r}
                    f.write(json.dumps(record) + "\n")
                    f.flush()

    def save_adapter(self,):
        # Save the adapter
        ray.get(self.learner.save_adapter.remote())

    def compute_rewards(self, answers, solutions):
        return self.reward_function(answers, solutions)

    @staticmethod
    def calculate_chunk_sizes(batch_size, num_actors, learner_chunk_size=1):
        """
        Calculate chunk sizes for splitting data among actors.
        
        Args:
            batch_size (int): Total size of the batch to be split
            num_actors (int): Number of actors to distribute chunks among
            learner_chunk_size (int): Size of the last chunk (default: 1)
        
        Returns:
            list: List of chunk sizes, where last chunk is learner_chunk_size and remaining data
                is distributed equally among num_actors chunks
        """
        # Validate inputs
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if num_actors <= 0:
            raise ValueError("Number of actors must be positive")
        if learner_chunk_size <= 0:
            raise ValueError("Learner chunk size must be positive")
        if batch_size < num_actors + learner_chunk_size:
            #raise ValueError(f"Batch size ({batch_size}) must be greater than number of actors + learner_chunk_size ({num_actors + learner_chunk_size})")
            print(f"Batch size ({batch_size}) must be greater than number of actors + learner_chunk_size ({num_actors + learner_chunk_size})")
            # adapting learner_chunk size
            learner_chunk_size = batch_size - num_actors

        # Reserve items for the last chunk
        remaining_size = batch_size - learner_chunk_size
        
        # Calculate size for each regular chunk
        base_chunk_size = remaining_size // num_actors
        extra_items = remaining_size % num_actors
        
        # Create list of chunk sizes
        chunk_sizes = []
        for i in range(num_actors):
            # Distribute any remaining items one per chunk until exhausted
            if i < extra_items:
                chunk_sizes.append(base_chunk_size + 1)
            else:
                chunk_sizes.append(base_chunk_size)
        
        # Add the final chunk of specified size
        chunk_sizes.append(learner_chunk_size)
        
        return chunk_sizes

    @staticmethod
    def split_dict_lists(data, chunk_sizes):
        # If chunk_sizes is a single number, convert to list
        if isinstance(chunk_sizes, int):
            chunk_sizes = [chunk_sizes]
        
        # Get the length of any list (assuming all are same length)
        list_length = len(next(iter(data.values())))
        
        # Validate that all lists in the dictionary have the same length
        if not all(len(values) == list_length for values in data.values()):
            raise ValueError("All lists in the dictionary must have the same length")
        
        # Validate that chunk sizes sum up to the list length
        if sum(chunk_sizes) != list_length:
            raise ValueError(f"Sum of chunk sizes ({sum(chunk_sizes)}) must equal the length of lists ({list_length})")
        
        # Create chunks
        chunks = []
        start = 0
        
        for size in chunk_sizes:
            end = start + size
            chunk = {key: values[start:end] for key, values in data.items()}
            chunks.append(chunk)
            start = end
        
        return chunks

    def _generate_all_candidates(self, batch, sampling_params=None):
        # Generate and evaluate candidates for each round
        candidates, generation_duration = self._generate_round(batch, sampling_params)
        candidates, reward_duration = self._compute_round_rewards(candidates)

        return candidates, generation_duration, reward_duration

    def _generate_round(self, batch, sampling_params=None):
        """Generate one round of candidates using actors and learner"""
        start_time = time.time()
        if len(batch["problem"]) != self.batch_size:
            print(f"Warning: Actual batch size ({batch_size}) differs from configured batch size ({self.batch_size})")
            batch_size = len(batch["problem"]) # needs to adapt for last dataloader batch that might missmatch inital batch size
        # Compute how to chunk the batch size across actors, creates a list of individual batch sizes per actor
        chunk_sizes = self.calculate_chunk_sizes(batch_size, self.num_actors, self.learner_chunk_size)
        # split the inital batch into chunks for each actor
        chunked_batch = self.split_dict_lists(batch, chunk_sizes)
        actor_tasks = [
            actor.generate.remote(task, sampling_params)
            for actor, task in zip(self.actors, chunked_batch[: self.num_actors])
        ]
        learner_task = self.learner.generate.remote(chunked_batch[-1], sampling_params)

        # Get results with timeout
        generations = ray.get(actor_tasks + [learner_task], timeout=240)

        generation_duration = time.time() - start_time
        return generations, generation_duration

    def _compute_round_rewards(self, candidate_data):
        """Compute rewards for the current round of candidates"""
        start_time = time.time()

        for i, candidate in enumerate(candidate_data):
            rewards = []
            for batch_answers, batch_solutions in zip(candidate["answers"], candidate["solution"]):
                batch_rewards = self.compute_rewards(batch_answers, batch_solutions)
                rewards.append(batch_rewards)

            candidate_data[i]["rewards"] = rewards

        reward_duration = time.time() - start_time

        return candidate_data, reward_duration


    def train(self):
        total_batch_steps = 0

        # initialize wandb
        run = wandb.init(
            name=self.run_name, config=self.config, project=self.project_name
        )
        # initial eval
        if self.eval_every > 0:
            self.evaluate(wandb=run, total_steps=total_batch_steps)

        for episode in tqdm(range(self.episodes), desc="Training ..."):
            self.dataset = self.dataset.shuffle()
            loader = self.dataset.iter(batch_size=self.batch_size)

            for batch in loader:
                total_batch_steps += 1
                # generate responses
                candidates, generation_duration, reward_duration = (
                    self._generate_all_candidates(batch)
                )
                # compute metrics
                mean_task_acc_rewards = []
                mean_task_format_reward = []
                mean_task_token_length = []
                max_task_acc_rewards = []
                min_task_acc_rewards = []
                for i, candidate in enumerate(candidates):
                    baselines = []
                    summed_rewards = []
                    for batch_reward, batch_token_length in zip(candidate["rewards"], candidate["token_lengths"]):
                        baselines.append(np.mean(batch_reward.sum(axis=1)))
                        mean_task_acc_rewards.append(np.mean(batch_reward[:,1]))
                        mean_task_token_length.append(np.mean(batch_token_length))
                        max_task_acc_rewards.append(np.max(batch_reward[:,1]))
                        min_task_acc_rewards.append(np.min(batch_reward[:,1]))
                        mean_task_format_reward.append(np.mean(batch_reward[:,0]))
                        summed_rewards.append(batch_reward.sum(axis=1)) # TODO: test here leave one out reward normalization
                    candidates[i]["baselines"] = baselines
                    candidates[i]["rewards"] = summed_rewards

                # topk filter
                for i, candidate in enumerate(candidates):
                    filtered_answers = []
                    filtered_rewards = []
                    filtered_problems = []
                    for j, rewards in enumerate(candidate["rewards"]):
                        topk_idx = np.argsort(rewards)[-self.topk:]
                        # Only filter rewards and answers as we only need those for loss calc
                        filtered_answers.append([candidate["answers"][j][idx] for idx in topk_idx])
                        filtered_rewards.append(rewards[topk_idx])
                        filtered_problems.append(candidate["problem"][j][:self.topk]) # we only need topk amount. as they are all the same per batch topk filter is not needed
                    candidates[i]["answers"] = filtered_answers
                    candidates[i]["rewards"] = filtered_rewards
                    candidates[i]["problem"] = filtered_problems
                
                # Logs 
                print(f"Sample from the candidates: \nProblem: {candidates[0]['problem'][0][0]}")
                print(f"\nAnswer: {candidates[0]['answers'][0][0]}")
                print(f"\nReward: {candidates[0]['rewards'][0][0]}\n\n")      


                # update policy
                update_start_time = time.time()
                loss = ray.get(
                    self.learner.train.remote(candidates))
                update_duration = time.time() - update_start_time

                # Save adapter
                self.save_adapter()

                run.log(
                    {
                        "loss": loss,
                        "mean_format_reward": np.mean(mean_task_format_reward).item(),
                        "mean_accuracy_reward": np.mean(mean_task_acc_rewards).item(),
                        "min_accuracy_reward": np.mean(min_task_acc_rewards).item(),
                        "max_accuracy_reward": np.mean(max_task_acc_rewards).item(),
                        "mean_token_length": np.mean(
                            mean_task_token_length
                        ).item(),
                        "episode": episode,
                        "total_batch_steps": total_batch_steps,
                        "timing/update_duration": update_duration,
                        "timing/reward_duration": reward_duration,
                        "timing/generation_duration": generation_duration,
                    },
                    step=total_batch_steps,
                )

                # evaluate
                if self.eval_every > 0 and total_batch_steps % self.eval_every == 0:
                    self.evaluate(wandb=run, total_steps=total_batch_steps)

                # save policy
                if total_batch_steps % self.save_every == 0:
                    ray.get(self.learner.save_checkpoint.remote(os.path.join(self.run_directory, f"model_{total_batch_steps}")))

            # save final policy
            ray.get(
                self.learner.save_checkpoint.remote(os.path.join(self.run_directory, f"model_{total_batch_steps}")
                )
            )
        # cleanup
        ray.shutdown()


    def evaluate(self, wandb, total_steps,):
        eval_loader = self.test_dataset.iter(batch_size=self.batch_size)

        total_passed = 0
        total_problems = 0
        total_token_length = []

        for batch in tqdm(eval_loader, desc="Evaluating ..."):
            # Generate candidates   
            eval_candidates, eval_generation_duration, eval_reward_duration = (
                self._generate_all_candidates(batch, self.eval_sampling_params)
            )

            for candidate in eval_candidates:
                batch_rewards = np.vstack(candidate["rewards"])
                token_lengths = np.vstack(candidate["token_lengths"])
                total_token_length.append(np.mean(token_lengths))
                accuracy_reward = batch_rewards[:, 1]
                total_passed += accuracy_reward.sum()
                total_problems += batch_rewards.shape[0]

        overall_pass_rate = total_passed / total_problems
        overall_token_length = np.mean(total_token_length)
        wandb.log({f"eval/pass@1": overall_pass_rate, "eval/mean_token_length": overall_token_length},step=total_steps,)
        
