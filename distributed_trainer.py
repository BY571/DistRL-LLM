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
        self.actors, self.learners = create_actor_and_learner(
            config["number_of_actors"], config["number_of_learners"], config["model"], self.gen_config, config
        )
        assert len(self.actors) == config["number_of_actors"]
        self.num_actors = config["number_of_actors"]
        assert len(self.learners) == config["number_of_learners"]
        self.num_learners = config["number_of_learners"]

        # set params
        self.episodes = config["episodes"]
        self.batch_size = config["batch_size"]
        self.learner_chunk_size = config["learner_chunk_size"]
        self.num_candidates = config["num_candidates"]
        self.save_every = config["save_every"]
        self.eval_every = config["eval_every"]
        self.topk = config["topk"]  
        self.run_name = config["run_name"]
        self.project_name = config["project_name"]
        self.run_directory = f"run_{self.run_name}"
        self.learner_type = config["learner"]


        self.eval_sampling_params = SamplingParams(
            temperature=0.6, # we want deterministic behavior at eval, do we?
            max_tokens=config["max_new_tokens"],
            top_p=0.95,
            n=8,
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
        ray.get(self.learners[0].save_adapter.remote())

    def compute_rewards(self, answers, solutions):
        return self.reward_function(answers, solutions)

    @staticmethod
    def calculate_chunk_sizes(batch_size, num_actors, num_learners=1, learner_chunk_size=1):
        """
        Calculate chunk sizes for splitting data among actors and learners.
        
        Args:
            batch_size (int): Total size of the batch to be split
            num_actors (int): Number of actors to distribute chunks among
            learner_chunk_size (int): Size of each learner chunk (default: 1)
            num_learners (int): Number of learners (default: 1)
        
        Returns:
            list: List of chunk sizes for actors followed by learners
        """
        # Validate basic inputs
        if batch_size <= 0 or num_actors <= 0 or num_learners <= 0:
            raise ValueError("All parameters must be positive")
            
        # Calculate total size needed for learners
        total_learner_size = learner_chunk_size * num_learners
        
        # Check if we have enough data for all actors and learners
        if batch_size < num_actors + total_learner_size:
            print(f"Warning: Batch size ({batch_size}) is smaller than actors + learners need ({num_actors + total_learner_size})")
            
            # Prioritize actors: ensure each actor gets at least 1 item
            min_actor_size = min(batch_size, num_actors)
            
            # If we can fit all actors with at least 1 item each
            if min_actor_size == num_actors:
                # Calculate remaining space for learners
                remaining_for_learners = batch_size - num_actors
                
                # If we have space for at least one learner
                if remaining_for_learners > 0 and num_learners > 0:
                    learner_chunk_size = max(1, remaining_for_learners // num_learners)
                    num_learners = min(num_learners, remaining_for_learners // learner_chunk_size)
                    total_learner_size = learner_chunk_size * num_learners
                else:
                    # No space for learners
                    num_learners = 0
                    total_learner_size = 0
            else:
                # Can't fit all actors, reduce number of actors
                num_actors = min_actor_size
                # No space for learners
                num_learners = 0
                total_learner_size = 0
        
        # Calculate actor chunk sizes
        actor_size = batch_size - total_learner_size
        
        # Create chunks list for actors
        if num_actors > 0:
            base_size = actor_size // num_actors
            extra = actor_size % num_actors
            chunks = [base_size + 1 if i < extra else base_size for i in range(num_actors)]
        else:
            chunks = []
        
        # Add learner chunks if any
        if num_learners > 0:
            chunks.extend([learner_chunk_size] * num_learners)
        
        return chunks

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
            batch_size = len(batch["problem"]) # needs to adapt for last dataloader batch that might missmatch inital batch size
            print(f"Warning: Actual batch size ({batch_size}) differs from configured batch size ({self.batch_size})")
        else:
            batch_size = self.batch_size
        # Compute how to chunk the batch size across actors, creates a list of individual batch sizes per actor
        chunk_sizes = self.calculate_chunk_sizes(batch_size, self.num_actors, self.num_learners, self.learner_chunk_size)
        # split the inital batch into chunks for each actor
        chunked_batch = self.split_dict_lists(batch, chunk_sizes)
        actor_tasks = [
            actor.generate.remote(task, sampling_params)
            for actor, task in zip(self.actors, chunked_batch[: self.num_actors])
        ]
        learner_tasks = [
            learner.generate.remote(task, sampling_params)
            for learner, task in zip(self.learners, chunked_batch[-self.num_learners:])
        ]

        # Get results with timeout
        generations = ray.get(actor_tasks + learner_tasks, timeout=240)

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

    def merge_candidates(self, candidates):
        problems = []
        answers = []
        rewards = []
        for candidate in candidates:
            for a, p, r in zip(candidate["answers"], candidate["problem"], candidate["rewards"]):
                problems.extend(p)
                answers.extend(a)
                rewards.extend(r)
        return problems, answers, rewards
    
    def train(self):
        total_batch_steps = 0
        total_samples_processed = 0

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
                total_samples_processed += len(batch["problem"])
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
                    advantages = []
                    for batch_reward, batch_token_length in zip(candidate["rewards"], candidate["token_lengths"]):
                        baselines.append(np.mean(batch_reward.sum(axis=1)))
                        mean_task_acc_rewards.append(np.mean(batch_reward[:,1]))
                        mean_task_token_length.append(np.mean(batch_token_length))
                        max_task_acc_rewards.append(np.max(batch_reward[:,1]))
                        min_task_acc_rewards.append(np.min(batch_reward[:,1]))
                        mean_task_format_reward.append(np.mean(batch_reward[:,0]))
                        advantages.append((batch_reward.sum(axis=1) - np.mean(batch_reward.sum(axis=1))) / (np.std(batch_reward.sum(axis=1)) + 1e-8))
                        summed_rewards.append(batch_reward.sum(axis=1)) # TODO: test here leave one out reward normalization
                    if self.learner_type == "grpo":
                        candidates[i]["rewards"] = advantages
                    else:
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

                # If there's only one learner, no need to split
                if self.num_learners == 1:
                    loss = ray.get(self.learners[0].train.remote(candidates))
                else:
                    # Merge problem, answer and rewards from candidates
                    problems, answers, rewards = self.merge_candidates(candidates)
                    # Split candidates across learners evenly
                    chunk_sizes = [len(problems) // self.num_learners] * self.num_learners
                    for i in range(len(problems) % self.num_learners):
                        chunk_sizes[i] += 1  # Distribute remainder

                    # Create chunks
                    start = 0
                    candidate_chunks = []
                    for size in chunk_sizes:
                        candidate_chunks.append((problems[start:start + size], answers[start:start + size], rewards[start:start + size]))

                        start += size

                    # Compute gradients independently
                    gradients_futures = [
                        learner.compute_gradients.remote(chunk) for learner, chunk in zip(self.learners, candidate_chunks)
                    ]

                    # Gather gradients from all learners - process one at a time to avoid OOM
                    gradients = []
                    losses = []
                    for future in gradients_futures:
                        gradient, loss = ray.get(future, timeout=240)
                        gradients.append(gradient)
                        losses.append(loss)
                        # Clear memory after each gradient computation
                        del future

                    loss = sum(losses) / len(losses)

                    # Merge gradients into the first learner
                    ray.get(self.learners[0].apply_merged_gradients.remote(gradients))
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
                        "total_samples_processed": total_samples_processed,
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
                    ray.get(self.learners[0].save_checkpoint.remote(os.path.join(self.run_directory, f"model_{total_batch_steps}")))

            # save final policy
            ray.get(
                self.learners[0].save_checkpoint.remote(os.path.join(self.run_directory, f"model_{total_batch_steps}")
                )
            )
        # cleanup
        ray.shutdown()
        
    def evaluate(self, wandb, total_steps,):
        eval_time = time.time()
        eval_loader = self.test_dataset.iter(batch_size=self.batch_size)

        total_passed = 0
        total_max_passed = 0
        total_problems = 0
        total_token_length = []

        for batch in tqdm(eval_loader, desc="Evaluating ..."):
            # Generate candidates   
            eval_candidates, eval_generation_duration, eval_reward_duration = (
                self._generate_all_candidates(batch, self.eval_sampling_params)
            )

            for candidate in eval_candidates:
                for r, token in zip(candidate["rewards"], candidate["token_lengths"]):

                    total_token_length.append(np.mean(token))
                    accuracy_reward = np.mean(r[:, 1])
                    total_passed += accuracy_reward
                    total_max_passed += np.max(r[:, 1])
                    total_problems += 1

        overall_pass_rate = total_passed / total_problems
        overall_max_pass_rate = total_max_passed / total_problems
        overall_token_length = np.mean(total_token_length)
        eval_duration = time.time() - eval_time
        wandb.log({f"eval/pass@1(mean{self.eval_sampling_params.n})": overall_pass_rate,
                   f"eval/BoN({self.eval_sampling_params.n})": overall_max_pass_rate,
                   "eval/mean_token_length": overall_token_length,
                   "timing/eval_duration": eval_duration},step=total_steps)
        
