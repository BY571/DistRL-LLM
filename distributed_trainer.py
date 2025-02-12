import ray
import numpy as np
import wandb
from transformers import GenerationConfig
from tqdm import tqdm
from buffer import RankedLists
import time
import glob
import os
import json
import shutil  # Make sure to import shutil at the top of your file
import itertools
from distributed_actors import create_actor_and_learner

class Trainer:
    def __init__(self, dataset, reward_function, config):
        self.dataset = dataset
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
        self.actors, self.learner = create_actor_and_learner(config["number_of_actors"], config["model"], self.gen_config, config)
        assert len(self.actors) == config["number_of_actors"]
        self.num_actors = config["number_of_actors"]


        # set params
        self.episodes = config["episodes"]
        self.batch_size = config["batch_size"]
        self.num_candidates = config["num_candidates"]
        self.max_monkey_rounds = config["max_monkey_rounds"]
        self.save_every = config["save_every"]
        self.eval_every = config["eval_every"]
        self.keep_last_x = config["keep_last_x"]

        self.run_name = config["run_name"]
        self.project_name = config["project_name"]

    def save_solutions(self, problems, solutions, answers, rewards, reward_threshold=0.5):
        with open("solutions.json", "a") as f: 
            for s, p, a, r in zip(solutions, problems, answers, rewards):
                if r > reward_threshold:
                    record={'problem': p, 'solution': s, 'answer': a, 'reward': r}
                    f.write(json.dumps(record) + "\n")
                    f.flush() 

    def save_adapter_with_limit(self, total_batch_steps):
        # Save the adapter
        model_path = f"models/chem_pg_model_{total_batch_steps}"
        ray.get(self.learner.save_adapter.remote(model_path))

        # Check and remove older models if necessary
        saved_models = sorted(glob.glob("models/chem_pg_model_*"), key=os.path.getmtime)
        
        # Identify models that should be preserved (those saved due to save_every)
        preserved_models = {f"models/chem_pg_model_{i}" for i in range(0, total_batch_steps + 1, self.save_every)}

        # Remove old models that are not in the preserved set
        if len(saved_models) > self.keep_last_x:
            for model in saved_models[:-self.keep_last_x]:
                if model not in preserved_models:
                    shutil.rmtree(model)  # Use shutil.rmtree to remove directories
                    print(f"Removed old model directory: {model}")

    def compute_rewards(self, candidates, solutions):
        return self.reward_function(candidates, solutions)

    def adapt_batch_size(self, batch, batch_size):
        target_size = self.num_actors + 1 # +1 for the learner as we want to utilize the learner when its not updating weights
        if self.batch_size != target_size:
            # Calculate how many times each element should be repeated
            repeats = (target_size + self.batch_size - 1) // self.batch_size
            
            # Create new batch by repeating each element appropriately
            new_batch = {}
            for k in batch:
                extended = []
                for item in batch[k]:
                    extended.extend([item] * repeats)
                # Trim to exactly match target_size
                new_batch[k] = extended[:target_size]
            batch = new_batch
            batch_size = target_size
        else:
            batch_size = self.batch_size
        return batch, batch_size
    
    def _generate_all_candidates(self, batch):
        batch, batch_size = self.adapt_batch_size(batch, self.batch_size)
        
        # Initialize data structures
        candidate_data = {
            'ranked_lists': [RankedLists(sort_index=0) for _ in range(batch_size)],
            'step_solutions': [[batch["solution"][i]] * self.num_candidates for i in range(batch_size)],
            'candidates': [[] for _ in range(batch_size)],
            'token_data': [[] for _ in range(batch_size)],
            'rewards': [[] for _ in range(batch_size)],
            'problems': [[batch["problem"][i]] * (self.num_candidates * self.max_monkey_rounds) for i in range(batch_size)],
            'solutions': [[batch["solution"][i]] * (self.num_candidates * self.max_monkey_rounds) for i in range(batch_size)]
        }
        
        timings = {'generation': 0, 'reward': 0}
        
        # Generate and evaluate candidates for each round
        for _ in range(self.max_monkey_rounds):
            self._generate_round(batch, candidate_data, timings)
            self._compute_round_rewards(candidate_data, timings)
        
        # Combine results
        self._combine_results(candidate_data)
        
        return candidate_data['ranked_lists'], timings['generation'], timings['reward']

    def _generate_round(self, batch, data, timings):
        """Generate one round of candidates using actors and learner"""
        start_time = time.time()
        
        # Prepare and execute generation tasks
        actor_tasks = [actor.generate.remote(problem) 
                    for actor, problem in zip(self.actors, batch["problem"][:self.num_actors])]
        learner_task = self.learner.generate.remote(batch["problem"][-1])
        
        # Get results with timeout
        generations = ray.get(actor_tasks + [learner_task], timeout=240)
        
        # Store generated candidates and tokens
        for candidates, tokens, gen in zip(data['candidates'], data['token_data'], generations):
            candidates.extend(gen[0])
            tokens.extend(gen[1])
        
        timings['generation'] += time.time() - start_time

    def _compute_round_rewards(self, data, timings):
        """Compute rewards for the current round of candidates"""
        start_time = time.time()
        
        for candidates, solutions, rewards in zip(data['candidates'], 
                                            data['step_solutions'], 
                                            data['rewards']):
            rewards.append(self.compute_rewards(candidates, solutions))
        
        timings['reward'] += time.time() - start_time

    def _combine_results(self, data):
        """Combine all generated candidates and their rewards into ranked lists"""
        for ranked_list, rewards, problems, solutions, candidates, tokens in zip(
                data['ranked_lists'],
                data['rewards'],
                data['problems'],
                data['solutions'],
                data['candidates'],
                data['token_data']):
            ranked_list.add((np.vstack(rewards), problems, solutions, candidates, tokens))

    def train(self):
        total_batch_steps = 0

        # initialize wandb
        run = wandb.init(name=self.run_name, config=self.config, project=self.project_name)

        for episode in tqdm(range(self.episodes), desc="PG Training ..."):
            self.dataset = self.dataset.shuffle()
            loader = self.dataset.iter(batch_size=self.batch_size)

            for batch in loader:
                total_batch_steps += 1
                # generate responses
                candidates, generation_duration, reward_duration = self._generate_all_candidates(batch)

                all_rewards, all_problems, all_solutions, all_candidates, all_token_lengths = [], [], [], [], []
                for c in candidates:
                    c_rewards, c_problems, c_solutions, c_candidates, c_token_lengths = c.get_all()
                    c_rewards = np.array(c_rewards)
                    self.save_solutions(problems=c_problems, solutions=c_solutions, answers=c_candidates, rewards=c_rewards[:, 1])
                    all_rewards.append(c_rewards)
                    all_problems.append(c_problems)
                    all_solutions.append(c_solutions)
                    all_candidates.append(c_candidates)
                    all_token_lengths.append(c_token_lengths)

                # 5. update policy
                all_rewards = np.vstack(all_rewards)
                update_start_time = time.time()
                loss = ray.get(self.learner.train.remote(list(itertools.chain.from_iterable(all_problems)), list(itertools.chain.from_iterable(all_candidates)), all_rewards))
                update_duration = time.time() - update_start_time

                # Save adapter
                self.save_adapter_with_limit(total_batch_steps)
                
                load_start_time = time.time()
                # Load adapter for actors
                try:
                    load_futures = [actor.load_adapter.remote(f"models/chem_pg_model_{total_batch_steps}") for actor in self.actors]
                    # Wait for all load requests to complete
                    ray.get(load_futures, timeout=120)
                except Exception as e:
                    print(f"Error loading adapter: {str(e)}")
                load_duration = time.time() - load_start_time

                run.log(
                    {
                        "loss": loss,
                        "mean_format_reward":  np.mean(all_rewards[:, 0]).item(),
                        "mean_accuracy_reward": np.mean(all_rewards[:, 1]).item(),
                        "min_accuracy_reward": np.min(all_rewards[:, 1]).item(),
                        "max_accuracy_reward": np.max(all_rewards[:, 1]).item(),
                        "mean_token_length": np.mean(np.array(all_token_lengths)).item(),
                        "episode": episode,
                        "total_batch_steps": total_batch_steps,
                        "timing/update_duration": update_duration,
                        "timing/reward_duration": reward_duration,
                        "timing/generation_duration": generation_duration,
                        "timing/load_duration": load_duration,
                    },
                    step=total_batch_steps,
                )

            # save policy
            if total_batch_steps % self.save_every == 0:
                ray.get(self.learner.save_adapter.remote(f"models/chem_pg_model_{total_batch_steps}"))


            # save final policy
            ray.get(self.learner.save_adapter.remote(f"models/chem_pg_model_{total_batch_steps}"))
        # cleanup    
        ray.shutdown()
