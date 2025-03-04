from itertools import islice
import time
import bitsandbytes as bnb
import numpy as np
import ray
import torch
import torch.nn.functional as F
from helper import init_peft_model
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from unsloth import FastLanguageModel
from unsloth_zoo.vllm_utils import load_lora, save_lora
from vllm import SamplingParams
from tqdm import tqdm

DTYPE = torch.bfloat16
LOAD_IN_4BIT = True


class BaseActor:
    def __init__(
        self, actor_type, model_name, gpu_id, config, gen_config=None, gpu_usage=0.7
    ):
        self.actor_type = actor_type
        self.max_seq_length = config["max_prompt_tokens"] + config["max_new_tokens"]
        self.dtype = DTYPE
        self.load_in_4bit = LOAD_IN_4BIT
        self.model_gpu_id = gpu_id
        self.update_batch_size = config["train_batch_size"]
        self.gen_config = gen_config
        self.max_new_tokens = config["max_new_tokens"]
        self.num_candidates = gen_config.num_return_sequences
        self.use_vllm = config["use_vllm"]
        self.max_lora_rank = config["max_lora_rank"]
        self.lora_save_path = config["lora_save_path"]
        self.max_gpu_usage = gpu_usage
        assert self.max_new_tokens == self.gen_config.max_new_tokens

        # adapt generation config for vllm
        if self.use_vllm:
            self.sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.gen_config.temperature,
                n=self.num_candidates,
            )

        self.policy = None
        self.tokenizer = None
        self._initialize_model(model_name)

    def _initialize_model(self, model_name):
        try:

            # Initialize model and tokenizer
            model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
                fast_inference=self.use_vllm,  # Enable vLLM fast inference
                max_lora_rank=self.max_lora_rank,
                gpu_memory_utilization=self.max_gpu_usage,  # Reduce if out of memory
            )

            # Initialize PEFT model
            self.policy = init_peft_model(model, lora_rank=self.max_lora_rank)

            # Verify initialization
            if self.policy is None:
                raise RuntimeError("Failed to initialize policy model")

            print(f"{self.actor_type} initialized")
            print(f"{self.actor_type} device: {self.policy.device}")
        except Exception as e:
            print(
                f"Error initializing {self.actor_type} on GPU {self.model_gpu_id}: {str(e)}"
            )
            # Clean up any partially initialized state
            raise

    def save_adapter(self, ):
        save_lora(self.policy, self.lora_save_path)
        print(f"Adapter saved for {self.actor_type} {self.model_gpu_id} at {self.lora_save_path}")

    @staticmethod
    def combine_lists(list_of_lists, num_candidate):
        result = []
        current_group = []
        
        for sublist in list_of_lists:
            current_group.append(sublist[0])  # Assuming each sublist has only one element
            
            if len(current_group) == num_candidate:
                result.append(current_group)
                current_group = []
        
        # Handle any remaining items if list length isn't a multiple of num_candidate
        if current_group:
            result.append(current_group)
        
        return result
    
    @torch.no_grad()
    def base_generate(self, messages):
        raise NotImplementedError("Base generate needs to be updated")
        try:
            # Encode prompt
            inputs = self.tokenizer.batch_encode_plus(
                [messages], return_tensors="pt", padding="longest"
            ).to("cuda")
            input_lens = (
                inputs["attention_mask"].sum(-1).max()
            )  # we only need to know the longest

            print(f"Generating...")
            outputs = self.policy.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.gen_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
            )
            only_answer_tokens = outputs[
                :,
                input_lens:,
            ]
            out_texts = []
            token_lengths = []
            for answer in only_answer_tokens:
                out_texts.append(
                    self.tokenizer.decode(answer, skip_special_tokens=True)
                )
                token_lengths.append(
                    (answer != self.tokenizer.pad_token_id).sum().item()
                )
                # token_lengths.append(len(answer))
            torch.cuda.empty_cache()
            return out_texts, token_lengths
        except Exception as e:
            print(f"Detailed error: {str(e)}")
            return [""], [0]

    def vllm_generate(self, task, sampling_params=None):
        completions = self.policy.fast_generate(task["problem"],
                                                sampling_params=self.sampling_params if sampling_params is None else sampling_params,
                                                lora_request=load_lora(self.policy, self.lora_save_path))
        sampled_candidates = sampling_params.n if sampling_params is not None else self.num_candidates
        # stack outputs
        total_out_texts = []
        total_token_lengths = []
        for c in completions:
            task_texts = []
            task_token_lengths = []
            for o in c.outputs:
                task_texts.append(o.text)
                task_token_lengths.append(len(o.token_ids))

            total_out_texts.append(task_texts)
            total_token_lengths.append(task_token_lengths)

        task["answers"] = total_out_texts
        task["token_lengths"] = total_token_lengths
        # adapt solutions and task id and problem
        # for solution in task["solution"] repeat it self.num_candidates times so we have lists per task, problem, solution etc
        task["solution"] = [[s for _ in range(sampled_candidates)] for s in task["solution"]]
        task["problem"] = [[p for _ in range(sampled_candidates)] for p in task["problem"]]

        return task

    def generate(self, messages, sampling_params=None):
        # TODO: check if messages is always a single sting!
        FastLanguageModel.for_inference(self.policy)
        if self.use_vllm:
            return self.vllm_generate(messages, sampling_params)
        else:
            return self.base_generate(messages)


@ray.remote(num_gpus=1, num_cpus=1)
class Generator(BaseActor):
    def __init__(self, model_name, gpu_id, config, gen_config=None, gpu_usage=0.8):
        super().__init__(
            actor_type="Generator",
            model_name=model_name,
            gpu_id=gpu_id,
            config=config,
            gen_config=gen_config,
            gpu_usage=gpu_usage,
        )


@ray.remote(num_gpus=1, num_cpus=1)
class Learner(BaseActor):
    def __init__(self, model_name, gpu_id, config, gen_config=None, gpu_usage=0.7):
        super().__init__(
            actor_type="Learner",
            model_name=model_name,
            gpu_id=gpu_id,
            config=config,
            gen_config=gen_config,
            gpu_usage=gpu_usage,
        )
        # Note: With 8-bit optimizers, large models can be finetuned with 75% less GPU memory without losing any accuracy compared to training with standard 32-bit optimizer
        # https://huggingface.co/docs/bitsandbytes/optimizers
        # self.optimizer = torch.optim.Adam(
        #     self.policy.parameters(), lr=config["lr"]#, weight_decay=0.01
        # )
        self.optimizer = bnb.optim.Adam8bit(
            self.policy.parameters(), lr=config["lr"]#, weight_decay=0.1
        )
        self.batch_size = config["batch_size"]
        self.max_prompt_tokens = config["max_prompt_tokens"]

    def compute_current_policy_probs(self, policy, messages, answers):
        # recreate input with question and answer but we only compute the probabilities for the answer
        inputs = self.tokenizer.batch_encode_plus(
            messages, return_tensors="pt", padding="max_length", padding_side="left", max_length=self.max_prompt_tokens, truncation=True
        ).to("cuda")
        #input_lens = inputs["attention_mask"].sum(-1).max()
        input_lens = self.max_prompt_tokens
        tokenized_answer = self.tokenizer.batch_encode_plus(
            answers,
            return_tensors="pt",
            max_length=self.max_new_tokens,
            padding="max_length",
            padding_side="right",
            truncation=True,
        ).to("cuda")
        answer_mask = tokenized_answer["attention_mask"]
        # we only compute the logprob for the answers saves memory
        # combine inputs and candidates
        full_inputs = torch.cat(
            [inputs["input_ids"], tokenized_answer["input_ids"]], dim=1
        )
        full_attention_mask = torch.cat(
            [inputs["attention_mask"], tokenized_answer["attention_mask"]],
            dim=1,
        )

        logits = policy(
            input_ids=full_inputs, attention_mask=full_attention_mask
        ).logits

        logits = logits[:, :-1, :]  # shift the logits to the right by one token
        input_ids = full_inputs[:, 1:]  # remove the first token
        # Compute logprob only on answer token, -1 for the shift
        logits = logits[:, input_lens - 1 :] 
        input_ids = input_ids[:, input_lens - 1 :]
        #print(f"Logits shape: {logits.shape}, Input ids shape: {input_ids.shape}")
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)

            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
        action_log_probs = torch.stack(per_token_logps)
        return action_log_probs, answer_mask

    def save_checkpoint(self, path):
        self.policy.save_pretrained(path)

    @staticmethod
    def compute_entropy_bonus(log_probs, alpha):
        """
        Computes the entropy bonus for reinforcement learning loss.

        Args:
            log_probs (torch.Tensor): Log probabilities of shape (batch_size, sequence_length, vocab_size).
            alpha (float): Entropy weighting coefficient.

        Returns:
            torch.Tensor: Scalar entropy bonus value.
        """
        probs = log_probs.exp()  # Convert log probabilities to probabilities
        entropy = -(probs * log_probs).sum(dim=-1)  # Sum over vocabulary dimension
        entropy_bonus = alpha * entropy.mean()  # Take mean over batch and sequence length
        return entropy_bonus

    def compute_loss(self, messages, answers, rewards):
        rewards = torch.tensor(rewards).to("cuda")

        total_loss = 0
        batch_size = len(messages)
        num_batches = (
            batch_size + self.update_batch_size - 1
        ) // self.update_batch_size

        self.optimizer.zero_grad()

        for i in tqdm(range(num_batches), desc="Update Policy..."):
            start_idx = i * self.update_batch_size
            end_idx = min((i + 1) * self.update_batch_size, batch_size)

            batch_messages = messages[start_idx:end_idx]
            batch_answers = answers[start_idx:end_idx]
            batch_rewards = rewards[start_idx:end_idx]
            if batch_rewards.all() == 0:
                # skip if batched rewards are 0
                continue
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_probs, mask = self.compute_current_policy_probs(
                    self.policy, batch_messages, batch_answers
                )
                # mask padding parts out and normalize and take mean over batch
                loss = -(((log_probs * mask).sum(-1) / mask.sum(-1)) * batch_rewards).mean()

                # TODO: test add entropy here
                #entropy = self.compute_entropy_bonus(log_probs, alpha=0.01)
                #loss = loss + entropy
                
                # Scale the loss by the number of batches to maintain the same overall magnitude
                loss = loss / num_batches

                # Accumulate gradients
                loss.backward()

                total_loss += (
                    loss.item() * num_batches
                )  # Unscale the loss for return value

        # # Update parameters after accumulating gradients from all batches
        # self.optimizer.step()
        # self.optimizer.zero_grad()

        return total_loss

    def train(self, candidates):
        FastLanguageModel.for_training(self.policy)
        problems = []
        answers = []
        rewards = []
        for candidate in candidates:
            for a, p, r, b in zip(candidate["answers"], candidate["problem"], candidate["rewards"], candidate["baselines"]):
                problems.extend(p)
                answers.extend(a)
                rewards.extend(r - b)
        print(f"Training on {len(problems)} samples")
        loss =  self.compute_loss(
            problems,
            answers,
            rewards,
        )
        # Update parameters after accumulating gradients from all batches
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    
    def _compute_gradients(self, problems, answers, rewards):
        """Computes gradients without applying weight updates."""
        self.optimizer.zero_grad()  # Reset gradients
        _ = self.compute_loss(problems, answers, rewards)

        # Collect gradients for all trainable LoRA parameters
        gradients = {
            name: param.grad.clone().detach()
            for name, param in self.policy.named_parameters()
            if param.requires_grad
        }
        return gradients

    def compute_gradients(self, candidates):
        FastLanguageModel.for_training(self.policy)
        problems = []
        answers = []
        rewards = []
        for candidate in candidates:
            for a, p, r in zip(candidate["answers"], candidate["problem"], candidate["rewards"]):
                problems.extend(p)
                answers.extend(a)
                rewards.extend(r)
        print(f"Learner {self.model_gpu_id}: Computing gradients on {len(problems)} samples.")
        return self._compute_gradients(problems, answers, rewards)
    
    def apply_merged_gradients(self, gradients_list):
        """Aggregates gradients from multiple learners and applies them."""
        if not gradients_list:
            print("No gradients to merge.")
            return

        # Initialize merged gradients
        merged_gradients = {name: torch.zeros_like(gradients_list[0][name]) for name in gradients_list[0]}

        # Sum gradients from all learners
        for gradients in gradients_list:
            for name in gradients:
                merged_gradients[name] += gradients[name]

        # Apply merged gradients
        for name, param in self.policy.named_parameters():
            if name in merged_gradients and param.requires_grad:
                param.grad = merged_gradients[name]

        # Perform optimizer step and clear gradients
        self.optimizer.step()
        self.optimizer.zero_grad()


@ray.remote(num_gpus=1, num_cpus=1)
class GRPOLearner(BaseActor):
    def __init__(self, model_name, gpu_id, config, gen_config=None, gpu_usage=0.7):
        super().__init__(
            actor_type="Learner",
            model_name=model_name,
            gpu_id=gpu_id,
            config=config,
            gen_config=gen_config,
            gpu_usage=gpu_usage,
        )
        # Note: With 8-bit optimizers, large models can be finetuned with 75% less GPU memory without losing any accuracy compared to training with standard 32-bit optimizer
        # https://huggingface.co/docs/bitsandbytes/optimizers
        self.optimizer = bnb.optim.Adam8bit(
            self.policy.parameters(), lr=config["lr"]#, weight_decay=0.1
        )
        self.batch_size = config["batch_size"]
        self.max_prompt_tokens = config["max_prompt_tokens"]

    def compute_current_policy_probs(self, policy, messages, answers):
        # recreate input with question and answer but we only compute the probabilities for the answer
        inputs = self.tokenizer.batch_encode_plus(
            messages, return_tensors="pt", padding="max_length", padding_side="left", max_length=self.max_prompt_tokens, truncation=True
        ).to("cuda")
        #input_lens = inputs["attention_mask"].sum(-1).max()
        input_lens = self.max_prompt_tokens
        tokenized_answer = self.tokenizer.batch_encode_plus(
            answers,
            return_tensors="pt",
            max_length=self.max_new_tokens,
            padding="max_length",
            padding_side="right",
            truncation=True,
        ).to("cuda")
        answer_mask = tokenized_answer["attention_mask"]
        # we only compute the logprob for the answers saves memory
        # combine inputs and candidates
        full_inputs = torch.cat(
            [inputs["input_ids"], tokenized_answer["input_ids"]], dim=1
        )
        full_attention_mask = torch.cat(
            [inputs["attention_mask"], tokenized_answer["attention_mask"]],
            dim=1,
        )

        logits = policy(
            input_ids=full_inputs, attention_mask=full_attention_mask
        ).logits

        logits = logits[:, :-1, :]  # shift the logits to the right by one token
        input_ids = full_inputs[:, 1:]  # remove the first token
        # Compute logprob only on answer token, -1 for the shift
        logits = logits[:, input_lens - 1 :] 
        input_ids = input_ids[:, input_lens - 1 :]
        #print(f"Logits shape: {logits.shape}, Input ids shape: {input_ids.shape}")
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)

            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
        action_log_probs = torch.stack(per_token_logps)
        return action_log_probs, answer_mask

    def save_checkpoint(self, path):
        self.policy.save_pretrained(path)

    @staticmethod
    def compute_entropy_bonus(log_probs, alpha):
        """
        Computes the entropy bonus for reinforcement learning loss.

        Args:
            log_probs (torch.Tensor): Log probabilities of shape (batch_size, sequence_length, vocab_size).
            alpha (float): Entropy weighting coefficient.

        Returns:
            torch.Tensor: Scalar entropy bonus value.
        """
        probs = log_probs.exp()  # Convert log probabilities to probabilities
        entropy = -(probs * log_probs).sum(dim=-1)  # Sum over vocabulary dimension
        entropy_bonus = alpha * entropy.mean()  # Take mean over batch and sequence length
        return entropy_bonus

    def compute_loss(self, messages, answers, rewards):
        rewards = torch.tensor(rewards).to("cuda")

        total_loss = 0
        batch_size = len(messages)
        num_batches = (
            batch_size + self.update_batch_size - 1
        ) // self.update_batch_size

        self.optimizer.zero_grad()

        for i in tqdm(range(num_batches), desc="Update Policy..."):
            start_idx = i * self.update_batch_size
            end_idx = min((i + 1) * self.update_batch_size, batch_size)

            batch_messages = messages[start_idx:end_idx]
            batch_answers = answers[start_idx:end_idx]
            batch_rewards = rewards[start_idx:end_idx]
            if batch_rewards.all() == 0:
                # skip if batched rewards are 0
                continue
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_probs, mask = self.compute_current_policy_probs(
                    self.policy, batch_messages, batch_answers
                )

                importance_logprob = torch.exp(log_probs - log_probs.detach())

                # mask padding parts out and normalize and take mean over batch
                loss = -(((importance_logprob * mask).sum(-1) / mask.sum(-1)) * batch_rewards).mean()

                # TODO: test add entropy here
                #entropy = self.compute_entropy_bonus(log_probs, alpha=0.01)
                #loss = loss + entropy
                
                # Scale the loss by the number of batches to maintain the same overall magnitude
                loss = loss / num_batches

                # Accumulate gradients
                loss.backward()

                total_loss += (
                    loss.item() * num_batches
                )  # Unscale the loss for return value

        # Update parameters after accumulating gradients from all batches
        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss

    def train(self, candidates):
        FastLanguageModel.for_training(self.policy)
        problems = []
        answers = []
        rewards = []
        for candidate in candidates:
            for a, p, r in zip(candidate["answers"], candidate["problem"], candidate["rewards"]):
                problems.extend(p)
                answers.extend(a)
                rewards.extend(r)
        print(f"Training on {len(problems)} samples")
        return self.compute_loss(
            problems,
            answers,
            rewards,
        )


def create_actor_and_learner(
    number_of_actors=1,
    number_of_learners=1,
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    gen_config=None,
    config=None,
):
    """
    Run the script with specified GPUs.
    Args:
        gpu_ids: List of GPU IDs to use (e.g., [0, 1] for first two GPUs)
    """
    available_gpus = list(range(torch.cuda.device_count()))
    if len(available_gpus) < number_of_actors + number_of_learners:
        raise RuntimeError(
            f"Not enough GPUs available. Available: {len(available_gpus)}, Required: {number_of_actors + number_of_learners}"
        )
    # Use the first available GPUs for actors
    gpu_ids = available_gpus[:number_of_actors]
    # Use the next available GPUs for learners
    learner_gpu_ids = available_gpus[number_of_actors:number_of_actors + number_of_learners]

    print(f"Using GPUs for actors: {gpu_ids}")
    print(f"Using GPUs for learners: {learner_gpu_ids}")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Create placement group with specific GPU mappings
    pg = placement_group(
        name="llm_pg",
        bundles=[
            {"GPU": 1, "CPU": 1} for _ in range(number_of_actors + number_of_learners)
        ],  # +1 for learner
        strategy="STRICT_PACK",
    )

    # Wait for placement group with timeout
    ready = ray.get(pg.ready(), timeout=60)
    if not ready:
        raise TimeoutError("Placement group creation timed out")

    # Initialize actors with specific GPU assignments
    # Create Actors
    actors = []
    for i, gpu_id in enumerate(gpu_ids):
        actor = Generator.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i
            )
        ).remote(model_name, gpu_id, config, gen_config, gpu_usage=config["actor_gpu_usage"])
        actors.append(actor)
    time.sleep(5)
    assert config["learner"] in ["grpo", "pg"], "Learner can be only 'pg' or 'grpo'!"
    # Create Learner
    if config["learner"] == "grpo":
        LEARNER = GRPOLearner
    else:
        LEARNER = Learner
    learner = []
    for i in range(number_of_learners):
        learner.append(LEARNER.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=number_of_actors + i
            )
        ).remote(model_name, learner_gpu_ids[i], config, gen_config, gpu_usage=config["learner_gpu_usage"]))

    return actors, learner
