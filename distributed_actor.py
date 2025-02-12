import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from unsloth import FastLanguageModel
import torch
from helper import init_peft_model
import bitsandbytes as bnb
import numpy as np
import torch.nn.functional as F
from vllm import LLM, SamplingParams


MAX_SEQ_LENGTH = 1024
DTYPE = torch.bfloat16
LOAD_IN_4BIT = True

# Function used from trl.trainer.utils import selective_log_softmax
def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


class BaseActor:
    def __init__(self, actor_type, model_name, gpu_id, config, gen_config=None, gpu_usage=0.7):
        self.actor_type = actor_type
        self.max_seq_length = MAX_SEQ_LENGTH
        self.dtype = DTYPE
        self.load_in_4bit = LOAD_IN_4BIT
        self.model_gpu_id = gpu_id
        self.update_batch_size = config["train_batch_size"]
        self.gen_config = gen_config
        self.cot_decoding = config["cot_decode"]
        self.max_new_tokens = config["max_new_tokens"]
        self.num_candidates = gen_config.num_return_sequences
        self.use_vllm = config["use_vllm"]
        self.max_lora_rank = 32
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
                fast_inference = self.use_vllm, # Enable vLLM fast inference
                max_lora_rank = self.max_lora_rank,
                gpu_memory_utilization = self.max_gpu_usage, # Reduce if out of memory

            )

            # Initialize PEFT model
            self.policy = init_peft_model(model, lora_rank=self.max_lora_rank)
            
            # Verify initialization
            if self.policy is None:
                raise RuntimeError("Failed to initialize policy model")

            print(f"{self.actor_type} initialized")
            print(f"{self.actor_type} device: {self.policy.device}")
        except Exception as e:
            print(f"Error initializing {self.actor_type} on GPU {self.model_gpu_id}: {str(e)}")
            # Clean up any partially initialized state
            raise

    def load_adapter(self, adapter_name):
        peft_config = self.policy.active_peft_config
        old_adapter = self.policy.active_adapter  # Get the current active adapter
        self.policy.add_adapter(adapter_name, peft_config)
        self.policy.set_adapter(adapter_name)
        print(f"Adapter loaded for {self.actor_type} {self.model_gpu_id}")
        if old_adapter:
            self.policy.delete_adapter(old_adapter)  # Remove the old adapter


    def save_adapter(self, adapter_name):
        self.policy.save_pretrained(adapter_name)
        self.tokenizer.save_pretrained(adapter_name)
        print(f"Adapter saved for {self.actor_type} {self.model_gpu_id}")

    def genCoTDecoding(self, inputs):

        outputs = self.policy(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        first_step_logits = outputs.logits[:, -1, :]
        
        top_k_first_tokens = torch.topk(first_step_logits, k=10).indices
        
        # Randomly select self.num_candidates from top_k_first_tokens
        random_indices = torch.randint(0, top_k_first_tokens.size(1), (self.num_candidates,))  # Shape: (num_candidates,)
        init_tokens = top_k_first_tokens[0, random_indices]  # Select tokens based on random indices
        
        # Reshape init_tokens to match: [batch_size, num_candidates, 1]
        init_tokens = init_tokens.view(-1, 1)  # Reshape to (64, 1)
        
        # Reshape input_ids to accommodate batch and candidate dimensions
        expanded_inputs = inputs["input_ids"].expand(self.num_candidates, -1)
                
        # Flatten batch and candidate dimensions for concatenation
        new_inputs = torch.cat([expanded_inputs, init_tokens], dim=-1)
        outputs = self.policy.generate(
            input_ids=new_inputs,
            max_new_tokens=self.gen_config.max_new_tokens - 1,  # as we generated one already: -1
            temperature=self.gen_config.temperature,
            num_return_sequences=1,  # as we have split
        )
        return outputs

    @torch.no_grad()
    def base_generate(self, messages):
        try:
            # Encode prompt
            inputs = self.tokenizer.batch_encode_plus([messages], return_tensors="pt", padding="longest").to("cuda")
            input_lens = inputs["attention_mask"].sum(-1).max() # we only need to know the longest

            if self.cot_decoding:
                outputs = self.genCoTDecoding(inputs)
            else:
                print(f"Generating...")
                outputs = self.policy.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=self.gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                )
            only_answer_tokens = outputs[:, input_lens:,]
            out_texts = []
            token_lengths = []
            for answer in only_answer_tokens:
                out_texts.append(self.tokenizer.decode(answer, skip_special_tokens=True))
                token_lengths.append((answer != self.tokenizer.pad_token_id).sum().item())
                #token_lengths.append(len(answer))
            torch.cuda.empty_cache()
            return out_texts, token_lengths
        except Exception as e:
            print(f"Detailed error: {str(e)}")
            return [""], [0]

    def vllm_generate(self, messages):
        completions = self.policy.fast_generate(messages, self.sampling_params)
        # stack outputs
        out_texts = []
        token_lengths = []
        for o in completions[0].outputs:
            out_texts.append(o.text)
            token_lengths.append(len(o.token_ids))
        return out_texts, token_lengths

    
    def generate(self, messages):
        # TODO: check if messages is always a single sting!
        FastLanguageModel.for_inference(self.policy)
        if self.use_vllm:
            return self.vllm_generate(messages)
        else:
            return self.base_generate(messages)


@ray.remote(num_gpus=1, num_cpus=1)
class Generator(BaseActor):
    def __init__(self, model_name, gpu_id, config, gen_config=None, gpu_usage=0.8):
        super().__init__(actor_type="Generator", model_name=model_name, gpu_id=gpu_id, config=config, gen_config=gen_config, gpu_usage=gpu_usage)


@ray.remote(num_gpus=1, num_cpus=1)
class Learner(BaseActor):
    def __init__(self, model_name, gpu_id, config, gen_config=None, gpu_usage=0.7):
        super().__init__(actor_type="Learner", model_name=model_name, gpu_id=gpu_id, config=config, gen_config=gen_config, gpu_usage=gpu_usage)
        # Note: With 8-bit optimizers, large models can be finetuned with 75% less GPU memory without losing any accuracy compared to training with standard 32-bit optimizer
        # https://huggingface.co/docs/bitsandbytes/optimizers
        self.optimizer = bnb.optim.Adam8bit(self.policy.parameters(), lr=config["lr"], weight_decay=0.01)
        self.batch_size = config["batch_size"]
        self.top_k_train = config["top_k_train"]

    def compute_current_policy_probs(self, policy, messages, answers):
        # recreate input with question and answer but we only compute the probabilities for the answer
        inputs = self.tokenizer.batch_encode_plus(messages, return_tensors="pt", padding="longest", padding_side="left").to("cuda")
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
        logits_to_keep = tokenized_answer["input_ids"].size(1)
        # combine inputs and candidates
        full_inputs = torch.cat([inputs["input_ids"], tokenized_answer["input_ids"]], dim=1)
        full_attention_mask = torch.cat(
            [inputs["attention_mask"], tokenized_answer["attention_mask"]],
            dim=1,
        )

        # Compute logits of the policy model
        logits = policy(
            input_ids=full_inputs, attention_mask=full_attention_mask, logits_to_keep=logits_to_keep+1
        ).logits

        logits = logits[:, :-1, :]  # shift the logits to the right by one token
        input_ids = full_inputs[:, -logits_to_keep:]
        #input_ids = full_inputs[:, 1:]  # remove the first token
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids), answer_mask

    def compute_loss(self, messages, answers, rewards, baseline=None):
        if baseline is None:
            baseline = 0
        rewards = torch.tensor(rewards).to("cuda")
        rewards = rewards.sum(dim=1)
        
        total_loss = 0
        batch_size = len(messages)
        num_batches = (batch_size + self.update_batch_size - 1) // self.update_batch_size
        
        self.optimizer.zero_grad()
        
        for i in range(num_batches):
            start_idx = i * self.update_batch_size
            end_idx = min((i + 1) * self.update_batch_size, batch_size)
            
            batch_messages = messages[start_idx:end_idx]
            batch_answers = answers[start_idx:end_idx]
            batch_rewards = rewards[start_idx:end_idx]
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_probs, mask = self.compute_current_policy_probs(
                    self.policy, batch_messages, batch_answers
                )
                #print(f"Batch {i+1}/{num_batches} Log probs: {log_probs.mean()}")
                # Compute loss for current batch
                loss = -(log_probs * (batch_rewards-baseline).unsqueeze(-1))
                
                # mask padding parts out and normalize and take mean over batch
                loss = ((loss * mask).sum(-1) / mask.sum(-1)).mean()

                # Scale the loss by the number of batches to maintain the same overall magnitude
                loss = loss / num_batches
                
                # Accumulate gradients
                loss.backward()
                
                total_loss += loss.item() * num_batches  # Unscale the loss for return value
        
        # Update parameters after accumulating gradients from all batches
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return total_loss

    def train(self, messages, answers, rewards):
        FastLanguageModel.for_training(self.policy)

        # split per batch size and do topk sampling

        mean_format = np.mean(rewards[:, 0])
        mean_accuracy = np.mean(rewards[:, 1])
        baseline = mean_format + mean_accuracy
        return self.compute_loss(messages, answers, rewards, baseline)

@ray.remote(num_gpus=1, num_cpus=1)
class GRPOLearner(BaseActor):
    def __init__(self, model_name, gpu_id, config, gen_config=None, gpu_usage=0.7):
        super().__init__(actor_type="Learner", model_name=model_name, gpu_id=gpu_id, config=config, gen_config=gen_config, gpu_usage=gpu_usage)
        # Note: With 8-bit optimizers, large models can be finetuned with 75% less GPU memory without losing any accuracy compared to training with standard 32-bit optimizer
        # https://huggingface.co/docs/bitsandbytes/optimizers
        self.optimizer = bnb.optim.Adam8bit(self.policy.parameters(), lr=config["lr"], weight_decay=0.1)
        self.beta = 0.1 #0.4

    def compute_current_policy_probs(self, policy, messages, answers):
        # recreate input with question and answer but we only compute the probabilities for the answer
        inputs = self.tokenizer.batch_encode_plus(messages, return_tensors="pt", padding="longest", padding_side="left").to("cuda")
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
        logits_to_keep = tokenized_answer["input_ids"].size(1)
        # combine inputs and candidates
        full_inputs = torch.cat([inputs["input_ids"], tokenized_answer["input_ids"]], dim=1)
        full_attention_mask = torch.cat(
            [inputs["attention_mask"], tokenized_answer["attention_mask"]],
            dim=1,
        )

        # Compute logits of the policy model
        logits = policy(
            input_ids=full_inputs, attention_mask=full_attention_mask, logits_to_keep=logits_to_keep+1 # for shifting
        ).logits

        logits = logits[:, :-1, :]  # shift the logits to the right by one token
        input_ids = full_inputs[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids), answer_mask

    def compute_loss(self, messages, answers, advantage):
        advantage = torch.tensor(advantage).to("cuda")
        
        total_loss = 0
        batch_size = len(messages)
        num_batches = (batch_size + self.update_batch_size - 1) // self.update_batch_size
        
        self.optimizer.zero_grad()
        for i in range(num_batches):
            start_idx = i * self.update_batch_size
            end_idx = min((i + 1) * self.update_batch_size, batch_size)
            
            batch_messages = messages[start_idx:end_idx]
            batch_answers = answers[start_idx:end_idx]
            batch_advantages = advantage[start_idx:end_idx]
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # compute reference logprobs
                with torch.no_grad():
                    with self.policy.disable_adapter():
                        ref_log_prob, mask = self.compute_current_policy_probs(
                        self.policy, batch_messages, batch_answers
                    )
                        
                log_probs, mask = self.compute_current_policy_probs(
                    self.policy, batch_messages, batch_answers
                )
                #print(f"Batch {i+1}/{num_batches} Log probs: {log_probs.mean()}")
                

                per_token_kl = torch.exp(ref_log_prob - log_probs) - (ref_log_prob - log_probs) - 1
                #print("KL: ", per_token_kl.mean())
                # Compute loss for current batch
                loss = torch.exp(log_probs - log_probs.detach()) * batch_advantages.unsqueeze(1)

                loss = -(loss - self.beta * per_token_kl)

                loss = ((loss * mask).sum(-1) / mask.sum(-1)).mean()
                
                # Scale the loss by the number of batches to maintain the same overall magnitude
                loss = loss / num_batches
                
                # Accumulate gradients
                loss.backward()
                
                total_loss += loss.item() * num_batches  # Unscale the loss for return value
        
        # Update parameters after accumulating gradients from all batches
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return total_loss

    def train(self, messages, answers, rewards):
        FastLanguageModel.for_training(self.policy)
        advantage = ((rewards - np.mean(rewards, axis=0))/ (np.std(rewards, axis=0) + 1e-10)).sum(axis=-1)
        return self.compute_loss(messages, answers, advantage)




def create_actor_and_learner(number_of_actors=1, model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit", gen_config=None, config=None):
    """
    Run the script with specified GPUs.
    Args:
        gpu_ids: List of GPU IDs to use (e.g., [0, 1] for first two GPUs)
    """
    available_gpus = list(range(torch.cuda.device_count()))
    if len(available_gpus) < number_of_actors + 1:
        raise RuntimeError(f"Not enough GPUs available. Available: {len(available_gpus)}, Required: {number_of_actors + 1}")
    
    # Use the next available GPUs for actors
    gpu_ids = available_gpus[:number_of_actors]
    # Use the last available GPU for the learner
    learner_gpu_id = available_gpus[number_of_actors]
    
    print(f"Using GPUs for actors: {gpu_ids}")
    print(f"Using GPU for learner: {learner_gpu_id}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create placement group with specific GPU mappings
    pg = placement_group(
        name="llm_pg",
        bundles=[{
            "GPU": 1,
            "CPU": 1
        } for _ in range(number_of_actors+1)], # +1 for learner
        strategy="STRICT_PACK"
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
                placement_group=pg,
                placement_group_bundle_index=i
            )
        ).remote(model_name, gpu_id, config, gen_config, gpu_usage=0.85)
        actors.append(actor)

    assert config["learner"] in ["grpo", "pg"], "Learner can be only 'pg' or 'grpo'!"
    # Create Learner
    if config["learner"] == "grpo":
        LEARNER = GRPOLearner
    else:
        LEARNER = Learner
    learner = LEARNER.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=number_of_actors
        )
    ).remote(model_name, learner_gpu_id, config, gen_config, gpu_usage=0.65)

    return actors, learner
