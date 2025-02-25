import argparse

import numpy as np
from datasets import load_dataset
from distributed_trainer import Trainer
from helper import process_dataset
from prompts import r0_preprompt, r1_preprompt
from reward_function import EVAL_FUNCTIONS, extract_between_tokens
from unsloth.tokenizer_utils import load_correct_tokenizer

REWARD_FORMAT = 0.1
REWARD_FORMAT_PARSED = 0.1

def format_reward(completions):
    """Reward function that checks if the completion has a specific format."""
    format_rewards = []
    extracted_answers = []
    extracted_thoughts = []
    for c in completions:
        think = extract_between_tokens(c, "<think>", "</think>")
        answer = extract_between_tokens(c, "<answer>", "</answer>")
        reward = REWARD_FORMAT
        if len(answer) == 0 or len(think) == 0:#one of the two was not parsed correctly
            reward = 0.0
            think = answer = ""
        format_rewards.append(reward)
        extracted_answers.append(answer)
        extracted_thoughts.append(think)
    return format_rewards, extracted_answers, extracted_thoughts


def assign_rewards(solutions, extracted_answers, extracted_thoughts):
    """Reward function that checks if the completion is the same as the ground truth."""
    accuracy_rewards = []
    format_rewards = []
    problem_types = []

    for answer, sol in zip(extracted_answers, solutions):
        try:
            fxn_name, ideal, problem_type = sol.split("!:!", 2)
            if answer == "": # extracted answer not parsed correctly
                format_reward = 0.0
                accuracy_reward = 0.0
            else: # parsed correctly
                reward = EVAL_FUNCTIONS[fxn_name](answer, ideal)
                format_reward = REWARD_FORMAT
                accuracy_reward = reward
        except Exception: # eval function failed, but it did parse correctly
            format_reward = REWARD_FORMAT_PARSED
            accuracy_reward = 0.0
            problem_type = None

        format_rewards.append(format_reward)
        accuracy_rewards.append(accuracy_reward)
        problem_types.append(problem_type)

    return accuracy_rewards, format_rewards, problem_types


def reward_function(completions, solutions):
    _, extracted_answers, extracted_thoughts = format_reward(completions=completions)
    acc_reward, form_reward, problem_types = assign_rewards(
        solutions=solutions,
        extracted_answers=extracted_answers,
        extracted_thoughts=extracted_thoughts,
    )
    return np.array(list(zip(form_reward, acc_reward)))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model", type=str, default="unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    )
    args.add_argument("--dataset", type=str, default="easy", choices=["train", "easy"])
    args.add_argument(
        "--task",
        type=str,
        default="reaction-prediction",
        choices=["all", "molecule-completion", "reaction-prediction", "molecule-name"],
    )
    args.add_argument("--run_name", type=str)  # forcing to be set by user
    args.add_argument("--project_name", type=str, default="chem-reason")
    args.add_argument("--lr", type=float, default=4e-5)
    args.add_argument("--max_new_tokens", type=int, default=1000)
    args.add_argument("--max_prompt_tokens", type=int, default=331) # 303 for easy
    args.add_argument("--temperature", type=float, default=0.8)
    args.add_argument("--episodes", type=int, default=20)
    args.add_argument(
        "--num_candidates",
        type=int,
        default=6,
        help="Number of sampled candidate per monkey iteration",
    )
    args.add_argument("--batch_size", type=int, default=8, help="Total batch size for all actors and learner that is later split into chunks") # 224 total per learner 
    args.add_argument("--learner_chunk_size", type=int, default=1, help="Number of samples per learner chunk")
    args.add_argument("--train_batch_size", type=int, default=4)
    args.add_argument(
        "--max_monkey_rounds",
        type=int,
        default=1,
        help="Number of generation rounds to sample candidates",
    )
    args.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save the model every x training steps",
    )
    args.add_argument(
        "--eval_every",
        type=int,
        default=100,
        help="Evaluate the model every x training steps",
    )
    args.add_argument(
        "--number_of_actors",
        type=int,
        default=3,
        help="Number of actors to use, default is 0. Only uses the learner to generate and train.",
    )
    args.add_argument(
        "--keep_last_x",
        type=int,
        default=10,
        help="Number of last model checkpoints to keep",
    )
    args.add_argument("--learner", type=str, choices=["pg", "grpo"], default="pg")
    args.add_argument("--max_lora_rank", type=int, default=32)
    args.add_argument("--topk", type=int, default=16)

    args = args.parse_args()

    if args.number_of_actors == 0:
        assert args.batch_size == 1, "Batch size must be 1 if number of actors is 0"

    raw_dataset = load_dataset("Acellera/molrqa", token="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")
    
    if args.task != "all":
        raw_dataset = raw_dataset.filter(lambda x: args.task in x["problem_type"])

    raw_train_dataset = raw_dataset[args.dataset]  #easy or train
    # if args.select_train_range is not None:
    #     start, end, step = map(int, args.select_train_range.split(","))
    #     raw_train_dataset = raw_train_dataset.select(range(start, end, step))
    #     print(f"Selected train range: {start} to {end} with step {step}")

    raw_eval_dataset = raw_dataset["test"].select(range(0, 100))  
    
    print(f"\nNumber of train samples: {len(raw_train_dataset)}\n\n")    


    config = {
        "run_name": args.run_name,
        "project_name": args.project_name,
        "lr": args.lr,
        "max_new_tokens": args.max_new_tokens,
        "episodes": args.episodes,
        "num_candidates": args.num_candidates,
        "batch_size": args.batch_size,
        "train_batch_size": args.train_batch_size,
        "temperature": args.temperature,
        "max_monkey_rounds": args.max_monkey_rounds,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "model": args.model,
        "dataset": args.dataset,
        "number_of_actors": args.number_of_actors,
        "keep_last_x": args.keep_last_x,
        "learner": args.learner,
        "use_vllm": True,
        "max_lora_rank": args.max_lora_rank,
        "topk": args.topk,
        "learner_chunk_size": args.learner_chunk_size,
    }

    # TODO: ugly that we need to load the tokenizer before
    tokenizer = load_correct_tokenizer(args.model)
    # process dataset
    if args.task == "reaction-prediction":
        postprompt = "Answer with the SMILES of the product."
    else:
        postprompt = ""

    train_dataset = process_dataset(tokenizer, raw_train_dataset, r1_preprompt, postprompt)

    trainer = Trainer(train_dataset, reward_function, config)
    trainer.train()
