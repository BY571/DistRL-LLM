import argparse

import numpy as np
from datasets import load_dataset
from distributed_trainer import Trainer
from helper import process_dataset, r1_preprompt
from unsloth.tokenizer_utils import load_correct_tokenizer
from reward_functions import reward_function


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    args.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500")
    args.add_argument("--run_name", type=str)
    args.add_argument("--project_name", type=str, default="math-reasoning")
    args.add_argument("--lora_save_path", type=str, default="lora_request_math")
    args.add_argument("--lr", type=float, default=2e-5)
    args.add_argument("--max_new_tokens", type=int, default=1200)
    args.add_argument("--max_prompt_tokens", type=int, default=350) # max: 865, mean 144
    args.add_argument("--temperature", type=float, default=0.8) # TODO: test >1 
    args.add_argument("--episodes", type=int, default=20)
    args.add_argument("--num_candidates", type=int, default=16, help="Number of sampled candidate per monkey iteration")
    args.add_argument("--batch_size", type=int, default=32, help="Total batch size for all actors and learner that is later split into chunks") # 224 total per learner 
    args.add_argument("--learner_chunk_size", type=int, default=6, help="Number of samples per learner chunk")
    args.add_argument("--train_batch_size", type=int, default=8)
    args.add_argument("--save_every", type=int, default=100, help="Save the model every x training steps")
    args.add_argument("--eval_every", type=int, default=10, help="Evaluate the model every x training steps")
    args.add_argument("--number_of_actors", type=int, default=2, help="Number of actors to use, default is 0. Only uses the learner to generate and train.")
    args.add_argument("--learner", type=str, choices=["pg", "grpo"], default="pg")
    args.add_argument("--max_lora_rank", type=int, default=64)
    args.add_argument("--topk", type=int, default=6, help="Number of top k generated candidates per task to consider to consider for training, filtered based on reward") 

    args = args.parse_args()

    if args.number_of_actors == 0:
        assert args.batch_size == 1, "Batch size must be 1 if number of actors is 0"

    raw_dataset = load_dataset(args.dataset)["test"] # math only has test
    
    # drop initial solution colum and rename answer column to solution
    raw_dataset = raw_dataset.map(lambda x: {"solution": x["answer"], "answer": x["answer"]})
    raw_dataset = raw_dataset.remove_columns(["answer"])
    
    raw_dataset = raw_dataset.train_test_split(test_size=0.1)
    # TODO: ugly that we need to load the tokenizer before
    tokenizer = load_correct_tokenizer(args.model)
    train_dataset = process_dataset(tokenizer, raw_dataset["train"], r1_preprompt, postprompt="")
    test_dataset = process_dataset(tokenizer, raw_dataset["test"], r1_preprompt, postprompt="")
    
    print(f"\nNumber of train samples: {len(train_dataset)}\n\n")
    print(f"\nNumber of test samples: {len(test_dataset)}\n\n")


    config = {
        "run_name": args.run_name,
        "project_name": args.project_name,
        "lora_save_path": args.lora_save_path,
        "lr": args.lr,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "episodes": args.episodes,
        "num_candidates": args.num_candidates,
        "batch_size": args.batch_size,
        "train_batch_size": args.train_batch_size,
        "temperature": args.temperature,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "model": args.model,
        "dataset": args.dataset,
        "number_of_actors": args.number_of_actors,
        "learner": args.learner,
        "use_vllm": True,
        "max_lora_rank": args.max_lora_rank,
        "topk": args.topk,
        "learner_chunk_size": args.learner_chunk_size,
    }


    trainer = Trainer(train_dataset, test_dataset, reward_function, config)
    trainer.train()
