from datasets import load_dataset
from helper import process_dataset
from reward_function import EVAL_FUNCTIONS, extract_between_tokens
import numpy as np
import argparse
from prompts import r0_preprompt, r1_preprompt
from distributed_trainer import Trainer
from unsloth.tokenizer_utils import load_correct_tokenizer

REWARD_FORMAT = 0.01


from reward_function import product_eval
import re

def filter_problem(data):
    if data["problem_type"] != "reaction-prediction":
        return True

    reaction = extract_reaction(clean_text(data["problem"]))
    reactants = reaction.split(">")[0].split(".")
    solution = extract_solution(data["solution"])

    # check reactants not in solution
    tofilter = True
    for mol in reactants:
        try:
            rew_reaction = product_eval(mol, solution)  
        except:
            rew_reaction = 0.0

        if rew_reaction == 1.0:
            tofilter = False
            break
    return tofilter

def clean_text(text: str) -> str:
    """
    Remove conversation tokens enclosed in angle brackets (e.g., <|im_start|>, <think>, etc.)
    from the input text.
    Args:
        text (str): The input text possibly containing conversation tokens.
    Returns:
        str: The cleaned text without any tokens enclosed in angle brackets.
    """
    # Remove any substring starting with '<' and ending with '>'
    return re.sub(r'<[^>]+>', '', text)

def extract_reaction(text: str) -> str:
    """
    Args:
        text (str): Cleaned text that may contain a reaction string.
    
    Returns:
        str or None: The extracted reaction string if found; otherwise, None.
    """
    # Split the text into tokens using any whitespace as a delimiter.
    tokens = re.findall(r'\S+', text)

    # Helper function: remove common trailing punctuation.
    def clean_token(token: str) -> str:
        return token.strip('.,;:?!"\'')

    candidates = []
    for token in tokens:
        cleaned_token = clean_token(token)
        # Check if the token contains exactly two '>' symbols.
        if cleaned_token.count('>') == 2:
            # Ensure the reactants part (before the first '>') is not empty.
            if cleaned_token.split('>', 1)[0]:
                candidates.append(cleaned_token)

    if candidates:
        # Choose the longest candidate, assuming it is the complete reaction string.
        return max(candidates, key=len)

    return None

def extract_solution(sol):
    fxn_name, ideal, problem_type = sol.split("!:!", 2)
    return ideal

def format_reward(completions):
    """
    Reward function that checks if each completion strictly follows the format:
      <think>...</think><answer>...</answer>
    Optionally, whitespace is allowed between </think> and <answer>, but nothing else.
    """

    pattern = re.compile(r'^<think>(.*?)</think>[ \t\n]*<answer>(.*?)</answer>$', re.DOTALL)

    format_rewards = []
    extracted_answers = []
    extracted_thoughts = []

    for c in completions:
        # Remove any leading/trailing whitespace from the completion.
        c = c.strip()
        match = pattern.match(c)
        if match:
            # The valid format was found; extract the text between the tags.
            think = match.group(1)
            answer = match.group(2)
            reward = REWARD_FORMAT  # valid format reward (assumed to be defined elsewhere)
        else:
            # Format is invalid; no extraction and no reward.
            think = ""
            answer = ""
            reward = 0.0

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
                format_reward = REWARD_FORMAT
                reward = EVAL_FUNCTIONS[fxn_name](answer, ideal)
                accuracy_reward = reward
        except Exception: # eval function failed, but it did parse correctly
            format_reward = REWARD_FORMAT
            accuracy_reward = 0.0
            problem_type = None

        format_rewards.append(format_reward)
        accuracy_rewards.append(accuracy_reward)
        problem_types.append(problem_type)

    return accuracy_rewards, format_rewards, problem_types


def reward_function(completions, solutions):
    _, extracted_answers, extracted_thoughts = format_reward(completions=completions)
    acc_reward, form_reward, problem_types = assign_rewards(solutions=solutions, extracted_answers=extracted_answers, extracted_thoughts=extracted_thoughts)
    return np.array(list(zip(form_reward, acc_reward)))

easy_solution_ids = ["121d7b00-3ed4-5d40-bb3c-bf88c1204a5b",
"30e16bec-0f24-5a04-b3de-930c0f7556f5",
"1e06817e-cc4e-51c2-962a-a149a29295a0",
"b9538349-b1c7-5616-a954-bc2ecfc372fc",
"e80bc2f3-d058-59b0-aa31-fa886553e855",
"4edcd554-5968-5e9f-b035-1f1e5c93df48",
"a6965515-2aa3-5e59-adb6-13516b8eb88d",
"eb731b56-cba8-5563-aac1-247b8e51c76d",
"d8aef821-cead-5597-a999-a62b20771fd1",
"0c12cae2-2899-57c8-97b4-775cf9b975f3",
"2e7d6c82-fca2-5c1d-b010-5b526f439e77",
"5e535277-46aa-55dc-924b-da46795be2ab",
"c59177ac-be15-5e2f-b946-5003d1f87582",
"0e25b50a-2cca-5873-b9b6-b0c13c56223c",
"457b8385-501c-5b5c-b8d9-e57304032463",
"4c9017be-ab35-5ae9-8769-0f9b4fe5e860",
"7dc910d2-d980-573d-a023-9a8d2ac04000",
"5c587430-7911-5129-a2d8-21365a8c4cd0",
"0ef7f16e-8c90-53be-86bc-4afee1eeca2a",
"fd3dfed3-6f28-56bf-80cb-a5c0e64c06e1",
"29f09fa0-9cba-5787-ba28-503818a2bdec",
"fb13a770-41de-500c-84cf-3e001db36eeb",
"f611c65c-f9d9-50e5-a3da-ed132cbcf3a5",
"29d2ec47-fb49-5572-a93c-f2dad4753687",
"ef06da35-b628-5944-8348-971c0bd683c1",
"95b19a9c-051e-5a42-bf83-6e393e49a9ad",
"406ed69b-9ef0-51a1-ae29-01f17b0e14a4",
"9ec1a243-1291-57a5-96a1-52ea4cd7eefc",
"03bcc39d-d84a-5b55-a49e-754c0ddf3bbc",
"1a0714d5-3c09-5e79-94a5-960f1ea2c0ad",
"a2c71521-1a3e-5eb8-9210-05631ba4c4e4",
"0eb7a2fe-a1dc-5f40-baa2-779dc322524f",
"7bff6e4f-c17c-531e-b2a4-76c52153d344",
"93644cd4-138c-508f-9cef-3ba4bb0c40bd",
"db78c8ba-2e5a-5d21-8853-1a3884e44696",
"15469447-758a-5f7b-82f0-bb8dbc2c6e06",
"7a7bded0-cad6-56d4-a6a9-04c2bb722045",
"133a3456-5702-5ac0-a2ec-326db5297ef6",
"f9d1b6ab-987f-5653-937b-375fbab81e7c",
"70afd062-3048-5808-bd09-e4c7a91a45c7",
"8ed4f205-7fb4-5cdc-a565-35d5aff48102",
"31d6a0b4-1a19-5b7e-80d0-cb8f70e25012",
"612b411c-0306-59da-849b-e59dab09f174",
"9e61b225-7233-50a9-b3da-bb670f9ae23c",
"3024b3ef-0278-5ad5-9b2e-4e2220b0ccaf",
"396f92eb-517e-5de8-912c-ce5a48d1717c",
"0c51eddd-80a9-5205-a4b1-263657ab16c0",
"6470d656-c9fb-509a-a4eb-5d96879ab78d",
"2d759916-128e-5cab-b3c9-d0a69661e457",
"b7c8b6af-8f79-5dd1-a22e-c040b0c239a3",
"d186f04f-6689-5928-981b-b06667c8a9f0",
"00b2383e-5c95-5919-8c6b-f08d3011483d",
"e780a429-481a-5f5a-a9f6-fa9b4b3a21b0",
"5dd5eff0-3dd0-5508-be66-f0bacc342b98",
"4309836d-f9c3-5c01-b7cd-681cc6937087",
"e75d9ac8-6d43-5ce8-9666-16b814a2952e",
"89014d17-5e61-57b8-bfb0-bf803d109763",
"edbcb3f1-ed16-5074-b606-a328d0d7cb4a",
"e2745c66-de10-5643-8317-2634c588cbe1",
"602e1f1a-3f13-5b8f-937b-0a2d0922d941",
"9e78e86c-3bad-574c-b06e-83007e69ab81",
"4d329108-fd9e-5ec9-88df-bc475fbb4585",
"e3502940-2136-505a-8ce9-6b8cb1b7aa3c",
"06c93d08-e196-5127-96e2-c0fd4b02603e",
"2df2d3df-d8d6-5bc0-8ff0-bb7fa98f47a1",
"1f9c1a2e-72e2-5205-bfee-accf0587a282",
"a862406f-705e-59fe-86bc-a52fe45553bd",
"5780ba23-9ef2-5a42-b83a-69ef6a3670fc",
"aa641f16-a43c-5b95-9ee2-98924d5e0af7",
"77fbfb0e-6b74-5f5a-888b-adebb5e894b1",
"3a6e0baf-4873-56be-b5a8-64c24d71b31f",
"068b9f55-25f0-5547-b421-cf6fbd4fab19",
"302cd359-7508-54ae-b9f0-a9475700f792",
"a9e6ecb5-2f40-5591-be64-3e1a78e47e67",
"4c6ab410-051e-5c4c-8d45-3cc48f2d5400",
"f1610e35-832c-5dce-a311-ec9b0682f7f0",
"ef480e07-6376-5dde-8ba1-c7916a826523",
"670067f2-154c-51c9-a571-ed905b37bc6e",
"10f41453-111d-5478-b5b4-85e3ab59eaeb",
"b3239a86-5c36-5d2f-beaf-9c4e2132d436",
"5c6d67bb-8a6b-5769-a6f0-14324fec7779",
"4f28445d-d4ff-57b5-aa9c-98ac06634785",
"cec519b1-17d3-5222-a9a3-ea9e374029fd",
"2cfe4e0b-4334-553c-8dac-ead6a1bef8b7",
"fdeb2d65-0bc5-500b-a5d2-ad7eaf4255ee",
"a7f9494c-b6e5-597d-89b9-087a09e1eb47",
"d277bf6c-399a-5ae6-b358-38a5616cba9e",
"98a18cbd-ab17-5041-8a7c-69027ae5eecc",
"cac213ea-fed9-5675-98fe-351bc6179f7a",
"10e084e6-28c5-595c-a617-34d76cf6311b",
"263491b6-0cac-5475-8c52-e9e633c2a9a9",
"7c242655-5b34-5b2f-9ae3-6b53964421f8",
"c412ed98-ce53-5384-99d4-8de0f7812d1a",
"2e967118-9860-5752-b3b4-cd247f1c1820",
"1609a2b0-9655-5052-bba1-359f946b3354",
"fe0319e6-dd7d-5f42-a48f-7deaf423dba1",
"1c6468e0-455a-53dc-9f58-f0a2a0f3a3a0",
"98479ae2-f343-52be-b6b4-941a4f39da82",
"c4882263-ce8b-5737-abbc-194f11515749",
"013667c1-0e7d-526b-b375-2b13d54d45a8"]


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct-bnb-4bit") #unsloth/Qwen2.5-7B-Instruct-bnb-4bit") unsloth/Llama-3.2-3B-Instruct-bnb-4bit
    args.add_argument("--dataset", type=str, default="train", choices=["train", "easy"])
    args.add_argument("--task", type=str, default="reaction-prediction", choices=["all", "molecule-completion", "reaction-prediction", "molecule-name"])
    args.add_argument("--run_name", type=str) #forcing to be set by user
    args.add_argument("--project_name", type=str, default="chem-reason")
    args.add_argument("--lr", type=float, default=2e-5)
    args.add_argument("--max_new_tokens", type=int, default=300)
    args.add_argument("--temperature", type=float, default=0.8)
    args.add_argument("--episodes", type=int, default=10)
    args.add_argument("--num_candidates", type=int, default=32, help="Number of sampled candidate per monkey iteration")
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--train_batch_size", type=int, default=16)
    args.add_argument("--cot_decode", action='store_true', default=False)
    args.add_argument("--max_monkey_rounds", type=int, default=1, help="Number of generation rounds to sample candidates")
    args.add_argument("--save_every", type=int, default=100, help="Save the model every x training steps")
    args.add_argument("--eval_every", type=int, default=100, help="Evaluate the model every x training steps")
    args.add_argument("--number_of_actors", type=int, default=3, help="Number of actors to use, default is 0. Only uses the learner to generate and train.")
    args.add_argument("--keep_last_x", type=int, default=10, help="Number of last model checkpoints to keep")
    args.add_argument("--learner", type=str, choices=["pg", "grpo"], default="pg")
    args.add_argument("--use_vllm", action='store_true', default=False)
    args.add_argument("--max_lora_rank", type=int, default=32)
    args.add_argument("--top_k_train", type=int, default=4, help="Top k training samples to per task/actor")
    
    args = args.parse_args()
    
    if args.number_of_actors == 0:
        assert args.batch_size == 1, "Batch size must be 1 if number of actors is 0"

    raw_dataset = load_dataset("whitead/molecule-rewards")
    if args.task != "all":
        raw_dataset = raw_dataset.filter(lambda x: args.task in x["problem_type"])

    raw_train_dataset = raw_dataset[args.dataset]  #easy or train
    raw_eval_dataset = raw_dataset["test"]  #!!! I AM USING THE TEST SET AS THE EVAL SET
    

    cleaned_data = raw_train_dataset.filter(filter_problem, batch_size=1)

    subset_data = cleaned_data.filter(lambda x: x["id"] in easy_solution_ids)

    # subset_data = raw_train_dataset
    print(f"Number of train samples: {len(subset_data)}\n\n")    
    
    config = {    "run_name": args.run_name,
                  "project_name": args.project_name,
                  "lr": args.lr,
                  "max_new_tokens": args.max_new_tokens,
                  "episodes": args.episodes,
                  "num_candidates": args.num_candidates,
                  "cot_decode": args.cot_decode,
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
                  "top_k_train": args.top_k_train,
                  }

    #TODO: ugly that we need to load the tokenizer before 
    tokenizer = load_correct_tokenizer(args.model)
    # process dataset
    train_dataset = process_dataset(tokenizer, subset_data, r1_preprompt)
    #eval_dataset = process_dataset(tokenizer, raw_eval_dataset, r1_preprompt)

    trainer = Trainer(train_dataset, reward_function, config)
    trainer.train()


