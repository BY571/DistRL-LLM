import re
import numpy as np

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(completions, solutions) -> list[float]:
    extracted_responses = [extract_xml_answer(c) for c in completions]
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, solutions)]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, c) for c in completions]
    return [0.1 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, c) for c in completions]
    return [0.1 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.05
    if text.count("\n</think>\n") == 1:
        count += 0.05
    if text.count("\n<answer>\n") == 1:
        count += 0.05
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.05
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]


def reward_function(completions, solutions):
    acc_reward = correctness_reward_func(completions, solutions)
    form_reward = strict_format_reward_func(completions)
    xml_reward = xmlcount_reward_func(completions)
    form_reward = form_reward + xml_reward
    return np.array(list(zip(form_reward, acc_reward)))
