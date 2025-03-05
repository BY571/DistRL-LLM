from unsloth import FastLanguageModel

r1_preprompt = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "The assistant first thinks about the reasoning process and then provides the user with the answer.\n" 
    "The response must follow this format:\n"
    "<think> reasoning process here </think>\n"
    "<answer> answer here </answer>\n"
)

def process_dataset(tokenizer, dataset, preprompt="", postprompt=""):
    def generate_messages(examples):

        messages = [[{"role": "system", "content": preprompt}, {"role": "user", "content": p+' '+ postprompt}] for p in examples["problem"]]
        chat_messages = [tokenizer.apply_chat_template(
                m,
                add_generation_prompt=True,
                tokenize=False,
            ) for m in messages]

        return {"problem": chat_messages}
    
    return dataset.map(generate_messages, batched=True)
