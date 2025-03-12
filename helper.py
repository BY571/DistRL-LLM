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

def init_peft_model(base_model, lora_rank=32, lora_alpha=16):
    return FastLanguageModel.get_peft_model(
        base_model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )