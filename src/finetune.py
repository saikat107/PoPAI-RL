import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import numpy as np

import torch
import random
from trl import SFTConfig, SFTTrainer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

model_name = "Qwen/QwQ-32B"
max_seq_length = 16000
output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 1
optim = "adamw_torch"
save_steps = 50
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1000
warmup_ratio = 0.1
lr_scheduler_type = "cosine"

tokenizer = AutoTokenizer.from_pretrained(model_name)

data_path = os.path.realpath("./thought_sft_mixed")
dataset = load_dataset(path=data_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
)

model.config.use_cache = False
def format_text(example):
    messages = example["prompt"] + example["completion"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer.encode(text)
    example["text"] = text
    example["input_ids"] = tokens
    example["num_tokens"] = len(tokens)
    return example


train_data = dataset["train"].map(format_text, num_proc=20)
valid_data = dataset["test"].map(format_text, num_proc=20)
print("Train data size:", len(train_data))
train_data = train_data.filter(lambda x: x["num_tokens"] < max_seq_length, num_proc=20)
print("Train data size after filter:", len(train_data))
print("Valid data size:", len(valid_data))
valid_data = valid_data.filter(lambda x: x["num_tokens"] < max_seq_length, num_proc=20)
print("Valid data size after filter:", len(valid_data))


training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

# print model is local_rank 0
if torch.distributed.get_rank() == 0:
    print(model.dtype)
    
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=valid_data,
    args=training_arguments,
)


trainer.train()
