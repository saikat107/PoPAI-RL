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
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
parser.add_argument("--model_name", type=str, default="Qwen/QwQ-32B", help="Model name or path.")
parser.add_argument("--max_seq_length", type=int, default=16000, help="Maximum sequence length.")
parser.add_argument("--output_dir", type=str, default="./results", help="Output directory.")
parser.add_argument("--final_ckpt", type=str, default="./finetune_final_ckpt", help="Final checkpoint directory.")
parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps.")
parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer type.")
parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every N steps.")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient norm.")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps.")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
args = parser.parse_args()



set_seed(42)

model_name = args.model_name
max_seq_length = args.max_seq_length
output_dir = args.output_dir
per_device_train_batch_size = args.per_device_train_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
optim = args.optim
save_steps = args.save_steps
logging_steps = args.logging_steps
learning_rate = args.learning_rate
max_grad_norm = args.max_grad_norm
max_steps = args.max_steps
warmup_ratio = args.warmup_ratio
lr_scheduler_type = args.lr_scheduler_type


tokenizer = AutoTokenizer.from_pretrained(model_name)

data_path = os.path.realpath("./thought_sft_mixed")
dataset = load_dataset(path=data_path)

data_type = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=data_type,
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
    bf16=True,
    save_total_limit=1
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

from accelerate import Accelerator
accelerator = Accelerator()

if accelerator.is_main_process:
    model.save_pretrained(args.final_ckpt)
    tokenizer.save_pretrained(args.final_ckpt)
                                         