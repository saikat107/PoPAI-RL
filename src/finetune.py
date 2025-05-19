import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import numpy as np
import json
import torch
import random
from trl import SFTConfig, SFTTrainer
import argparse
from prompt_util import SystemPromptPopulator

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
parser.add_argument("--model_name", type=str, default="Qwen/QwQ-32B", help="Model name or path.")
parser.add_argument(
    "--input_field", type=str, default="prompt", help="The input to use for generation.",
)
parser.add_argument(
    "--prompt_type", type=str, default="thought", 
    help="The prompt to use for generation. Options: 'no_thought', 'thought', 'reflection', 'emulation'.",
    choices=['no_thought', "thought", "reflection", "emulation", "verification"],
)
parser.add_argument("--max_seq_length", type=int, default=16000, help="Maximum sequence length.")
parser.add_argument("--output_dir", type=str, default="./results", help="Output directory.")
parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps.")
parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer type.")
parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every N steps.")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient norm.")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps.")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
parser.add_argument("--data_path", type=str, default="./thought_sft_mixed", help="Path to the dataset.")

parser.add_argument(
    "--languages_to_train_on", type=str, help="Languages to train on.", choices=["verus", "fstar"], nargs="+",
    default=["verus", "fstar"]
)
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

if not output_dir.rstrip("/").endswith(args.prompt_type):
    output_dir = os.path.join(output_dir, args.prompt_type)
print("Output directory:", output_dir)
os.makedirs(output_dir, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(model_name)

data_path = os.path.realpath(args.data_path)
dataset = load_dataset(path=data_path)

data_type = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=data_type,
)


def appropriate_defn(name, defn):
    global args
    if args.prompt_type != "no_thought":
        return defn
    else:
        content = defn[0]["content"]
        assert "<answer>" in content and "</answer>" in content
        start = content.index("<answer>") + len("<answer>")
        end = content.index("</answer>") 
        content = content[start:end].strip()
        content = f"<answer>\n{content}\n</answer>"
        defn[0]["content"] = content
        return defn 

model.config.use_cache = False

def which_language(name):
    if name.startswith("VERUS"):
        return "verus"
    else:
        return "fstar"

prompt_populator = SystemPromptPopulator(
    input_field=args.input_field,
    prompt_type=args.prompt_type,
)
    
def format_text(example):
    example = prompt_populator.add_system_prompt(example)
    messages = (
        example[args.prompt_type] + 
        appropriate_defn(example["name"], example["completion"])
    )
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer.encode(text)
    example["text"] = text
    example["input_ids"] = tokens
    example["num_tokens"] = len(tokens)
    example["language"] = which_language(example["name"])
    return example


train_data = dataset["train"].map(format_text, num_proc=20)
valid_data = dataset["test"].map(format_text, num_proc=20)
print("Train data size:", len(train_data))
train_data = train_data.filter(
    lambda x: (x["num_tokens"] < max_seq_length and x["language"] in args.languages_to_train_on), 
    num_proc=20
)
print("Train data size after filter:", len(train_data))
print("Valid data size:", len(valid_data))
valid_data = valid_data.filter(
    lambda x: (x["num_tokens"] < max_seq_length and x["language"] in args.languages_to_train_on),  
    num_proc=20
)
print("Valid data size after filter:", len(valid_data))


training_arguments = SFTConfig(
    do_train=True,
    do_eval=True,
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
try:
    if torch.distributed.get_rank() == 0:
        print(model)
        print("=================================")
        print(train_data[0].keys())
        print("=================================")
        print(json.dumps(train_data[0][args.prompt_type], indent=2))
        print("=================================")
except Exception as e:
    import traceback
    traceback.print_exc()
    pass
    
    
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=valid_data,
    args=training_arguments,
)


trainer.train()
                                         
