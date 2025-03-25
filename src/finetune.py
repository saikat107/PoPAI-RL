from dataclasses import dataclass, field
import json
import os
import shutil
import sys
from typing import Dict, List, Optional
import numpy as np

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
import wandb

import warnings
import traceback

warnings.filterwarnings("ignore")
tqdm.pandas()
accelerator = Accelerator()

MODEL_MAX_LENGTH = {
    "Qwen/Qwen2.5-7B-Instruct-1M": 16000,
    "Qwen/QwQ-32B": 16000,
}


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(
        default="Qwen/QwQ-32B",
        metadata={
            "choices": ["Qwen/QwQ-32B", "Qwen/Qwen2.5-7B-Instruct-1M"],
            "help": "the model name",
        },
    )
    model_path: Optional[str] = field(
        default=None, metadata={"help": "the path of the model"}
    )
    data_path: Optional[str] = field(
        default="",
        metadata={
            "help": "the path of the dataset. We assume that the data_path contains a file named 'train_sft.json'",
            "required": True,
        },
    )
    data_cache_path: Optional[str] = field(
        default=None, metadata={"help": "the path of the dataset cache"}
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )
    report_to: Optional[str] = field(
        default="wandb", metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    num_warmup_steps: Optional[int] = field(
        default=100, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.05, metadata={"help": "the weight decay"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    seq_length: Optional[int] = field(
        default=None, metadata={"help": "Input sequence length"}
    )
    max_prompt_length: Optional[int] = field(
        default=None, metadata={"help": "Prompt max length"}
    )
    which_prompt: Optional[str] = field(
        default="full_prompt",
        metadata={
            "choices": [
                "no_context_prompt",
                "base_prompt",
                "l1_type_prompt",
                "l2_type_prompt",
                "related_only_prompt",
                "full_prompt",
            ],
            "help": "the type of prompt to use",
        },
    )
    eight_bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    four_bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory", "required": True}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    logging_steps: Optional[int] = field(
        default=5, metadata={"help": "the number of logging steps"}
    )
    num_train_epochs: Optional[int] = field(
        default=-1, metadata={"help": "the number of training epochs"}
    )
    max_steps: Optional[int] = field(
        default=1000, metadata={"help": "the number of training steps"}
    )
    save_steps: Optional[int] = field(
        default=40,
        metadata={"help": "Number of updates steps before two checkpoint saves"},
    )
    save_total_limit: Optional[int] = field(
        default=1, metadata={"help": "Limits total number of checkpoints."}
    )
    workers: Optional[int] = field(
        default=20, metadata={"help": "Number of workers for the data loader"}
    )
    merge_only: Optional[bool] = field(
        default=False, metadata={"help": "Wether to merge the model only or not"}
    )
    do_not_use_completion_only: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use completion only or not"}
    )


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

print("=" * 100)
print("Arguments: ", args)
print("=" * 100)

os.makedirs(args.output_dir, exist_ok=True)
experiment_name = (
    args.output_dir.split("/")[-1] if "/" in args.output_dir else args.output_dir
)

if args.report_to == "wandb":
    wandb.init(project="fstar-synthesis", name=experiment_name)
    # wandb.config.update(args)


tokenizer = AutoTokenizer.from_pretrained(
    args.model_path if args.model_path is not None else args.model_name,
    trust_remote_code=True,
)


if args.eight_bit and args.four_bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif args.eight_bit or args.four_bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=args.eight_bit, load_in_4bit=args.four_bit
    )
    # Copy the model to each device
    device_map = {"": accelerator.local_process_index}
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None
model = AutoModelForCausalLM.from_pretrained(
    args.model_path if args.model_path is not None else args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
)

print(model)

max_allowed_seq_length = (
    MODEL_MAX_LENGTH[args.model_name]
    if args.seq_length is None
    else args.seq_length
)
data_dir = args.data_path
cache_dir = args.data_cache_path

if cache_dir is None:
    cache_dir = os.path.join(data_dir, "cached-" + experiment_name)
os.makedirs(cache_dir, exist_ok=True)
data_files = {"full": [os.path.join(data_dir, "train_sft.json")]}
raw_datasets = load_dataset(
    path=data_dir,
    data_files=data_files,
    cache_dir=cache_dir,
)["full"]

training_args = SFTConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.num_warmup_steps,
    optim=args.optimizer_type,
    weight_decay=args.weight_decay,
    logging_steps=args.logging_steps,
    num_train_epochs=args.num_train_epochs,
    eval_steps=args.save_steps,
    max_steps=args.max_steps,
    report_to=args.report_to,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    evaluation_strategy="steps",
    metric_for_best_model="eval_loss",
    bf16=True,
    max_seq_length=max_allowed_seq_length,
    dataset_text_field=args.dataset_text_field,
)


def preprocess_function(entry):
    global args, tokenizer
    prompt = entry[args.which_prompt]
    completion = entry["completion"]
    tokens = tokenizer.apply_chat_template(prompt + completion, tokenize=True)
    text = tokenizer.decode(tokens)
    num_tokens = len(tokens)
    entry[args.dataset_text_field] = text
    entry['num_tokens'] = num_tokens
    return entry



splits = raw_datasets.train_test_split(test_size=0.05, seed=42)
train_dataset = splits["train"]
valid_dataset = splits["test"]

with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=False,
        num_proc=args.workers,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )
    print("Train dataset size before filtering: ", len(train_dataset))
    train_dataset = train_dataset.filter(
        lambda x: x['num_tokens'] > 0 and x['num_tokens'] < max_allowed_seq_length,
        num_proc=args.workers,
        load_from_cache_file=True,
        desc="Filtering train dataset",
    )
    print("Train dataset size after filtering: ", len(train_dataset))

with training_args.main_process_first(desc="valid dataset map pre-processing"):
    valid_dataset = valid_dataset.map(
        preprocess_function,
        batched=False,
        num_proc=args.workers,
        load_from_cache_file=True,
        desc="Running tokenizer on valid dataset",
    )
    print("Valid dataset size before filtering: ", len(valid_dataset))
    valid_dataset = valid_dataset.filter(
        lambda x: x['num_tokens'] > 0 and x['num_tokens'] < max_allowed_seq_length,
        num_proc=args.workers,
        load_from_cache_file=True,
        desc="Filtering valid dataset",
    )
    print("Valid dataset size after filtering: ", len(valid_dataset))
    print(valid_dataset[20]['text'])
    

if args.use_peft:
    peft_config = LoraConfig(
        r=args.peft_lora_r,
        lora_alpha=args.peft_lora_alpha,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None
    
    
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer
)

if not args.merge_only:
    print(args.output_dir)
    existing_checkpoint = len([
        f for f in os.listdir(args.output_dir) 
        if "checkpoint" in f and os.path.isdir(os.path.join(args.output_dir, f) and not "final" in f)
    ]) > 0
    retry_count = 0
    while retry_count < 5:
        try:
            if existing_checkpoint:
                print("Loading from an existing checkpoint")
                trainer.train(resume_from_checkpoint=True)
            else:
                print("Starting the fresh training")
                trainer.train()
            break
        except torch.OutOfMemoryError as e:
            print("Out of memory error")
            traceback.print_exc();
            break
        except Exception as e:
            traceback.print_exc();
            if accelerator.is_main_process:
                existing_checkpoint = True
                torch.cuda.empty_cache()
            retry_count += 1
            print("Error occured : ", e, f" Retrying {retry_count}")

    if accelerator.is_main_process:
        print("Saving final model")
        best_model_ckpt = trainer.state.best_model_checkpoint
        output_directory = os.path.join(args.output_dir, "ckpt-final")
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        shutil.copytree(best_model_ckpt, output_directory)


def get_all_checkpoints(output_dir):
    return [f for f in os.listdir(output_dir) if "checkpoint" in f and os.path.isdir(os.path.join(output_dir, f))]

if accelerator.is_main_process and args.use_peft:
    output_directories = get_all_checkpoints(training_args.output_dir) + ["ckpt-final"]
    merged_dir = os.path.join(training_args.output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    for odir in output_directories:
        try:
            print(f"Loading and Merging the Peft Model from {odir}")
            model = AutoPeftModelForCausalLM.from_pretrained(
                os.path.join(training_args.output_dir, odir), trust_remote_code=True)
            model = model.merge_and_unload()

            print("Saving the merged model")
            output_merged_dir = os.path.join(merged_dir, odir)
            model.save_pretrained(output_merged_dir)
            tokenizer.save_pretrained(output_merged_dir)
        except Exception as e:
            print(f"Error occured while merging the model from {odir} : {e}")
            traceback.print_exc();
            continue