from datasets import load_dataset
import sys
import os
import json

project_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
print(project_path)
dataset = load_dataset(os.path.join(project_path, "rl_data"))
print(dataset)

import sys

sys.path.append(project_path)
from evaluation.FStar.fstar_evaluation import Evaluator as FStarEvaluator
fstar_eval = FStarEvaluator()

import json
name = "FStar.Math.Lemmas.multiple_modulo_lemma"
solution = "let multiple_modulo_lemma (a:int) (n:pos) = cancel_mul_mod a n"

result, details = fstar_eval.check_solution(name, solution)
print("Success:", result)
print(json.dumps(details, indent=2))

from evaluation.Verus.verus_evaluation import Evaluator as VerusEvaluator
verus_eval = VerusEvaluator()
name = "VERUS:13232"
solution = "use vstd::prelude::*;\n\nfn main() {}\n\nverus!{\nfn func(a: usize, b: usize, c: usize) -> (r: bool)\n    requires\n        1 <= a && a <= 100,\n        1 <= b && b <= 100,\n        1 <= c && c <= 100\n    ensures\n        r ==> (a + b == c || a + c == b || b + c == a)\n{\n    proof {\n        assert(1 <= a && a <= 100);\n        assert(1 <= b && b <= 100);\n        assert(1 <= c && c <= 100);\n    }\n    return a + b == c || a + c == b || b + c == a;\n}\n}"

result, details = verus_eval.check_solution(name, solution)
print("Success:", result)
print(json.dumps(details, indent=2))

train_dataset = dataset['train']
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import torch
model_name = "Qwen/QwQ-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def which_language(name):
    if name.startswith("VERUS"):
        return "verus"
    else:
        return "fstar"

def format_text(example):
    messages = (
        example["prompt"] +
        example["completion"]
    )
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer.encode(text)
    example["text"] = text
    example["input_ids"] = tokens
    example["num_tokens"] = len(tokens)
    example["language"] = which_language(example["name"])
    return example

train_data = dataset["train"].map(format_text, num_proc=20)
print("Train data size:", len(train_data))
train_data = train_data.filter(
    lambda x: (x["num_tokens"] < 16000 and x["language"] in ['fstar', 'verus']),
    num_proc=20
)
print("Train data size after filter:", len(train_data))

data_type = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=data_type,
)

print(model)
import re
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

def extract_solution(soln):
    if "<answer>" in soln:
        soln = soln[(soln.index("<answer>") + len("<answer>")):]
    if "</answer>" in soln:
        soln = soln[:soln.index("</answer>")]
    return soln


def verification_reward(completions, **kwargs):
    names = kwargs['name']
    languages = kwargs['language']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for lang, name, sol in zip(languages, names, completion_contents):
        print(lang, name, sol)
        sol = extract_solution(sol)
        print(sol)
        if lang == 'verus':
            ev = verus_eval
        else:
            ev = fstar_eval
        result, details = ev.check_solution(name, sol)
        print(result)
        print(details)
        if result == True:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


kwargs = train_dataset[0]
for k in kwargs.keys():
    kwargs[k] = [kwargs[k]]
completions = [train_dataset[0]['completion']]

print(verification_reward(completions, **kwargs))

from trl import GRPOConfig

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="results/Qwen-GRPO-test",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,

    # Parameters that control de data preprocessing
    max_completion_length=4096, # default: 256
    num_generations=2, # default: 8
    max_prompt_length=12000, # default: 512

    # Parameters related to reporting and saving
    #report_to="wandb",
    logging_steps=10,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=5
)

from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, verification_reward],
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
