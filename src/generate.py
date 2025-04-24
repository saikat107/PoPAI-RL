import argparse
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json


parser = argparse.ArgumentParser(description="Generate text using a language model.")
parser.add_argument(
    "--model_name", type="str", default=None, help="Name of the model to use for generation (only when using a pretrained model)."
)
parser.add_argument(
    "--model_path", type=str, default=None, help="Path to the model to use for generation (only when using a local model).",
)
parser.add_argument(
    "--test_file", type=str, required=True, help="Path to the test file containing the input prompts for generation.",
)
parser.add_argument(
    "--which_prompt", type=str, default="prompt", help="The prompt to use for generation. Options: 'prompt', 'no_thought_prompt'",
    choices=['prompt', 'no_thought_prompt'],
)
parser.add_argument(
    "--output_dir", type=str, default=None, help="Directory to save the generated outputs.",
)
parser.add_argument(
    "--max_length", type=int, default=4096, help="Maximum length of the generated text.",
)
parser.add_argument(
    "--temperature", type=float, default=0.7, help="Temperature for sampling.",
)
parser.add_argument(
    "--top_p", type=float, default=0.9, help="Top-p sampling parameter.",
)
parser.add_argument(
    "--top_k", type=int, default=50, help="Top-k sampling parameter.",
)
parser.add_argument(
    "--num_generations", "--n", "-n", type=int, default=20, help="Number of sequences to return for each input prompt.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility.",
)
parser.add_argument(
    "--num_gpus", type=int, default=8, help="Number of GPUs to use for generation."
)

args = parser.parse_args()
assert args.model_name or args.model_path, "Either --model_name or --model_path must be provided."
assert not (args.model_name and args.model_path), "Only one of --model_name or --model_path can be provided."

args.output_dir = args.output_dir or os.path.join(os.getcwd(), "outputs")
os.makedirs(args.output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)


llm = LLM(
    model=args.model_name if args.model_name is not None else args.model_path,
    tensor_parallel_size=args.num_gpus,
)
sampling_params = SamplingParams(
    temperature=args.temperature, 
    top_p=args.top_p,
    top_k=args.top_k, 
    max_tokens=args.max_length,
    n=args.num_generations,
)

test_data = json.load(open(args.test_file, "r"))

for ti, test in enumerate(tqdm(test_data)):
    prompt = tokenizer.apply_chat_template(test[args.which_prompt], tokenize=False)
    request = {
        "id": f'{ti}',
        "prompt": prompt,
    }
    response = llm.generate([request], sampling_params)[0]
    generations = []
    for output in responses.outputs:
        generations.append(output.text)
    test["generated_response"] = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "which_prompt": args.which_prompt,
        "responses": generations,
    }
    
with open(os.path.join(args.output_dir, os.path.basename(agrs.test_file)), "w") as f:
    json.dump(test_data, f, indent=4)
print(f"Generated responses saved to {os.path.join(args.output_dir, os.path.basename(args.test_file))}")



