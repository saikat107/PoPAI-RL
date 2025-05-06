import argparse
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from prompt_util import populate_system_prompt


parser = argparse.ArgumentParser(description="Generate text using a language model.")
parser.add_argument(
    "--model_name", type=str, default=None, help="Name of the model to use for generation (only when using a pretrained model)."
)
parser.add_argument(
    "--model_path", type=str, default=None, help="Path to the model to use for generation (only when using a local model).",
)
parser.add_argument(
    "--test_file", type=str, required=True, help="Path to the test file containing the input prompts for generation.",
)
parser.add_argument(
    "--which_prompt", type=str, default="thought", 
    help="The prompt to use for generation. Options: 'no_thought', 'thought', 'reflection', 'emulation'.",
    choices=['no_thought', "thought", "reflection", "emulation", "verification"],
)
parser.add_argument(
    "--which_input", type=str, default="prompt", help="The input to use for generation.",
)
parser.add_argument(
    "--output_dir", type=str, default=None, help="Directory to save the generated outputs.", required=True
)
parser.add_argument("--output_file", type=str, default=None, help="File to save the generated outputs.", required=True)
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
parser.add_argument(
    "--append_to_existing_results", action="store_true",
    help="If set, append the new results to the existing results file instead of overwriting it.",
)
args = parser.parse_args()
assert args.model_name or args.model_path, "Either --model_name or --model_path must be provided."
assert not (args.model_name and args.model_path), "Only one of --model_name or --model_path can be provided."

if args.which_prompt == "verification":
    raise NotImplementedError(
        "Verification prompt is not implemented yet. Please use one of the other prompts: "
        "'no_thought', 'thought', 'reflection', or 'emulation'."
    )

args.output_dir = args.output_dir or os.path.join(os.getcwd(), "outputs")
os.makedirs(args.output_dir, exist_ok=True)

result_path = os.path.join(args.output_dir, args.output_file)
test_data = None
if os.path.exists(result_path):
    if args.append_to_existing_results:
        print(f"Existing results found at {result_path}. We will append the new results to this file.")
        test_data = json.load(open(result_path, "r"))
    else:
        print(f"Existing results found at {result_path}. We will overwrite this file.")
    
if test_data is None:
    print(f"Loading test data from {args.test_file}.")
    test_data = json.load(open(args.test_file, "r"))
    test_data = populate_system_prompt(
        _data=test_data, 
        which_input=args.which_input, 
        which_prompt=args.which_prompt
    )

tokenizer = AutoTokenizer.from_pretrained(args.model_name if args.model_name is not None else args.model_path)

llm = LLM(
    model=args.model_name if args.model_name is not None else args.model_path,
    tensor_parallel_size=args.num_gpus,
    dtype='auto'
)
sampling_params = SamplingParams(
    temperature=args.temperature, 
    top_p=args.top_p,
    top_k=args.top_k, 
    max_tokens=args.max_length,
    n=args.num_generations,
)

for ti, test in enumerate(tqdm(test_data)):
    prompt = tokenizer.apply_chat_template(test[args.which_prompt], tokenize=False)
    request = {
        "id": f'{ti}',
        "prompt": prompt,
    }
    for response in llm.generate([request], sampling_params):
        generations = []
        for output in response.outputs:
            generations.append(output.text)
        if "generated_response" not in test:
            test["generated_response"] = []
        elif isinstance(test["generated_response"], dict):
            test["generated_response"] = [test["generated_response"]]
        test["generated_response"].append(
            {
                "model_name": args.model_name,
                "model_path": args.model_path,
                "which_prompt": args.which_prompt,
                "responses": generations,
            }
        )
    
    with open(result_path, "w") as f:
        json.dump(test_data, f, indent=4)
print(f"Generated responses saved to {result_path}")



