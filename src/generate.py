import argparse
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json


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
    "--which_prompt", type=str, default="prompt", help="The prompt to use for generation. Options: 'prompt', 'no_thought_prompt'",
    choices=['prompt', 'no_thought_prompt', "cot", "emulate"],
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

args = parser.parse_args()
assert args.model_name or args.model_path, "Either --model_name or --model_path must be provided."
assert not (args.model_name and args.model_path), "Only one of --model_name or --model_path can be provided."

args.output_dir = args.output_dir or os.path.join(os.getcwd(), "outputs")
os.makedirs(args.output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_name if args.model_name is not None else args.model_path)

VERUS_NO_THOUGHT_SYSTEM_PROMPT = (
    "You are an experienced formal language programming assistant. "
    "You are very familiar with Verus, which is a tool for verifying "
    "the correctness of code written in Rust. Your mission is to write "
    "correct proof code, including loop invariants and assertions to "
    "the given Rust code, so that Verus can verify the give function "
    "behaves exact what is described in the specifications, which is "
    "`requires` and `ensures`. The given verus code is missing proofs. "
    "The assistant only provides the verified rust code inside <answer> "
    "and </answer> tags. The assistant should not provide any explanation "
    "or reasoning in the <answer> tag."
)

FSTAR_NO_THOUGHT_SYSTEM_PROMPT = (
    "Suppose you are a F* programming assistant. The user asks to "
    "write the definition of a F* term from its type declaration. "
    "The user provides the type declaration and some other information, "
    "such as the context, other definitions in the type etc., and "
    "the Assistant writes the definition so that the input type is satisfied. "
    "The assistant only provides the complete satisfyable definition of the "
    "term inside <answer> and </answer> tags. The assistant should not "
    "provide any explanation or reasoning in the <answer> tag."   
)

FSTAR_COT_SYSTEM_PROMPT = (
    "Suppose you are a F* programming assistant. The user asks to write the "
    "definition of a F* term from its type declaration. The user provides the "
    "type declaration and some other information, such as the context, other "
    "definitions in the type etc., and the Assistant writes the definition so "
    "that the input type is satisfied. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here "
    "</think><answer> answer here </answer>. Inside the <think> tag, the assistant "
    "provides a list of reasoning steps that it will take to arrive at the answer. "
    "While the assistant is reasoning about synthesizing a verifiable definition, "
    "it should first reflect on the overall problem and sketch a solution strategy. "
    "Then it should follow the high level strategy and provide a list of actionable "
    "steps to arrive at the solution. The assistant should format its reasoning as follows: \n"
    "<think> \n"
    "    <reflection> \n"
    "        The high level strategy is to synthesize a verifiable definition. \n"
    "    </reflection> \n"
    "    <steps> \n"
    "        <step> \n"
    "            The first step ... \n"
    "        </step> \n"
    "        <step> \n"
    "            The second step ... \n"
    "        </step> \n"
    "        ...\n"
    "    </steps> \n"
    "</think> \n"
    "Note that these steps should in such a way that a human can follow them. "
    "In the <answer> tag, the assistant only provides the complete "
    "satisfyable definition of the term. "
)

FSTAR_EMULATE_SYSTEM_PROMPT = (
    "Suppose you are a F* programming assistant. The user asks to write the "
    "definition of a F* term from its type declaration. The user provides the "
    "type declaration and some other information, such as the context, other "
    "definitions in the type etc., and the Assistant writes the definition so "
    "that the input type is satisfied. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here "
    "</think><answer> answer here </answer>. Inside the <think> tag, the assistant "
    "provides a list of reasoning steps that it will take to arrive at the answer. "
    "While the assistant is reasoning about synthesizing a verifiable definition, "
    "it should first reflect on the overall problem and sketch a solution strategy. "
    "Then it should follow the high level strategy and provide a list of actionable "
    "steps to arrive at the solution. After that, the assistant should emulate these "
    "steps and think what would be the state of the program following each individual step. "
    "The assistant should format its reasoning as follows: \n"
    "<think> \n"
    "    <reflection> \n"
    "        The high level strategy is to synthesize a verifiable definition. \n"
    "    </reflection> \n"
    "    <steps> \n"
    "        <step> \n"
    "            The first step ... \n"
    "        </step> \n"
    "        <step> \n"
    "            The second step ... \n"
    "        </step> \n"
    "        ...\n"
    "    </steps> \n"
    "    <emulation> \n"
    "        <step> \n"
    "            Program state before and after taking step 1 ... \n"
    "        </step> \n"
    "        <step> \n"
    "            Program state before and after taking step 2 ... \n"
    "        </step> \n"
    "        ...\n"
    "    </emulation> \n"
    "</think> \n"
    "Note that these steps should in such a way that a human can follow them. "
    "Ideally, at the last step of the emulation, the assistant should be able to "
    "provide the final definition. Fianlly, n the <answer> tag, the assistant only "
    "provides the complete satisfyable definition of the term. "
)


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

def prepare_test_data(_test_data):
    for test in _test_data:
        if args.which_prompt in test:
            prompt = test[args.which_prompt]
        else:
            prompt = test["prompt"]
        if args.which_prompt == "no_thought_prompt":
            prompt[0]["content"] = (
                VERUS_NO_THOUGHT_SYSTEM_PROMPT if test["name"].startswith("VERUS") else FSTAR_NO_THOUGHT_SYSTEM_PROMPT
            )
        elif args.which_prompt == "cot":
            prompt[0]["content"] = (
                VERUS_COT_SYSTEM_PROMPT if test["name"].startswith("VERUS") else FSTAR_COT_SYSTEM_PROMPT
            )
        elif args.which_prompt == "emulate":
            prompt[0]["content"] = (
                VERUS_EMULATE_SYSTEM_PROMPT if test["name"].startswith("VERUS") else FSTAR_EMULATE_SYSTEM_PROMPT
            )
        test[args.which_prompt] = prompt
    return _test_data

result_path = os.path.join(args.output_dir, args.output_file)
if os.path.exists(result_path):
    print(f"Existing results found at {result_path}. We will append the new results to this file.")
    test_data = json.load(open(result_path, "r"))
else:
    print(f"Loading test data from {args.test_file}.")
    test_data = json.load(open(args.test_file, "r"))
    test_data = prepare_test_data(test_data)

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
print(f"Generated responses saved to {os.path.join(args.output_dir, args.output_file)}")



