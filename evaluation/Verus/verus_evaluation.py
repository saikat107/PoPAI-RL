import argparse
import json
import os
import traceback
from multiprocessing import Pool
import copy
import re
import tqdm

from typing import (
    Generic,
    Literal,
    TypeVar,
    TypedDict,
    Any,
    Optional,
    Callable,
    Iterable,
    Union,
    cast,
)
import subprocess
import json
import sys
import os
import threading
import multiprocessing
import random
import queue
import tempfile
import numpy as np


current_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
bin_path = os.path.join(current_dir, "helpers")
os.environ["PATH"] = bin_path + ":" + os.environ["PATH"]

def add_to_path(p):
    paths = os.environ["PATH"].split(":")
    if p not in paths:
        os.environ["PATH"] = p + ":" + os.environ["PATH"]

def remove_from_path(p):
    paths = os.environ["PATH"].split(":")
    if p in paths:
        paths.remove(p)
    os.environ["PATH"] = ":".join(paths)

def eprint(msg):
    # sys.stderr.write(str(msg) + "\n")
    # sys.stderr.flush()
    pass


def run_verus(solution, timeout):
     try:
         # Write the solution to a temporary file and run the Verus command on it
         with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix=".rs") as temp_file:
             temp_file.write(solution)
             temp_file.flush()
             file_name = temp_file.name
             # Run the Verus command
             process = subprocess.Popen(
                 [f"verus", file_name],
                 stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE,
                 text=True,
             )
             stdout, stderr = process.communicate(timeout=timeout)
             if process.returncode == 0:
                 return True, {"status": "success", "response": stdout}
             else:
                 return False, {"status": "failure", "response": stderr}
     except subprocess.TimeoutExpired:
         process.kill()
         return False, {"status": "timeout", "response": "Process timed out"}
     except Exception as e:
         return False, {"status": "error", "response": str(e)}

def analyze_solution(
    entry,
    solution: str,
    timeout: int,
):
    name = entry["name"]
    if solution.strip() == "":
        result = False
        detail = {
            "kind": "none",
            "query-id": "none",
            "status": "failure",
            "response": [
                {
                    "level": "error",
                    "number": 998,
                    "message": "Empty string as solution",
                    "ranges": [],
                }
            ],
        }
    result, detail = run_verus(solution, timeout=timeout)
    logged_solution = {
        "name": name,
        "solution": solution,
        "result": result,
        "detail": detail,
    }
    return logged_solution


def evaluation_function(truths):
    if truths is None or len(truths) == 0:
        truths = [False]
    k_values = range(1, len(truths) + 1)
    metrics = {f"pass@{k}": any(truths[:k]) for k in k_values}
    metrics["pass@any"] = any(truths)
    return metrics


def remove_block_comment(response):
    if "(*" in response:
        first_part = response[: response.index("(*")]
        second_part = response[response.index("(*") + 2 :]
        if "*)" in second_part:
            remaining = second_part[second_part.index("*)") + 2 :]
        else:
            remaining = ""
        response = first_part + remaining
        return remove_block_comment(response)
    return response


def remove_line_comment(response):
    lines = response.split("\n")
    taken_lines = []
    for l in lines:
        if "//" in l:
            l = l[: l.index("//")]
            if l.strip() != "":
                taken_lines.append(l)
        else:
            taken_lines.append(l)
    return "\n".join(taken_lines)


def sanitize(response):
    # If there is a <ansewer> </answer> tag, only take what is in between
    if "<answer>" in response:
        response = response[response.index("<answer>") + 8 :]
        if "</answer>" in response:
            response = response[: response.index("</answer>")]
    response = remove_block_comment(response)
    response = remove_line_comment(response)
    return response


def check_example(inp):
    example, _, solution_key, timeout = inp
    name = example["name"]
    if "." in name:
        name = name.split(".")[-1]

    responses = example[solution_key]
    results = [None] * len(responses)
    truths = [False] * len(responses)
    for ri, response in enumerate(responses):
        try:
            res = analyze_solution(
                entry=example,
                solution=response,
                timeout=timeout,
            )
            res["checked_solution"] = response
            results[ri] = res
            truths[ri] = res["result"] if res is not None else False
        except Exception as e:
            traceback.print_exc()
            if isinstance(e, KeyboardInterrupt):
                raise e
            res = None
            results[ri] = res
            truths[ri] = False
        except:
            traceback.print_exc()
            res = None
            results[ri] = res
            truths[ri] = False
    return (example, results, truths, evaluation_function(truths))


class Evaluator:
    def __init__(self):
        current_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
        self.bin_path = os.path.join(current_dir, "helpers")
        
    def check_solution(self, example_name: str, solution: str, timeout: int = 300):
        remove_from_path(self.bin_path)
        add_to_path(self.bin_path)
        example = {"name": example_name}
        example["generated_response"] = [solution]
        example, res, truths, _ = check_example(
            (example, False, "generated_response", timeout)
        )
        remove_from_path(self.bin_path)
        return truths[0], res[0]


def summarize_metrics(metrics_list, short=False):
    summary = {}
    for metric in metrics_list[0]:
        values = [m[metric] if metric in m else False for m in metrics_list]
        summary[metric] = round((sum(values) * 100 / len(values)), 2)
    if short:
        metrics = list(metrics_list[0].keys())
        summary = {metrics[0]: summary[metrics[0]], metrics[-1]: summary[metrics[-1]]}
    return summary


def extract_json_with_keys(text, keys):
    # Escape keys for regex safety
    escaped_keys = [re.escape(key) for key in keys]
    # Build regex pattern dynamically
    key_patterns = [rf'"{key}"\s*:\s*".*?"' for key in escaped_keys]
    combined_pattern = r"\{[^}]*" + "[^}]*".join(key_patterns) + r"[^}]*\}"
    pattern = re.compile(combined_pattern, re.DOTALL)
    matches = pattern.findall(text)
    valid_json_objects = []
    for match in matches:
        try:
            json_obj = json.loads(match)

            # Double-check the keys are present
            if all(k in json_obj for k in keys):
                valid_json_objects.append(json_obj)

        except json.JSONDecodeError:
            continue
    return valid_json_objects

def extract_inside_a_pattern(texts, tag):
    extractions = []
    begin_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    for text in texts:
        if begin_tag in text:
            text = text[text.index(begin_tag) + len(begin_tag) :]
            if end_tag in text:
                text = text[: text.index(end_tag)]
        extractions.append(text.strip())
    return extractions


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", "-i", type=str, required=True, nargs="+")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--output_dir", "-d", type=str)
    parser.add_argument("--check_ground_truth", "-g", action="store_true")
    parser.add_argument(
        "--solution_key", "-k", type=str, default="generated_response/responses"
    )
    # Use the following two arguments ONLY if the generated response is a json object
    # and you want to extract a specific key from it, by default the key is "definition"
    parser.add_argument(
        "--extract_from_solution", "-e", action="store_true", help="Whether to extract json from the solution",
    )
    parser.add_argument(
        "--solution_tag", "-a", type=str, help="tag to extract from the solution",
        default="answer"
    )
    parser.add_argument("--timeout", "-t", type=float, default=100000)
    parser.add_argument("--num_workers", "-w", type=int, default=80)
    return parser.parse_args()


def find_solution(entry, solution_key):
    try:
        if "/" not in solution_key:
            response = entry[solution_key]
        else:
            solution_key_parts = solution_key.split("/")
            responses = entry
            for k in solution_key_parts:
                if not isinstance(responses, dict) and isinstance(responses, list):
                    responses = responses[0]
                responses = responses[k]
            assert isinstance(responses, list) or isinstance(responses, str), f"Expected list or string, got {type(responses)}"
        if isinstance(responses, str):
            responses = [responses]
        return responses
    except Exception as e:
        print(f"Error finding solution: {e}")
        traceback.print_exc()
        return None

def calculate_pass_at_k(truths, k=1):
    n = len(truths)
    c = sum([1 for t in truths if t])
    if n - c == 0:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def pass_at_k(list_of_truths, k=1):
    return round(
        np.mean([calculate_pass_at_k(truths, k) for truths in list_of_truths]) * 100, 2
    )

def main():
    args = get_argument()
    generated_files = args.input_files
    tasks = []
    for gf in generated_files:
        with open(gf, "r") as fp:
            tasks.extend(json.load(fp))
            fp.close()

    for t in tasks:
        if not args.check_ground_truth:
            solutions = find_solution(t, args.solution_key)
            tmp_sol_key = args.solution_key
            if "/" in tmp_sol_key:
                tmp_sol_key = tmp_sol_key.replace("/", "_")
                
            if solutions is None:
                print(
                    f"Solution key not found in the example {t['name']}. Adding empty solution"
                )
                t[tmp_sol_key] = ""
            else:
                t[tmp_sol_key] = solutions
            
            if args.extract_from_solution:
                t[tmp_sol_key + "-extracted"] = extract_inside_a_pattern(
                    t[tmp_sol_key], args.solution_tag
                )
    if not args.check_ground_truth:
        if args.extract_from_solution:
            args.solution_key = args.solution_key + "-extracted"
        if "/" in args.solution_key:
            args.solution_key = args.solution_key.replace("/", "_")
    tasks = [
        (t, args.check_ground_truth, args.solution_key, args.timeout) for t in tasks
    ]
    pool = (
        Pool(args.num_workers)
        if args.num_workers is not None and args.num_workers > 1
        else None
    )
    mapping_function = pool.imap_unordered if pool is not None else map
    results = mapping_function(check_example, tasks)
    detailed_results, aggregate_metrics, aggregate_truths = [], [], []
    bar = tqdm.tqdm(results, total=len(tasks), desc="")
    for example, res, truths, metrics in bar:
        detailed_results.append(
            {"example": example, "results": res, "truths": truths, "metrics": metrics}
        )
        aggregate_metrics.append(metrics)
        aggregate_truths.append(truths)
        bar.set_description(json.dumps(summarize_metrics(aggregate_metrics, short=True)))
    if pool:
        pool.close()

    final_summary_metrics = summarize_metrics(aggregate_metrics)
    print(json.dumps(final_summary_metrics))
    pass_at_1 = pass_at_k(aggregate_truths, k=1)
    pass_at_2 = pass_at_k(aggregate_truths, k=2)
    pass_at_5 = pass_at_k(aggregate_truths, k=5)
    print("Expected Metrics:")
    print(f"Pass@1: {pass_at_1}")
    print(f"Pass@2: {pass_at_2}")
    print(f"Pass@5: {pass_at_5}")
    final_summary_metrics["expected_pass@1"] = pass_at_1
    final_summary_metrics["expected_pass@2"] = pass_at_2
    final_summary_metrics["expected_pass@5"] = pass_at_5
    if args.output_dir is not None:
        if len(generated_files) > 1:
            assert (
                args.output is not None
            ), "Output file must be specified when multiple input files are provided."
        output_file_name = (
            os.path.basename(args.input_files[0])
            if len(args.input_files) == 1
            else args.output
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(
            os.path.join(
                args.output_dir,
                output_file_name.replace(".json", "_detailed_results.json"),
            ),
            "w",
        ) as fp:
            json.dump(detailed_results, fp, indent=4)
            fp.close()
        with open(
            os.path.join(
                args.output_dir,
                output_file_name.replace(".json", "_aggregate_metrics.json"),
            ),
            "w",
        ) as fp:
            json.dump(aggregate_metrics, fp, indent=4)
            fp.close()
        with open(
            os.path.join(
                args.output_dir,
                output_file_name.replace(".json", "_summary_metrics.json"),
            ),
            "w",
        ) as fp:
            json.dump(final_summary_metrics, fp, indent=4)


if __name__ == "__main__":
    main()
    # evaluator = Evaluator()
    # # with open(
    # #     "/home/saikatc/workspace/PoPAI/Fully-Checked-Augmented-DataSet-V2/sample_int.json",
    # #     "r",
    # # ) as fp:
    # #     tasks = json.load(fp)
    # #     t = tasks[63]
    # verdict, res = evaluator.check_solution(
    #     "Spec.Ed25519.Lemmas.aff_point_double_lemma",
    #     "let aff_point_double_lemma p =\n  let x, y = p in\n  assert (is_on_curve p);\n  let (x_add, y_add) = aff_point_add p p in\n  let (x_double, y_double) = aff_point_double p in\n  calc (==) {\n    x_double;\n    (==) { } (2 *% x *% y) /% (y *% y -% x *% x);\n    (==) { } (x_add);\n    (==) { }\n      (x *% y +% y *% x) /% (1 +% d *% (x *% x) *% (y *% y));\n    (==) { }\n      (2 *% x *% y) /% (y *% y -% x *% x) by is_on_curve;\n  };\n  calc (==) {\n    y_double;\n    (==) { } (y *% y +% x *% x) /% (2 -% y *% y +% x *% x);\n    (==) { } (y_add);\n    (==) { }\n      (y *% y +% x *% x) /% (1 -% d *% (x *% x) *% (y *% y));\n    (==) { lemma_aff_double_aux x y };\n    (==) { } (y *% y +% x *% x) /% (2 -% y *% y +% x *% x);\n  };\n  trefl();",
    #     timeout=300
    # )
    # print(verdict)
    # print(json.dumps(res, indent=4))
