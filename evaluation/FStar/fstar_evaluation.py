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
    NotRequired,
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
import numpy as np


current_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
DATSET_DIR = os.path.join(current_dir, "helpers/support_files")
bin_path = os.path.join(current_dir, "helpers/bin")
os.environ["PATH"] = bin_path + ":" + os.environ["PATH"]
full_data_path = os.path.join(current_dir, "helpers/full_data.json")
FULL_DATA = json.load(open(full_data_path, "r"))

class Dependency(TypedDict):
    source_file: str
    checked_file: str
    interface_file: bool  # Whether the file depends on its interface file
    dependencies: list[str]


class Open(TypedDict):
    open: str


class Abbrev(TypedDict):
    abbrev: str
    full_module: str


OpenOrAbbrev = Union[Open, Abbrev]


class Vconfig(TypedDict):
    initial_fuel: int
    max_fuel: int
    initial_ifuel: int
    max_ifuel: int
    detail_errors: bool
    detail_hint_replay: bool
    no_smt: bool
    quake_lo: int
    quake_hi: int
    quake_keep: bool
    retry: bool
    smtencoding_elim_box: bool
    smtencoding_nl_arith_repr: str
    smtencoding_l_arith_repr: str
    smtencoding_valid_intro: bool
    smtencoding_valid_elim: bool
    tcnorm: bool
    no_plugins: bool
    no_tactics: bool
    z3cliopt: list[str]
    z3smtopt: list[str]
    z3refresh: bool
    z3rlimit: int
    z3rlimit_factor: int
    z3seed: int
    z3version: str
    trivial_pre_for_unannotated_effectful_fns: bool
    reuse_hint_for: Optional[str]


class Range(TypedDict):
    file_name: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int


class DefinitionToCheck(TypedDict):
    file_name: (
        str  # Filename where the definition logically appears (after interleaving)
    )
    name: str  # Fully-qualified name
    opens_and_abbrevs: list[OpenOrAbbrev]
    vconfig: Optional[Vconfig]
    source_type: str  # val block
    source_definition: str  # let block


class Definition(DefinitionToCheck):
    source_range: Range  # The range where the definition's source code is located
    interleaved: bool  # True if this definition was copied from the fsti file
    definition: str
    effect: str
    effect_flags: list[str]
    mutual_with: list[str]
    premises: list[str]
    proof_features: list[str]
    is_simple_lemma: bool
    is_div: bool  # ML/Div (i.e., allows general recursion)
    is_proof: bool  # Whether the type is a prop / squash / has the Lemma effect
    is_simply_typed: bool  # Whether the type is polymorphically simply typed (e.g. `t:Type -> list t -> nat`)
    is_type: bool  # Whether the definition is a type (i.e., the type is of the form `... -> Type/logical/prop`)
    type: str
    prompt: str
    expected_response: str


class Source(TypedDict):
    # Git repository name, e.g. `hacl-star`
    project_name: str
    # File name relative to the repository, e.g. `code/curve25519/Hacl.Impl.Curve25519.Lemmas.fst`
    file_name: str
    # Revision of the git repository
    git_rev: str
    # Url of the git repository
    git_url: str


class InsightFileFirstPass(TypedDict):
    defs: list[Definition]
    dependencies: Dependency


class InsightFile(InsightFileFirstPass):
    source: Source


def eprint(msg):
    # sys.stderr.write(str(msg) + "\n")
    # sys.stderr.flush()
    pass


T = TypeVar("T")


class IdeResponse(TypedDict("Detail", {"query-id": str}), Generic[T]):
    kind: str
    status: str
    response: T


IssueLevel = Literal["error", "warning", "info", "not-implemented"]

IssuePos = tuple[int, int]


class IssueRange(TypedDict):
    fname: str
    beg: IssuePos
    end: IssuePos


class Issue(TypedDict):
    level: IssueLevel
    number: NotRequired[int]
    message: str
    ranges: list[IssueRange]


PushResponse = IdeResponse[list[Issue]]


class Result(TypedDict):
    name: str
    goal_statement: Optional[str]
    full_solution: Optional[str]
    result: bool
    detail: Optional[PushResponse]
    server_crashed: NotRequired[Any]


class UnexpectedResponse(Exception):
    @property
    def response(self):
        return self.args[0]


def assert_response(condition: bool, response: Any):
    if not condition:
        raise UnexpectedResponse(response)


class FStarIdeProcess:
    pushed_until_lid: Optional[str] = None

    def __init__(self, args: list[str], timeout: Optional[float] = None):
        self.process: Any = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            encoding="UTF-8",
        )

        self.qid = 0
        self.timeout = timeout
        self.timeout_happened = False
        # Consume initialization message
        res = self._read_msg()
        assert_response(res["kind"] == "protocol-info", res)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.process.__exit__(exc_type, exc_value, traceback)

    def on_message(self, msg):
        if msg.get("level") != "progress":
            # eprint(msg["contents"])
            pass

    def _read_msg(self) -> Any:
        while True:
            if self.timeout_happened:
                return {"status": "timeout"}
            line = self.process.stdout.readline()
            if line.startswith("{"):
                return json.loads(line)
            elif line == "":
                if self.timeout_happened:
                    return {"status": "timeout"}
                raise Exception(
                    f"fstar terminated with exit code {self.process.poll()}"
                )
            else:
                # fstar writes some --debug output to stdout.
                sys.stderr.write(line)

    def _write_msg(self, msg: Any):
        try:
            json.dump(msg, self.process.stdin)
            self.process.stdin.write("\n")
            self.process.stdin.flush()
        except:
            if self.timeout_happened:
                return {"status": "timeout"}

    def _next_qid(self):
        self.qid += 1
        return str(self.qid)

    def call_simple(self, query: str, args: Any) -> IdeResponse[Any]:
        qid = self._next_qid()
        self._write_msg({"query-id": qid, "query": query, "args": args})
        while True:
            try:
                res = self._read_msg()
                if res["kind"] == "message":
                    self.on_message(res)
                elif res["kind"] == "response":
                    assert_response(res["query-id"] == qid, (res, qid))
                    # eprint(f'result {json.dumps(res)}')
                    return res
                else:
                    raise Exception("Unexpected message from fstar: " + json.dumps(res))
            except:
                if self.timeout_happened:
                    return {"status": "timeout"}

    def call_checked(self, query: str, args: Any):
        res = self.call_simple(query, args)
        # assert_response(res['status'] == 'success', res)
        return res

    def pop_partial_checked(self):
        assert self.pushed_until_lid
        self.call_checked("pop", {})
        self.pushed_until_lid = None

    def load_partial_checked_until(self, until_lid: str):
        if self.pushed_until_lid:
            self.pop_partial_checked()
        self.call_checked("push-partial-checked-file", {"until-lid": until_lid})
        self.pushed_until_lid = until_lid

    def check_snippet_at_decl(
        self, decl_name: str, snippet: str
    ) -> tuple[bool, PushResponse]:
        def timeout_handler():
            self.timeout_happened = True
            self.process.terminate()

        self.load_partial_checked_until(decl_name)

        timer = threading.Timer(self.timeout, timeout_handler)
        timer.start()
        res = self.call_simple(
            "push", {"kind": "full", "line": 0, "column": 0, "code": snippet}
        )
        if res["status"] == "success":
            self.call_checked("pop", {})
        try:
            success = res["status"] == "success"
            if any(err["number"] == Warning_WarnOnUse for err in res["response"]):
                success = False
        except:
            success = False
        finally:
            timer.cancel()
            self.process.terminate()
        return success, res


Warning_WarnOnUse = 335

from dataclasses import dataclass

PoolTask = DefinitionToCheck


@dataclass
class TodoItem:
    task: PoolTask
    on_done: Callable[[Result], None]

    @property
    def file(self):
        return os.path.basename(self.task["file_name"])

    @property
    def defn(self):
        return self.task["name"]


class FStarPool:
    mutex = threading.Lock()
    cv = threading.Condition(mutex)
    todo: dict[str, dict[str, list[TodoItem]]] = {}
    workers: list[threading.Thread]
    cur_worker_file: list[Optional[str]]
    cur_worker_defn: list[Optional[str]]
    _terminated: bool = False

    def __init__(self, dataset_dir: str, extra_args: list[str] = [], nworkers=None):
        if not nworkers:
            nworkers = multiprocessing.cpu_count()
        self.dataset_dir = dataset_dir
        self.extra_args = extra_args
        with self.mutex:
            self.cur_worker_file = [None] * nworkers
            self.cur_worker_defn = [None] * nworkers
            self.workers = []
            for i in range(nworkers):
                thr = threading.Thread(
                    name=f"fstar-worker-{i}",
                    target=self._worker,
                    args=(i,),
                    daemon=True,
                )
                self.workers.append(thr)
                thr.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        with self.cv:
            if self._terminated:
                return
            self._terminated = True
            self.cv.notify_all()
        for thr in self.workers:
            thr.join()

    def _get_next(self, i):
        if len(self.todo) == 0:
            return None

        cur_file = self.cur_worker_file[i]
        cur_defn = self.cur_worker_defn[i]

        def go(f, d):
            item = self.todo[f][d].pop()
            if len(self.todo[f][d]) == 0:
                del self.todo[f][d]
            if len(self.todo[f]) == 0:
                del self.todo[f]
            return item

        if cur_file in self.todo:
            if cur_defn in self.todo[cur_file]:
                return go(cur_file, cur_defn)

            for d in self.todo[cur_file].keys():
                if d not in self.cur_worker_defn:
                    return go(cur_file, d)

            return go(cur_file, random.choice(list(self.todo[cur_file])))

        for f in self.todo.keys():
            if f not in self.cur_worker_file:
                return go(f, next(iter(self.todo[f].keys())))

        f = random.choice(list(self.todo))
        return go(f, random.choice(list(self.todo[f])))

    def _worker(self, i):
        cur_file: Optional[str] = None
        proc: Optional[FStarIdeProcess] = None

        while True:
            with self.cv:
                item = self._get_next(i)
                while item is None:
                    if self._terminated:
                        return
                    else:
                        self.cv.wait()
                        item = self._get_next(i)
                item_file = item.file
                self.cur_worker_defn[i] = item.task["name"]
                self.cur_worker_file[i] = item_file

            if cur_file != item_file or proc is None:
                if proc is not None:
                    proc.process.terminate()
                proc = create_fstar_process_for_dataset(
                    item_file, self.dataset_dir, self.extra_args
                )
                cur_file = item_file

            try:
                proc.load_partial_checked_until(item.task["name"])
                res = process_one_instance(item.task, proc)
                item.on_done(res)
            except BaseException as e:
                proc = None
                cur_file = None
                if isinstance(e, UnexpectedResponse):
                    detail = e.response
                else:
                    detail = str(e)
                item.on_done(
                    {
                        "name": item.defn,
                        "result": False,
                        "detail": None,
                        "server_crashed": detail,
                        "goal_statement": None,
                        "full_solution": None,
                    }
                )

    def _enqueue(self, item: TodoItem):
        item_file = item.file
        if item_file not in self.todo:
            self.todo[item_file] = {}
        if item.defn not in self.todo[item_file]:
            self.todo[item_file][item.defn] = []
        self.todo[item_file][item.defn].append(item)

    def _submit(self, items: Iterable[TodoItem]):
        with self.cv:
            for item in items:
                self._enqueue(item)
            self.cv.notify_all()  # TODO: try to wake up workers with same file first

    def process_instances_unordered_enumerated(
        self, tasks: list[PoolTask]
    ) -> Iterable[tuple[int, Result]]:
        q = queue.SimpleQueue()

        def mk_item(i: int, task: PoolTask) -> TodoItem:
            return TodoItem(on_done=lambda res: q.put((i, res)), task=task)

        self._submit(mk_item(i, task) for i, task in enumerate(tasks))
        for _ in range(len(tasks)):
            yield q.get()

    def process_instances_unordered(self, tasks: list[PoolTask]) -> Iterable[Result]:
        for _, r in self.process_instances_unordered_enumerated(tasks):
            yield r

    def process_instances(
        self, tasks: list[PoolTask], progressbar=False
    ) -> list[Result]:
        result: list = [None] * len(tasks)
        stream = self.process_instances_unordered_enumerated(tasks)
        if progressbar:
            from tqdm import tqdm

            stream = tqdm(stream, total=len(tasks))
        for i, r in stream:
            result[i] = r
        return result


def build_options_scalfolding(entry):
    # translate vconfig to an option string
    # for each key/value pair in vconfig, add an element to an array of strings with the key and value
    options = []
    for key, value in (entry["vconfig"] or {}).items():
        match key:
            case "z3cliopt" | "z3smtopt":
                for val in value:
                    options.append("--" + key)
                    options.append(f"'{val}'")
                continue
            case (
                "initial_fuel"
                | "max_fuel"
                | "initial_ifuel"
                | "max_ifuel"
                | "z3rlimit"
                | "z3rlimit_factor"
                | "z3seed"
            ):
                value = str(value)
            case "z3refresh":
                if value:
                    options.append("--z3refresh")
                    continue
                else:
                    continue
            case "smtencoding_elim_box":
                key = "smtencoding.elim_box"
                value = "true" if value else "false"
            case "smtencoding_nl_arith_repr":
                key = "smtencoding.nl_arith_repr"
                value = str(value)
            case "smtencoding_l_arith_repr":
                key = "smtencoding.l_arith_repr"
                value = str(value)
            case "smtencoding_valid_intro":
                key = "smtencoding.valid_intro"
                value = "true" if value else "false"
            case "smtencoding_valid_elim":
                key = "smtencoding.valid_elim"
                value = "true" if value else "false"
            case (
                "retry"
                | "detail_errors"
                | "reuse_hint_for"
                | "no_plugins"
                | "no_tactics"
                | "no_smt"
                | "quake_lo"
                | "quake_hi"
                | "quake_keep"
                | "tcnorm"
                | "trivial_pre_for_unannotated_effectful_fns"
                | "detail_hint_replay"
            ):
                continue
            case _:
                continue
        options.append("--" + key)
        options.append(str(value))
    options_string = " ".join(options)
    scaffolding = f'#push-options "{options_string}"\n'
    return scaffolding


def build_scaffolding(entry: DefinitionToCheck):
    scaffolding = ""
    module_name = os.path.splitext(os.path.basename(entry["file_name"]))[0]
    if module_name == "prims":
        module_name = "Prims"
    opens, abbrevs = [], []
    if module_name != "Prims":
        for oa in entry["opens_and_abbrevs"][::-1]:
            if "abbrev" in oa and oa["abbrev"]:
                key = "short_module" if "short_module" in oa else "abbrev"
                abbrevs.append("module " + oa[key] + " = " + oa["full_module"] + "\n")
            else:
                module_key = "open" if "open" in oa else "full_module"
                opens.append("open " + oa[module_key] + "\n")
        scaffolding += "".join(opens)
        scaffolding += "".join(abbrevs)
        scaffolding += "open " + module_name + "\n"
    scaffolding += build_options_scalfolding(entry)
    return scaffolding


def process_one_instance(
    entry: DefinitionToCheck, fstar_process: FStarIdeProcess
) -> Result:
    scaffolding = build_scaffolding(entry)
    lemma_long_name = entry["name"]
    goal = entry["source_type"]
    if goal == "<UNK>":
        goal = ""
    solution = entry["source_definition"]
    full_soln = f"{scaffolding}\n{goal}\n{solution}"
    result, detail = fstar_process.check_snippet_at_decl(entry["name"], full_soln)
    return {
        "name": lemma_long_name,
        "goal_statement": goal,
        "full_solution": full_soln,
        "result": result,
        "detail": detail,
    }


def analyze_solution(
    entry: Definition,
    goal_statement: str,
    solution: str,
    fstar_process: FStarIdeProcess,
    check_name_match: bool = True,
):
    scaffolding = build_scaffolding(entry)
    lemma_long_name = entry["name"]
    name = lemma_long_name
    if "." in name:
        name = name.split(".")[-1].strip()
    mandatory_part = f"{name}"
    solution = "\n" + solution.strip()
    full_soln = f"{scaffolding}\n#restart-solver\n{goal_statement} {solution}"
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
    if not check_name_match or mandatory_part in solution.strip():
        result, detail = fstar_process.check_snippet_at_decl(entry["name"], full_soln)
    else:
        result = False
        detail = {
            "kind": "none",
            "query-id": "none",
            "status": "failure",
            "response": [
                {
                    "level": "error",
                    "number": 999,
                    "message": "Wrong name in solution",
                    "ranges": [],
                }
            ],
        }
    logged_solution = {
        "name": lemma_long_name,
        "goal_statement": goal_statement,
        "full_solution": full_soln,
        "result": result,
        "detail": detail,
    }
    return logged_solution


def should_ignore(entry: Definition) -> Optional[str]:
    if entry["interleaved"]:
        return "interleaved"
    if entry["definition"].startswith("<"):
        return "nondefinition"
    if "=" not in entry["source_definition"]:
        # QueryCheckedFile messages up `type =` declarations.
        return "nondefinition (type)"
    if entry["file_name"] == "dummy":
        return "unreal lemma"
    return None


def create_fstar_process_for_dataset(
    file_name: str,
    dataset_dir: str,
    extra_args: list[str] = [],
    timeout: Optional[float] = None,
) -> FStarIdeProcess:
    return FStarIdeProcess(
        [
            "fstar.exe",
            "--ide",
            os.path.basename(file_name),
            "--report_assumes",
            "warn",
            "--include",
            dataset_dir,
        ]
        + extra_args,
        timeout=timeout,
    )


def create_fstar_process_for_json_file(
    json_data: InsightFile, dataset_dir: str, extra_args: list[str] = []
) -> FStarIdeProcess:
    return create_fstar_process_for_dataset(
        json_data["dependencies"]["source_file"], dataset_dir, extra_args
    )


# for each entry in the json file, send the query to fstar insights
def send_queries_to_fstar(json_data: InsightFile, dataset_dir: str):
    outputs = []
    extra_args = [
        # '--trace_error',
        # '--debug', 'FStar.Array',
        # '--debug_level', 'Rel,RelCheck,High',
    ]
    with create_fstar_process_for_json_file(
        json_data, dataset_dir, *extra_args
    ) as fstar_process:
        # for each entry in the json file
        for entry in json_data["defs"]:
            if reason := should_ignore(entry):
                # eprint(f'Ignoring {entry["name"]}: {reason}')
                continue
            # send the query to fstar insights
            out = process_one_instance(entry, fstar_process)
            # if out['result']:
            #     eprint(f'Verified {out["name"]}')
            # else:
            #     eprint(f'Failed {out["name"]}')
            outputs.append(out)
        return outputs


def pool_tasks_of_file(json_data: InsightFile, warn=False) -> list[PoolTask]:
    items: list[PoolTask] = []
    for entry in json_data["defs"]:
        if reason := should_ignore(entry):
            # if warn:
            #     eprint(f'Ignoring {entry["name"]}: {reason}')
            continue
        items.append(entry)
    return items


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
    example, check_ground_truth, solution_key, timeout = inp
    name = example["name"]
    if "." in name:
        name = name.split(".")[-1]

    if check_ground_truth:
        responses = [example["source_definition"]]
    else:
        responses = example[solution_key]
        responses = [sanitize(r) for r in responses]
    results = [None] * len(responses)
    truths = [False] * len(responses)
    for ri, response in enumerate(responses):
        proc = create_fstar_process_for_dataset(
            example["file_name"], DATSET_DIR, [], timeout=timeout
        )
        try:
            proc.load_partial_checked_until(example["name"])
            goal_stmt = example["original_source_type"]
            goal_stmt = goal_stmt.strip()
            if goal_stmt == "" or goal_stmt == "<UNK>":
                goal_stmt = f""
            res = analyze_solution(
                entry=example,
                goal_statement=goal_stmt,
                solution=response,
                fstar_process=proc,
                check_name_match=not check_ground_truth,
            )
            res["checked_solution"] = response
            results[ri] = res
            truths[ri] = res["result"] if res is not None else False
        except Exception as e:
            traceback.print_exc()
            if proc is not None:
                proc.process.terminate()
            if isinstance(e, KeyboardInterrupt):
                raise e
            res = None
            results[ri] = res
            truths[ri] = False
        except:
            traceback.print_exc()
            if proc is not None:
                proc.process.terminate()
            res = None
            results[ri] = res
            truths[ri] = False
        if proc is not None:
            proc.process.terminate()
    return (example, results, truths, evaluation_function(truths))


class Evaluator:
    def __init__(self):
        current_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
        self.full_data_path = os.path.join(current_dir, "helpers/full_data.json")
        self.full_data = json.load(open(self.full_data_path, "r"))

    def check_solution(self, example_name: str, solution: str, timeout: int = 300):
        example = copy.copy(self.full_data[example_name])
        example["generated_response"] = [solution]
        example, res, truths, _ = check_example(
            (example, False, "generated_response", timeout)
        )
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
    return np.mean([calculate_pass_at_k(truths, k) for truths in list_of_truths])

def main():
    args = get_argument()
    generated_files = args.input_files
    tasks = []
    for gf in generated_files:
        with open(gf, "r") as fp:
            tasks.extend(json.load(fp))
            fp.close()

    for t in tasks:
        if "file_name" not in t:
            tmp_t = FULL_DATA[t["name"]]
            for k in tmp_t:
                if k not in t:
                    t[k] = tmp_t[k]
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
