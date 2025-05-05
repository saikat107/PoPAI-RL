import os
import sys
import json


# Add the project directory to the Python path
current_dir = os.getcwd()
project_dir = os.path.realpath(os.path.join(current_dir, ".."))
sys.path.append(project_dir)

# FSrar Evaluation
"""
Expected output (Printed in the console):

Success: True
{
  "name": "FStar.Math.Lemmas.multiple_modulo_lemma",
  "goal_statement": "val multiple_modulo_lemma (a:int) (n:pos) : Lemma ((a * n) % n = 0)",
  "full_solution": "open FStar\nopen Prims\nopen FStar.Pervasives\nopen FStar.Math\nopen FStar.Math\nopen FStar.Mul\nopen FStar.Math.Lib\nopen FStar.Math.Lemmas\n#push-options \"--initial_fuel 0 --max_fuel 0 --initial_ifuel 0 --max_ifuel 0 --smtencoding.elim_box false --smtencoding.nl_arith_repr boxwrap --smtencoding.l_arith_repr boxwrap --smtencoding.valid_intro true --smtencoding.valid_elim false --z3rlimit 5 --z3rlimit_factor 1 --z3seed 0\"\n\n#restart-solver\nval multiple_modulo_lemma (a:int) (n:pos) : Lemma ((a * n) % n = 0) \nlet multiple_modulo_lemma (a:int) (n:pos) = cancel_mul_mod a n",
  "result": true,
  "detail": {
    "kind": "response",
    "query-id": "4",
    "status": "success",
    "response": []
  },
  "checked_solution": "let multiple_modulo_lemma (a:int) (n:pos) = cancel_mul_mod a n"
}
"""
from evaluation.FStar.fstar_evaluation import Evaluator as FStarEvaluator
fstar_eval = FStarEvaluator()

name = "FStar.Math.Lemmas.multiple_modulo_lemma"
solution = "let multiple_modulo_lemma (a:int) (n:pos) = ()"

result, details = fstar_eval.check_solution(name, solution)
print("Success:", result)
print(json.dumps(details, indent=2))

exit()


# Verus Evaluation
"""
Expected output (Printed in the console):

Success: True
{
  "name": "VERUS:13232",
  "solution": "use vstd::prelude::*;\n\nfn main() {}\n\nverus!{\nfn func(a: usize, b: usize, c: usize) -> (r: bool)\n    requires\n        1 <= a && a <= 100,\n        1 <= b && b <= 100,\n        1 <= c && c <= 100\n    ensures\n        r ==> (a + b == c || a + c == b || b + c == a)\n{\n    proof {\n        assert(1 <= a && a <= 100);\n        assert(1 <= b && b <= 100);\n        assert(1 <= c && c <= 100);\n    }\n    return a + b == c || a + c == b || b + c == a;\n}\n}",
  "result": true,
  "detail": {
    "status": "success",
    "response": "verification results:: 1 verified, 0 errors\n"
  },
  "checked_solution": "use vstd::prelude::*;\n\nfn main() {}\n\nverus!{\nfn func(a: usize, b: usize, c: usize) -> (r: bool)\n    requires\n        1 <= a && a <= 100,\n        1 <= b && b <= 100,\n        1 <= c && c <= 100\n    ensures\n        r ==> (a + b == c || a + c == b || b + c == a)\n{\n    proof {\n        assert(1 <= a && a <= 100);\n        assert(1 <= b && b <= 100);\n        assert(1 <= c && c <= 100);\n    }\n    return a + b == c || a + c == b || b + c == a;\n}\n}"
}
"""
from evaluation.Verus.verus_evaluation import Evaluator as VerusEvaluator
verus_eval = VerusEvaluator()
name = "VERUS:13232"
solution = "use vstd::prelude::*;\n\nfn main() {}\n\nverus!{\nfn func(a: usize, b: usize, c: usize) -> (r: bool)\n    requires\n        1 <= a && a <= 100,\n        1 <= b && b <= 100,\n        1 <= c && c <= 100\n    ensures\n        r ==> (a + b == c || a + c == b || b + c == a)\n{\n    proof {\n        assert(1 <= a && a <= 100);\n        assert(1 <= b && b <= 100);\n        assert(1 <= c && c <= 100);\n    }\n    return a + b == c || a + c == b || b + c == a;\n}\n}"

result, details = verus_eval.check_solution(name, solution)
print("Success:", result)
print(json.dumps(details, indent=2))
