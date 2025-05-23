import openai
import nltk
from nltk.tokenize import word_tokenize
from textwrap import dedent
import re
import copy
import xml.etree.ElementTree as ET
import textwrap

problem_definition = """
The type of the term is:
```
val snd : x: (_ * _) -> _
```
#####

The file context where I am writing this term is:
```
module Duplex.PCM

open FStar.PCM

open Steel.Channel.Protocol
module P = FStar.Preorder
module R = FStar.ReflexiveTransitiveClosure


open FStar.Ghost
open Steel.Memory
open Steel.Effect.Atomic
open Steel.Effect
open Steel.PCMReference
module Mem = Steel.Memory
module PR = Steel.PCMReference

#####

A premise is other terms either defined in the same file or imported from other files.
Here are complete type and definition of the premises in the input type:
#####

Here are complete type and definition of the other premises. These premises are used to define the terms that make up the current input type:
#####

Given the above information, please provide the definition of the following type:
```
val snd : x: (_ * _) -> _
```


Start the definition with 
```
let snd =
```
"""

resp = """<think>
    <reflection>
        The high level strategy is to synthesize a verifiable definition for the F* term `snd`, which has the type `x: (_ * _) -> _`. Since the provided context includes various modules and definitions, I need to ensure that the solution adheres to F*'s syntax and any specific conventions used in the context. However, the type signature of `snd` is straightforward\u2014it should take a pair and return its second component. The standard F* library already has a `snd` function, but if we need to define it explicitly here, it can be done simply by pattern matching or using existing functions. The challenge here is to ensure that the definition is compatible with the given context and any type inference requirements. Since the return type is inferred (`_`), the function must correctly return the second element of the pair without any additional constraints from the context provided. The context includes many protocol and channel-related definitions, but since the `snd` function is generic, it should work with any pair type. Therefore, the solution is to directly implement the standard `snd` function.
    </reflection>
    <steps>
        <step>
            Identify that the function `snd` is supposed to return the second element of a pair.
        </step>
        <step>
            Recall that in F*, pairs are tuples and the second element can be accessed via pattern matching or using the built-in `snd` function.
        </step>
        <step>
            Since the problem requires writing the definition explicitly (even though F* might have a built-in), we need to define it using a lambda or a function that extracts the second component.
        </step>
        <step>
            Check if there are any constraints in the provided context that might affect the definition. The context includes many protocol definitions, but the `snd` function here is generic and doesn't depend on them.
        </step>
        <step>
            Decide to write the definition using pattern matching for clarity and correctness.
        </step>
    </steps>
    <emulation>
        <step>
            Starting with the skeleton `let snd = ...`, the function needs to take a pair `x` of any two types.
        </step>
        <step>
            Define the function using a lambda that pattern matches on the pair: `fun (a, b) -> b`.
        </step>
        <step>
            Ensure that the type annotation matches the given type `x: (_ * _) -> _`, which it does since the lambda's type is exactly `('a * 'b) -> 'b`, fitting the universal quantification
        </step>
        <step>
            Verify that no additional F* features (like effects or refinements) are needed here, as the problem doesn't specify any beyond the basic pair decomposition.
        </step>
    </emulation>
</think>
<answer>
    let snd = fun (x, y) -> y
</answer>"""


# Ensure the tokenizer resources are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def sanitize(text):
    lines = text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(lines)


def step_answer_match(step, answer):
    a_tokens = word_tokenize(step)
    b_tokens = word_tokenize(answer)
    # If either string is empty, return 0
    if not a_tokens or not b_tokens:
        return 0.0
    # Classic DP solution for LCS
    dp = [[0] * (len(b_tokens) + 1) for _ in range(len(a_tokens) + 1)]
    for i in range(1, len(a_tokens) + 1):
        for j in range(1, len(b_tokens) + 1):
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    l = dp[len(a_tokens)][len(b_tokens)]
    return (float(l) / len(b_tokens)) if len(b_tokens) > 0 else 0


def normalize_all_tags(text: str) -> str:
    """
    Normalize the text removing the tags and replacing them with sections.
    First find the tags and then replace them with the corresponding sections.
    Replace all the tags
    """
    def first_letter_uppercase(word):
        return word[0].upper() + word[1:].lower()

    def replace_tag(match):
        tag = match.group(1)
        return f"# {first_letter_uppercase(tag)}\n"

    # Replace all tags with their normalized form
    text = re.sub(r"[ \t]*<\s*([a-zA-Z0-9_]+)\s*>\s*", replace_tag, text)
    text = re.sub(r"[ \t]*</\s*([a-zA-Z0-9_]+)\s*>\s*", "", text)
    return text.strip()


def xml_to_markdown(xml_string):
    def format_text(text):
        if not text or not text.strip():
            return ""
        return text

    def parse_node(node, level=1):
        md = ""
        indent = "#" * level
        title = node.tag.replace("_", " ").title()
        md += f"\n{indent} {title}\n"
        if node.text and node.text.strip():
            md += f"{format_text(node.text)}\n"
        children = list(node)
        if children:
            for i, child in enumerate(children, start=1):
                md += parse_node(child, level + 1)
        return md

    root = ET.fromstring(xml_string)
    markdown_output = parse_node(root)
    markdown_output = re.sub(f"[\n]+", "\n", markdown_output)
    markdown_output = re.sub(r"[ \t]+", " ", markdown_output)
    lines = markdown_output.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    markdown_output = "\n".join(lines)
    return markdown_output.strip()


def try_parse_and_normalize_with_xml_parser(_think_block: str) -> str:
    think_block = copy.deepcopy(_think_block)
    think_block = f"<Reasoning>{think_block}</Reasoning>"
    try:
        return xml_to_markdown(think_block)
    except ET.ParseError:
        return None
        

def parse_think_xml_with_optional_blocks(xml_text: str):
    xml_text = dedent(xml_text.strip())
    def extract_block(tag: str) -> str:
        """Extracts a single block like <reflection>...</reflection>"""
        pattern = fr"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, xml_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_step_list(tag: str) -> list[str]:
        """Extracts all <step>...</step> within a parent block"""
        block = extract_block(tag)
        step_pattern = r"<step>\s*(.*?)\s*</step>"
        return [s.strip() for s in re.findall(step_pattern, block, re.DOTALL | re.IGNORECASE)]

    def extract_think_block() -> str:
        text = copy.deepcopy(xml_text)
        if "<think>" in xml_text:
            text = text[(text.index("<think>") + len("<think>")):]
        if "</think>" in text:
            text = text[:text.index("</think>")]
        return text.strip()

    think_block = extract_think_block()
    normalized_text = try_parse_and_normalize_with_xml_parser(think_block)
    if normalized_text is None:
        normalized_text = normalize_all_tags(think_block)

    return {
        "reflection": sanitize(extract_block("reflection")),
        "steps": [sanitize(s) for s in extract_step_list("steps")],
        "emulation": [sanitize(s) for s in extract_step_list("emulation")],
        "answer": sanitize(extract_block("answer")),
        "normalized_think_block": normalized_text,
    }


class FormattedResponse:
    def __init__(self, problem_prefect: str, problem_desciption: str, response: str):
        if problem_prefect == "fstar":
            self.problem_prefect = "The problem is about synthesizing a verifiable definition for a F* term given the type of the term."
        elif problem_prefect == "verus":
            self.problem_prefect = "The problem is about synthesizing verified rust code with necessary invariants and assertion given a rust code with specification."
        else:
            raise ValueError(
                "Invalid problem prefix. Expected 'fstar' or 'verus'.")
        self.problem_description = problem_desciption
        self.original_text = response

        result = parse_think_xml_with_optional_blocks(response)
        self.reflection = result["reflection"]
        self.steps = result["steps"]
        self.emulation = result["emulation"]
        self.answer = result["answer"]
        self.full_reasoning = result["normalized_think_block"]
        try:
            self.client = openai.OpenAI(
                base_url="http://localhost:8000/v1", api_key="sk-")
            self.model = self.client.models.list().data[0].id
        except Exception as e:
            self.client = None
            self.model = None
            print("Error initializing OpenAI client:", e)

    def __str__(self):
        ret_val = f"""Reflection:
        {self.reflection}
                
        Steps:
        =======================================
        """
        for si, step in enumerate(self.steps):
            ret_val += f"Step {si + 1}: {step}\n"
        ret_val += f"""Emulation:
        =======================================
        """
        for ei, emu in enumerate(self.emulation):
            ret_val += f"Emulation {ei + 1}: {emu}\n"
        ret_val += f"""Answer:
        =======================================
        {self.answer}"""
        return ret_val

    def is_well_formed(self, reflection_expected: bool = False, steps_expected: bool = False, emulation_expected: bool = False) -> bool:
        """
        Check if the response is well-formed based on the expected presence of reflection, steps, and emulation.
        """
        if reflection_expected and (self.reflection is None or self.reflection == ""):
            return False
        if steps_expected and (len(self.steps) == 0 or any(s.strip() == "" for s in self.steps)):
            return False
        if emulation_expected and (len(self.emulation) == 0 or any(s.strip() == "" for s in self.emulation)):
            return False
        if self.answer.strip() == "":
            return False
        return True

    def does_steps_match_emulation(self) -> bool:
        return len(self.steps) == len(self.emulation)

    def answer_matches_steps(self) -> float:
        """
        Check if the answer matches the steps.
        """
        return step_answer_match(
            step=self.emulation[-1], answer=self.answer
        ) if len(self.emulation) > 0 and self.answer != "" else 0.0

    def find_score_in_response(self, response_text: str) -> float:
        # Score can be in the format of <score> float_score </score> format
        # or it is a floating point number after the last </think> tag
        score_match = re.search(r"<score>(.*?)</score>", response_text)
        if score_match:
            return float(score_match.group(1))
        else:
            # If no <score> tag, try to find the last floating point number in the response
            last_number_match = re.search(r"[-+]?\d*\.\d+|\d+", response_text)
            if last_number_match:
                return float(last_number_match.group(0))
            else:
                raise ValueError("No score found in the response text.")

    def call_model_with_retry(self, prompt, max_trial=10, temparature=0.0):
        while max_trial > 0:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    max_tokens=5000,
                    temperature=temparature,
                )
                response_text = response.choices[0].message.content
                # Extract the score from the response
                score = self.find_score_in_response(response_text)
                print("====================")
                print(response_text)
                print("====================")
                return score
            except Exception as e:
                max_trial -= 1
                temparature += 0.1

        return 0.0

    def step_emulation_conformity_score(self) -> float:
        """
        Check if the steps and emulation conform to each other.
        """
        # Create a prompt for the model
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant that is experiences with logical reasoning. The user is trying to solve a problem related to synthesizing a verifiable code from a given code or specification. In order to do so, the user came up with a list of steps. The user also has simulated these steps in their mind and articulated the simulation in a natural language. The user is asking you to check if the steps and the emulation are conforming to each other. Please answer with a score between 0 and 1, where 0 means they do not conform at all and 1 means they are exactly the same. You should first think about the conformity between the steps and the emulation. If there is a conformity, think about the reasoning behind it. If there is a disagreement, think about the reason why they do not conform. Then, give a score between 0 and 1. You should also articulate your thought process for the score you wil give. You should format your answer in \n<think>\n\tprovide your thought process of scoring\n</think>\n<score>\n\tYour given score between 0 and 1\n</score>\n format. Do not include any other text in between <score> and </score> tags.",
            }
        ]
        user_prompt = f"{self.problem_prefect}\n\n" + \
            self.problem_description + "\n"
        user_prompt += "\nThe steps are: "
        for i, step in enumerate(self.steps):
            user_prompt += f"\nStep {i + 1}: {step}"

        user_prompt += "\nThe user simulation of these steps are: "
        for i, emu in enumerate(self.emulation):
            user_prompt += f"\nSimulation {i + 1}: {emu}"

        prompt.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        return self.call_model_with_retry(prompt)
    
    def full_reasoning_lead_to_answer_score(self) -> float:
        # Here we will check if given a problem definition, the developers reasoning does indeed reach the correct answer. If we can exactly reach the answer, we will return 1.0, if we can reach a similar answer, we will return a score between 0 and 1. If we cannot reach the answer at all, we will return 0.0.
        # Create a prompt for the model
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant that is experiences with logical reasoning. The user is trying to solve a problem related to synthesizing a verifiable code from a given code or specification. In order to do so, the user came up with some reasoning. The user is asking you to check if the reasoning leads to the answer. Please answer with a score between 0 and 1, where 0 means you cannot derive the answer following the reasoning and 1 means you can derive the answer by following the reasoning. While scoring the reasoning, you should consider the following.\n1. If the reasoning is logically sound, a sound reasoning should get higher score than unsound one\n2. If the reasoning is aligned with solving the problem and if the reasonign is making unjust assumption. A reasoning that aligns with solving the problem with fewer unjust assumption should get higher score.\n3. If the reasoning does indeed reach to the solution. If someone with no experience could independently follow the reasoning process and reach to the solution that reasoning should get higher score.\n You should also articulate your thought process for the score you wil give. You should format your answer in \n<think>\n\tprovide your thought process of scoring\n</think>\n<score>\n\tYour given score between 0 and 1\n</score>\n format. Do not include any other text in between <score> and </score> tags.",
            }
        ]
        user_prompt = f"{self.problem_prefect}\n\n" + \
            self.problem_description + "\n"
        user_prompt += "\nThe developers reasoning is: "
        for i, step in enumerate(self.steps):
            user_prompt += f"\nStep {i + 1}: {step}"

        prompt.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        return self.call_model_with_retry(prompt)

    def step_answer_conformity_score(self) -> float:
        # Here we will check if given a problem definition, and list of steps, if we could reach the answer. If we can exactly reach the answer, we will return 1.0, if we can reach a similar answer, we will return a score between 0 and 1. If we cannot reach the answer at all, we will return 0.0.
        # Create a prompt for the model
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant that is experiences with logical reasoning. The user is trying to solve a problem related to synthesizing a verifiable code from a given code or specification. In order to do so, the user came up with a list of steps. The user alse has prepared an answer. The user is asking you to check if following the steps, the user could actually reach the answer. Please answer with a score between 0 and 1, where 0 means you cannot derive the answer following the steps and 1 means you can derive the answer by following the steps. While scoring the steps, you should think about the logical deduction for each of the steps and how each of these step transform the problem. You should deeply reason about why the list of given steps are correct or not to reach the answer. Then, give a score between 0 and 1. You should also articulate your thought process for the score you wil give. You should format your answer in \n<think>\n\tprovide your thought process of scoring\n</think>\n<score>\n\tYour given score between 0 and 1\n</score>\n format. Do not include any other text in between <score> and </score> tags.",
            }
        ]
        user_prompt = f"{self.problem_prefect}\n\n" + \
            self.problem_description + "\n"
        user_prompt += "\nThe steps are: "
        for i, step in enumerate(self.steps):
            user_prompt += f"\nStep {i + 1}: {step}"
        user_prompt += "\nThe answer is: "
        user_prompt += f"\n{self.answer}"
        prompt.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        return self.call_model_with_retry(prompt)


# Example usage
fmt = FormattedResponse(problem_prefect="fstar",
                        problem_desciption=problem_definition, response=resp)
            
# print(fmt)
print("Is well-formed (Reflection Only): ",
      fmt.is_well_formed(reflection_expected=True))
print("Is well-formed (Steps Only): ", fmt.is_well_formed(steps_expected=True))
print("Is well-formed (Emulation Only): ",
      fmt.is_well_formed(emulation_expected=True))
print("Is well-formed (Reflection + Steps): ",
      fmt.is_well_formed(reflection_expected=True, steps_expected=True))
print("Is well-formed (Reflection + Emulation): ",
      fmt.is_well_formed(reflection_expected=True, emulation_expected=True))
print("Is well-formed (Steps + Emulation): ",
      fmt.is_well_formed(steps_expected=True, emulation_expected=True))
print("Is well-formed (All): ", fmt.is_well_formed(reflection_expected=True,
      steps_expected=True, emulation_expected=True))
print("Step Emulation match emulation:", fmt.step_emulation_conformity_score())
print("Step Answer conformity check:", fmt.step_answer_conformity_score())
print("Full reasoning lead to answer score:", fmt.full_reasoning_lead_to_answer_score())

