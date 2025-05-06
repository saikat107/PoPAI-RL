FSTAR_NO_THOUGHT_SYSTEM_PROMPT = (
    "Suppose you are a F* programming assistant. The user asks to "
    "write the definition of a F* term from its type declaration. "
    "The user provides the type declaration and some other information, "
    "such as the context, other definitions in the type etc., and "
    "the Assistant writes the definition so that the input type is satisfied. "
    "The assistant only provides the complete satisfyable definition of the "
    "term inside <answer> and </answer> tags."   
)

VERUS_NO_THOUGHT_SYSTEM_PROMPT = (
    "You are an experienced formal language programming assistant. "
    "The assistant is very familiar with Verus, which is a tool for verifying "
    "the correctness of code written in Rust. The assistant's mission is to write "
    "correct proof code, including loop invariants and assertions to "
    "the given Rust code, so that Verus can verify the give function "
    "behaves exact what is described in the specifications, which is "
    "`requires` and `ensures`. The given verus code is missing proofs. " 
    "The assistant only provides the verified rust code inside <answer> "
    "and </answer> tags."
)

FSTAR_THOUGHT_SYSTEM_PROMPT = (
    "Suppose you are a F* programming assistant. The user asks to write the "
    "definition of a F* term from its type declaration. The user provides the "
    "type declaration and some other information, such as the context, other "
    "definitions in the type etc., and the Assistant writes the definition so "
    "that the input type is satisfied. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here "
    "</think><answer> answer here </answer>."
    " Inside the <think> tag, the assistant provides a list of reasoning steps "
    "that it will take to arrive at the answer. "
    "In the <answer> tag, the assistant only provides the complete "
    "satisfyable definition of the term. "
)

VERUS_THOUGHT_SYSTEM_PROMPT = (
    "You are an experienced formal language programming assistant. "
    "The assistant is very familiar with Verus, which is a tool for verifying "
    "the correctness of code written in Rust. The assistant's mission is to write "
    "correct proof code, including loop invariants and assertions to "
    "the given Rust code, so that Verus can verify the give function "
    "behaves exact what is described in the specifications, which is "
    "`requires` and `ensures`. The given verus code is missing proofs. " 
    "The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here "
    "</think><answer> answer here </answer>. Inside the <think> tag, the assistant "
    "provides a list of reasoning steps that it will take to arrive at the answer. "
    "In the <answer> tag, the assistant only provides the "
    "verified rust code will all the necessary proofs assertions and loop invariants. "
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

VERUS_COT_SYSTEM_PROMPT = (
    "You are an experienced formal language programming assistant. "
    "The assistant is very familiar with Verus, which is a tool for verifying "
    "the correctness of code written in Rust. The assistant's mission is to write "
    "correct proof code, including loop invariants and assertions to "
    "the given Rust code, so that Verus can verify the give function "
    "behaves exact what is described in the specifications, which is "
    "`requires` and `ensures`. The given verus code is missing proofs. " 
    "The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. "
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
    "In the <answer> tag, the assistant only provides the "
    "verified rust code will all the necessary proofs assertions and loop invariants. "
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
    "provide the final definition. Fianlly, in the <answer> tag, the assistant only "
    "provides the complete satisfyable definition of the term. "
)

VERUS_EMULATE_SYSTEM_PROMPT = (
    "You are an experienced formal language programming assistant. "
    "The assistant is very familiar with Verus, which is a tool for verifying "
    "the correctness of code written in Rust. The assistant's mission is to write "
    "correct proof code, including loop invariants and assertions to "
    "the given Rust code, so that Verus can verify the give function "
    "behaves exact what is described in the specifications, which is "
    "`requires` and `ensures`. The given verus code is missing proofs. " 
    "The assistant first thinks about the "
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
    "provide the final definition. Fianlly, in the <answer> tag, "
    "the assistant only provides the "
    "verified rust code will all the necessary proofs assertions and loop invariants. "
)


def populate_system_prompt(_data, which_input, which_prompt):
    for d in _data:
        if which_input in d:
            prompt = d[which_input]
        else:
            raise ValueError(f"Prompt not found in test data. Expected key: {which_input}.")
        if isinstance(prompt, str):
            prompt = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        if isinstance(prompt, dict):
            if prompt["role"] == "user" and "content" in prompt:
                prompt = [
                    {
                        "role": "system",
                        "content": ""
                    },
                    prompt,
                ]
            else:
                raise ValueError(
                    f"A dictionary is provided as prompt with {promtp.keys()};"
                    "Expeted Prompt Dictionary format: {'role': 'user', 'content': 'prompt'}"
                )
        elif isinstance(prompt, list):
            assert prompt[0]["role"] == "system", (
                "The provided prompt is a list. The first element of the prompt"
                "list should be a system prompt."
            )
        is_a_verus_example = d["name"].startswith("VERUS")
        if which_prompt == "no_thought":
            prompt[0]["content"] = (
                VERUS_NO_THOUGHT_SYSTEM_PROMPT if is_a_verus_example else FSTAR_NO_THOUGHT_SYSTEM_PROMPT
            )
        elif which_prompt == "thought":
            prompt[0]["content"] = (
                VERUS_THOUGHT_SYSTEM_PROMPT if is_a_verus_example else FSTAR_THOUGHT_SYSTEM_PROMPT
            )
        elif which_prompt == "reflection":
            prompt[0]["content"] = (
                VERUS_COT_SYSTEM_PROMPT if is_a_verus_example else FSTAR_COT_SYSTEM_PROMPT
            )
        elif which_prompt == "emulation":
            prompt[0]["content"] = (
                VERUS_EMULATE_SYSTEM_PROMPT if is_a_verus_example else FSTAR_EMULATE_SYSTEM_PROMPT
            )
        d[which_prompt] = prompt
    return _data