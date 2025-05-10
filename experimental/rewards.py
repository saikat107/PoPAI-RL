resp = """<think>
    <reflection>
        The high level strategy is to synthesize a verifiable definition for the lemma `serialize_get_protected_bits`, which ensures that the serialized header's first byte contains the protected bits correctly. The lemma requires two parts: proving the serialized sequence is non-empty and showing the protected bits match the bitfield extracted from the first byte.

        First, I need to recall that the `serialize_header` function produces a sequence of bytes starting with the first byte (from `first_byte_of_header`). The first byte's structure depends on whether the header is short or long, and the protected bits are embedded in specific bits of this first byte.

        The key steps involve using existing lemmas and functions to break down the problem. The `serialize_header_eq` lemma provides an equation for the serialized header, which includes the first byte and the body. This can be used to show that the first byte is indeed the head of the sequence.

        Next, using the `first_byte` structure defined in the context, we can analyze how the protected bits are encoded. For a short header, the protected bits occupy 5 bits starting at 0, while for a long header, they occupy 4 bits. The `LPB.get_bitfield` function can extract these bits from the first byte of the serialized sequence.

        To ensure the sequence length is greater than zero, since the first byte is always present, `Seq.length sq > 0` holds because `LP.serialize` of the header includes at least the first byte.

        The protected bits extraction must align with the `get_protected_bits` function's return value, which is directly the protected bits field in both header variants. Thus, connecting the serialized first byte's bitfield to this field is crucial.

        I should also leverage existing definitions like `first_byte_of_header` and `mk_header` to ensure that the first byte's structure correctly encodes the protected bits. Additionally, using the `LPB.get_bitfield` properties from the premises will help equate the extracted bits with the header's protected bits.
    </reflection>
    <steps>
        <step>
            Start by invoking `serialize_header_eq` for the given `short_dcid_len` and `h`. This lemma gives an equation for the serialized header as the concatenation of the first byte (serialized via `LP.serialize_u8`) and the body's serialization. This ensures the first byte is the head of the sequence `sq`.
        </step>
        <step>
            Extract the first byte from the serialized sequence `sq` using `Seq.head`, which is valid since the length is proven to be >0 via the first part of the lemma's assertion.
        </step>
    </steps>

    <emulation>
    <step>
       ldskjfals
    </step>

    <step>
        The first byte's value is synthesized using `LPB.synth_bitsum'_recip` and `first_byte_of_header`. This gives the first byte's bit representation, which is crucial for the next steps.
        The first byte is then serialized using `LP.serialize_u8`, and we assert that the length of the serialized sequence is greater than zero, confirming that the first byte is indeed present.

        Next, I need to extract the protected bits from the first byte. Depending on whether the header is short or long, the protected bits occupy different bit positions. For a short header, they are in bits 0-4, while for a long header, they are in bits 4-8. The `LPB.get_bitfield` function will help extract these bits.
        The extracted bits are then compared with the protected bits in the header `h`. This is done by checking if the first byte's value matches the expected enum representation for short or long headers. The assertions ensure that the extracted bits align with the header's protected bits.

        Finally, I need to ensure that the assertions hold true for both cases of headers (short and long). This will involve checking the bit positions and values accordingly.
        The final assertion checks that the extracted protected bits from the first byte match the expected value in the header `h`. This completes the verification of the lemma.
    
    let serialize_get_protected_bits (short_dcid_len: short_dcid_len_t) (h: header' short_dcid_len)
    : Lemma
    (let sq = LP.serialize (serialize_header short_dcid_len) h in
      Seq.length sq > 0 /\
      get_protected_bits h == LPB.uint8.LPB.get_bitfield (Seq.head sq) 0 (if PShort? h then 5 else 4)) =
  let sq = LP.serialize (serialize_header short_dcid_len) h in
  serialize_header_eq short_dcid_len h;
  let first_byte_val = LPB.synth_bitsum'_recip first_byte (first_byte_of_header short_dcid_len h) in
  let first_byte_in_seq = LP.serialize_u8 first_byte_val in
  assert (sq = first_byte_in_seq `Seq.append` LP.serialize (serialize_header_body short_dcid_len (LPB.bitsum'_key_of_t first_byte (first_byte_of_header short_dcid_len h))) (mk_header_body short_dcid_len (first_byte_of_header short_dcid_len h) h));
  assert (Seq.length first_byte_in_seq = 1);
  assert (Seq.length sq = 1 + (Seq.length (LP.serialize (serialize_header_body ...) ...)));
  assert (Seq.length sq > 0);
  let first_byte = Seq.head sq in
  let protected_bits_length = if PShort? h then 5 else 4 in
  let extracted_pb = LPB.uint8.LPB.get_bitfield first_byte 0 protected_bits_length in
  begin
    match h with
    | PShort pb _ _ ->
        assert (first_byte_val <: U8.t == header_form.Short.enum_repr);
        assert (LPB.get_bitfield8 first_byte_val 0 5 == pb);
        assert (extracted_pb == pb)
    | PLong pb _ _ _ _ ->
        assert (first_byte_val <: U8.t == header_form.Long.enum_repr);
        assert (LPB.get_bitfield8 first_byte_val 4 8 == pb);
        assert (extracted_pb == pb)
  end;
  ()
  end;
  ()
    </step>
        
</emulation>
</think>


<answer>
let serialize_get_protected_bits (short_dcid_len: short_dcid_len_t) (h: header' short_dcid_len)
    : Lemma
    (let sq = LP.serialize (serialize_header short_dcid_len) h in
      Seq.length sq > 0 /\
      get_protected_bits h == LPB.uint8.LPB.get_bitfield (Seq.head sq) 0 (if PShort? h then 5 else 4)) =
  let sq = LP.serialize (serialize_header short_dcid_len) h in
  serialize_header_eq short_dcid_len h;
  let first_byte_val = LPB.synth_bitsum'_recip first_byte (first_byte_of_header short_dcid_len h) in
  let first_byte_in_seq = LP.serialize_u8 first_byte_val in
  assert (sq = first_byte_in_seq `Seq.append` LP.serialize (serialize_header_body short_dcid_len (LPB.bitsum'_key_of_t first_byte (first_byte_of_header short_dcid_len h))) (mk_header_body short_dcid_len (first_byte_of_header short_dcid_len h) h));
  assert (Seq.length first_byte_in_seq = 1);
  assert (Seq.length sq = 1 + (Seq.length (LP.serialize (serialize_header_body ...) ...)));
  assert (Seq.length sq > 0);
  let first_byte = Seq.head sq in
  let protected_bits_length = if PShort? h then 5 else 4 in
  let extracted_pb = LPB.uint8.LPB.get_bitfield first_byte 0 protected_bits_length in
  begin
    match h with
    | PShort pb _ _ ->
        assert (first_byte_val <: U8.t == header_form.Short.enum_repr);
        assert (LPB.get_bitfield8 first_byte_val 0 5 == pb);
        assert (extracted_pb == pb)
    | PLong pb _ _ _ _ ->
        assert (first_byte_val <: U8.t == header_form.Long.enum_repr);
        assert (LPB.get_bitfield8 first_byte_val 4 8 == pb);
        assert (extracted_pb == pb)
  end;
  ()
</answer>"""

import re
from textwrap import dedent

from nltk.tokenize import word_tokenize
import nltk
from vllm import LLM

# Ensure the tokenizer resources are available
nltk.download('punkt', quiet=True)

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

    return {
        "reflection": sanitize(extract_block("reflection")),
        "steps": [sanitize(s) for s in extract_step_list("steps")],
        "emulation": [sanitize(s) for s in extract_step_list("emulation")],
        "answer": sanitize(extract_block("answer")),
    }


class FormattedResponse:
    def __init__(self, response: str):
        self.original_text = response
        self.reflection = ""
        self.steps = []
        self.emulation = []
        self.answer = ""

        result = parse_think_xml_with_optional_blocks(response)
        self.reflection = result["reflection"]
        self.steps = result["steps"]
        self.emulation = result["emulation"]
        self.answer = result["answer"]

    def __str__(self):
        return f"Reflection:\n{self.reflection}\n\n\nSteps:\n====\n{'\n'.join(self.steps)}\n\n\nEmulation:\n====\n{'\n'.join(self.emulation)}\n\n\nAnswer:\n====\n{self.answer}"

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
        return step_answer_match(step=self.emulation[-1], answer=self.answer) if len(self.emulation) > 0 and self.answer != "" else 0.0
    
    def step_answer_conformity_match(self) -> float:
        # We will have to call the 
        pass

    
        
        

# Example usage

fmt = FormattedResponse(resp)
# print(fmt)
print("Is well-formed (Reflection Only): ", fmt.is_well_formed(reflection_expected=True))
print("Is well-formed (Steps Only): ", fmt.is_well_formed(steps_expected=True))
print("Is well-formed (Emulation Only): ", fmt.is_well_formed(emulation_expected=True))
print("Is well-formed (Reflection + Steps): ", fmt.is_well_formed(reflection_expected=True, steps_expected=True))
print("Is well-formed (Reflection + Emulation): ", fmt.is_well_formed(reflection_expected=True, emulation_expected=True))
print("Is well-formed (Steps + Emulation): ", fmt.is_well_formed(steps_expected=True, emulation_expected=True))
print("Is well-formed (All): ", fmt.is_well_formed(reflection_expected=True, steps_expected=True, emulation_expected=True))
print("Does steps match emulation:", fmt.does_steps_match_emulation())
print("Answer matches steps:", fmt.answer_matches_steps())
# print("Reflection:", fmt.reflection)
# print("Steps:", fmt.steps)
# print("Emulation:", fmt.emulation)
# print("Answer:", fmt.answer)
