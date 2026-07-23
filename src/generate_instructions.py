"""
generate_instructions.py  —  Stage 3a
=======================================
Self-Instruct: prompt Ollama ONCE with a meta-prompt to generate
NUM_TEMPLATES (default 50) diverse instruction templates for mode
choice prediction.

Each template has the same structure:
  <task definition> + <data description> + <thinking guidance>
But with different linguistic styles:
  imperative, question, conditional, analytical, role-play, etc.

Output: instruction_dataset/instruction_templates.json
  A JSON list of 50 template strings (the {placeholder} variables
  are filled in Stage 3b by build_instructions.py).
"""

import json
import re
import time

from config import INSTRUCTION_DIR, NUM_TEMPLATES, OLLAMA_MODEL
from utils import get_logger, get_ollama_client

logger = get_logger(__name__, log_file="generate_instructions.log")
client = get_ollama_client()

# ─────────────────────────────────────────────────────────────────────────────
# BASE EXAMPLE INSTRUCTION  (Stage 6 prediction prompt — used as length/style
# reference for the meta-prompt)
# ─────────────────────────────────────────────────────────────────────────────
BASE_INSTRUCTION = """\
Task definition:
Your task is to predict a traveler's transport mode choice for a target trip \
based on their inferred behavioral persona and the attributes of the available \
transport options.

Data description:
<persona> contains the assigned behavioral profile: persona_label (3-8 word archetype), \
identity_profile (socio-demographic context), preference_profile (time/cost/comfort/habit attitudes).
<demographics> provides static attributes: female (1=female,0=male), business (1=business,0=leisure), income (£).
<target> contains available_modes and full attributes for each: \
time_* (travel minutes), cost_* (£), access_* (wait minutes, bus/air/rail), \
service_air / service_rail (1=no-frills, 2=wifi, 3=meal service). \
Only modes in available_modes have valid values; all others are None.

Thinking guidance:
1. Behavioral persona: the traveler's known attitudes toward time vs. cost trade-offs, \
   comfort, and habitual mode loyalty.
2. Demographic profile: business=time priority+possible reimbursement; \
   leisure=cost sensitivity; high income=reduced fare sensitivity.
3. Trip attributes: compare only modes in available_modes on time, cost, access, and service.
4. Persona-option alignment: which available mode best matches the preference_profile \
   given the concrete trade-offs on this trip?

Output format:
Output a JSON object with keys "prediction" (one of: car, bus, air, rail) and \
"reason" (one concise sentence explaining the prediction). No line breaks.

Data inputs:
<persona>: {persona_label} | {identity_profile} | {preference_profile}
<demographics>: female={female}, business={business}, income=£{income}
<target>: {target_trip_attributes}, next_mode=<unknown>
"""

# ─────────────────────────────────────────────────────────────────────────────
# META-PROMPT  (sent once to Ollama to generate all 50 templates)
# ─────────────────────────────────────────────────────────────────────────────
META_PROMPT = f"""\
You are asked to come up with a set of {NUM_TEMPLATES} diverse task instructions \
related to transport mode choice prediction. These task instructions will be given \
to a fine-tuned LLaMA model and we will evaluate the model for completing the instructions.

Here are the requirements:

1. The instructions must be written in English.

2. Each generated instruction should be approximately the same length as the base \
   example instruction given below. Give a corresponding task title before each instruction.

3. Assuming that you are a transport mobility research scientist, each instruction must \
   be related to predicting which transport mode (car, bus, air, or rail) a traveler \
   will choose for a given trip, based on their behavioral persona and trip attributes.

4. The language style used across instructions must be diverse. You should mix: \
   direct imperative instructions (e.g., 'Predict the mode...'), \
   open questions (e.g., 'Which mode would this traveler choose?'), \
   conditional phrasings (e.g., 'Given the persona below, determine...'), \
   and analytical framings (e.g., 'Analyse the trip options and identify...').

5. A LLaMA language model must be able to complete each instruction from text input alone. \
   Do not ask for visual, audio, or external tool output.

6. The main structure of each instruction must include three sections: \
   <task definition>, <data description>, and <thinking guidance>.

7. <data description> must always describe exactly these input elements and no others: \
   persona_label, identity_profile, preference_profile, female, business, income, \
   available_modes, time_*, cost_*, access_*, service_air, service_rail.

8. <thinking guidance> must always guide the model to reason about: \
   (1) the traveler's behavioral persona, (2) the demographic profile, \
   (3) the available mode attributes, and (4) the alignment between persona and options.

9. Output format must always be a JSON object with keys: \
   "prediction" (one of: car, bus, air, rail) and \
   "reason" (one concise sentence explaining the prediction). \
   Do not include line breaks in the output.

10. Each instruction must be self-contained: a model receiving only that instruction \
    and the filled data should be able to produce a correct prediction without any \
    additional context.

Base example instruction (use this as the length and structure reference):
---
{BASE_INSTRUCTION}
---

Now generate the list of {NUM_TEMPLATES} task instructions.
Format each as:

TITLE: <task title>
INSTRUCTION:
<full instruction body with task definition, data description, thinking guidance, output format, data inputs>

Separate each entry with a blank line. Do not number them — use TITLE / INSTRUCTION markers only.
"""


# ─────────────────────────────────────────────────────────────────────────────
# PARSING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def parse_templates_from_response(raw: str) -> list[dict]:
    """
    Parse the LLM response into a list of template dicts.
    Each dict: {"title": str, "template": str}

    The template string contains {placeholder} variables that will be filled
    by build_instructions.py (e.g. {persona_label}, {female}, {target_trip_attributes}).
    """
    templates = []
    # Split on TITLE: marker
    blocks = re.split(r"\nTITLE:", raw)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract title (first line) and instruction body
        lines = block.split("\n")
        title = lines[0].strip()

        # Find INSTRUCTION: marker
        body_lines = []
        in_body = False
        for line in lines[1:]:
            if line.strip().startswith("INSTRUCTION:"):
                in_body = True
                rest = line.replace("INSTRUCTION:", "", 1).strip()
                if rest:
                    body_lines.append(rest)
            elif in_body:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()
        if title and body:
            templates.append({"title": title, "template": body})

    return templates


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("STAGE 3a: SELF-INSTRUCT — Generate Instruction Templates")
    logger.info(f"  Requesting {NUM_TEMPLATES} diverse templates from {OLLAMA_MODEL}")
    logger.info("=" * 60)

    # Call Ollama once with the meta-prompt
    # max_tokens is generous; LLaMA needs space for 50 full instructions
    logger.info("Sending meta-prompt to Ollama (this may take a few minutes)...")
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": META_PROMPT}],
        temperature=0.7,    # higher temperature for stylistic diversity
        max_tokens=16000,   # enough for 50 full instructions
    )
    raw = response.choices[0].message.content
    logger.info(f"Received {len(raw)} characters from model")

    # Parse templates
    templates = parse_templates_from_response(raw)
    logger.info(f"Parsed {len(templates)} instruction templates")

    if len(templates) < NUM_TEMPLATES:
        logger.warning(
            f"Expected {NUM_TEMPLATES} templates but only got {len(templates)}. "
            "The model may have truncated. Consider re-running or increasing max_tokens."
        )

    # Save templates to JSON
    out_path = f"{INSTRUCTION_DIR}instruction_templates.json"
    with open(out_path, "w") as f:
        json.dump(templates, f, indent=2)

    logger.info(f"Saved {len(templates)} templates → {out_path}")

    # Log sample titles
    logger.info("Sample template titles:")
    for t in templates[:5]:
        logger.info(f"  • {t['title']}")

    logger.info("Stage 3a complete.")


if __name__ == "__main__":
    main()
