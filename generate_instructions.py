"""
generate_instructions.py

Generates N diverse instruction variants per task using Qwen2.5-7B-Instruct
directly via HuggingFace transformers. No API key or internet connection
required after the model is downloaded once.

Reads TWO plain-text files per task (authored by you):
    prompt_store/gen/gen_pred_{task}_instruct.txt   -- generation rules
    prompt_store/seed_task/seed_prompt_{task}.txt   -- worked example

N is parsed from the line "List of N tasks:" inside the gen file.

Output:
    prompt_store/instruction/{task}/{task}_instruction_01.txt
    ...
    prompt_store/instruction/{task}/{task}_instruction_NN.txt
    prompt_store/instruction/{task}/{task}_instructions_raw.txt

Usage:
    python generate_instructions.py                 # all 3 tasks
    python generate_instructions.py --task dest     # one task
    python generate_instructions.py --force         # regenerate cached
"""

import argparse
import os
import re

import config

DEFAULT_N = 10
COUNT_PATTERN = re.compile(r"list\s+of\s+(\d+)\s+tasks", re.IGNORECASE)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_prompt_files(task):
    gen_file  = config.gen_path(task)
    seed_file = config.seed_prompt_path(task)

    for path, label in [(gen_file, "gen rules"), (seed_file, "seed prompt")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{task}] {label} file not found: {path}\n"
                f"Please author this file before running generate_instructions.py."
            )

    with open(gen_file)  as f: gen_text  = f.read().strip()
    with open(seed_file) as f: seed_text = f.read().strip()
    return gen_text, seed_text


def parse_n(gen_text, task):
    m = COUNT_PATTERN.search(gen_text)
    if m:
        return int(m.group(1))
    print(f"[{task}] WARNING: 'List of N tasks' not found in gen file. "
          f"Defaulting to {DEFAULT_N}.")
    return DEFAULT_N


def build_prompt(gen_text, seed_text):
    return (
        gen_text
        + "\n\nWrap EACH instruction in <instruction> and </instruction> tags. "
          "Example of one instruction (generate all 10 in exactly this structure, "
          "no data, description only):\n\n"
          "<instruction>\n" + seed_text + "\n</instruction>"
    )


# ---------------------------------------------------------------------------
# Open-source generation via HuggingFace (Qwen2.5-7B-Instruct)
# ---------------------------------------------------------------------------

def generate_with_hf(prompt):
    """Loads Qwen2.5-7B-Instruct on the available GPU, generates, then
    frees VRAM so subsequent training steps have the full GPU budget."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config.GENERATION_MODEL_ID
    print(f"  Loading {model_id} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
            temperature=config.GENERATION_TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)

    # Free GPU memory before training scripts run
    del model, inputs, output_ids
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Parsing the model response
# ---------------------------------------------------------------------------

def parse_instructions(raw_text, n):
    # Extract everything between <instruction> and </instruction> tags
    blocks = re.findall(
        r"<instruction>(.*?)</instruction>",
        raw_text,
        re.DOTALL
    )

    if not blocks:
        # Fallback: model ignored tags, save raw and warn
        print("WARNING: No <instruction> tags found in model output.")
        print("Check the raw file and update your gen rules to enforce the format.")
        return [{"title": "Instruction 1", "instruction": raw_text.strip()}]

    results = []
    for i, block in enumerate(blocks[:n], start=1):
        block = block.strip()
        results.append({
            "title": f"Instruction {i}",
            "instruction": block
        })

    if len(results) < n:
        print(f"WARNING: Expected {n} instructions, got {len(results)}. "
              f"Increase GENERATION_MAX_NEW_TOKENS in config.py and re-run with --force.")

    return results


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def cached_count(task):
    d = config.task_instruction_dir(task)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d)
                if re.match(rf"^{re.escape(task)}_instruction_\d+\.txt$", f)])


# ---------------------------------------------------------------------------
# Main per-task routine
# ---------------------------------------------------------------------------

def generate_for_task(task, force=False):
    out_dir = config.task_instruction_dir(task)
    gen_text, seed_text = load_prompt_files(task)
    n = parse_n(gen_text, task)

    if not force and cached_count(task) >= n:
        print(f"[{task}] {n} instruction files already exist. "
              f"Skipping (use --force to regenerate).")
        return

    print(f"[{task}] Generating {n} instruction variants ...")
    prompt   = build_prompt(gen_text, seed_text)
    raw      = generate_with_hf(prompt)

    parsed = parse_instructions(raw, n)
    if len(parsed) < n:
        print(f"[{task}] WARNING: only parsed {len(parsed)}/{n} instructions. "
              f"Check {out_dir}/{task}_instructions_raw.txt")

    os.makedirs(out_dir, exist_ok=True)
    width = max(2, len(str(n)))

    # Save raw response for debugging
    with open(os.path.join(out_dir, f"{task}_instructions_raw.txt"), "w") as f:
        f.write(raw)

    # Save each instruction as its own file
    for i, item in enumerate(parsed, start=1):
        path = os.path.join(out_dir, f"{task}_instruction_{i:0{width}d}.txt")
        with open(path, "w") as f:
            f.write(f"Task title: {item['title']}\n\n{item['instruction']}\n")

    print(f"[{task}] Saved {len(parsed)} files to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse task instructions using Qwen2.5-7B-Instruct"
    )
    parser.add_argument("--task", default="all",
                        choices=["ori","dest","entry_time","all"])
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if cached files exist")
    args = parser.parse_args()

    tasks = config.TASKS if args.task == "all" else [args.task]
    for t in tasks:
        generate_for_task(t, force=args.force)


if __name__ == "__main__":
    main()
