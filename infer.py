"""
infer.py
Runs predictions on the test set using a trained LoRA adapter.

Usage:
    python infer.py --task dest --use_4bit
    python infer.py --task ori  --use_4bit --limit 100   # quick check
    python infer.py --task dest --smoke_test --use_4bit  # single example
    python infer.py --task dest --no_adapter --use_4bit  # base model baseline
"""

import argparse
import ast
import json
import os
import re

import pandas as pd

import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",         default="dest",
                   choices=["ori","dest","entry_time"])
    p.add_argument("--base_model",   default=config.BASE_MODEL_ID)
    p.add_argument("--adapter_path", default=None,
                   help="Path to trained adapter. Defaults to final_adapter for the task.")
    p.add_argument("--no_adapter",   action="store_true",
                   help="Run raw base model (baseline, no adapter)")
    p.add_argument("--input_json",   default=None)
    p.add_argument("--output_csv",   default=None)
    p.add_argument("--max_new_tokens", default=config.MAX_NEW_TOKENS, type=int)
    p.add_argument("--limit",        default=None, type=int)
    p.add_argument("--device_map",   default="auto")
    p.add_argument("--use_4bit",     action="store_true")
    p.add_argument("--smoke_test",   action="store_true")
    args = p.parse_args()

    if not args.no_adapter and args.adapter_path is None:
        args.adapter_path = os.path.join(
            config.train_output_dir(args.task), "final_adapter"
        )
    if args.input_json is None:
        args.input_json = config.task_json_path(args.task, "test")
    if args.output_csv is None:
        args.output_csv = os.path.join(
            config.OUTPUT_DIR, f"{args.task}_predictions.csv"
        )
    return args


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    hf_token = getattr(config, "HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb,
            device_map=args.device_map, use_cache=True,
            trust_remote_code=True, token=hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            device_map=args.device_map, use_cache=True,
            trust_remote_code=True, token=hf_token,
        )

    if not args.no_adapter:
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction, input_text, max_new_tokens):
    import torch
    # Use the tokenizer's own chat template -- works for any model
    messages = [{"role": "user", "content": instruction + input_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def extract_prediction(raw):
    """Robustly parse the prediction value from model output JSON."""
    has_braces = "{" in raw and "}" in raw
    if not has_braces:
        return None, "PARSE_FAILURE"

    text = raw[raw.find("{"):raw.find("}")+1]

    try:
        obj = json.loads(text)
        if "prediction" in obj:
            return str(obj["prediction"]), None
    except Exception:
        pass

    try:
        obj = ast.literal_eval(text)
        if "prediction" in obj:
            return str(obj["prediction"]), None
    except Exception:
        pass

    for seg in text.strip("{}").split(","):
        if "prediction" in seg:
            val = seg.split(":", 1)[-1].strip().strip('"').strip("'")
            return val, None

    return None, "PARSE_FAILURE"


def smoke_test(model, tokenizer, args):
    meta        = config.TASK_META[args.task]
    placeholder = meta["placeholder"]
    pred_field  = meta["pred_field"]

    seed_path = config.seed_prompt_path(args.task)
    instr_dir = config.task_instruction_dir(args.task)

    # Use first generated instruction if available, else fall back to seed prompt
    example_text = None
    if os.path.isdir(instr_dir):
        files = sorted(f for f in os.listdir(instr_dir)
                       if f.endswith(".txt") and "_instruction_" in f)
        if files:
            with open(os.path.join(instr_dir, files[0])) as f:
                t = f.read().strip()
            lines = t.split("\n", 1)
            example_text = lines[1].strip() if lines[0].lower().startswith("task title:") else t

    if not example_text:
        if not os.path.exists(seed_path):
            raise FileNotFoundError(f"Seed prompt not found: {seed_path}")
        with open(seed_path) as f:
            example_text = f.read().strip()

    instruction = (example_text + "\n"
                   + f'Please organize your answer in a JSON object '
                     f'containing the key: "prediction" ({pred_field}).')

    if args.task == "dest":
        target = f"('Male', 28, '2024-06-17 08:12:00', '2024-06-17 08:38:00', 'Monday', 'Uttara North', '{placeholder}')"
    elif args.task == "ori":
        target = f"('Male', 28, '2024-06-17 08:12:00', None, 'Monday', '{placeholder}', None)"
    else:
        target = f"('Male', 28, '{placeholder}', '2024-06-17 08:38:00', 'Monday', 'Uttara North', 'Farmgate')"

    inp = (
        "<persona>: (45, 30, 6, 7, 28.5, 32, 6) \n"
        "<history>: [('Male', 28, '2024-06-03 08:05:00', '2024-06-03 08:32:00', "
        "'Monday', 'Uttara North', 'Farmgate')] \n"
        "<context>: [('Male', 28, '2024-06-10 08:10:00', '2024-06-10 08:35:00', "
        "'Monday', 'Uttara North', 'Farmgate')] \n"
        f"<target_stay>: {target} \n"
    )

    raw = generate(model, tokenizer, instruction, inp, args.max_new_tokens)
    print("\n=== RAW OUTPUT ===\n", raw)
    pred, err = extract_prediction(raw)
    print(f"\n=== PARSED ===\nprediction: {pred}  |  parse_error: {err}")


def main():
    args = parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    model, tokenizer = load_model(args)

    if args.smoke_test:
        smoke_test(model, tokenizer, args)
        return

    with open(args.input_json) as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[:args.limit]

    print(f"Running inference on {len(samples)} samples ...")
    rows = []
    for i, s in enumerate(samples):
        raw  = generate(model, tokenizer, s["instruction"], s["input"],
                        args.max_new_tokens)
        pred, err = extract_prediction(raw)
        gt, _     = extract_prediction(s["output"])
        rows.append({"index": i, "raw_output": raw,
                     "prediction": pred, "ground_truth": gt,
                     "parse_error": err})
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(samples)}")

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved → {args.output_csv}")


if __name__ == "__main__":
    main()
