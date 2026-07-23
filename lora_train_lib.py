"""
lora_train_lib.py
Shared QLoRA fine-tuning logic used by train_lora_ori.py,
train_lora_dest.py, and train_lora_entry_time.py.
Do not run this file directly.
"""

import argparse
import os

import pandas as pd
from datasets import Dataset

import config


def load_train_data(task):
    path = config.task_json_path(task, "train")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run build_dataset.py first."
        )
    return Dataset.from_pandas(pd.read_json(path))


def make_process_fn(tokenizer, max_len):
    def process(example):
        # Use the tokenizer's own chat template -- works for any model
        # (Llama, Qwen, Mistral) without hardcoding template strings.
        messages = [
            {"role": "user",      "content": example["instruction"] + example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize the full conversation
        full = tokenizer(full_text, add_special_tokens=False)

        # Tokenize only the user part to know where to start unmasking labels
        user_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["instruction"] + example["input"]}],
            tokenize=False,
            add_generation_prompt=True,  # includes the assistant turn opener
        )
        user_ids = tokenizer(user_only, add_special_tokens=False)["input_ids"]
        n_user   = len(user_ids)

        ids    = full["input_ids"]    + [tokenizer.pad_token_id]
        mask   = full["attention_mask"] + [1]
        labels = [-100] * n_user + full["input_ids"][n_user:] + [tokenizer.pad_token_id]

        if len(ids) > max_len:
            ids    = ids[:max_len]
            mask   = mask[:max_len]
            labels = labels[:max_len]

        return {"input_ids": ids, "attention_mask": mask, "labels": labels}
    return process


def build_parser(task):
    p = argparse.ArgumentParser(
        description=f"QLoRA fine-tune on MRT '{task}' prediction task"
    )
    p.add_argument("--base_model",  default=config.BASE_MODEL_ID)
    p.add_argument("--output_dir",  default=config.train_output_dir(task))
    p.add_argument("--no_qlora",    action="store_true",
                   help="Use plain bf16 LoRA instead of 4-bit QLoRA")
    p.add_argument("--epochs",      default=config.NUM_TRAIN_EPOCHS,     type=int)
    p.add_argument("--batch_size",  default=config.PER_DEVICE_TRAIN_BATCH_SIZE, type=int)
    p.add_argument("--grad_accum",  default=config.GRADIENT_ACCUMULATION_STEPS, type=int)
    p.add_argument("--lr",          default=config.LEARNING_RATE,        type=float)
    p.add_argument("--device_map",  default="auto")
    return p


def run_training(task, args):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               BitsAndBytesConfig, TrainingArguments,
                               Trainer, DataCollatorForSeq2Seq)
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    use_qlora = config.USE_QLORA and not args.no_qlora
    print(f"Task: {task} | QLoRA: {use_qlora} | Output: {args.output_dir}")
    print("=" * 60)
    print(f"  BASE MODEL : {args.base_model}")
    print(f"  TASK       : {task}")
    print(f"  QLoRA      : {use_qlora}")
    print(f"  OUTPUT     : {args.output_dir}")
    print("=" * 60)
    
    ds = load_train_data(task)
    print(f"Training samples: {len(ds)}")

    hf_token = getattr(config, "HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=True,
        token=hf_token,
    )
    # Qwen uses a dedicated pad token; Llama does not -- handle both
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb,
            device_map=args.device_map, use_cache=False,
            trust_remote_code=True, token=hf_token,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            device_map=args.device_map, use_cache=False,
            trust_remote_code=True, token=hf_token,
        )

    tokenized = ds.map(make_process_fn(tokenizer, config.MAX_SEQ_LENGTH),
                       remove_columns=ds.column_names)

    # Auto-detect LoRA target modules if not set in config
    target_modules = config.LORA_TARGET_MODULES
    if target_modules is None:
        # Inspect the model's named modules to find attention projection layers
        all_modules = [name for name, _ in model.named_modules()]
        if any("q_proj" in m for m in all_modules):
            # Llama / Mistral / Qwen style
            target_modules = [
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"
            ]
        elif any("query_key_value" in m for m in all_modules):
            # Falcon / GPT-NeoX style
            target_modules = [
                "query_key_value","dense",
                "dense_h_to_4h","dense_4h_to_h"
            ]
        elif any("c_attn" in m for m in all_modules):
            # GPT-2 style
            target_modules = ["c_attn","c_proj","c_fc"]
        else:
            # Generic fallback -- target all linear layers
            from peft import get_linear_layer_names
            target_modules = get_linear_layer_names(model)
        print(f"Auto-detected LoRA target modules: {target_modules}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=config.LOGGING_STEPS,
        num_train_epochs=args.epochs,
        save_steps=config.SAVE_STEPS,
        learning_rate=args.lr,
        gradient_checkpointing=True,
        bf16=True,
        report_to="none",
    )

    Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    ).train()

    final = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\n[{task}] Adapter saved → {final}")

    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()
