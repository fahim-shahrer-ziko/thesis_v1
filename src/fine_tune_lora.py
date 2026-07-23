"""
fine_tune_lora.py  —  Stage 4
================================
LoRA fine-tuning of LLaMA 3 8B Instruct on the full 50×N instruction dataset.

Uses:
  - 4-bit NF4 quantisation (bitsandbytes) to reduce GPU memory
  - PEFT/LoRA for parameter-efficient fine-tuning (~1% trainable params)
  - HuggingFace Trainer with cosine LR schedule

Input : instruction_dataset/train_instructions_hf/
Output: outputs/lora_adapter_final/   (LoRA weights + tokenizer)
        outputs/lora_checkpoints/     (intermediate checkpoints)

Requirements:
  - NVIDIA GPU with ≥16 GB VRAM (A100/A6000 recommended)
  - HF_TOKEN set in .env if using gated LLaMA model
"""

import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import (
    BASE_MODEL,
    BATCH_SIZE,
    EPOCHS,
    GRAD_ACCUM,
    HF_TOKEN,
    INSTRUCTION_DIR,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LR_SCHEDULER,
    MAX_SEQ_LEN,
    OUTPUTS_DIR,
    TARGET_MODULES,
    WARMUP_RATIO,
)
from utils import get_logger

logger = get_logger(__name__, log_file="fine_tune_lora.log")


# ─────────────────────────────────────────────────────────────────────────────
# CHAT TEMPLATE FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def apply_chat_template(example: dict, tokenizer) -> dict:
    """
    Apply the LLaMA 3 chat template to one training example.
    Converts {system, user, assistant} into a single formatted string.
    Appends EOS token so the model learns when to stop generating.
    """
    messages = [
        {"role": "system",    "content": example["system"]},
        {"role": "user",      "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    # tokenize=False → returns the formatted string, not token IDs
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": prompt + tokenizer.eos_token}


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(examples: dict, tokenizer) -> dict:
    """
    Tokenize the 'text' field with truncation and right-side padding.
    Called via dataset.map(batched=True).
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("STAGE 4: LORA FINE-TUNING")
    logger.info(f"  Base model : {BASE_MODEL}")
    logger.info(f"  LoRA rank  : {LORA_R} | alpha: {LORA_ALPHA}")
    logger.info(f"  Epochs     : {EPOCHS} | LR: {LEARNING_RATE}")
    logger.info(f"  Batch size : {BATCH_SIZE} × accum {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
    logger.info("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    logger.info("1. Loading instruction dataset...")
    dataset = load_from_disk(f"{INSTRUCTION_DIR}train_instructions_hf")
    logger.info(f"   {len(dataset)} examples")

    # ── 2. Load tokenizer ─────────────────────────────────────────────────────
    logger.info(f"2. Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
        token=HF_TOKEN or None,
    )
    tokenizer.pad_token    = tokenizer.eos_token  # LLaMA has no dedicated pad token
    tokenizer.padding_side = "right"              # right padding for causal LM

    # ── 3. Apply chat template ────────────────────────────────────────────────
    logger.info("3. Applying Llama 3 chat template...")
    dataset = dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Applying chat template",
    )

    # ── 4. Tokenize ───────────────────────────────────────────────────────────
    logger.info("4. Tokenizing...")
    dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # ── 5. Load base model with 4-bit quantisation ────────────────────────────
    logger.info(f"5. Loading {BASE_MODEL} with 4-bit NF4 quantisation...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # nested quantisation saves extra memory
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",                # auto-distribute across available GPUs
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )

    # ── 6. Prepare model for k-bit training ───────────────────────────────────
    model = prepare_model_for_kbit_training(model)

    # ── 7. Apply LoRA ─────────────────────────────────────────────────────────
    logger.info("6. Applying LoRA adapters...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,    # all attention projection matrices
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"   Trainable: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")

    # ── 8. Training arguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=f"{OUTPUTS_DIR}lora_checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        bf16=True,                        # bfloat16 training (requires Ampere+ GPU)
        fp16=False,
        remove_unused_columns=False,
        report_to="none",                 # set to "wandb" to enable W&B logging
        dataloader_num_workers=2,
    )

    # ── 9. Data collator (language modelling, no MLM masking) ─────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,             # slightly improves throughput
    )

    # ── 10. Train ─────────────────────────────────────────────────────────────
    logger.info("7. Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()

    # ── 11. Save LoRA adapter ─────────────────────────────────────────────────
    adapter_path = f"{OUTPUTS_DIR}lora_adapter_final"
    logger.info(f"8. Saving LoRA adapter → {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Stage 4 complete.")


if __name__ == "__main__":
    main()
