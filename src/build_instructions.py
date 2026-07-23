"""
build_instructions.py  —  Stage 3b
=====================================
Self-Instruct fill step: combine the 50 diverse instruction templates
(from Stage 3a) with ALL train trip rows to produce the full fine-tuning dataset.

Logic:
  50 templates  ×  N train trips  =  50 × N instruction examples

Each example is a 3-part chat:
  system    → traveler's persona (from train_personas_raw.csv)
  user      → filled instruction template (demographics + trip attributes)
  assistant → ground-truth JSON {"prediction": mode, "reason": sentence}

Output:
  instruction_dataset/train_instructions.jsonl  — JSONL for LoRA fine-tuning
  instruction_dataset/train_instructions_hf/    — HuggingFace Dataset format
"""

import json

import pandas as pd
from datasets import Dataset

from config import (
    DEMOGRAPHIC_COLS,
    INSTRUCTION_DIR,
    LABEL_COL,
    MODE_MAP,
    PERSONAS_DIR,
    PROCESSED_DIR,
    TRAVELER_ID_COL,
)
from utils import (
    build_gold_reason,
    format_available_modes,
    format_demographics,
    format_target_trip,
    get_logger,
)

logger = get_logger(__name__, log_file="build_instructions.log")


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM MESSAGE BUILDER
# Fixed per traveler — contains their persona from Stage 2
# ─────────────────────────────────────────────────────────────────────────────

def build_system(row: pd.Series) -> str:
    """
    Build the system message for one training example.
    The system message sets the persona context.
    It is the same for all trips of the same traveler.
    """
    return (
        "You are a transportation mode choice prediction model simulating "
        "a specific traveler persona inferred from their behavioral history.\n\n"
        "<persona>\n"
        f"Label:       {row.get('persona_label', 'unknown')}\n"
        f"Identity:    {row.get('identity_profile', '')}\n"
        f"Preferences: {row.get('preference_profile', '')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# USER MESSAGE BUILDER
# Fills a template string with actual trip + demographic data
# ─────────────────────────────────────────────────────────────────────────────

def fill_template(template: str, row: pd.Series) -> str:
    """
    Fill a template string with actual data values from a trip row.

    The template uses {placeholder} variables that match these keys:
      {persona_label}, {identity_profile}, {preference_profile}
      {female}, {business}, {income}
      {demographics}
      {target_trip_attributes}
      {available_modes_text}

    Any unfilled placeholders are left as-is (safe fallback).
    """
    gender  = "Female" if row.get("female", 0) == 1 else "Male"
    purpose = "Business" if row.get("business", 0) == 1 else "Personal/Leisure"
    income  = f"£{row['income']:,.0f}" if pd.notna(row.get("income")) else "unknown"

    replacements = {
        "{persona_label}":           str(row.get("persona_label", "")),
        "{identity_profile}":        str(row.get("identity_profile", "")),
        "{preference_profile}":      str(row.get("preference_profile", "")),
        "{female}":                  str(int(row.get("female", 0))),
        "{business}":                str(int(row.get("business", 0))),
        "{income}":                  income,
        "{demographics}":            f"female={int(row.get('female',0))}, business={int(row.get('business',0))}, income={income}",
        "{target_trip_attributes}":  format_target_trip(row),
        "{available_modes_text}":    format_available_modes(row),
        "{gender}":                  gender,
        "{purpose}":                 purpose,
    }

    filled = template
    for key, val in replacements.items():
        filled = filled.replace(key, val)
    return filled


# ─────────────────────────────────────────────────────────────────────────────
# ASSISTANT MESSAGE BUILDER
# Ground-truth label + deterministic reason for supervised fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def build_assistant(row: pd.Series) -> str:
    """
    Build the assistant message (ground-truth response).
    Returns a JSON string: {"prediction": mode, "reason": sentence}
    The reason is generated deterministically from the factual data
    (no hallucination) via build_gold_reason() in utils.py.
    """
    mode   = row.get("chosen_mode", "car")
    reason = build_gold_reason(row)
    return json.dumps({"prediction": mode, "reason": reason})


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# Core cross-product: 50 templates × N train trips
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(merged_df: pd.DataFrame, templates: list[dict]) -> list[dict]:
    """
    Build the full instruction dataset.

    For each of the 50 instruction templates, iterate over every row in
    merged_df (train_trips joined with train_personas). Each (template, row)
    pair produces one JSONL example.

    Total examples = len(templates) × len(merged_df)

    Args:
        merged_df : DataFrame with trip rows + persona fields attached
        templates : list of {"title": str, "template": str} dicts

    Returns:
        list of {"system": str, "user": str, "assistant": str} dicts
    """
    examples = []
    n_trips = len(merged_df)
    n_templates = len(templates)
    logger.info(f"Building {n_templates} × {n_trips} = {n_templates * n_trips} examples...")

    for t_idx, tmpl in enumerate(templates, 1):
        template_str = tmpl["template"]
        template_title = tmpl["title"]

        for _, row in merged_df.iterrows():
            # Skip rows where persona info is missing (should be rare)
            if pd.isna(row.get("persona_label")):
                continue

            examples.append({
                "system":    build_system(row),
                "user":      fill_template(template_str, row),
                "assistant": build_assistant(row),
                # Metadata — not used in training but useful for debugging
                "_template_title": template_title,
                "_traveler_id":    str(row.get(TRAVELER_ID_COL, "")),
                "_trip_id":        str(row.get("trip_id", "")),
                "_true_mode":      str(row.get("chosen_mode", "")),
            })

        if t_idx % 10 == 0:
            logger.info(f"  Processed {t_idx}/{n_templates} templates ({len(examples)} examples so far)")

    logger.info(f"Total examples built: {len(examples)}")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("STAGE 3b: BUILD INSTRUCTION DATASET (Self-Instruct fill)")
    logger.info("=" * 60)

    # ── Load train trips ──────────────────────────────────────────────────────
    train_trips = pd.read_csv(f"{PROCESSED_DIR}train_trips.csv")
    if "chosen_mode" not in train_trips.columns:
        train_trips["chosen_mode"] = train_trips[LABEL_COL].map(MODE_MAP)
    logger.info(f"Loaded {len(train_trips)} train trips")

    # ── Load personas (one per traveler) ──────────────────────────────────────
    personas = pd.read_csv(f"{PERSONAS_DIR}train_personas_raw.csv")
    logger.info(f"Loaded {len(personas)} train personas")

    # ── Merge: broadcast each traveler's persona to all their trips ───────────
    merged = train_trips.merge(
        personas[[TRAVELER_ID_COL, "persona_label", "identity_profile", "preference_profile"]],
        on=TRAVELER_ID_COL,
        how="left",
    )
    logger.info(f"Merged dataset: {len(merged)} rows "
                f"({merged['persona_label'].isna().sum()} missing personas)")

    # ── Load instruction templates (from Stage 3a) ────────────────────────────
    templates_path = f"{INSTRUCTION_DIR}instruction_templates.json"
    with open(templates_path) as f:
        templates = json.load(f)
    logger.info(f"Loaded {len(templates)} instruction templates from {templates_path}")

    # ── Build the full cross-product dataset ──────────────────────────────────
    examples = build_dataset(merged, templates)

    # ── Save as JSONL (primary format for LoRA training) ──────────────────────
    jsonl_path = f"{INSTRUCTION_DIR}train_instructions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            # Remove metadata keys before saving to JSONL for training
            training_ex = {k: v for k, v in ex.items() if not k.startswith("_")}
            f.write(json.dumps(training_ex, ensure_ascii=False) + "\n")
    logger.info(f"Saved JSONL → {jsonl_path}")

    # ── Save as HuggingFace Dataset (optional, for datasets.load_from_disk) ───
    # Strip metadata for HF format as well
    hf_examples = [{k: v for k, v in ex.items() if not k.startswith("_")} for ex in examples]
    hf_dataset = Dataset.from_list(hf_examples)
    hf_path = f"{INSTRUCTION_DIR}train_instructions_hf"
    hf_dataset.save_to_disk(hf_path)
    logger.info(f"Saved HuggingFace dataset → {hf_path}")

    # ── Log sample ────────────────────────────────────────────────────────────
    ex = examples[0]
    logger.info(f"\nSample (template: '{ex['_template_title']}')")
    logger.info(f"  SYSTEM:    {ex['system'][:120]}...")
    logger.info(f"  USER:      {ex['user'][:120]}...")
    logger.info(f"  ASSISTANT: {ex['assistant'][:120]}")
    logger.info("Stage 3b complete.")


if __name__ == "__main__":
    main()
