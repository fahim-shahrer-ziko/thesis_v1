"""
test_predict.py  —  Stages 5 & 6
===================================
Stage 5: Assign personas to test travelers using ONLY demographics.
         No trip history used — prevents data leakage.
         Uses the fine-tuned LLaMA model to match demographics → persona.

Stage 6: Predict mode for each test trip using the assigned persona
         + trip attributes. Output is saved as a full CSV including
         all original test columns plus predicted_mode and reason.

Output:
  data/personas/test_personas.csv        — assigned personas for test travelers
  outputs/test_predictions_full.csv      — full predictions CSV (Stage 7 input)
"""

import json
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    BASE_MODEL,
    DEMOGRAPHIC_COLS,
    HF_TOKEN,
    LABEL_COL,
    MAX_SEQ_LEN,
    MODE_MAP,
    OLLAMA_MODEL,
    OUTPUTS_DIR,
    PERSONAS_DIR,
    PROCESSED_DIR,
    TRAVELER_ID_COL,
    VALID_MODES,
)
from utils import (
    extract_mode_from_text,
    format_demographics,
    format_target_trip,
    get_logger,
    get_ollama_client,
)

logger = get_logger(__name__, log_file="test_predict.log")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — PERSONA ASSIGNMENT PROMPT  (demographics only, no history)
# ─────────────────────────────────────────────────────────────────────────────

def build_persona_assignment_prompt(demo_row: pd.Series, known_personas: list[dict]) -> str:
    """
    Build the persona assignment prompt for one test traveler.
    The model receives demographics + the list of known train personas
    and must assign the best-matching persona.

    Args:
        demo_row       : one row from test_traveler_demographics.csv
        known_personas : list of {persona_label, identity_profile, preference_profile}
                         dicts derived from train_personas_raw.csv

    Returns:
        Filled prompt string
    """
    demo_text = format_demographics(demo_row)

    # Format the known persona list for the prompt
    persona_list_text = "\n".join([
        f"  {i+1}. Label: {p['persona_label']}\n"
        f"     Identity: {p['identity_profile']}\n"
        f"     Preferences: {p['preference_profile']}"
        for i, p in enumerate(known_personas)
    ])

    return f"""\
Task definition:
You are a transportation behavior expert. Based solely on a traveler's \
demographic profile, assign the most appropriate behavioral mobility persona \
from the list of known persona types that were inferred from the training data.

Data description:
<demographics> provides the traveler's static attributes:
  female: 1 = female, 0 = male.
  business: 1 = business trip, 0 = personal / leisure.
  income: Annual income in £.

<known_personas> is the list of persona types learned from the training set:
{persona_list_text}

Thinking guidance:
1. Demographic interpretation: what does gender, trip purpose, and income level \
   imply about this traveler's likely transport preferences?
2. Persona matching: which known persona's preference_profile best aligns with \
   the demographic implications? Consider income-cost sensitivity, \
   business vs. leisure time-sensitivity.
3. Confidence check: if multiple personas are plausible, choose the one whose \
   preference_profile most closely matches the demographic profile.

Output format:
Output ONLY a valid JSON object with exactly 3 keys. No markdown fences, no preamble:
{{
  "persona_label":      "<assigned persona label from known_personas>",
  "identity_profile":   "<identity_profile of the assigned persona>",
  "preference_profile": "<preference_profile of the assigned persona>"
}}

Data inputs:
<demographics>: {demo_text}
<known_personas>: see list above
"""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — PREDICTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_prompt(row: pd.Series) -> tuple[str, str]:
    """
    Build the system and user messages for mode choice prediction.
    Mirrors the instruction template format used during fine-tuning
    to ensure alignment between training and inference.

    Returns:
        (system_msg, user_msg) tuple
    """
    system_msg = (
        "You are a transportation mode choice prediction model simulating "
        "a specific traveler persona inferred from their behavioral history.\n\n"
        "<persona>\n"
        f"Label:       {row.get('persona_label', 'unknown')}\n"
        f"Identity:    {row.get('identity_profile', '')}\n"
        f"Preferences: {row.get('preference_profile', '')}"
    )

    gender  = "Female" if row.get("female", 0) == 1 else "Male"
    purpose = "Business" if row.get("business", 0) == 1 else "Personal/Leisure"
    income  = f"£{row['income']:,.0f}" if pd.notna(row.get("income")) else "unknown"
    target  = format_target_trip(row)

    user_msg = f"""\
Task definition:
Your task is to predict a traveler's transport mode choice for a target trip \
based on their inferred behavioral persona and the attributes of the available \
transport options.

Data description:
<persona> contains: persona_label, identity_profile, preference_profile (see system).
<demographics>: female={int(row.get('female',0))}, business={int(row.get('business',0))}, income={income}
<target>: {target}

Thinking guidance:
1. Behavioral persona: the traveler's known attitudes toward time vs. cost trade-offs, \
   comfort, and habitual mode loyalty.
2. Demographic profile: business=time priority+possible reimbursement; \
   leisure=cost sensitivity; high income=reduced fare sensitivity.
3. Trip attributes: compare only modes in available_modes on time, cost, access, service.
4. Persona-option alignment: which available mode best matches the preference_profile?

Output format:
Output a JSON object with keys "prediction" (one of: car, bus, air, rail) and \
"reason" (one concise sentence explaining the prediction). No line breaks.

Data inputs:
<persona>: {row.get('persona_label','')} | {row.get('identity_profile','')} | {row.get('preference_profile','')}
<demographics>: female={int(row.get('female',0))}, business={int(row.get('business',0))}, income={income}
<target>: {target}
"""
    return system_msg, user_msg


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — ASSIGN PERSONAS TO TEST TRAVELERS (demographics only)
# ─────────────────────────────────────────────────────────────────────────────

def assign_test_personas(test_demos: pd.DataFrame, known_personas: list[dict],
                          model, tokenizer, device) -> pd.DataFrame:
    """
    For each test traveler, call the fine-tuned model to assign a persona
    using only their demographic profile.

    Args:
        test_demos     : test_traveler_demographics.csv as DataFrame
        known_personas : unique personas from train_personas_raw.csv
        model          : loaded fine-tuned model
        tokenizer      : corresponding tokenizer
        device         : torch device

    Returns:
        test_demos extended with columns: persona_label, identity_profile, preference_profile
    """
    results = []
    n = len(test_demos)
    logger.info(f"Stage 5: Assigning personas to {n} test travelers...")

    for i, (_, row) in enumerate(test_demos.iterrows(), 1):
        if i % 10 == 0:
            logger.info(f"  [{i}/{n}] Assigning persona...")

        prompt = build_persona_assignment_prompt(row, known_personas)

        # Format as single-turn chat (no system message for this task)
        messages = [{"role": "user", "content": prompt}]
        filled   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs   = tokenizer(filled, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0], skip_special_tokens=True)

        # Parse the persona JSON from model output
        try:
            start = raw.index("{")
            end   = raw.rindex("}") + 1
            obj   = json.loads(raw[start:end])
            assigned = {
                "persona_label":      obj.get("persona_label", known_personas[0]["persona_label"]),
                "identity_profile":   obj.get("identity_profile", ""),
                "preference_profile": obj.get("preference_profile", ""),
            }
        except (ValueError, json.JSONDecodeError):
            # Fallback: assign the most common persona
            logger.warning(f"  Parse failed for traveler {row[TRAVELER_ID_COL]}; using fallback persona")
            assigned = {
                "persona_label":      known_personas[0]["persona_label"],
                "identity_profile":   known_personas[0]["identity_profile"],
                "preference_profile": known_personas[0]["preference_profile"],
            }

        results.append({TRAVELER_ID_COL: row[TRAVELER_ID_COL], **assigned})

    personas_df = pd.DataFrame(results)
    logger.info(f"Persona distribution:\n{personas_df['persona_label'].value_counts().to_string()}")
    return test_demos.merge(personas_df, on=TRAVELER_ID_COL, how="left")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — PREDICT MODE FOR EACH TEST TRIP
# ─────────────────────────────────────────────────────────────────────────────

def predict_mode(row: pd.Series, model, tokenizer, device) -> tuple[str, str]:
    """
    Predict the transport mode for one test trip using the fine-tuned model.

    Returns:
        (predicted_mode, reason) tuple
    """
    system_msg, user_msg = build_prediction_prompt(row)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0], skip_special_tokens=True)

    # Extract prediction and reason from JSON response
    mode   = extract_mode_from_text(raw)
    reason = ""
    try:
        start  = raw.index("{")
        end    = raw.rindex("}") + 1
        obj    = json.loads(raw[start:end])
        reason = obj.get("reason", "")
    except (ValueError, json.JSONDecodeError):
        pass

    return mode, reason


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_finetuned_model():
    """
    Load the base LLaMA model with 4-bit quantisation and apply the LoRA adapter.
    Returns (model, tokenizer, device).
    """
    adapter_path = f"{OUTPUTS_DIR}lora_adapter_final"
    logger.info(f"Loading fine-tuned model from {adapter_path}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN or None,
    )
    model  = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Model loaded on {device}")
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("STAGES 5 & 6: PERSONA ASSIGNMENT + MODE PREDICTION")
    logger.info("=" * 60)

    # ── Load test data ────────────────────────────────────────────────────────
    test_trips = pd.read_csv(f"{PROCESSED_DIR}test_trips.csv")
    test_demos = test_trips[[TRAVELER_ID_COL] + DEMOGRAPHIC_COLS].drop_duplicates()
    if "chosen_mode" not in test_trips.columns:
        test_trips["true_mode"] = test_trips[LABEL_COL].map(MODE_MAP)
    else:
        test_trips["true_mode"] = test_trips["chosen_mode"]
    logger.info(f"Test set: {len(test_trips)} trips | {len(test_demos)} travelers")

    # ── Load known personas (from train set) for Stage 5 ─────────────────────
    train_personas = pd.read_csv(f"{PERSONAS_DIR}train_personas_raw.csv")
    known_personas = (
        train_personas[["persona_label", "identity_profile", "preference_profile"]]
        .drop_duplicates("persona_label")
        .to_dict("records")
    )
    logger.info(f"Known persona types from train: {len(known_personas)}")

    # ── Load fine-tuned model ─────────────────────────────────────────────────
    model, tokenizer, device = load_finetuned_model()

    # ── Stage 5: Assign personas to test travelers ────────────────────────────
    test_demos_with_persona = assign_test_personas(test_demos, known_personas, model, tokenizer, device)

    # Save test personas
    persona_out = f"{PERSONAS_DIR}test_personas.csv"
    test_demos_with_persona.to_csv(persona_out, index=False)
    logger.info(f"Saved test personas → {persona_out}")

    # Merge personas back to full trip rows
    test_trips = test_trips.merge(
        test_demos_with_persona[[TRAVELER_ID_COL, "persona_label", "identity_profile", "preference_profile"]],
        on=TRAVELER_ID_COL,
        how="left",
    )

    # ── Stage 6: Predict mode for each test trip ──────────────────────────────
    logger.info(f"Stage 6: Predicting modes for {len(test_trips)} test trips...")
    predictions, reasons = [], []

    for i, (_, row) in enumerate(test_trips.iterrows(), 1):
        if i % 50 == 0:
            logger.info(f"  [{i}/{len(test_trips)}] Predicting...")
        try:
            pred, reason = predict_mode(row, model, tokenizer, device)
        except Exception as e:
            logger.warning(f"  Trip {i} prediction failed: {e}; defaulting to 'car'")
            pred, reason = "car", ""
        predictions.append(pred)
        reasons.append(reason)

    test_trips["predicted_mode"] = predictions
    test_trips["reason"]         = reasons

    # ── Save full predictions CSV ─────────────────────────────────────────────
    # All original test columns + persona_label + predicted_mode + reason
    out_path = f"{OUTPUTS_DIR}test_predictions_full.csv"
    test_trips.to_csv(out_path, index=False)
    logger.info(f"Saved predictions → {out_path}")

    # Quick accuracy report
    acc = (test_trips["predicted_mode"] == test_trips["true_mode"]).mean()
    logger.info(f"Quick accuracy: {acc:.3f} ({acc*100:.1f}%)")
    logger.info("Stages 5 & 6 complete. Run evaluate.py for full metrics.")


if __name__ == "__main__":
    main()
