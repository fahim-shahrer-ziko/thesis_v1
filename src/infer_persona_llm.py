"""
infer_persona_llm.py  —  Stage 2
==================================
For each unique traveler in the TRAIN set, call the local Ollama LLaMA
model to infer a 3-field behavioral persona from:
  - demographics (gender, trip purpose, income)
  - a sample of their historical trip choices (with full mode attributes)

Output: data/personas/train_personas_raw.csv
  Columns: traveler_id, persona_label, identity_profile, preference_profile

IMPORTANT — no test leakage:
  Only train travelers are processed here.
  Test-set personas are assigned in Stage 5 using demographics only.
"""

import json
import time

import pandas as pd

from config import (
    DEMOGRAPHIC_COLS,
    LABEL_COL,
    MAX_TRIP_SAMPLE,
    MODE_MAP,
    OLLAMA_MODEL,
    PERSONAS_DIR,
    PROCESSED_DIR,
    TRAVELER_ID_COL,
)
from utils import (
    format_demographics,
    format_trip_history,
    get_logger,
    get_ollama_client,
    parse_json_response,
)

logger = get_logger(__name__, log_file="infer_persona.log")

# ── Ollama client (OpenAI-compatible, local) ──────────────────────────────────
client = get_ollama_client()

# ─────────────────────────────────────────────────────────────────────────────
# PERSONA GENERATION PROMPT  (matches Stage 2 in the project guide)
# ─────────────────────────────────────────────────────────────────────────────
PERSONA_PROMPT = """\
Your task is to infer a behavioral mobility persona for a traveler based on \
their static demographic profile and a chronological sample of their historical \
transport mode choices, including the full attributes of every available mode \
for each trip. You are NOT asked to predict a future trip. \
The output is a concise structured persona profile.

<demographics>
{demographics_text}

<history>
{trips_text}

---
Thinking guidance (internal only — do NOT include in output):

1. Socio-demographic context: What does gender, trip purpose, and income imply?
   Business travel = time sensitivity + possible reimbursement;
   high income = lower cost sensitivity.

2. Revealed preference analysis: For each trip, which mode was chosen?
   Were cheaper or faster alternatives rejected? Identify consistent patterns.

3. Cost vs. time trade-off: Does this traveler consistently choose
   faster-but-costlier or slower-but-cheaper modes?

4. Comfort and service sensitivity: Does the traveler prefer high-service
   options (wifi, meals) at extra cost? Do access times matter significantly?

5. Habitual loyalty vs. rational switching: Does the traveler stick to one
   mode regardless of attribute changes, or switch when an alternative dominates?

6. Behavioral archetype synthesis: Combine all findings into one coherent
   persona label and description.

Rules:
- Do NOT invent numeric values absent from the input.
- Only reason about modes listed in available_modes per trip; ignore None values.
- If fewer than 3 trips, note limited evidence in preference_profile.

---
Output ONLY a valid JSON object with exactly 3 keys.
No markdown fences, no preamble, no line breaks inside values:

{{
  "persona_label":      "<3-8 word mobility archetype>",
  "identity_profile":   "<1-2 sentences on socio-demographic factors. Max 60 words.>",
  "preference_profile": "<2-3 sentences on time/cost/comfort attitudes. Max 80 words.>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# CORE INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def infer_persona(traveler_id, demos_df: pd.DataFrame, trips_df: pd.DataFrame,
                  max_retries: int = 3) -> dict:
    """
    Call Ollama to infer persona for one traveler.

    Args:
        traveler_id  : the traveler's ID value
        demos_df     : train_traveler_demographics.csv as DataFrame
        trips_df     : train_trips.csv as DataFrame (filtered to train set)
        max_retries  : number of retry attempts on JSON parse failure

    Returns:
        dict with keys: traveler_id, persona_label,
                        identity_profile, preference_profile
    """
    # Get this traveler's demographics (single row)
    demo_row = demos_df[demos_df[TRAVELER_ID_COL] == traveler_id].iloc[0]
    demo_text = format_demographics(demo_row)

    # Get this traveler's past trips in chronological order
    traveler_trips = (
        trips_df[trips_df[TRAVELER_ID_COL] == traveler_id]
        .sort_values("trip_id")
    )
    trips_text = format_trip_history(traveler_trips, n=MAX_TRIP_SAMPLE)

    # Fill prompt template
    prompt = PERSONA_PROMPT.format(
        demographics_text=demo_text,
        trips_text=trips_text,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                # Ollama's JSON mode: forces structured output
                extra_body={"format": "json"},
            )
            raw = response.choices[0].message.content
            result = parse_json_response(raw)

            # Validate required keys exist
            required = {"persona_label", "identity_profile", "preference_profile"}
            missing = required - result.keys()
            if missing:
                raise ValueError(f"Missing keys: {missing}")

            result["traveler_id"] = traveler_id
            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"  Traveler {traveler_id} attempt {attempt}: parse error — {e}")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"  Traveler {traveler_id} attempt {attempt}: API error — {e}")
            time.sleep(2)

    # Fallback after all retries exhausted
    logger.error(f"  Traveler {traveler_id}: giving up after {max_retries} attempts")
    return {
        "traveler_id": traveler_id,
        "persona_label": "unknown traveler",
        "identity_profile": "",
        "preference_profile": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("STAGE 2: PERSONA GENERATION (Ollama LLaMA)")
    logger.info(f"  Model  : {OLLAMA_MODEL}")
    logger.info(f"  Backend: local Ollama — no token limits, no cost")
    logger.info("=" * 60)

    # Load train data
    train_trips = pd.read_csv(f"{PROCESSED_DIR}train_trips.csv")
    train_demos = pd.read_csv(f"{PROCESSED_DIR}train_traveler_demographics.csv")

    # Ensure chosen_mode column exists
    if "chosen_mode" not in train_trips.columns:
        train_trips["chosen_mode"] = train_trips[LABEL_COL].map(MODE_MAP)

    unique_ids = train_demos[TRAVELER_ID_COL].unique()
    logger.info(f"Inferring personas for {len(unique_ids)} train travelers...")

    personas = []
    for i, tid in enumerate(unique_ids, 1):
        logger.info(f"  [{i}/{len(unique_ids)}] Traveler {tid}")
        persona = infer_persona(tid, train_demos, train_trips)
        personas.append(persona)
        # Brief pause to avoid overwhelming local Ollama server
        time.sleep(0.1)

    # Save results
    out_path = f"{PERSONAS_DIR}train_personas_raw.csv"
    pd.DataFrame(personas).to_csv(out_path, index=False)
    logger.info(f"Saved {len(personas)} personas → {out_path}")

    # Log a sample
    s = personas[0]
    logger.info(f"Sample — {s['traveler_id']}: {s['persona_label']}")
    logger.info("Stage 2 complete.")


if __name__ == "__main__":
    main()
