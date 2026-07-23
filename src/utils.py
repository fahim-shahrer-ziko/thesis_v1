"""
utils.py
========
Shared formatting and helper functions reused across all pipeline stages.
Import from here to keep each stage script clean and DRY.
"""

import json
import logging
import os
import sys
from typing import Optional

import pandas as pd

from config import (
    AVAILABILITY_COLS,
    DEMOGRAPHIC_COLS,
    LOGS_DIR,
    MODE_MAP,
    SERVICE_LABELS,
    TRAVELER_ID_COL,
    VALID_MODES,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a logger that writes to stdout and optionally to a file.
    Call once per module: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on reimport
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        os.makedirs(LOGS_DIR, exist_ok=True)
        fh = logging.FileHandler(os.path.join(LOGS_DIR, log_file))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# TRIP ATTRIBUTE FORMATTERS
# Reused by: infer_persona_llm, generate_instructions, build_instructions,
#            test_predict
# ─────────────────────────────────────────────────────────────────────────────

def format_demographics(row: pd.Series) -> str:
    """
    Convert a traveler's demographic row into a readable string.

    Example output:
        "Gender: Female; Trip purpose: Business; Annual income: £45,000"
    """
    parts = []
    if pd.notna(row.get("female")):
        parts.append(f"Gender: {'Female' if row['female'] == 1 else 'Male'}")
    if pd.notna(row.get("business")):
        parts.append(f"Trip purpose: {'Business' if row['business'] == 1 else 'Personal/Leisure'}")
    if pd.notna(row.get("income")):
        parts.append(f"Annual income: £{row['income']:,.0f}")
    return "; ".join(parts) if parts else "Demographics: unknown"


def format_available_modes(row: pd.Series) -> str:
    """
    Format only the modes that are available for a given trip row.
    Unavailable modes (av_* == 0 or NaN) are silently skipped.

    Example output (multi-line string):
        • CAR:  45 min travel, £30 cost
        • RAIL: 62 min travel, £18 cost, 10 min access, service: wifi
    """
    lines = []

    if row.get("av_car", 0) == 1 and pd.notna(row.get("time_car")):
        lines.append(f"• CAR:  {row['time_car']:.0f} min travel, £{row['cost_car']:.0f} cost")

    if row.get("av_bus", 0) == 1 and pd.notna(row.get("time_bus")):
        lines.append(
            f"• BUS:  {row['time_bus']:.0f} min travel, £{row['cost_bus']:.0f} cost, "
            f"{row['access_bus']:.0f} min access"
        )

    if row.get("av_air", 0) == 1 and pd.notna(row.get("time_air")):
        svc = SERVICE_LABELS.get(int(row["service_air"]), "standard")
        lines.append(
            f"• AIR:  {row['time_air']:.0f} min travel, £{row['cost_air']:.0f} cost, "
            f"{row['access_air']:.0f} min access, service: {svc}"
        )

    if row.get("av_rail", 0) == 1 and pd.notna(row.get("time_rail")):
        svc = SERVICE_LABELS.get(int(row["service_rail"]), "standard")
        lines.append(
            f"• RAIL: {row['time_rail']:.0f} min travel, £{row['cost_rail']:.0f} cost, "
            f"{row['access_rail']:.0f} min access, service: {svc}"
        )

    return "\n".join(lines) if lines else "No modes available for this trip."


def format_single_trip(trip: pd.Series, idx: int) -> str:
    """
    Format a single historical trip (for persona inference history block).
    Shows the chosen mode's attributes AND the alternatives that were rejected.

    Example output:
        Trip 1: Chose RAIL  |  Available: car, rail
          Chosen  → rail: 62 min, £18, access 10 min, wifi
          Rejected→ car: 45 min, £30
    """
    mode = trip.get("chosen_mode", "unknown")
    avail = [m for m in VALID_MODES if trip.get(f"av_{m}", 0) == 1]
    lines = [f"Trip {idx}: Chose {mode.upper()}  |  Available: {', '.join(avail)}"]

    # Chosen mode detail
    if mode == "car":
        lines.append(f"  Chosen  → car:  {trip['time_car']} min, £{trip['cost_car']}")
    elif mode == "bus":
        lines.append(f"  Chosen  → bus:  {trip['time_bus']} min, £{trip['cost_bus']}, access {trip['access_bus']} min")
    elif mode == "air":
        svc = SERVICE_LABELS.get(int(trip.get("service_air", 1)), "std")
        lines.append(f"  Chosen  → air:  {trip['time_air']} min, £{trip['cost_air']}, access {trip['access_air']} min, {svc}")
    elif mode == "rail":
        svc = SERVICE_LABELS.get(int(trip.get("service_rail", 1)), "std")
        lines.append(f"  Chosen  → rail: {trip['time_rail']} min, £{trip['cost_rail']}, access {trip['access_rail']} min, {svc}")

    # Rejected alternatives
    alts = []
    if mode != "car"  and trip.get("av_car",  0) == 1: alts.append(f"car {trip['time_car']} min £{trip['cost_car']}")
    if mode != "bus"  and trip.get("av_bus",  0) == 1: alts.append(f"bus {trip['time_bus']} min £{trip['cost_bus']} (access {trip['access_bus']} min)")
    if mode != "air"  and trip.get("av_air",  0) == 1: alts.append(f"air {trip['time_air']} min £{trip['cost_air']} (access {trip['access_air']} min)")
    if mode != "rail" and trip.get("av_rail", 0) == 1: alts.append(f"rail {trip['time_rail']} min £{trip['cost_rail']} (access {trip['access_rail']} min)")

    if alts:
        lines.append(f"  Rejected→ {' | '.join(alts)}")
    return "\n".join(lines)


def format_trip_history(trips_df: pd.DataFrame, n: int = 5) -> str:
    """
    Combine up to n historical trips into a block for the persona prompt.
    Trips are already in chronological order (sorted by trip_id upstream).
    """
    blocks = [
        format_single_trip(row, i + 1)
        for i, (_, row) in enumerate(trips_df.head(n).iterrows())
    ]
    return "\n\n".join(blocks)


def format_target_trip(row: pd.Series) -> str:
    """
    Format a target trip for the prediction prompt (Stage 6).
    Same as format_available_modes but also includes the structured data tag.
    """
    avail = [m for m in VALID_MODES if row.get(f"av_{m}", 0) == 1]
    attrs = f"available_modes=[{', '.join(avail)}]"

    if "car"  in avail: attrs += f", time_car={row['time_car']:.0f}, cost_car={row['cost_car']:.0f}"
    if "bus"  in avail: attrs += f", time_bus={row['time_bus']:.0f}, cost_bus={row['cost_bus']:.0f}, access_bus={row['access_bus']:.0f}"
    if "air"  in avail:
        svc = SERVICE_LABELS.get(int(row.get("service_air", 1)), "std")
        attrs += f", time_air={row['time_air']:.0f}, cost_air={row['cost_air']:.0f}, access_air={row['access_air']:.0f}, service_air={svc}"
    if "rail" in avail:
        svc = SERVICE_LABELS.get(int(row.get("service_rail", 1)), "std")
        attrs += f", time_rail={row['time_rail']:.0f}, cost_rail={row['cost_rail']:.0f}, access_rail={row['access_rail']:.0f}, service_rail={svc}"

    return attrs + ", next_mode=<unknown>"


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CLIENT FACTORY
# Returns an openai.OpenAI client pointed at the local Ollama server.
# Reused by: infer_persona_llm, generate_instructions, test_predict
# ─────────────────────────────────────────────────────────────────────────────

def get_ollama_client():
    """
    Return an OpenAI-compatible client connected to the local Ollama server.
    The 'openai' SDK is reused — only base_url changes.
    api_key is a dummy string; Ollama ignores it.
    """
    from openai import OpenAI
    from config import OLLAMA_BASE_URL

    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING HELPER
# Handles cases where the model wraps output in ```json ... ``` fences
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict:
    """
    Safely parse a JSON string returned by an LLM.
    Strips accidental markdown code fences before parsing.

    Raises json.JSONDecodeError if parsing still fails after cleanup.
    """
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# MODE EXTRACTOR
# Parses the final predicted mode from any LLM text output
# Reused by: test_predict
# ─────────────────────────────────────────────────────────────────────────────

def extract_mode_from_text(text: str, fallback: str = "car") -> str:
    """
    Extract predicted transport mode from free-form LLM output.

    Search strategy (in order):
      1. JSON key "prediction" in the response
      2. "Final answer: X" pattern anywhere in text
      3. Last occurrence of a valid mode word
      4. fallback (default: "car")
    """
    low = text.lower()

    # Strategy 1: parse JSON block if present
    try:
        # Find first { ... } block
        start = low.index("{")
        end = low.rindex("}") + 1
        obj = json.loads(text[start:end])
        pred = obj.get("prediction", "").strip().lower()
        if pred in VALID_MODES:
            return pred
    except (ValueError, json.JSONDecodeError, KeyError):
        pass

    # Strategy 2: "final answer: X" or "prediction: X"
    for line in reversed(low.splitlines()):
        if "final answer" in line or "prediction" in line:
            for m in VALID_MODES:
                if m in line:
                    return m

    # Strategy 3: last mode word in entire text
    for word in reversed(low.split()):
        word = word.strip(".,!?:\"'")
        if word in VALID_MODES:
            return word

    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# REASON GENERATOR (deterministic, for training assistant messages)
# ─────────────────────────────────────────────────────────────────────────────

def build_gold_reason(row: pd.Series) -> str:
    """
    Generate a deterministic one-sentence reason for the ground-truth chosen_mode.
    Used in build_instructions.py to fill the assistant's 'reason' field
    without hallucinating — derived purely from the row's factual data.

    Example output:
        "The traveler chose rail as it aligns with their cost-conscious persona
         — rail (£18) was cheaper than car (£30), consistent with their
         preference for lower-cost modes."
    """
    mode = row.get("chosen_mode", "car")
    persona = row.get("persona_label", "this traveler")

    # Collect (mode, time, cost) for available modes
    avail = []
    for m in VALID_MODES:
        if row.get(f"av_{m}", 0) == 1 and pd.notna(row.get(f"time_{m}")):
            avail.append((m, float(row[f"time_{m}"]), float(row[f"cost_{m}"])))

    if not avail:
        return f"The traveler chose {mode} as it was the only available option."

    cheapest = min(avail, key=lambda x: x[2])
    fastest  = min(avail, key=lambda x: x[1])

    if mode == cheapest[0] and mode != fastest[0]:
        return (f"The traveler chose {mode} as it aligns with the '{persona}' persona "
                f"— {mode} (£{cheapest[2]:.0f}) was the cheapest option, consistent "
                f"with a cost-sensitive preference over the faster {fastest[0]}.")
    elif mode == fastest[0] and mode != cheapest[0]:
        return (f"The traveler chose {mode} as it aligns with the '{persona}' persona "
                f"— {mode} ({fastest[1]:.0f} min) was the fastest option, consistent "
                f"with a time-sensitive preference over the cheaper {cheapest[0]}.")
    elif mode == cheapest[0] and mode == fastest[0]:
        return (f"The traveler chose {mode} as it aligns with the '{persona}' persona "
                f"— {mode} was both the cheapest (£{cheapest[2]:.0f}) and fastest "
                f"({fastest[1]:.0f} min) option available.")
    else:
        return (f"The traveler chose {mode} as it aligns with the '{persona}' persona "
                f"— despite not being cheapest ({cheapest[0]}) or fastest ({fastest[0]}), "
                f"{mode} best matches their comfort and service preferences.")
