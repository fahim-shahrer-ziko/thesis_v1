"""
prepare_data.py  —  Stage 1
============================
Load the Apollo Mode Choice dataset, filter by SP/RP type,
and split into train/test at the TRAVELER level (not trip level)
to prevent data leakage.

Outputs (all in data/processed/):
  train_trips.csv                — all train trip rows
  test_trips.csv                 — all test trip rows
  train_traveler_demographics.csv — one row per train traveler
  test_traveler_demographics.csv  — one row per test traveler
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ALL_ATTR_COLS,
    AVAILABILITY_COLS,
    DATA_PATH,
    DATA_TYPE,
    DEMOGRAPHIC_COLS,
    LABEL_COL,
    MODE_MAP,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TRAVELER_ID_COL,
)
from utils import get_logger

logger = get_logger(__name__, log_file="prepare_data.log")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(file_path: str) -> pd.DataFrame:
    """
    Load CSV, replace 'NA' strings with NaN, convert numeric columns,
    filter by DATA_TYPE, and add derived columns (trip_id, chosen_mode).
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"  Raw shape: {df.shape}")

    # Replace 'NA' strings (not Python None) with actual NaN
    df = df.replace("NA", np.nan)

    # Convert all numeric columns in bulk
    numeric_cols = DEMOGRAPHIC_COLS + AVAILABILITY_COLS + ALL_ATTR_COLS + [LABEL_COL]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Filter by SP / RP type ───────────────────────────────────────────────
    logger.info(f"  Filtering by DATA_TYPE={DATA_TYPE}")
    orig_n = len(df)

    if DATA_TYPE == "SP_ONLY":
        df = df[df["SP"] == 1].copy()
        df["trip_id"] = df["SP_task"].astype(str)
        logger.info(f"  SP only: {len(df)} trips (from {orig_n})")

    elif DATA_TYPE == "RP_ONLY":
        df = df[df["RP"] == 1].copy()
        df["trip_id"] = df["RP_journey"].astype(str)
        logger.info(f"  RP only: {len(df)} trips (from {orig_n})")

    else:  # BOTH
        df["trip_id"] = df.apply(
            lambda x: f"RP_{x['RP_journey']}" if x["RP"] == 1 else f"SP_{x['SP_task']}",
            axis=1,
        )
        logger.info(f"  SP+RP combined: {len(df)} trips")

    # ── Choice column cleanup ────────────────────────────────────────────────
    df[LABEL_COL] = df[LABEL_COL].astype(float).astype("Int64")
    df["chosen_mode"] = df[LABEL_COL].map(MODE_MAP)

    # Drop rows missing traveler ID or choice label
    before = len(df)
    df = df.dropna(subset=[TRAVELER_ID_COL, LABEL_COL])
    logger.info(f"  Dropped {before - len(df)} rows missing ID or choice")

    logger.info(f"  Final: {len(df)} trips | {df[TRAVELER_ID_COL].nunique()} travelers")
    logger.info(f"  Mode distribution:\n{df['chosen_mode'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRAVELER-LEVEL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_by_traveler(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset at the TRAVELER level so that all trips of a traveler
    go entirely into train OR test — never both.
    This is critical to prevent feature leakage.
    """
    unique_travelers = df[TRAVELER_ID_COL].unique()
    train_ids, test_ids = train_test_split(
        unique_travelers,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_df = df[df[TRAVELER_ID_COL].isin(train_ids)].copy()
    test_df  = df[df[TRAVELER_ID_COL].isin(test_ids)].copy()

    logger.info(
        f"Split: {len(train_ids)} train travelers ({len(train_df)} trips) | "
        f"{len(test_ids)} test travelers ({len(test_df)} trips)"
    )
    logger.info(f"Train mode distribution:\n{train_df['chosen_mode'].value_counts().to_string()}")
    logger.info(f"Test  mode distribution:\n{test_df['chosen_mode'].value_counts().to_string()}")
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA PREPARATION")
    logger.info("=" * 60)

    # Load and clean
    df = load_and_clean(DATA_PATH)

    # Split
    train_df, test_df = split_by_traveler(df)

    # Save trip-level splits
    train_df.to_csv(f"{PROCESSED_DIR}train_trips.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}test_trips.csv",   index=False)

    # Save traveler-level demographics (one row per traveler)
    # These are used later by Stage 5 for test persona assignment
    train_demos = train_df[[TRAVELER_ID_COL] + DEMOGRAPHIC_COLS].drop_duplicates()
    test_demos  = test_df[[TRAVELER_ID_COL]  + DEMOGRAPHIC_COLS].drop_duplicates()
    train_demos.to_csv(f"{PROCESSED_DIR}train_traveler_demographics.csv", index=False)
    test_demos.to_csv(f"{PROCESSED_DIR}test_traveler_demographics.csv",   index=False)

    logger.info("Saved:")
    logger.info(f"  {PROCESSED_DIR}train_trips.csv                ({len(train_df)} rows)")
    logger.info(f"  {PROCESSED_DIR}test_trips.csv                 ({len(test_df)} rows)")
    logger.info(f"  {PROCESSED_DIR}train_traveler_demographics.csv ({len(train_demos)} travelers)")
    logger.info(f"  {PROCESSED_DIR}test_traveler_demographics.csv  ({len(test_demos)} travelers)")
    logger.info("Stage 1 complete.")


if __name__ == "__main__":
    main()
