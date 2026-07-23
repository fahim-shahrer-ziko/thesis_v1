"""
check_data.py
Validates trip_data.csv and persona_data.csv before running the pipeline.
Usage:  python check_data.py
"""

import argparse
import re

import pandas as pd
import config


def check(trip_csv, persona_csv):
    issues, warnings = [], []

    print(f"Loading {trip_csv} ...")
    trip_df = pd.read_csv(trip_csv)
    print(f"  {len(trip_df)} rows, {len(trip_df.columns)} columns")

    print(f"Loading {persona_csv} ...")
    persona_df = pd.read_csv(persona_csv)
    print(f"  {len(persona_df)} rows, {len(persona_df.columns)} columns")

    # 1. Required columns
    req_trip = [config.COL_PASSENGER_ID, config.COL_GENDER, config.COL_AGE,
                config.COL_ENTRY_DT, config.COL_EXIT_DT, config.COL_DAYNAME,
                config.COL_ENTRY_STATION]
    for c in req_trip:
        if c not in trip_df.columns:
            issues.append(f"trip_data.csv missing column: '{c}'")

    has_exit = config.COL_EXIT_STATION in trip_df.columns
    has_typo = config.COL_EXIT_STATION_TYPO in trip_df.columns
    if not has_exit and not has_typo:
        issues.append(f"trip_data.csv: exit station column not found "
                      f"(expected '{config.COL_EXIT_STATION}' or '{config.COL_EXIT_STATION_TYPO}')")
    elif has_typo and not has_exit:
        warnings.append(f"Using typo column '{config.COL_EXIT_STATION_TYPO}' — will be auto-renamed.")

    if config.COL_PASSENGER_ID not in persona_df.columns:
        issues.append(f"persona_data.csv missing join key: '{config.COL_PASSENGER_ID}'")
    for c in config.PERSONA_COLUMNS:
        if c not in persona_df.columns:
            issues.append(f"persona_data.csv missing column: '{c}'")

    if issues:
        print("\n--- COLUMN ISSUES (fix before continuing) ---")
        for i in issues: print(f"  ❌ {i}")
        return

    # 2. Datetime parsing
    for col in [config.COL_ENTRY_DT, config.COL_EXIT_DT]:
        parsed = pd.to_datetime(trip_df[col], errors="coerce")
        bad = parsed.isna().sum() - trip_df[col].isna().sum()
        if bad > 0:
            issues.append(f"'{col}': {bad} values could not be parsed as datetime")

    # 3. Missing values
    for col in req_trip + ([config.COL_EXIT_STATION] if has_exit else [config.COL_EXIT_STATION_TYPO]):
        n = trip_df[col].isna().sum()
        if n > 0:
            issues.append(f"'{col}': {n} missing values in trip_data.csv")

    # 4. Join coverage
    trip_ids    = set(trip_df[config.COL_PASSENGER_ID].unique())
    persona_ids = set(persona_df[config.COL_PASSENGER_ID].unique())
    missing     = trip_ids - persona_ids
    if missing:
        warnings.append(f"{len(missing)} passengers in trip CSV have no persona row → will be dropped")

    # 5. Trip count distribution
    counts = trip_df.groupby(config.COL_PASSENGER_ID).size()
    n_total  = len(counts)
    n_below  = (counts < config.MIN_TRIPS_PER_PASSENGER).sum()
    print(f"\n--- TRIP COUNT DISTRIBUTION ---")
    print(f"  Total passengers:    {n_total}")
    print(f"  min={counts.min()}  median={counts.median():.0f}  "
          f"mean={counts.mean():.1f}  max={counts.max()}")
    print(f"  Below MIN_TRIPS ({config.MIN_TRIPS_PER_PASSENGER}): "
          f"{n_below} ({100*n_below/n_total:.1f}%) → will be dropped")
    print(f"  Estimated retained:  ~{n_total - n_below - len(missing)}")
    if config.MAX_PASSENGERS:
        print(f"  MAX_PASSENGERS={config.MAX_PASSENGERS} → will subsample to that")

    # 6. Exit before entry
    entry = pd.to_datetime(trip_df[config.COL_ENTRY_DT], errors="coerce")
    exit_ = pd.to_datetime(trip_df[config.COL_EXIT_DT],  errors="coerce")
    bad_order = ((exit_ < entry) & entry.notna() & exit_.notna()).sum()
    if bad_order > 0:
        issues.append(f"{bad_order} trips have Exit_DT earlier than Entry_DT")

    print("\n" + "="*55)
    if issues:
        print(f"❌ {len(issues)} ISSUE(S) — fix before running build_dataset.py:")
        for i in issues: print(f"  - {i}")
    else:
        print("✅ No blocking issues found.")
    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNING(S):")
        for w in warnings: print(f"  - {w}")
    print("="*55)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trip_csv",    default=config.TRIP_CSV_PATH)
    p.add_argument("--persona_csv", default=config.PERSONA_CSV_PATH)
    args = p.parse_args()
    check(args.trip_csv, args.persona_csv)


if __name__ == "__main__":
    main()
