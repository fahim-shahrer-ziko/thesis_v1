"""
build_dataset.py

Joins trip_data.csv + persona_data.csv on Card_Id, builds sliding-window
(history, context, target) samples for each task, assigns one randomly
chosen instruction style per passenger, and saves:

    data4FT/{task}_train.json
    data4FT/{task}_val.json
    data4FT/{task}_test.json
    data4FT/passenger_split.json

Run AFTER generate_instructions.py.

Usage:
    python build_dataset.py              # all 3 tasks
    python build_dataset.py --task dest  # one task only
"""

import argparse
import json
import os
import random

import pandas as pd

import config
from data_utils import (stay_tuple_full, stay_tuple_target,
                        persona_tuple, get_exit_station_column)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data():
    trip_df    = pd.read_csv(config.TRIP_CSV_PATH)
    persona_df = pd.read_csv(config.PERSONA_CSV_PATH)

    exit_col = get_exit_station_column(trip_df)
    if exit_col != config.COL_EXIT_STATION:
        trip_df = trip_df.rename(columns={exit_col: config.COL_EXIT_STATION})

    trip_df[config.COL_ENTRY_DT] = pd.to_datetime(trip_df[config.COL_ENTRY_DT])
    trip_df[config.COL_EXIT_DT]  = pd.to_datetime(trip_df[config.COL_EXIT_DT])
    return trip_df, persona_df


def join_and_filter(trip_df, persona_df):
    for df, name in [(trip_df, "trip_data.csv"), (persona_df, "persona_data.csv")]:
        if config.COL_PASSENGER_ID not in df.columns:
            raise KeyError(
                f"{name}: column '{config.COL_PASSENGER_ID}' not found. "
                f"Columns present: {list(df.columns)}"
            )

    persona_df = persona_df.set_index(config.COL_PASSENGER_ID)
    counts     = trip_df.groupby(config.COL_PASSENGER_ID).size()
    valid      = [p for p in counts[counts >= config.MIN_TRIPS_PER_PASSENGER].index
                  if p in persona_df.index]

    if config.MAX_PASSENGERS and len(valid) > config.MAX_PASSENGERS:
        rng   = random.Random(config.RANDOM_SEED)
        valid = rng.sample(valid, config.MAX_PASSENGERS)

    trip_df    = trip_df[trip_df[config.COL_PASSENGER_ID].isin(valid)]
    persona_df = persona_df.loc[valid]
    print(f"Passengers after filtering (join key='{config.COL_PASSENGER_ID}', "
          f"min trips={config.MIN_TRIPS_PER_PASSENGER}): {len(valid)}")
    return trip_df, persona_df


def split_passengers(pids):
    rng  = random.Random(config.RANDOM_SEED)
    pids = list(pids); rng.shuffle(pids)
    n    = len(pids)
    n_tr = int(n * config.TRAIN_FRAC)
    n_va = int(n * config.VAL_FRAC)
    return pids[:n_tr], pids[n_tr:n_tr+n_va], pids[n_tr+n_va:]


# ---------------------------------------------------------------------------
# Sample formatting
# ---------------------------------------------------------------------------

def format_sample(instruction, persona, history, context, target, label, pred_field):
    full_instr = (instruction + "\n"
                  + f'Please organize your answer in a JSON object containing '
                    f'the following key: "prediction" ({pred_field}).')
    inp = (f"<persona>: {persona} \n"
           f"<history>: {history} \n"
           f"<context>: {context} \n"
           f"<target_stay>: {target} \n")
    out = 'Pred results: {\n' + f'"prediction": "{label}"' + '\n}'
    return {"instruction": full_instr, "input": inp, "output": out}


# ---------------------------------------------------------------------------
# Per-passenger sliding window
# ---------------------------------------------------------------------------

def pid_int(persona_row):
    try:    return int(persona_row.name)
    except: return 0


def build_windows(stays, persona_row, task, instr, placeholder, pred_field):
    ptuple = persona_tuple(persona_row)
    n      = len(stays)
    samples= []

    candidates = list(range(config.NUM_CONTEXT_STAY, n))
    if not candidates:
        return samples

    rng = random.Random(hash((pid_int(persona_row), task)) & 0xFFFFFFFF)
    rng.shuffle(candidates)

    for ti in candidates[:config.MAX_WINDOWS_PER_PASSENGER]:
        c_start  = max(0, ti - config.NUM_CONTEXT_STAY)
        h_start  = max(0, c_start - config.MAX_HISTORY_STAY)

        history  = [stay_tuple_full(r) for _, r in stays.iloc[h_start:c_start].iterrows()]
        context  = [stay_tuple_full(r) for _, r in stays.iloc[c_start:ti].iterrows()]
        target, label = stay_tuple_target(stays.iloc[ti], task, placeholder)

        samples.append(format_sample(instr, ptuple, history, context,
                                     target, label, pred_field))
    return samples


# ---------------------------------------------------------------------------
# Instruction loading
# ---------------------------------------------------------------------------

def load_instructions(task):
    d = config.task_instruction_dir(task)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"{d}/ not found. Run `python generate_instructions.py --task {task}` first."
        )
    files = sorted(f for f in os.listdir(d)
                   if f.startswith(f"{task}_instruction_") and f.endswith(".txt"))
    if not files:
        raise ValueError(
            f"No instruction files in {d}/. "
            f"Run `python generate_instructions.py --task {task} --force`."
        )
    styles = []
    for fn in files:
        with open(os.path.join(d, fn)) as f:
            text = f.read().strip()
        lines = text.split("\n", 1)
        body  = lines[1].strip() if lines[0].lower().startswith("task title:") else text
        styles.append(body)
    return styles


# ---------------------------------------------------------------------------
# Build one split
# ---------------------------------------------------------------------------

def build_split(task, pids, trip_df, persona_df, styles, split_name):
    meta      = config.TASK_META[task]
    placeholder = meta["placeholder"]
    pred_field  = meta["pred_field"]

    rng       = random.Random(config.RANDOM_SEED + hash((task, split_name)) % 100000)
    style_map = {p: rng.choice(styles) for p in pids}

    grouped  = trip_df.groupby(config.COL_PASSENGER_ID)
    samples  = []

    for pid in pids:
        stays = grouped.get_group(pid)\
                       .sort_values(config.COL_ENTRY_DT)\
                       .reset_index(drop=True)
        prow  = persona_df.loc[pid].rename(pid)
        wins  = build_windows(stays, prow, task,
                              style_map[pid], placeholder, pred_field)
        samples.extend(wins)

    print(f"  [{task}/{split_name}] pax={len(pids)}  samples={len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build instruction-tuning dataset")
    parser.add_argument("--task", default="all",
                        choices=["ori","dest","entry_time","all"])
    args = parser.parse_args()
    tasks = config.TASKS if args.task == "all" else [args.task]

    random.seed(config.RANDOM_SEED)
    os.makedirs(config.DATA4FT_DIR, exist_ok=True)

    print("Loading CSVs ...")
    trip_df, persona_df = load_raw_data()
    trip_df, persona_df = join_and_filter(trip_df, persona_df)

    train_p, val_p, test_p = split_passengers(persona_df.index)
    print(f"Split → train:{len(train_p)}  val:{len(val_p)}  test:{len(test_p)}")

    with open(config.SPLIT_MANIFEST_PATH, "w") as f:
        json.dump({"train": [str(p) for p in train_p],
                   "val":   [str(p) for p in val_p],
                   "test":  [str(p) for p in test_p]}, f, indent=2)

    for task in tasks:
        print(f"\nBuilding task: {task}")
        styles = load_instructions(task)
        for split_name, pids in [("train",train_p),("val",val_p),("test",test_p)]:
            samples = build_split(task, pids, trip_df, persona_df, styles, split_name)
            with open(config.task_json_path(task, split_name), "w") as f:
                json.dump(samples, f, indent=2)
            print(f"    Saved {config.task_json_path(task, split_name)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
