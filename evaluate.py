"""
evaluate.py
Computes accuracy metrics on predictions from infer.py.

Usage:
    python evaluate.py --predictions_csv output/dest_predictions.csv --task dest
    python evaluate.py --predictions_csv output/entry_time_predictions.csv \
        --task entry_time --time_tolerance_min 15
"""

import argparse
import re

import pandas as pd


def normalize(v):
    if v is None: return ""
    return re.sub(r"\s+", " ", str(v).strip().lower())


def to_minutes(dt_str):
    if dt_str is None: return None
    try:
        parsed = pd.to_datetime(str(dt_str).strip())
        return parsed.hour*60 + parsed.minute + parsed.second/60
    except Exception:
        return None


def time_diff(t1, t2):
    m1, m2 = to_minutes(t1), to_minutes(t2)
    if m1 is None or m2 is None: return None
    d = abs(m1-m2)
    return min(d, 1440-d)


def exact_match(df):
    valid = df[df["parse_error"].isna() | (df["parse_error"]=="")]
    if len(valid)==0:
        return {"n_total":len(df),"n_parsed":0,"exact_match_acc":None,
                "parse_failure_rate":1.0}
    matches = valid.apply(
        lambda r: normalize(r["prediction"])==normalize(r["ground_truth"]), axis=1
    )
    return {"n_total":len(df), "n_parsed":len(valid),
            "exact_match_acc": round(matches.mean(),4),
            "parse_failure_rate": round(1-len(valid)/len(df),4)}


def time_tolerance(df, tol):
    valid = df[df["parse_error"].isna() | (df["parse_error"]=="")].copy()
    if len(valid)==0:
        return {"n_total":len(df),"n_parsed":0,"within_tolerance_acc":None}
    diffs = valid.apply(lambda r: time_diff(r["prediction"],r["ground_truth"]), axis=1)
    vd    = diffs.dropna()
    if len(vd)==0:
        return {"n_total":len(df),"n_parsed":len(valid),
                "within_tolerance_acc":None,
                "note":"Could not parse times"}
    return {"n_total":len(df), "n_parsed":len(valid),
            "n_time_parsed":len(vd),
            "within_tolerance_acc": round((vd<=tol).mean(),4),
            "mean_abs_diff_min":    round(float(vd.mean()),2),
            "median_abs_diff_min":  round(float(vd.median()),2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--task", default="dest",
                   choices=["ori","dest","entry_time","all"])
    p.add_argument("--time_tolerance_min", default=15, type=int)
    args = p.parse_args()

    df = pd.read_csv(args.predictions_csv)
    print(f"\nLoaded {len(df)} predictions from {args.predictions_csv}\n")

    print("=== Exact-match metrics ===")
    for k, v in exact_match(df).items():
        print(f"  {k}: {v}")

    if args.task in ("entry_time","all"):
        print(f"\n=== Time-tolerance metrics (+/- {args.time_tolerance_min} min) ===")
        for k, v in time_tolerance(df, args.time_tolerance_min).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
