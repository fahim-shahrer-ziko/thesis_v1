"""
generate_sample_data.py

Creates small SYNTHETIC sample CSVs that match the real schema described by the user:

trip_data.csv (one row per trip):
    Card_Id, Gender, Age_On_June_2024, Entry_DT, Exit_DT, DayName,
    Entry_Station_Name, Exit_Station_Name

persona_data.csv (one row per passenger):
    Card_Id, Total_Trips, Active_Days, Unique_Origin, Unique_Dests,
    Avg_TT, Peak_Trips, Weekend_Trips

This is ONLY for local testing of the pipeline (config.py, build_dataset.py,
train_lora.py, evaluate.py, infer.py) before you point the pipeline at your
real ~500k+ passenger smart-card data. Replace dataset/trip_data.csv and
dataset/persona_data.csv with your real files (same column names) to run
on real data.
"""

import os
import random
from datetime import datetime, timedelta

import pandas as pd

random.seed(42)

OUT_DIR = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)

STATIONS = [
    "Uttara North", "Uttara Center", "Uttara South", "Pallabi", "Mirpur 11",
    "Mirpur 10", "Kazipara", "Shewrapara", "Agargaon", "Bijoy Sarani",
    "Farmgate", "Karwan Bazar", "Shahbagh", "Dhaka University", "Bangladesh Secretariat",
    "Motijheel",
]

DAYNAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

N_PASSENGERS = 60          # small number for a fast local test run
TRIPS_PER_PASSENGER = (15, 40)  # min, max trips per passenger
DATE_START = datetime(2024, 6, 1)
DATE_END = datetime(2024, 8, 31)


def random_datetime_between(start, end):
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def make_trip_row(passenger_id, gender, age, home_station, work_station):
    """Generate one synthetic trip with a commuter-like pattern:
    most trips are home<->work, with some noise."""
    entry_dt = random_datetime_between(DATE_START, DATE_END)
    dayname = DAYNAMES[entry_dt.weekday()]

    is_weekend = dayname in ("Saturday", "Sunday")
    if not is_weekend and random.random() < 0.75:
        # commuter pattern: morning home->work, evening work->home
        if entry_dt.hour < 14:
            entry_dt = entry_dt.replace(hour=random.randint(7, 9), minute=random.randint(0, 59))
            origin, dest = home_station, work_station
        else:
            entry_dt = entry_dt.replace(hour=random.randint(17, 19), minute=random.randint(0, 59))
            origin, dest = work_station, home_station
    else:
        # noisy / weekend / leisure trip
        origin, dest = random.sample(STATIONS, 2)
        entry_dt = entry_dt.replace(hour=random.randint(9, 21), minute=random.randint(0, 59))

    travel_minutes = random.randint(15, 45)
    exit_dt = entry_dt + timedelta(minutes=travel_minutes)

    return {
        "Card_Id": passenger_id,
        "Gender": gender,
        "Age_On_June_2024": age,
        "Entry_DT": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Exit_DT": exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "DayName": DAYNAMES[entry_dt.weekday()],
        "Entry_Station_Name": origin,
        "Exit_Station_Name": dest,
    }


def main():
    trip_rows = []
    persona_rows = []

    for pid in range(1, N_PASSENGERS + 1):
        gender = random.choice(["Male", "Female"])
        age = random.randint(18, 65)
        home_station, work_station = random.sample(STATIONS, 2)

        n_trips = random.randint(*TRIPS_PER_PASSENGER)
        passenger_trips = [
            make_trip_row(pid, gender, age, home_station, work_station)
            for _ in range(n_trips)
        ]
        trip_rows.extend(passenger_trips)

        df_p = pd.DataFrame(passenger_trips)
        df_p["Entry_DT_parsed"] = pd.to_datetime(df_p["Entry_DT"])
        df_p["Exit_DT_parsed"] = pd.to_datetime(df_p["Exit_DT"])
        travel_time_min = (df_p["Exit_DT_parsed"] - df_p["Entry_DT_parsed"]).dt.total_seconds() / 60.0

        persona_rows.append({
            "Card_Id": pid,
            "Total_Trips": len(df_p),
            "Active_Days": df_p["Entry_DT_parsed"].dt.date.nunique(),
            "Unique_Origin": df_p["Entry_Station_Name"].nunique(),
            "Unique_Dests": df_p["Exit_Station_Name"].nunique(),
            "Avg_TT": round(travel_time_min.mean(), 1),
            "Peak_Trips": int(((df_p["Entry_DT_parsed"].dt.hour >= 7) & (df_p["Entry_DT_parsed"].dt.hour <= 9) |
                                (df_p["Entry_DT_parsed"].dt.hour >= 17) & (df_p["Entry_DT_parsed"].dt.hour <= 19)).sum()),
            "Weekend_Trips": int(df_p["DayName"].isin(["Saturday", "Sunday"]).sum()),
        })

    trip_df = pd.DataFrame(trip_rows)
    persona_df = pd.DataFrame(persona_rows)

    trip_path = os.path.join(OUT_DIR, "trip_data.csv")
    persona_path = os.path.join(OUT_DIR, "persona_data.csv")
    trip_df.to_csv(trip_path, index=False)
    persona_df.to_csv(persona_path, index=False)

    print(f"Wrote {len(trip_df)} trip rows for {N_PASSENGERS} passengers -> {trip_path}")
    print(f"Wrote {len(persona_df)} persona rows -> {persona_path}")


if __name__ == "__main__":
    main()
