"""
data_utils.py
Formatting helpers shared across build_dataset, train, infer, evaluate.

All three tasks use the same 7-element stay tuple:
    (Gender, Age_On_June_2024, Entry_DT, Exit_DT, DayName,
     Entry_Station_Name, Exit_Station_Name)

Plus a separate <persona> block (always present):
    (Total_Trips, Active_Days, Unique_Origin, Unique_Dests,
     Avg_TT, Peak_Trips, Weekend_Trips)

target_stay masks exactly one field per task:
    ori        -> Entry_Station_Name  = <entry_station_name>
                  Exit_DT and Exit_Station_Name also set to None
    dest       -> Exit_Station_Name   = <next_station_name>
    entry_time -> Entry_DT            = <next_entry_time>
"""

import pandas as pd
import config


def format_dt(dt):
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def persona_tuple(persona_row):
    return (
        int(persona_row[config.COL_TOTAL_TRIPS]),
        int(persona_row[config.COL_ACTIVE_DAYS]),
        int(persona_row[config.COL_UNIQUE_ORIGIN]),
        int(persona_row[config.COL_UNIQUE_DESTS]),
        round(float(persona_row[config.COL_AVG_TT]), 1),
        int(persona_row[config.COL_PEAK_TRIPS]),
        int(persona_row[config.COL_WEEKEND_TRIPS]),
    )


def stay_tuple_full(row):
    """All 7 fields known — used for <history> and <context>."""
    return (
        row[config.COL_GENDER],
        int(row[config.COL_AGE]),
        format_dt(row[config.COL_ENTRY_DT]),
        format_dt(row[config.COL_EXIT_DT]),
        row[config.COL_DAYNAME],
        row[config.COL_ENTRY_STATION],
        row[config.COL_EXIT_STATION],
    )


def stay_tuple_target(row, task, placeholder):
    """
    Returns (target_tuple, ground_truth_label).

    ori:
        (Gender, Age, Entry_DT, None, DayName, <entry_station_name>, None)
        label = real Entry_Station_Name

    dest:
        (Gender, Age, Entry_DT, Exit_DT, DayName, Entry_Station, <next_station_name>)
        label = real Exit_Station_Name

    entry_time:
        (Gender, Age, <next_entry_time>, Exit_DT, DayName, Entry_Station, Exit_Station)
        label = real Entry_DT string
    """
    gender    = row[config.COL_GENDER]
    age       = int(row[config.COL_AGE])
    entry_dt  = format_dt(row[config.COL_ENTRY_DT])
    exit_dt   = format_dt(row[config.COL_EXIT_DT])
    dayname   = row[config.COL_DAYNAME]
    origin    = row[config.COL_ENTRY_STATION]
    dest      = row[config.COL_EXIT_STATION]

    if task == "ori":
        target = (gender, age, entry_dt, None, dayname, placeholder, None)
        label  = origin
    elif task == "dest":
        target = (gender, age, entry_dt, exit_dt, dayname, origin, placeholder)
        label  = dest
    elif task == "entry_time":
        target = (gender, age, placeholder, exit_dt, dayname, origin, dest)
        label  = entry_dt
    else:
        raise ValueError(f"Unknown task: {task}")

    return target, label


def get_exit_station_column(df):
    if config.COL_EXIT_STATION in df.columns:
        return config.COL_EXIT_STATION
    if config.COL_EXIT_STATION_TYPO in df.columns:
        return config.COL_EXIT_STATION_TYPO
    raise KeyError(
        f"Exit-station column not found. Expected '{config.COL_EXIT_STATION}' "
        f"or '{config.COL_EXIT_STATION_TYPO}'. Found: {list(df.columns)}"
    )
