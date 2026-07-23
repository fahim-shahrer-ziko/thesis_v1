"""
config.py
Single source of truth for all settings.

TO SWITCH MODELS: change BASE_MODEL_ID only.
Everything else (LoRA targets, chat template, pad token) is
detected automatically at runtime.

Tested with:
    Qwen/Qwen2.5-7B-Instruct       (no HF_TOKEN needed)
    Qwen/Qwen2.5-1.5B-Instruct     (no HF_TOKEN needed, fastest)
    mistralai/Mistral-7B-Instruct-v0.3  (no HF_TOKEN needed)
    meta-llama/Meta-Llama-3.1-8B-Instruct  (needs HF_TOKEN + license)
    meta-llama/Meta-Llama-3.2-3B-Instruct  (needs HF_TOKEN + license)
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))

PROMPT_STORE_DIR = os.path.join(PROJECT_ROOT, "prompt_store")
GEN_DIR          = os.path.join(PROMPT_STORE_DIR, "gen")
SEED_TASK_DIR    = os.path.join(PROMPT_STORE_DIR, "seed_task")
INSTRUCTION_DIR  = os.path.join(PROMPT_STORE_DIR, "instruction")

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Check whether we are running on Kaggle
if os.path.exists("/kaggle/input"):
    # Change this to your Kaggle dataset name
    DATASET_DIR = "/kaggle/input/datasets/fahimshahrerziko/dataset-v2"
else:
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

DATA4FT_DIR    = os.path.join(PROJECT_ROOT, "data4FT")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "modelSave")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "output")
LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")

TRIP_CSV_PATH    = os.path.join(DATASET_DIR, "trip_data.csv")
PERSONA_CSV_PATH = os.path.join(DATASET_DIR, "persona_data.csv")

# DATASET_DIR    = os.path.join(PROJECT_ROOT, "dataset")
# DATA4FT_DIR    = os.path.join(PROJECT_ROOT, "data4FT")
# MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "modelSave")
# OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "output")
# LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")

# TRIP_CSV_PATH    = os.path.join(DATASET_DIR, "trip_data.csv")
# PERSONA_CSV_PATH = os.path.join(DATASET_DIR, "persona_data.csv")

SPLIT_MANIFEST_PATH = os.path.join(DATA4FT_DIR, "passenger_split.json")

def task_json_path(task, split):
    return os.path.join(DATA4FT_DIR, f"{task}_{split}.json")

def gen_path(task):
    return os.path.join(GEN_DIR, f"gen_pred_{task}_instruct.txt")

def seed_prompt_path(task):
    return os.path.join(SEED_TASK_DIR, f"seed_prompt_{task}.txt")

def task_instruction_dir(task):
    return os.path.join(INSTRUCTION_DIR, task)

def train_output_dir(task):
    return os.path.join(MODEL_SAVE_DIR, f"mrt_lora_{task}")

# ---------------------------------------------------------------------------
# CSV column names
# ---------------------------------------------------------------------------
COL_PASSENGER_ID      = "Card_Id"
COL_GENDER            = "Gender"
COL_AGE               = "Age_On_June_2024"
COL_ENTRY_DT          = "Entry_DT"
COL_EXIT_DT           = "Exit_DT"
COL_DAYNAME           = "DayName"
COL_ENTRY_STATION     = "Entry_Station_Name"
COL_EXIT_STATION      = "Exit_Station_Name"
COL_EXIT_STATION_TYPO = "Exit_Satation_Name"

PERSONA_COLUMNS = [
    "Total_Trips","Active_Days","Unique_Origins",
    "Unique_Dests","Avg_TT","Peak_Trips","Weekend_Trips",
]
COL_TOTAL_TRIPS   = "Total_Trips"
COL_ACTIVE_DAYS   = "Active_Days"
COL_UNIQUE_ORIGIN = "Unique_Origins"
COL_UNIQUE_DESTS  = "Unique_Dests"
COL_AVG_TT        = "Avg_TT"
COL_PEAK_TRIPS    = "Peak_Trips"
COL_WEEKEND_TRIPS = "Weekend_Trips"

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
TASKS = ["ori", "dest", "entry_time"]

TASK_META = {
    "ori": {
        "pred_field" : "entry station",
        "placeholder": "<entry_station_name>",
        "masked_field": "Entry_Station_Name",
    },
    "dest": {
        "pred_field" : "exit station",
        "placeholder": "<next_station_name>",
        "masked_field": "Exit_Station_Name",
    },
    "entry_time": {
        "pred_field" : "entry time",
        "placeholder": "<next_entry_time>",
        "masked_field": "Entry_DT",
    },
}

# ---------------------------------------------------------------------------
# HuggingFace token
# Only needed for gated models (e.g. Llama).
# Set to None for open models (Qwen, Mistral).
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # e.g. "hf_xxxxxxxxxxxx"

# ---------------------------------------------------------------------------
# Instruction generation model
# This model generates the 10 diverse instruction variants from your
# seed prompts. Runs locally on GPU -- no API key needed.
# ---------------------------------------------------------------------------
GENERATION_MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
GENERATION_MAX_NEW_TOKENS = 8000
GENERATION_TEMPERATURE    = 0.9

# ---------------------------------------------------------------------------
# Fine-tuning model
# Change ONLY this line to switch the model being fine-tuned.
# LoRA target modules are detected automatically -- no other change needed.
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# LoRA hyperparameters -- same for all models
LORA_R       = 64
LORA_ALPHA   = 64
LORA_DROPOUT = 0.0

# LORA_TARGET_MODULES: set to None to auto-detect for the loaded model,
# or set explicitly if you want to override:
#   Llama / Mistral: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
#   Qwen:            ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
#   Falcon:          ["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"]
#   GPT-NeoX:        ["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"]
LORA_TARGET_MODULES = None   # None = auto-detect

USE_QLORA = True   # 4-bit QLoRA; set False if bitsandbytes unavailable

MAX_SEQ_LENGTH = 1024

# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
NUM_CONTEXT_STAY          = 5
MIN_TRIPS_PER_PASSENGER   = 8
MAX_HISTORY_STAY          = 40
MAX_WINDOWS_PER_PASSENGER = 5

MAX_PASSENGERS = None   # None = all passengers; set e.g. 20000 to subsample
RANDOM_SEED    = 111

TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
TEST_FRAC  = 0.1

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS            = 3
LEARNING_RATE               = 1e-4
LOGGING_STEPS               = 10
SAVE_STEPS                  = 100

# ---------------------------------------------------------------------------
# Inference / evaluation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 200
GEN_DO_SAMPLE  = False
