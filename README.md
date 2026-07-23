# MRT Passenger Prediction -- Instruction Tuning

End-to-end instruction-tuning pipeline for predicting a metro passenger's
next origin station, exit station, and entry time. Fine-tunes
meta-llama/Meta-Llama-3.1-8B-Instruct with QLoRA -- one separate adapter
per task, 3 models total.

## Project structure

```
dataset/
    trip_data.csv               your real trip CSV (Card_Id join key)
    persona_data.csv            your real persona CSV (Card_Id join key)

prompt_store/
    gen/
        gen_pred_ori_instruct.txt       YOU author this (gen rules for ori)
        gen_pred_dest_instruct.txt      YOU author this (gen rules for dest)
        gen_pred_entry_time_instruct.txt  YOU author this (gen rules for entry_time)
    seed_task/
        seed_prompt_ori.txt             YOU author this (example instruction for ori)
        seed_prompt_dest.txt            YOU author this (example instruction for dest)
        seed_prompt_entry_time.txt      YOU author this (example instruction for entry_time)
    instruction/                        generate_instructions.py writes here
        ori/
            ori_instruction_01.txt ... ori_instruction_NN.txt
            ori_instructions_raw.txt
        dest/ ... entry_time/ (same pattern)
    instruction_EXAMPLE_ONLY/           IGNORE -- placeholder files used to build
                                        the worked-example data4FT/ in this delivery

data4FT/
    ori_train.json / ori_val.json / ori_test.json
    dest_train.json / dest_val.json / dest_test.json
    entry_time_train.json / entry_time_val.json / entry_time_test.json
    passenger_split.json                Card_Id split manifest

config.py                   all paths, column names, hyperparameters
data_utils.py               stay-tuple + persona formatting
generate_instructions.py    gen_rules + seed_prompt -> OpenAI -> N instruction files
build_dataset.py            CSV + instructions -> per-task train/val/test JSON
lora_train_lib.py           shared QLoRA training logic (used by all 3 scripts below)
train_lora_ori.py           train the origin-prediction adapter
train_lora_dest.py          train the destination-prediction adapter
train_lora_entry_time.py    train the entry-time-prediction adapter
infer.py                    run inference with a trained adapter
evaluate.py                 accuracy + time-tolerance metrics
check_data.py               validate CSVs before running the pipeline
run_pipeline.sh             orchestrate every stage end-to-end
scripts/generate_sample_data.py   make a 60-passenger synthetic dataset for testing
```

## CSV requirements

Both CSVs must share a common join key column: Card_Id.

Trip CSV (trip_data.csv):
    Card_Id, Gender, Age_On_June_2024, Entry_DT, Exit_DT, DayName,
    Entry_Station_Name, Exit_Station_Name
    (also accepts typo: Exit_Satation_Name -- auto-renamed)

Persona CSV (persona_data.csv):
    Card_Id, Total_Trips, Active_Days, Unique_Origin, Unique_Dests,
    Avg_TT, Peak_Trips, Weekend_Trips

Run python check_data.py to validate your CSVs before anything else.

## Prompt/data format

All three tasks use the same rich 7-element stay tuple:
    (Gender, Age_On_June_2024, Entry_DT, Exit_DT, DayName,
     Entry_Station_Name, Exit_Station_Name)

Plus a separate persona block (always present):
    (Total_Trips, Active_Days, Unique_Origin, Unique_Dests,
     Avg_TT, Peak_Trips, Weekend_Trips)

target_stay masks exactly one field per task:
    ori:        Entry_Station_Name  -> <next_station_name>
    dest:       Exit_Station_Name   -> <next_station_name>
    entry_time: Entry_DT            -> <next_entry_time>

Training output format (same for all 3):
    Pred results: {
    "prediction": "<value>"
    }

## How the prompt_store works

You author TWO plain-text files per task by hand:

1. prompt_store/gen/gen_pred_{task}_instruct.txt
   The self-instruct generation guideline: rules for the LLM to follow
   when generating diverse instructions, ending with a line like:
       "List of 10 tasks:"
   generate_instructions.py parses the number N from this line and
   generates exactly N instruction files. If no such line is found, it
   defaults to 10 and prints a warning.

2. prompt_store/seed_task/seed_prompt_{task}.txt
   The worked example instruction shown to the LLM as a reference -- the
   one complete instruction it should treat as the template for the N
   diverse ones it produces.

generate_instructions.py concatenates these two files into one prompt,
calls OpenAI, splits the response on lines containing only "###", and
saves each parsed instruction as:
    prompt_store/instruction/{task}/{task}_instruction_01.txt
    prompt_store/instruction/{task}/{task}_instruction_02.txt
    ... up to N files ...
    prompt_store/instruction/{task}/{task}_instructions_raw.txt  (raw LLM response)

Generation is cached: re-running generate_instructions.py skips tasks that
already have N instruction files, unless you pass --force.

## How to run

### Option A: one command
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# after you have authored your prompt_store/gen/ and prompt_store/seed_task/ files
./run_pipeline.sh
```

Flags:
    --use-sample-data       use a 60-passenger synthetic dataset instead of your real CSVs
    --skip-generate         skip OpenAI call, reuse existing instruction files
    --skip-train            skip training, just run inference + eval
    --task dest             only process one task (ori, dest, or entry_time)

### Option B: step by step

0. (Optional) Test with synthetic data:
   python scripts/generate_sample_data.py

1. Validate your CSVs:
   python check_data.py

2. Author your prompt_store files by hand (prompt_store/gen/ and prompt_store/seed_task/).
   See the "How the prompt_store works" section above.

3. Generate diverse instructions via OpenAI:
   export OPENAI_API_KEY=sk-...
   python generate_instructions.py
   python generate_instructions.py --task dest   # just one task
   python generate_instructions.py --force       # regenerate even if cached

4. Build the instruction-tuning datasets (saved to data4FT/):
   python build_dataset.py
   python build_dataset.py --task dest   # just one task

5. Train (one command per task, runs independently):
   python train_lora_dest.py
   python train_lora_ori.py
   python train_lora_entry_time.py
   # all accept: --no_qlora, --epochs N, --batch_size N, --lr N

   Adapters save to:
   modelSave/llama3_1_mrt_qlora_dest/final_adapter/
   modelSave/llama3_1_mrt_qlora_ori/final_adapter/
   modelSave/llama3_1_mrt_qlora_entry_time/final_adapter/

   GPU required: ~6-10GB VRAM for 4-bit QLoRA on an 8B model.
   Adjust PER_DEVICE_TRAIN_BATCH_SIZE in config.py to fit your hardware.

6. Inference:
   python infer.py --task dest --use_4bit
   python infer.py --task ori --use_4bit
   python infer.py --task entry_time --use_4bit
   python infer.py --task dest --smoke_test --use_4bit   # single prompt test

   Predictions saved to output/{task}_predictions.csv

7. Evaluate:
   python evaluate.py --predictions_csv output/dest_predictions.csv --task dest
   python evaluate.py --predictions_csv output/entry_time_predictions.csv \
       --task entry_time --time_tolerance_min 15
