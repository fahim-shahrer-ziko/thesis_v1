"""
train_lora_entry_time.py
Fine-tune on the entry_time prediction task.
Run: python train_lora_entry_time.py
"""
import lora_train_lib as lib

TASK = "entry_time"

def main():
    parser = lib.build_parser(TASK)
    lib.run_training(TASK, parser.parse_args())

if __name__ == "__main__":
    main()
