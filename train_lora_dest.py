"""
train_lora_dest.py
Fine-tune on the dest prediction task.
Run: python train_lora_dest.py
"""
import lora_train_lib as lib

TASK = "dest"

def main():
    parser = lib.build_parser(TASK)
    lib.run_training(TASK, parser.parse_args())

if __name__ == "__main__":
    main()
