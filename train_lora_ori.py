"""
train_lora_ori.py
Fine-tune on the ori prediction task.
Run: python train_lora_ori.py
"""
import lora_train_lib as lib

TASK = "ori"

def main():
    parser = lib.build_parser(TASK)
    lib.run_training(TASK, parser.parse_args())

if __name__ == "__main__":
    main()
