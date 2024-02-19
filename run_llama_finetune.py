import subprocess
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Script for fine-tuning LLaMA model")
    parser.add_argument("--output_dir", type=str,
                        default="./one_iter_data/workdirs_alpaca_lora_llama_finetune", help="Output directory")
    parser.add_argument("--base_model", type=str,
                        default="/mnt/share/share_data/PAT/datasets/llama_1B_oneiter/")
    args = parser.parse_args()

    current_working_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = 'alpaca_data.json'
    batch_size = 4
    micro_batch_size = 4
    cutoff_len = 64

    llama_finetune_command = [
        "python", os.path.join(current_working_directory, "finetune.py"),
        "--base_model", args.base_model,
        "--data_path", os.path.join(current_working_directory, data_path),
        "--batch_size", str(batch_size),
        "--micro_batch_size", str(micro_batch_size),
        "--cutoff_len", str(cutoff_len),
        "--output_dir", args.output_dir,
    ]

    subprocess.run(llama_finetune_command)


if __name__ == "__main__":
    main()