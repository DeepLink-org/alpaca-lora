#! /usr/bin/sh
accelerate launch --config_file  ./deepspeed_llama.yaml --main_process_port 29512 \
    ./finetune.py \
    --batch_size 32 \
    --micro_batch_size 1 \
    --cutoff_len 256 \
    --output_dir './out' \
    --torch_dtype float16 \
    --base_model '/mnt/lustrenew/share_data/PAT/datasets/llama2/70B'
