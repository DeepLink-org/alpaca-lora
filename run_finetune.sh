WANDB_MODE=disabled srun -n1 -p pat_dev --gres=gpu:1 --tasks-per-node 1 --job-name=lora \
python finetune.py \
--base_model '/mnt/lustrenew/liuyipeng/Llama7B/mnt/lustre/share_data/diopi-sh/llama_hf/' \
--data_path 'alpaca_data.json' \
--output_dir './lora-alpaca' 
