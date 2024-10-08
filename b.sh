torchrun --nproc_per_node=1 train.py --logdir=logs/scene_real2 --config ./logs/scene_real2/config.yaml --wandb --show_pbar --debugckpt;
torchrun --nproc_per_node=1 train.py --logdir=logs/scene_bigbound --config ./logs/scene_bigbound/config.yaml --wandb --show_pbar --debugckpt;
torchrun --nproc_per_node=1 train.py --logdir=logs/scene_low_dictsize --config ./logs/scene_low_dictsize/config.yaml --wandb --show_pbar --debugckpt;

