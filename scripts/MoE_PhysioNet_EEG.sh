#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/MoE_EEG_%j.out
#SBATCH --error=logs/MoE_EEG_%j.err
#SBATCH --job-name=MoE_EEG

# Prevent ~/.local packages from shadowing conda env
export PYTHONNOUSERSITE=1

# Optional: uncomment if CUDA OOM occurs
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

python DistributedEEGMoE.py \
    --num-experts=4 \
    --expert-hidden-dim=64 \
    --expert-layers=2 \
    --nhead=4 \
    --halo-size=2 \
    --num-classes=2 \
    --dropout=0.1 \
    --gating-noise-std=0.1 \
    --balance-loss-alpha=0.01 \
    --n-epochs=100 \
    --batch-size=32 \
    --lr=1e-3 \
    --wd=1e-5 \
    --patience=20 \
    --grad-clip=1.0 \
    --lr-scheduler=cosine \
    --sampling-freq=16 \
    --sample-size=109 \
    --seed=2025 \
    --job-id="${SLURM_JOB_ID:-MoE_local}" \
    --base-path=./checkpoints
