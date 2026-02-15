#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/fc_clustering_%j.out
#SBATCH --error=logs/fc_clustering_%j.err
#SBATCH --job-name=FC_cluster

# Prevent ~/.local packages from shadowing conda env
export PYTHONNOUSERSITE=1

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

python abcd_fc_clustering.py \
    --data-root=/pscratch/sd/j/junghoon/ABCD \
    --output-dir=./results/fc_clustering \
    --seed=2025 \
    --max-k=8 \
    --variance-threshold=0.95 \
    --use-fisher-z
