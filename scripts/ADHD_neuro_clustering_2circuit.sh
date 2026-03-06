#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/neuro_clustering_adhd2_%j.out
#SBATCH --error=logs/neuro_clustering_adhd2_%j.err
#SBATCH --job-name=NC_adhd2

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

python abcd_neuro_clustering.py \
    --circuit-config=adhd_2 \
    --output-dir=./results/neuro_clustering_adhd2 \
    --target-phenotype=ADHD_label \
    --regress-site \
    --use-fisher-z \
    --max-k=8 \
    --variance-threshold=0.95 \
    --seed=2025
