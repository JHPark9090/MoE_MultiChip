#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/FluidInt_coherence_clustering_site_regressed_%j.out
#SBATCH --error=logs/FluidInt_coherence_clustering_site_regressed_%j.err
#SBATCH --job-name=FI_coh_clust

# Prevent ~/.local packages from shadowing conda env
export PYTHONNOUSERSITE=1

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

python abcd_fc_clustering.py \
    --feature-type=coherence \
    --data-root=/pscratch/sd/j/junghoon/ABCD \
    --output-dir=./results/fluidint_coherence_clustering_site_regressed \
    --target-phenotype=nihtbx_fluidcomp_uncorrected \
    --tr=0.8 \
    --freq-band 0.01 0.1 \
    --regress-site \
    --seed=2025 \
    --max-k=8 \
    --variance-threshold=0.95
