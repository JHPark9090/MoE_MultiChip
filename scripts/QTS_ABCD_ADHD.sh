#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/QTS_ABCD_ADHD_%j.out
#SBATCH --error=logs/QTS_ABCD_ADHD_%j.err
#SBATCH --job-name=QTS_ABCD

# Prevent ~/.local packages from shadowing conda env
export PYTHONNOUSERSITE=1

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

python QTSTransformer_PhysioNet_EEG.py \
    --dataset=ABCD_fMRI \
    --parcel-type=HCP180 \
    --target-phenotype=ADHD_label \
    --n-qubits=8 \
    --n-layers=2 \
    --degree=3 \
    --dropout=0.1 \
    --n-epochs=100 \
    --batch-size=16 \
    --lr=1e-3 \
    --wd=1e-5 \
    --patience=20 \
    --sample-size=0 \
    --seed=2025 \
    --job-id="${SLURM_JOB_ID:-QTS_ABCD_local}" \
    --base-path=./checkpoints
