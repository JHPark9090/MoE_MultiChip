#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/SingleExpert_Quantum_ASD_%j.out
#SBATCH --error=logs/SingleExpert_Quantum_ASD_%j.err
#SBATCH --job-name=SE_Q_ASD

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

python SingleExpertBaseline_ABCD.py \
    --model-type=quantum \
    --task-type=binary \
    --target-phenotype=ASD_label \
    --expert-hidden-dim=64 \
    --n-qubits=8 \
    --n-ansatz-layers=2 \
    --degree=3 \
    --num-classes=2 \
    --dropout=0.2 \
    --n-epochs=100 \
    --batch-size=32 \
    --lr=1e-3 \
    --wd=1e-5 \
    --patience=20 \
    --grad-clip=1.0 \
    --lr-scheduler=cosine \
    --sample-size=0 \
    --seed=2025 \
    --job-id="${SLURM_JOB_ID:-SE_Q_ASD_local}" \
    --base-path=./checkpoints
