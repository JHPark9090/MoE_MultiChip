#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/CMv5_Q10d2_4e_ADHD_%j.out
#SBATCH --error=logs/CMv5_Q10d2_4e_ADHD_%j.err
#SBATCH --job-name=CMv5_Q10d2_4

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

# v5: Corrected Yeo17 mapping, 10-qubit, degree=2, 4-expert
python CircuitMoE_v2_ABCD.py \
    --model-type=quantum \
    --circuit-config=adhd_3 \
    --task-type=binary \
    --target-phenotype=ADHD_label \
    --expert-hidden-dim=64 \
    --n-qubits=10 \
    --n-ansatz-layers=2 \
    --degree=2 \
    --pool-factor=10 \
    --num-classes=2 \
    --dropout=0.2 \
    --gating-noise-std=0.1 \
    --balance-loss-alpha=0.1 \
    --n-epochs=100 \
    --batch-size=32 \
    --lr=1e-3 \
    --wd=1e-5 \
    --patience=20 \
    --grad-clip=1.0 \
    --lr-scheduler=cosine \
    --sample-size=0 \
    --seed=2025 \
    --job-id="${SLURM_JOB_ID:-CMv5_Q10d2_4e_ADHD_local}" \
    --base-path=./checkpoints
