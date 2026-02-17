#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/ASD_ClusterMoE_Classical_Soft_%j.out
#SBATCH --error=logs/ASD_ClusterMoE_Classical_Soft_%j.err
#SBATCH --job-name=ASD_CMoE_CS

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

python ClusterInformedMoE_ABCD.py \
    --model-type=classical \
    --routing=soft \
    --target-phenotype=ASD_label \
    --cluster-file=results/asd_coherence_clustering_site_regressed/cluster_assignments.csv \
    --cluster-column=km_cluster \
    --num-experts=2 \
    --expert-hidden-dim=64 \
    --expert-layers=2 \
    --nhead=4 \
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
    --job-id="${SLURM_JOB_ID:-ASD_ClusterMoE_CS_local}" \
    --base-path=./checkpoints
