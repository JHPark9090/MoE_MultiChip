#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/hetero_v5_classical_%j.out
#SBATCH --error=logs/hetero_v5_classical_%j.err
#SBATCH --job-name=hetero_v5cl

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

N_CLUSTERS=${N_CLUSTERS:-3}

echo "============================================"
echo "v5 Classical 4-expert (adhd_3)"
echo "============================================"
python analyze_heterogeneity.py \
    --checkpoint=checkpoints/CircuitMoE_classical_adhd_3_49777876.pt \
    --output-dir=analysis/heterogeneity_v5_classical_adhd_3 \
    --n-clusters=$N_CLUSTERS

echo ""
echo "============================================"
echo "v5 Classical 2-expert (adhd_2)"
echo "============================================"
python analyze_heterogeneity.py \
    --checkpoint=checkpoints/CircuitMoE_classical_adhd_2_49777878.pt \
    --output-dir=analysis/heterogeneity_v5_classical_adhd_2 \
    --n-clusters=$N_CLUSTERS

echo ""
echo "============================================"
echo "All v5 classical heterogeneity analyses complete!"
echo "============================================"
