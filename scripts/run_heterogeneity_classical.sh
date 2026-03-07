#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/heterogeneity_classical_%j.out
#SBATCH --error=logs/heterogeneity_classical_%j.err
#SBATCH --job-name=hetero_cl

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

N_CLUSTERS=${N_CLUSTERS:-3}

echo "============================================"
echo "Classical 4-expert (adhd_3)"
echo "============================================"
python analyze_heterogeneity.py \
    --checkpoint=checkpoints/CircuitMoE_classical_adhd_3_49731003.pt \
    --output-dir=analysis/heterogeneity_classical_adhd_3 \
    --n-clusters=$N_CLUSTERS

echo ""
echo "============================================"
echo "Classical 2-expert (adhd_2)"
echo "============================================"
python analyze_heterogeneity.py \
    --checkpoint=checkpoints/CircuitMoE_classical_adhd_2_49731010.pt \
    --output-dir=analysis/heterogeneity_classical_adhd_2 \
    --n-clusters=$N_CLUSTERS

echo ""
echo "============================================"
echo "All classical heterogeneity analyses complete!"
echo "============================================"
