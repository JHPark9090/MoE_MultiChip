#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/interp_v5_classical_%j.out
#SBATCH --error=logs/interp_v5_classical_%j.err
#SBATCH --job-name=interp_v5cl

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

echo "============================================"
echo "v5 Classical 4-expert (adhd_3)"
echo "============================================"
python analyze_circuit_moe.py \
    --checkpoint=checkpoints/CircuitMoE_classical_adhd_3_49777876.pt \
    --output-dir=analysis/v5_classical_adhd_3

echo ""
echo "============================================"
echo "v5 Classical 2-expert (adhd_2)"
echo "============================================"
python analyze_circuit_moe.py \
    --checkpoint=checkpoints/CircuitMoE_classical_adhd_2_49777878.pt \
    --output-dir=analysis/v5_classical_adhd_2

echo ""
echo "============================================"
echo "All v5 classical interpretability analyses complete!"
echo "============================================"
