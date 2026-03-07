#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/interpretability_analysis_%j.out
#SBATCH --error=logs/interpretability_analysis_%j.err
#SBATCH --job-name=interp

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

echo "============================================"
echo "Classical 4-expert (adhd_3)"
echo "============================================"
python analyze_circuit_moe.py \
    --checkpoint=checkpoints/CircuitMoE_classical_adhd_3_49731003.pt \
    --output-dir=analysis/classical_adhd_3

echo ""
echo "============================================"
echo "Classical 2-expert (adhd_2)"
echo "============================================"
python analyze_circuit_moe.py \
    --checkpoint=checkpoints/CircuitMoE_classical_adhd_2_49731010.pt \
    --output-dir=analysis/classical_adhd_2

echo ""
echo "============================================"
echo "Quantum v2 4-expert (adhd_3)"
echo "============================================"
python analyze_circuit_moe.py \
    --checkpoint=checkpoints/CircuitMoE_quantum_adhd_3_49767122.pt \
    --output-dir=analysis/quantum_v2_adhd_3

echo ""
echo "============================================"
echo "Quantum v2 2-expert (adhd_2)"
echo "============================================"
python analyze_circuit_moe.py \
    --checkpoint=checkpoints/CircuitMoE_quantum_adhd_2_49767123.pt \
    --output-dir=analysis/quantum_v2_adhd_2

echo ""
echo "============================================"
echo "All analyses complete!"
echo "============================================"
