#!/bin/bash
#SBATCH --account=m4807
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/interpretability_quantum_%j.out
#SBATCH --error=logs/interpretability_quantum_%j.err
#SBATCH --job-name=interp_qm

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

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
echo "All quantum interpretability analyses complete!"
echo "============================================"
