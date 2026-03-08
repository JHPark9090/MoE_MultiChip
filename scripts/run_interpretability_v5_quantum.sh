#!/bin/bash
#SBATCH --account=m4807
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/interp_v5_quantum_%j.out
#SBATCH --error=logs/interp_v5_quantum_%j.err
#SBATCH --job-name=interp_v5qm

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip

echo "============================================"
echo "v5 Quantum 8Q d3 4-expert (adhd_3)"
echo "============================================"
python analyze_circuit_moe.py \
    --checkpoint=checkpoints/CircuitMoE_quantum_adhd_3_49777879.pt \
    --output-dir=analysis/v5_quantum_8q_d3_adhd_3

echo ""
echo "============================================"
echo "v5 Quantum 8Q d3 2-expert (adhd_2)"
echo "============================================"
python analyze_circuit_moe.py \
    --checkpoint=checkpoints/CircuitMoE_quantum_adhd_2_49777880.pt \
    --output-dir=analysis/v5_quantum_8q_d3_adhd_2

echo ""
echo "============================================"
echo "All v5 quantum interpretability analyses complete!"
echo "============================================"
