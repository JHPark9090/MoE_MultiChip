#!/bin/bash
#SBATCH --account=m4807
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/expert_saliency_v5_%j.out
#SBATCH --error=logs/expert_saliency_v5_%j.err
#SBATCH --job-name=expert_sal

set -e
export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip
mkdir -p logs

# v5 checkpoints (seed=2025)
CL3="checkpoints/CircuitMoE_classical_adhd_3_49777876.pt"
CL2="checkpoints/CircuitMoE_classical_adhd_2_49777878.pt"
QM3="checkpoints/CircuitMoE_quantum_adhd_3_49777879.pt"
QM2="checkpoints/CircuitMoE_quantum_adhd_2_49777880.pt"

echo "================================================================"
echo "  Per-Expert Saliency Analysis — v5 Models (seed=2025)"
echo "================================================================"

echo ""
echo "--- Classical 4-expert (adhd_3) ---"
python analyze_expert_saliency.py \
    --checkpoint=$CL3 \
    --output-dir=analysis/expert_saliency_v5_classical_adhd_3

echo ""
echo "--- Classical 2-expert (adhd_2) ---"
python analyze_expert_saliency.py \
    --checkpoint=$CL2 \
    --output-dir=analysis/expert_saliency_v5_classical_adhd_2

echo ""
echo "--- Quantum 4-expert (adhd_3) ---"
python analyze_expert_saliency.py \
    --checkpoint=$QM3 \
    --output-dir=analysis/expert_saliency_v5_quantum_8q_d3_adhd_3

echo ""
echo "--- Quantum 2-expert (adhd_2) ---"
python analyze_expert_saliency.py \
    --checkpoint=$QM2 \
    --output-dir=analysis/expert_saliency_v5_quantum_8q_d3_adhd_2

echo ""
echo "================================================================"
echo "  All 4 per-expert saliency analyses complete."
echo "================================================================"
echo "Output directories:"
for d in analysis/expert_saliency_v5_*; do
    if [ -d "$d" ]; then
        echo "  $d"
    fi
done
