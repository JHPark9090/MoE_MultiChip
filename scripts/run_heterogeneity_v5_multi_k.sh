#!/bin/bash
#SBATCH --account=m4807
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/hetero_v5_multi_k_%j.out
#SBATCH --error=logs/hetero_v5_multi_k_%j.err
#SBATCH --job-name=hetero_multi_k

set -e
export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip
mkdir -p logs

# Checkpoints
CL3="checkpoints/CircuitMoE_classical_adhd_3_49777876.pt"
CL2="checkpoints/CircuitMoE_classical_adhd_2_49777878.pt"
QM3="checkpoints/CircuitMoE_quantum_adhd_3_49777879.pt"
QM2="checkpoints/CircuitMoE_quantum_adhd_2_49777880.pt"

for K in 2 4 5; do
    echo ""
    echo "================================================================"
    echo "  K=${K} — All 4 models"
    echo "================================================================"

    echo "--- Classical 4-expert (adhd_3), K=${K} ---"
    python analyze_heterogeneity.py \
        --checkpoint=$CL3 \
        --output-dir=analysis/heterogeneity_v5_classical_adhd_3_k${K} \
        --n-clusters=$K

    echo "--- Classical 2-expert (adhd_2), K=${K} ---"
    python analyze_heterogeneity.py \
        --checkpoint=$CL2 \
        --output-dir=analysis/heterogeneity_v5_classical_adhd_2_k${K} \
        --n-clusters=$K

    echo "--- Quantum 4-expert (adhd_3), K=${K} ---"
    python analyze_heterogeneity.py \
        --checkpoint=$QM3 \
        --output-dir=analysis/heterogeneity_v5_quantum_8q_d3_adhd_3_k${K} \
        --n-clusters=$K

    echo "--- Quantum 2-expert (adhd_2), K=${K} ---"
    python analyze_heterogeneity.py \
        --checkpoint=$QM2 \
        --output-dir=analysis/heterogeneity_v5_quantum_8q_d3_adhd_2_k${K} \
        --n-clusters=$K

    echo "  K=${K} complete."
done

echo ""
echo "================================================================"
echo "All heterogeneity analyses complete (K=2,4,5 × 4 models = 12 runs)"
echo "================================================================"
echo "Output directories:"
for d in analysis/heterogeneity_v5_*_k*; do
    if [ -d "$d" ]; then
        echo "  $d"
    fi
done
