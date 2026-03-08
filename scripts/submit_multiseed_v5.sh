#!/bin/bash
# Submit all multi-seed v5 training runs
# Seeds 2024 and 2026 (seed 2025 already completed)
# 6 models × 2 seeds = 12 jobs
#
# Models:
#   1. Classical Single Expert
#   2. Quantum Single Expert
#   3. Classical Circuit MoE 2-expert
#   4. Classical Circuit MoE 4-expert
#   5. Quantum Circuit MoE 2-expert
#   6. Quantum Circuit MoE 4-expert

set -e
cd /pscratch/sd/j/junghoon/MoE_MultiChip
mkdir -p logs

echo "Submitting multi-seed v5 robustness runs..."
echo "============================================="

for SEED in 2024 2026; do
    echo ""
    echo "=== Seed ${SEED} ==="

    # 1. Classical Single Expert
    JID1=$(sbatch --parsable scripts/multiseed/SE_Classical_seed${SEED}.sh)
    echo "  Classical Single Expert:     ${JID1}"

    # 2. Quantum Single Expert
    JID2=$(sbatch --parsable scripts/multiseed/SE_Quantum_seed${SEED}.sh)
    echo "  Quantum Single Expert:       ${JID2}"

    # 3. Classical Circuit MoE 2-expert
    JID3=$(sbatch --parsable scripts/multiseed/CMv5_Classical_2e_seed${SEED}.sh)
    echo "  Classical MoE 2-expert:      ${JID3}"

    # 4. Classical Circuit MoE 4-expert
    JID4=$(sbatch --parsable scripts/multiseed/CMv5_Classical_4e_seed${SEED}.sh)
    echo "  Classical MoE 4-expert:      ${JID4}"

    # 5. Quantum Circuit MoE 2-expert
    JID5=$(sbatch --parsable scripts/multiseed/CMv5_Quantum_2e_seed${SEED}.sh)
    echo "  Quantum MoE 2-expert:        ${JID5}"

    # 6. Quantum Circuit MoE 4-expert
    JID6=$(sbatch --parsable scripts/multiseed/CMv5_Quantum_4e_seed${SEED}.sh)
    echo "  Quantum MoE 4-expert:        ${JID6}"
done

echo ""
echo "============================================="
echo "All 12 jobs submitted. Check with: squeue -u \$USER"
echo "============================================="
