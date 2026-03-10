#!/bin/bash
#SBATCH --account=m4807
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/viz_expert_saliency_%j.out
#SBATCH --error=logs/viz_expert_saliency_%j.err
#SBATCH --job-name=viz_sal

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/MoE_MultiChip
mkdir -p logs

echo "================================================================"
echo "  Visualizing Per-Expert Saliency — All 4 Models + Cross-Model"
echo "================================================================"

python visualize_expert_saliency.py --all

echo ""
echo "================================================================"
echo "  Visualization complete."
echo "================================================================"
echo "Output directories:"
for d in analysis/expert_saliency_v5_*/figures; do
    if [ -d "$d" ]; then
        echo "  $d ($(ls "$d"/*.png 2>/dev/null | wc -l) figures)"
    fi
done
echo "  analysis/expert_saliency_cross_model"
