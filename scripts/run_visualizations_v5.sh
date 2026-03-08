#!/bin/bash
#SBATCH --account=m4807
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=viz_v5
#SBATCH --output=logs/viz_v5_%j.out
#SBATCH --error=logs/viz_v5_%j.err

set -e
export PYTHONNOUSERSITE=1

cd /pscratch/sd/j/junghoon/MoE_MultiChip
mkdir -p logs

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

echo "=== Generating v5 Interpretability Visualizations ==="

python visualize_interpretability.py --config adhd_3 --model-type classical --version v5
python visualize_interpretability.py --config adhd_2 --model-type classical --version v5
python visualize_interpretability.py --config adhd_3 --model-type quantum --version v5
python visualize_interpretability.py --config adhd_2 --model-type quantum --version v5

echo ""
echo "=== Generating v5 Heterogeneity Visualizations ==="

python visualize_heterogeneity.py --config adhd_3 --model-type classical --version v5
python visualize_heterogeneity.py --config adhd_2 --model-type classical --version v5
python visualize_heterogeneity.py --config adhd_3 --model-type quantum --version v5
python visualize_heterogeneity.py --config adhd_2 --model-type quantum --version v5

echo ""
echo "=== All visualizations complete ==="
echo "Output directories:"
for d in analysis/v5_*/figures analysis/heterogeneity_v5_*/figures; do
    if [ -d "$d" ]; then
        echo "  $d: $(ls $d | wc -l) files"
    fi
done
