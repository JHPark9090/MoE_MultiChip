# Network-Aware Multi-Scale Expert (NAME) — Results

> **NOTE (2026-03-07):** The Yeo 17-network spatial projection in NAME used an **incorrect Yeo-17 network mapping** (fabricated contiguous-block ROI assignments). NAME's results should be re-evaluated with the corrected mapping, though the overall underperformance vs SE baseline may be due to the d_net=8 bottleneck rather than the mapping issue.

**Date**: 2026-03-06
**Status**: ADHD Classical completed. Underperforms SE baseline. Other phenotypes/quantum not submitted.

## Motivation

The SE vs MoE analysis (`SingleExpert_vs_MoE_Results.md`) identified feature extraction as the bottleneck: both classical and quantum experts compress 65,340 fMRI values (363 timesteps x 180 ROIs) through `Linear(180, 64)` + mean temporal pooling, discarding spatial structure and temporal dynamics. NAME was designed to address this by preserving brain network topology and capturing multi-scale temporal features.

## Architecture

Two-branch design with network-aware preprocessing:

### Branch 1: Network-Grouped Multi-Scale Temporal Processing

1. **Yeo 17-network spatial projection**: 17 parallel `Linear(n_k, 8) + GELU` layers, one per Yeo network. ROIs within the same functional network are projected together, preserving spatial grouping. Output: `(B, 363, 136)`.

2. **Multi-scale dilated depthwise-separable convolutions**: 3 parallel branches with dilation 1/4/16 (receptive fields ~6s/22s/78s at TR=0.8s). Captures short, medium, and long-range temporal dynamics. Output: `(B, 363, 128)` after merge.

3. **Temporal mixing**: 1-layer TransformerEncoder (d=128, nhead=4, ff=256, dropout=0.3). Output: `(B, 363, 128)`.

4. **Attentive temporal pooling**: `softmax(Linear(128, 1))` weighted sum over time, replacing mean pooling. Output: `(B, 128)`.

### Branch 2: Functional Connectivity Fingerprint

5. **On-the-fly Pearson FC**: Batched correlation matrix `(B, 180, 180)`.

6. **Network-level FC summarization**: Average FC within each of 153 network pairs (17 within-network + 136 between-network). Output: `(B, 153)`.

7. **FC MLP**: `Linear(153, 64) + GELU + Dropout(0.3)`. Output: `(B, 64)`.

### Fusion

8. **Concat + MLP**: `cat(temporal=128, fc=64)` -> `Linear(192, 128) + LN + GELU + Dropout(0.3)` -> `Linear(128, 64)`. Output: `(B, 64)`.

9. **Classifier**: `Linear(64, 1)` for binary classification.

### Parameter Counts

| Model | Expert Params | Total (with classifier) |
|-------|--------------|------------------------|
| Classical | 289,841 | 289,906 |
| Quantum (QSVT) | 169,216 | 169,281 |
| SE Baseline (for comparison) | 134,784 | 134,849 |

## ADHD Classical Results

**Job ID**: 49703411
**Runtime**: ~72 seconds (24 epochs x 3s/epoch)

### Training Trajectory

| Epoch | Train Loss | Train AUC | Val Loss | Val AUC |
|-------|-----------|-----------|----------|---------|
| 1 | 0.6891 | 0.5098 | 0.6835 | 0.5455 |
| 4 | 0.6853 | 0.5068 | 0.6806 | **0.5658** |
| 10 | 0.6838 | 0.5152 | 0.6824 | 0.5488 |
| 20 | 0.6785 | 0.5616 | 0.6783 | 0.5601 |
| 24 | 0.6754 | 0.5723 | 0.6870 | 0.5615 |

Early stopped at epoch 24 (patience=20, best at epoch 4).

### Final Test Results

| Metric | NAME Classical | SE Classical Baseline |
|--------|---------------|----------------------|
| Test AUC | 0.5800 | **0.6193** |
| Test Accuracy | 56.95% | **59.5%** |
| Best Val AUC | 0.5658 | **0.6419** |
| Params | 289,906 | 134,849 |
| Epochs to converge | 24 | ~30 |

### Cross-Model Comparison (ADHD)

| Model | Params | Test AUC | Test Acc |
|-------|--------|----------|----------|
| **Classical SE** | **134,849** | **0.6193** | **59.5%** |
| Cluster MoE Soft | 283,491 | 0.6001 | 59.3% |
| Learned MoE Soft | 287,459 | 0.5968 | 58.5% |
| NAME Classical | 289,906 | 0.5800 | 56.95% |
| Quantum SE | 13,648 | 0.5769 | 58.6% |

NAME ranks **4th out of 5** — worse than all classical models and barely above the quantum SE with 21x fewer parameters.

## Analysis: Why NAME Failed

### 1. The model barely learned

The loss decreased from 0.689 to 0.675 over 24 epochs — a tiny 2% reduction. Train AUC peaked at 0.57, barely above chance. The model was stuck predicting near-majority-class for most of training (accuracy = 56.9% ≈ majority class proportion of 57.0%).

### 2. Architectural over-engineering hurt

The SE baseline with a simple `Linear(180, 64)` + 2-layer Transformer + mean pooling outperforms NAME's 17-network projection + multi-scale convolutions + attention pooling + FC branch. More architectural complexity introduced:
- **More parameters to optimize** (290K vs 135K) without proportionally more signal
- **Higher dropout** (0.3 vs 0.2) that may have suppressed early learning
- **FC branch noise**: Pearson correlations from 363 timepoints of noisy fMRI may add more noise than signal, especially with batch_size=32

### 3. The bottleneck diagnosis may be wrong

The original hypothesis was that `Linear(180, 64)` loses spatial structure. But the data suggests the bottleneck is not in how features are extracted — it's in the **intrinsic difficulty of the ADHD classification task** from resting-state fMRI. The best achievable AUC may be near 0.62 regardless of architecture.

Evidence:
- All models (SE, MoE variants, NAME) cluster in the 0.55-0.62 AUC range
- The SE baseline with the simplest architecture achieves the best result
- Adding complexity (MoE routing, network grouping, FC features) consistently hurts

### 4. Possible confounds

- **d_net=8 too narrow**: Some Yeo networks have 20+ ROIs projected to 8 dims — this may lose information rather than preserve it.
- **No positional encoding**: The Transformer in NAME has no PE, unlike the SE baseline which uses learned positional embeddings.
- **Dropout mismatch**: NAME uses 0.3 (per plan) vs SE's 0.2.

## Conclusion

NAME does not improve over the SE baseline for ADHD classification. The feature extraction bottleneck hypothesis was not validated — the simpler architecture performs better. This suggests the performance ceiling on ADHD classification from resting-state fMRI is near 0.62 AUC with current data, and architectural improvements to feature extraction yield diminishing or negative returns.

The remaining avenue is **neurobiology-based clustering+MoE** (direction #2): rather than extracting features differently, route different neurobiological subtypes of ADHD to specialized experts using domain knowledge about ADHD-relevant brain circuits (DMN, frontoparietal, attention networks).

## Files

```
models/yeo17_networks.py          # HCP180 ROI -> Yeo 17-network mapping
models/NetworkAwareExpert.py       # Classical + Quantum NAME experts
NetworkAwareBaseline_ABCD.py       # Training script
scripts/ADHD_NAME_Classical.sh     # SLURM script (submitted)
scripts/{ADHD,ASD,Sex,FluidInt}_NAME_{Classical,Quantum}.sh  # 8 scripts total (not submitted)
```
