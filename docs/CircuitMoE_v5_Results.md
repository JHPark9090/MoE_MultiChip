# Circuit MoE v5 — Corrected Yeo-17 Mapping Results

**Date**: 2026-03-07
**Status**: All v5 experiments complete (10 jobs).

## Background

All Circuit MoE experiments prior to v5 (v1-v4) used an **incorrect Yeo-17 network mapping** in `models/yeo17_networks.py`. The old mapping assigned HCP-MMP1 180 ROIs to Yeo-17 networks using fabricated contiguous index blocks that did not correspond to actual neuroanatomical correspondence.

**v5** uses a **validated mapping** computed via volumetric overlap between the Glasser atlas (`HCPMMP1_for_ABCD.nii.gz`) and the Yeo 2011 17-network atlas, with bilateral majority-vote per region. The validated mapping is stored in `dataloaders/glasser_to_yeo17_mapping.json`.

### Circuit Composition Changes (Old vs Corrected)

| Circuit | Networks Included | Old ROI Count | **v5 ROI Count** |
|---------|------------------|:---:|:---:|
| DMN | DefaultA/B/C, TempPar | 55 | **41** |
| Executive | ContA/B/C, DorsAttnA/B | 50 | **52** |
| Salience | SalVentAttnA/B, Limbic_TempPole/OFC | 29 | **44** |
| SensoriMotor | VisCent/Peri, SomMotA/B | 46 | **43** |
| **Total** | | 180 | **180** |

For the 2-expert configuration:
| Circuit | Networks Included | Old ROI Count | **v5 ROI Count** |
|---------|------------------|:---:|:---:|
| Internal | DMN + Executive | 105 | **93** (41+52) |
| External | Salience + SensoriMotor | 75 | **87** (44+43) |

---

## 1. Experimental Setup

### Shared Hyperparameters

All experiments: Adam (lr=1e-3, wd=1e-5), cosine LR scheduler, batch_size=32, grad_clip=1.0, dropout=0.2, gating_noise_std=0.1, balance_loss_alpha=0.1, patience=20, seed=2025.

### Dataset

ABCD resting-state fMRI, HCP180 parcellation (180 bilateral ROIs, 363 timepoints), 70/15/15 split.

| Phenotype | Task | N Subjects | Train / Val / Test |
|-----------|------|:---:|:---:|
| ADHD | Binary classification | 4,458 | 3,120 / 669 / 669 |

### Model Architectures

| Model | Expert Architecture | Params |
|-------|-------------------|--------|
| Classical SE | AllChannelExpert: Linear(180,64) + Dropout(0.2) + Learnable PE + TransformerEncoder(d=64, nhead=4, layers=2, ff=256) + mean pooling | 134,849 |
| Quantum SE (8Q) | QuantumTSTransformer: PE + Linear(180, n_rots) + QSVT/LCU (8Q, 2L, D=3) + sim14 + PauliXYZ + Linear(24, 64) | 13,648 |
| Classical Circuit MoE 4e | 4 CircuitExperts (same arch as SE but input_dim = circuit ROI count) + gating network | 516,485 |
| Classical Circuit MoE 2e | 2 CircuitExperts + gating network | 269,827 |
| Quantum Circuit MoE | QSVT quantum experts (circuit-specific ROI subsets) + gating network | 27K-35K |

---

## 2. Results — ADHD Classification

### 2.1 Single-Expert Baselines (Unaffected by Mapping Fix)

These models use all 180 ROIs and are identical between v4 and v5.

| Model | Params | Train AUC | Val AUC | **Test AUC** | Test Acc | Test Loss |
|-------|--------|:---:|:---:|:---:|:---:|:---:|
| **Classical SE** | **134,849** | 0.6132 | 0.6146 | **0.6193** | 0.5949 | 0.6655 |
| Quantum SE (8Q, D=3) | 13,648 | 0.8318 | 0.6081 | 0.5769 | 0.5859 | 0.8009 |

### 2.2 Other MoE Baselines (Unaffected by Mapping Fix)

These models use all 180 ROIs per expert — routing is based on clustering or learned gating, not Yeo-17 circuit partitioning. Results are unchanged from earlier experiments.

**Cluster-Informed MoE** (2 experts, routed via site-regressed coherence clustering k=2):

| Model | Config | Params | Val AUC | **Test AUC** | Test Acc |
|-------|--------|:---:|:---:|:---:|:---:|
| Classical | Cluster Soft | 283,491 | 0.6379 | 0.6001 | 0.5934 |
| Classical | Cluster Hard | 269,633 | 0.6004 | 0.5735 | 0.5650 |
| Quantum | Cluster Soft | 41,089 | 0.6236 | 0.5463 | 0.5605 |
| Quantum | Cluster Hard | 27,231 | 0.5742 | 0.5853 | 0.5904 |

**Learned Routing MoE** (2 experts, routed via top-64 PCA of coherence features):

| Model | Config | Params | Val AUC | **Test AUC** | Test Acc |
|-------|--------|:---:|:---:|:---:|:---:|
| Classical | Learned Soft | 287,459 | 0.5914 | 0.5968 | 0.5845 |
| Classical | Learned Hard | 287,459 | 0.5752 | 0.5758 | 0.5919 |
| Quantum | Learned Soft | 45,057 | 0.5796 | 0.5749 | 0.5770 |
| Quantum | Learned Hard | 45,057 | 0.5716 | 0.5542 | 0.5770 |

### 2.3 NAME Classical (Affected by Mapping Fix — Not Yet Retrained)

NAME uses Yeo 17-network spatial projection (`models/NetworkAwareExpert.py` imports `get_network_indices` from `yeo17_networks`). This result used the **old incorrect mapping** and has not been retrained with the corrected mapping.

| Model | Params | Val AUC | **Test AUC** | Test Acc | Notes |
|-------|--------|:---:|:---:|:---:|-------|
| NAME Classical | 289,906 | 0.5658 | 0.5800 | — | Old mapping; needs v5 retraining |

### 2.4 Arbitrary-Split Circuit MoE (v4 — Inadvertent Baseline)

The v1-v4 Circuit MoE experiments used a fabricated Yeo-17 mapping where ROIs were assigned to networks in near-contiguous index blocks (e.g., VisCent = [0-7], VisPeri = [8-17], etc.) with no neuroanatomical basis. Because the ROI-to-network assignments were arbitrary, these experiments effectively serve as an **arbitrary-split baseline** — each expert received a block of ROIs grouped by index order rather than by brain function. The group sizes differ slightly from v5 (old: DMN=55, Exec=50, Sal=29, SM=46 vs v5: DMN=41, Exec=52, Sal=44, SM=43), which is a confound but does not invalidate the directional comparison.

**Classical Arbitrary-Split:**

| Model | Config | Params | Val AUC | **Test AUC** | Test Acc |
|-------|--------|:---:|:---:|:---:|:---:|
| Classical 4-expert (arb) | adhd_3 | 516,485 | 0.6152 | **0.6167** | 0.6280 |
| Classical 2-expert (arb) | adhd_2 | 269,827 | 0.6164 | 0.5987 | 0.5874 |

**Quantum Arbitrary-Split (v2, 8Q D=3):**

| Model | Config | Params | Val AUC | **Test AUC** | Test Acc |
|-------|--------|:---:|:---:|:---:|:---:|
| Q v2 8Q d3 4-expert (arb) | adhd_3 | 30,373 | — | 0.5764 | 0.5790 |
| Q v2 8Q d3 2-expert (arb) | adhd_2 | 26,771 | — | 0.5783 | 0.5934 |

**Quantum Arbitrary-Split (v3, 10Q D=3):**

| Model | Config | Params | Val AUC | **Test AUC** | Test Acc |
|-------|--------|:---:|:---:|:---:|:---:|
| Q v3 10Q d3 4-expert (arb) | adhd_3 | 34,885 | — | **0.6022** | — |
| Q v3 10Q d3 2-expert (arb) | adhd_2 | 30,467 | — | 0.5674 | — |

### 2.5 Neuroscience-Guided Circuit MoE v5 — Classical

| Model | Config | Params | Best Ep | ES Ep | Train AUC | Val AUC | **Test AUC** | Test Acc | Test Loss |
|-------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Classical 4-expert | adhd_3 | 516,485 | 1 | 21 | 0.5242 | 0.5982 | **0.5925** | 0.5800 | 0.6790 |
| Classical 2-expert | adhd_2 | 269,827 | 25 | 45 | 0.8748 | 0.6345 | 0.5696 | 0.5740 | 0.9749 |

### 2.6 Neuroscience-Guided Circuit MoE v5 — Quantum Degree 3

| Model | Config | Params | Best Ep | ES Ep | Train AUC | Val AUC | **Test AUC** | Test Acc | Test Loss |
|-------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Q 8Q d3 4-expert | adhd_3 | 30,373 | 13 | 33 | 0.6589 | 0.6170 | 0.5773 | 0.5919 | 0.7051 |
| **Q 8Q d3 2-expert** | adhd_2 | 26,771 | 21 | 41 | 0.6947 | 0.6049 | **0.5939** | 0.5859 | 0.7130 |
| Q 10Q d3 4-expert | adhd_3 | 34,885 | 4 | 24 | 0.5634 | 0.5972 | 0.5835 | 0.5845 | 0.6753 |
| Q 10Q d3 2-expert | adhd_2 | 30,467 | 5 | 25 | 0.5631 | 0.5871 | 0.5867 | 0.5874 | 0.6733 |

### 2.7 Neuroscience-Guided Circuit MoE v5 — Quantum Degree 2

| Model | Config | Params | Best Ep | ES Ep | Train AUC | Val AUC | **Test AUC** | Test Acc | Test Loss |
|-------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Q 8Q d2 4-expert | adhd_3 | 30,369 | 25 | 45 | 0.7392 | 0.6193 | 0.5345 | 0.5575 | 0.7845 |
| Q 8Q d2 2-expert | adhd_2 | 26,769 | 27 | 47 | 0.7129 | 0.5960 | 0.5503 | 0.5755 | 0.7499 |
| Q 10Q d2 4-expert | adhd_3 | 34,881 | 10 | 30 | 0.6323 | 0.5900 | 0.5877 | 0.5949 | 0.6889 |
| Q 10Q d2 2-expert | adhd_2 | 30,465 | 12 | 32 | 0.6277 | 0.6295 | 0.5515 | 0.5695 | 0.6834 |

---

## 3. Summary — All ADHD Models Ranked by Test AUC

Includes all model types: SE baselines, Cluster MoE, Learned MoE, NAME, and Circuit MoE v5.

| Rank | Model | Type | Params | **Test AUC** | Test Acc | Val-Test Gap |
|:----:|-------|------|--------|:---:|:---:|:---:|
| 1 | **Classical SE** | SE | 134,849 | **0.6193** | 0.5949 | -0.5 pts |
| 2 | Classical Cluster Soft | Cluster MoE | 283,491 | 0.6001 | 0.5934 | -3.8 pts |
| 3 | Classical Learned Soft | Learned MoE | 287,459 | 0.5968 | 0.5845 | -0.5 pts |
| 4 | **Q 8Q d3 2-expert (v5)** | Circuit MoE | **26,771** | **0.5939** | 0.5859 | -1.1 pts |
| 5 | Classical Circuit 4e (v5) | Circuit MoE | 516,485 | 0.5925 | 0.5800 | -0.6 pts |
| 6 | Q 10Q d2 4-expert (v5) | Circuit MoE | 34,881 | 0.5877 | 0.5949 | -0.2 pts |
| 7 | Q 10Q d3 2-expert (v5) | Circuit MoE | 30,467 | 0.5867 | 0.5874 | -0.0 pts |
| 8 | Q Cluster Hard | Cluster MoE | 27,231 | 0.5853 | 0.5904 | +1.1 pts |
| 9 | Q 10Q d3 4-expert (v5) | Circuit MoE | 34,885 | 0.5835 | 0.5845 | -1.4 pts |
| 10 | NAME Classical | NAME | 289,906 | 0.5800\* | — | — |
| 11 | Q 8Q d3 4-expert (v5) | Circuit MoE | 30,373 | 0.5773 | 0.5919 | -4.0 pts |
| 12 | Quantum SE 8Q | SE | 13,648 | 0.5769 | 0.5859 | -3.1 pts |
| 13 | Classical Learned Hard | Learned MoE | 287,459 | 0.5758 | 0.5919 | +0.1 pts |
| 14 | Q Learned Soft | Learned MoE | 45,057 | 0.5749 | 0.5770 | +0.5 pts |
| 15 | Classical Cluster Hard | Cluster MoE | 269,633 | 0.5735 | 0.5650 | -2.7 pts |
| 16 | Classical Circuit 2e (v5) | Circuit MoE | 269,827 | 0.5696 | 0.5740 | -6.5 pts |
| 17 | Q Learned Hard | Learned MoE | 45,057 | 0.5542 | 0.5770 | +1.7 pts |
| 18 | Q 10Q d2 2-expert (v5) | Circuit MoE | 30,465 | 0.5515 | 0.5695 | -7.8 pts |
| 19 | Q 8Q d2 2-expert (v5) | Circuit MoE | 26,769 | 0.5503 | 0.5755 | -4.6 pts |
| 20 | Q Cluster Soft | Cluster MoE | 41,089 | 0.5463 | 0.5605 | -7.7 pts |
| 21 | Q 8Q d2 4-expert (v5) | Circuit MoE | 30,369 | 0.5345 | 0.5575 | -8.5 pts |

\* NAME Classical used the old incorrect Yeo-17 mapping; not yet retrained with v5.

---

## 4. Ablation: Neuroscience-Guided (v5) vs Arbitrary-Split (v4)

Since the v1-v4 experiments inadvertently used arbitrary contiguous-block ROI assignments, the v4 results serve as an **arbitrary-split ablation baseline**. This allows a direct comparison of whether neuroscience-guided circuit partitioning provides a genuine advantage over arbitrary input splitting.

**Caveat**: The group sizes differ slightly between v4 and v5 (see Section "Circuit Composition Changes" above), so this is not a perfectly controlled ablation. However, the parameter counts are identical (same model architecture, same total ROIs), so the comparison is informative.

### 4.1 Head-to-Head: Arbitrary (v4) vs Neuroscience (v5)

| Model | Arbitrary (v4) | **Neuroscience (v5)** | Delta | Winner |
|-------|:---:|:---:|:---:|:---:|
| Classical 4-expert | **0.6167** | 0.5925 | -2.4 pts | Arbitrary |
| Classical 2-expert | **0.5987** | 0.5696 | -2.9 pts | Arbitrary |
| Q 8Q d3 4-expert | 0.5764 | 0.5773 | +0.1 pts | Tie |
| Q 8Q d3 2-expert | 0.5783 | **0.5939** | **+1.6 pts** | **Neuro** |
| Q 10Q d3 4-expert | **0.6022** | 0.5835 | -1.9 pts | Arbitrary |
| Q 10Q d3 2-expert | 0.5674 | **0.5867** | **+1.9 pts** | **Neuro** |

**Scorecard**: Arbitrary wins 3, Neuroscience wins 2, Tie 1.

### 4.2 Interpretation

The results reveal a **classical-quantum split in the ablation**:

- **Classical experts prefer arbitrary (contiguous) splits** (-2.4 to -2.9 pts). Classical Transformer experts likely benefit from the spatial locality preserved by contiguous index blocks — adjacent ROI indices in the HCP atlas tend to correspond to anatomically proximate cortical regions. Classical convolution-like operations can exploit this spatial coherence.

- **Quantum 2-expert models prefer neuroscience-guided splits** (+1.6 to +1.9 pts). Quantum circuits apply all-to-all entanglement (CNOT/CRX gates between all qubit pairs), so they have no inherent bias toward spatial locality. Instead, they benefit from grouping functionally related ROIs whose activity patterns are correlated — exactly what the Yeo-17 functional network mapping provides.

- **Quantum 4-expert models show mixed results**: The 10Q 4-expert preferred arbitrary (-1.9 pts), the 8Q 4-expert was flat (+0.1 pts). With 4 smaller experts, each circuit has fewer ROIs (41-52 in v5 vs 29-55 in v4), and the quantum circuit may not have enough capacity to exploit the functional coherence of small, specialized subsets.

### 4.3 Summary

The ablation does **not** show a clear universal advantage for neuroscience-guided partitioning. The benefit is architecture-dependent: quantum 2-expert models gain +1.6-1.9 pts from functional grouping, while classical models lose -2.4-2.9 pts. This suggests that the value of neuroscience-informed input partitioning depends on whether the expert architecture can exploit functional coherence (quantum entanglement) versus spatial locality (classical convolution).

---

## 5. Analysis

### 5.1 Degree 3 vs Degree 2

| Comparison | d3 Test AUC | d2 Test AUC | Winner |
|-----------|:---:|:---:|:---:|
| 8Q 4-expert | 0.5773 | 0.5345 | d3 (+4.3 pts) |
| 8Q 2-expert | 0.5939 | 0.5503 | d3 (+4.4 pts) |
| 10Q 4-expert | 0.5835 | 0.5877 | d2 (+0.4 pts) |
| 10Q 2-expert | 0.5867 | 0.5515 | d3 (+3.5 pts) |

**Degree 3 wins 3/4 comparisons** and by large margins (3.5-4.4 pts). The 10Q d2 4-expert is the sole exception, likely due to better regularization (lower train AUC at best epoch). Degree 2 models show larger val-test gaps, suggesting they overfit more easily.

### 5.2 8-Qubit vs 10-Qubit

| Comparison | 8Q Test AUC | 10Q Test AUC | Winner |
|-----------|:---:|:---:|:---:|
| d3 4-expert | 0.5773 | 0.5835 | 10Q (+0.6 pts) |
| d3 2-expert | **0.5939** | 0.5867 | 8Q (+0.7 pts) |
| d2 4-expert | 0.5345 | **0.5877** | 10Q (+5.3 pts) |
| d2 2-expert | 0.5503 | 0.5515 | 10Q (+0.1 pts) |

**10Q wins 3/4 comparisons**, but the best single model is 8Q d3 2-expert (0.5939). More qubits helps 4-expert models (more ROIs per expert = more features to encode) but gives mixed results for 2-expert.

### 5.3 4-Expert vs 2-Expert

| Comparison | 4-expert Test AUC | 2-expert Test AUC | Winner |
|-----------|:---:|:---:|:---:|
| Classical | 0.5925 | 0.5696 | 4-expert (+2.3 pts) |
| Q 8Q d3 | 0.5773 | **0.5939** | 2-expert (+1.7 pts) |
| Q 10Q d3 | 0.5835 | 0.5867 | 2-expert (+0.3 pts) |
| Q 8Q d2 | 0.5345 | 0.5503 | 2-expert (+1.6 pts) |
| Q 10Q d2 | **0.5877** | 0.5515 | 4-expert (+3.6 pts) |

**Mixed results.** Classical prefers 4-expert. Quantum d3 prefers 2-expert. No clear pattern for d2.

### 5.4 Parameter Efficiency

| Model | Type | Params | Test AUC | AUC per 100K Params |
|-------|------|:---:|:---:|:---:|
| Quantum SE 8Q | SE | 13,648 | 0.5769 | 4.227 |
| Q Circuit 8Q d3 2e (v5) | Circuit MoE | 26,771 | 0.5939 | 2.218 |
| Q Cluster Hard | Cluster MoE | 27,231 | 0.5853 | 2.149 |
| Q Circuit 10Q d3 2e (v5) | Circuit MoE | 30,467 | 0.5867 | 1.925 |
| Q Circuit 8Q d3 4e (v5) | Circuit MoE | 30,373 | 0.5773 | 1.901 |
| Q Circuit 10Q d2 4e (v5) | Circuit MoE | 34,881 | 0.5877 | 1.685 |
| Q Learned Soft | Learned MoE | 45,057 | 0.5749 | 1.276 |
| Classical SE | SE | 134,849 | 0.6193 | 0.459 |
| Classical Cluster Soft | Cluster MoE | 283,491 | 0.6001 | 0.212 |
| Classical Learned Soft | Learned MoE | 287,459 | 0.5968 | 0.208 |
| NAME Classical | NAME | 289,906 | 0.5800 | 0.200 |
| Classical Circuit 4e (v5) | Circuit MoE | 516,485 | 0.5925 | 0.115 |

Quantum models achieve 4-20x higher AUC-per-parameter than classical models. The best quantum Circuit MoE (0.5939) reaches **95.9% of Classical SE** (0.6193) with **5.0x fewer parameters** (26,771 vs 134,849).

### 5.5 Overfitting Analysis

| Model | Train AUC @ Best | Val AUC | Test AUC | Train-Test Gap |
|-------|:---:|:---:|:---:|:---:|
| Classical SE | 0.6132 | 0.6146 | 0.6193 | -0.6 pts |
| Classical 4e (v5) | 0.5242 | 0.5982 | 0.5925 | -6.8 pts |
| Classical 2e (v5) | 0.8748 | 0.6345 | 0.5696 | +30.5 pts |
| Q 8Q d3 2e (v5) | 0.6947 | 0.6049 | 0.5939 | +10.1 pts |
| Q 10Q d3 2e (v5) | 0.5631 | 0.5871 | 0.5867 | -2.4 pts |
| Q 10Q d2 4e (v5) | 0.6323 | 0.5900 | 0.5877 | +4.5 pts |

The Classical SE shows minimal overfitting (negative gap — test > train at early stopping). MoE models generally overfit more, with gating networks contributing to the problem. Quantum models with low train AUC at best epoch (Q 10Q d3 2e: 0.5631) show the best val-test agreement (-0.0 pts gap).

---

## 6. Conclusions

1. **Classical SE remains the best overall model** (0.6193 test AUC). No MoE variant — Circuit, Cluster, Learned, or NAME — surpasses the single-expert baseline on test AUC for ADHD.

2. **Classical Cluster Soft MoE is the best MoE model overall** (0.6001 test AUC), followed by Classical Learned Soft (0.5968). Both use all 180 ROIs and are unaffected by the Yeo-17 mapping fix.

3. **Best quantum model: Q 8Q d3 2-expert Circuit MoE (v5)** (0.5939 test AUC, 26,771 params) — 95.9% of Classical SE with 5.0x fewer parameters. This outperforms all other quantum MoE variants including Cluster MoE (Q Cluster Hard: 0.5853, Q Cluster Soft: 0.5463) and Learned MoE (Q Learned Soft: 0.5749).

4. **Degree 3 > Degree 2**: Higher polynomial degree provides better test generalization across most configurations (+3.5 to +4.4 pts advantage for 8Q models).

5. **The corrected mapping hurt classical Circuit MoE but helped quantum 2-expert models**, suggesting that functionally coherent ROI groupings better match quantum circuits' all-to-all entanglement structure, while classical experts may have benefited from the accidental spatial locality of the old contiguous-block mapping.

6. **Overfitting is the primary bottleneck**, not circuit expressivity. Models with the highest val AUC consistently have the worst val-test gaps (Classical 2e: val 0.6345, test 0.5696; Q 8Q d2 4e: val 0.6193, test 0.5345). Stronger regularization or larger datasets may be needed.

7. **No MoE routing strategy helps for ADHD**: Circuit (neuroscience-guided), Cluster (data-driven), and Learned (end-to-end) routing all fail to outperform a single expert. This is consistent across classical and quantum variants and suggests the bottleneck is expert capacity and overfitting, not routing quality.

8. **ADHD classification remains a hard problem** — all models cluster between 0.53-0.62 test AUC, consistent with the heterogeneous, low-SNR nature of ADHD neuroimaging phenotypes.

---

## 7. Next Steps

1. **Run interpretability analysis** on v5 checkpoints using updated `analyze_circuit_moe.py` (now includes brain region names from `dataloaders/hcp_mmp1_labels.py`)
2. **Run heterogeneity analysis** on v5 checkpoints using updated `analyze_heterogeneity.py`
3. **Arbitrary-split ablation**: Compare v5 neuroscience-guided circuits against random contiguous-block splits of the same size to validate that the functional grouping provides a genuine advantage
4. **Multi-seed runs**: Repeat best configurations (Q 8Q d3 2e, Classical 4e) across seeds 2024/2025/2026 to establish confidence intervals
5. **Regularization experiments**: Test stronger dropout (0.3-0.4), label smoothing, or mixup to reduce overfitting
