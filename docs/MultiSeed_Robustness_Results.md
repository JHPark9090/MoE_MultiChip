# Multi-Seed Robustness Analysis — Circuit MoE v5

**Date**: 2026-03-08 (updated)
**Seeds**: 2024, 2025, 2026, 2027, 2028 (n=5, all complete)

## 1. Experimental Setup

### Models (6 configurations)

| Model | Script | Params | Expert Architecture |
|-------|--------|:------:|---------------------|
| Classical SE | `SingleExpertBaseline_ABCD.py` | 134,849 | AllChannelExpert: Linear(180,64) + TransformerEncoder(d=64, nhead=4, layers=2) |
| Quantum SE (8Q, D=3) | `SingleExpertBaseline_ABCD.py` | 13,648 | QuantumTSTransformer: QSVT/LCU (8Q, 2L, D=3) + sim14 |
| Classical Circuit MoE 2e | `CircuitMoE_ABCD.py` | 269,827 | 2 CircuitExperts (Internal: 93 ROIs, External: 87 ROIs) + gating |
| Classical Circuit MoE 4e | `CircuitMoE_ABCD.py` | 516,485 | 4 CircuitExperts (DMN: 41, Exec: 52, Sal: 44, SM: 43) + gating |
| Quantum Circuit MoE 2e (8Q, D=3) | `CircuitMoE_v2_ABCD.py` | 26,771 | 2 QSVT quantum experts + gating |
| Quantum Circuit MoE 4e (8Q, D=3) | `CircuitMoE_v2_ABCD.py` | 30,373 | 4 QSVT quantum experts + gating |

### Shared Hyperparameters

All runs: Adam (lr=1e-3, wd=1e-5), cosine LR scheduler, batch_size=32, grad_clip=1.0, dropout=0.2, patience=20, n_epochs=100. Circuit MoE additionally: gating_noise_std=0.1, balance_loss_alpha=0.1.

### Dataset

ABCD resting-state fMRI, ADHD binary classification. N=4,458 (train 3,120 / val 669 / test 669). HCP-MMP1 180 bilateral ROIs, 363 timepoints. Corrected Yeo-17 mapping (v5).

### Checkpoints

| Model | Seed 2024 | Seed 2025 | Seed 2026 | Seed 2027 | Seed 2028 |
|-------|-----------|-----------|-----------|-----------|-----------|
| Classical SE | `*_49799103.pt` | (original v5 run) | `*_49799112.pt` | `*_49806440.pt` | `*_49806447.pt` |
| Quantum SE | `*_49799104.pt` | (original v5 run) | `*_49799113.pt` | `*_49806441.pt` | `*_49806448.pt` |
| Classical MoE 2e | `*_49799105.pt` | `*_49777878.pt` | `*_49799114.pt` | `*_49806443.pt` | `*_49806449.pt` |
| Classical MoE 4e | `*_49799108.pt` | `*_49777876.pt` | `*_49799115.pt` | `*_49806444.pt` | `*_49806450.pt` |
| Quantum MoE 2e | `*_49799110.pt` | `*_49777880.pt` | `*_49799117.pt` | `*_49806445.pt` | `*_49806451.pt` |
| Quantum MoE 4e | `*_49799111.pt` | `*_49777879.pt` | `*_49799118.pt` | `*_49806446.pt` | `*_49806452.pt` |

---

## 2. Test AUC Results

### 2.1 Full Results Table

| Model | Params | Seed 2024 | Seed 2025 | Seed 2026 | Seed 2027 | Seed 2028 | **Mean** | **Std** | **95% CI** |
|-------|:------:|:---------:|:---------:|:---------:|:---------:|:---------:|:--------:|:-------:|:----------:|
| **Classical SE** | 134,849 | 0.5935 | **0.6193** | 0.5863 | 0.5565 | 0.5930 | **0.5897** | 0.0224 | [0.562, 0.618] |
| Classical MoE 2e | 269,827 | 0.5991 | 0.5696 | 0.5753 | 0.5592 | 0.5930 | 0.5792 | 0.0165 | [0.559, 0.600] |
| Classical MoE 4e | 516,485 | 0.5434 | 0.5925 | 0.5700 | 0.5708 | 0.6013 | 0.5756 | 0.0226 | [0.548, 0.604] |
| Quantum MoE 4e | 30,373 | 0.5809 | 0.5773 | 0.5611 | 0.5674 | 0.5469 | 0.5667 | 0.0136 | [0.550, 0.584] |
| Quantum SE (8Q, D=3) | 13,648 | 0.5657 | 0.5769 | 0.5365 | 0.5675 | 0.5511 | 0.5595 | 0.0158 | [0.540, 0.579] |
| Quantum MoE 2e | 26,771 | 0.5501 | 0.5939 | 0.5246 | 0.5681 | 0.5482 | 0.5570 | 0.0258 | [0.525, 0.589] |

*All n=5. 95% CI via t-distribution (df=4). Sorted by mean test AUC.*

### 2.2 Seed-by-Seed Rankings

| Seed | 1st | 2nd | 3rd | 4th | 5th | 6th |
|------|-----|-----|-----|-----|-----|-----|
| 2024 | C MoE 2e (0.599) | **C SE (0.594)** | Q MoE 4e (0.581) | Q SE (0.566) | Q MoE 2e (0.550) | C MoE 4e (0.543) |
| 2025 | **C SE (0.619)** | Q MoE 2e (0.594) | C MoE 4e (0.593) | Q MoE 4e (0.577) | Q SE (0.577) | C MoE 2e (0.570) |
| 2026 | **C SE (0.586)** | C MoE 2e (0.575) | C MoE 4e (0.570) | Q MoE 4e (0.561) | Q SE (0.537) | Q MoE 2e (0.525) |
| 2027 | C MoE 4e (0.571) | Q MoE 2e (0.568) | Q SE (0.568) | Q MoE 4e (0.567) | C MoE 2e (0.559) | **C SE (0.557)** |
| 2028 | C MoE 4e (0.601) | **C SE (0.593)** | C MoE 2e (0.593) | Q SE (0.551) | Q MoE 2e (0.548) | Q MoE 4e (0.547) |

Classical SE ranks 1st or 2nd in 4/5 seeds (exception: seed 2027, ranked 6th).

---

## 3. Statistical Analysis (all n=5, paired t-tests)

### 3.1 Classical SE vs All Others

| Comparison | Δ Mean | t-stat | p-value | Cohen's d | Significant? |
|-----------|:------:|:------:|:-------:|:---------:|:------------:|
| C SE vs Quantum SE | +0.030 | 2.771 | 0.050 | 1.55 | **borderline** |
| C SE vs Quantum MoE 2e | +0.033 | 2.622 | 0.059 | 1.35 | no (trend) |
| C SE vs Quantum MoE 4e | +0.023 | 2.215 | 0.091 | 1.24 | no |
| C SE vs Classical MoE 4e | +0.014 | 1.200 | 0.296 | 0.63 | no |
| C SE vs Classical MoE 2e | +0.011 | 1.028 | 0.362 | 0.53 | no |

### 3.2 Classical vs Quantum (Same Architecture)

| Comparison | Δ Mean | t-stat | p-value | Cohen's d | Significant? |
|-----------|:------:|:------:|:-------:|:---------:|:------------:|
| C SE vs Q SE | +0.030 | 2.771 | 0.050 | 1.55 | **borderline** |
| C MoE 2e vs Q MoE 2e | +0.022 | 1.384 | 0.239 | 1.03 | no |
| C MoE 4e vs Q MoE 4e | +0.009 | 0.606 | 0.577 | 0.48 | no |

### 3.3 Classical Internal Comparisons

| Comparison | Δ Mean | t-stat | p-value | Cohen's d | Significant? |
|-----------|:------:|:------:|:-------:|:---------:|:------------:|
| C SE vs C MoE 2e | +0.011 | 1.028 | 0.362 | 0.53 | no |
| C SE vs C MoE 4e | +0.014 | 1.200 | 0.296 | 0.63 | no |
| C MoE 2e vs C MoE 4e | +0.004 | 0.264 | 0.805 | 0.18 | no |

### 3.4 Quantum Internal Comparisons

| Comparison | Δ Mean | t-stat | p-value | Cohen's d | Significant? |
|-----------|:------:|:------:|:-------:|:---------:|:------------:|
| Q MoE 4e vs Q SE | +0.007 | 1.315 | 0.259 | 0.49 | no |
| Q MoE 4e vs Q MoE 2e | +0.010 | 0.954 | 0.394 | 0.47 | no |
| Q SE vs Q MoE 2e | +0.003 | 0.449 | 0.677 | 0.12 | no |

---

## 4. Key Findings

### 4.1 What We Can Claim (p≤0.05)

With n=5 seeds, **no pairwise comparison reaches strict statistical significance** (p<0.05). The closest is Classical SE vs Quantum SE at p=0.050 (borderline). This reflects the genuinely narrow performance band across all architectures on this task.

### 4.2 What We Cannot Claim (p>0.05)

1. **Classical SE is the best model**: While it has the highest mean AUC (0.590), the advantage over Classical MoE 2e (0.579, p=0.362) and Classical MoE 4e (0.576, p=0.296) is not significant.

2. **Classical > Quantum**: The SE comparison is borderline (p=0.050, d=1.55). In MoE configurations, no significant difference exists (p=0.24 for 2e, p=0.58 for 4e).

3. **MoE routing helps or hurts**: No MoE variant significantly differs from the corresponding SE baseline (all p>0.05).

4. **2-expert vs 4-expert**: No significant difference in either classical (p=0.81) or quantum (p=0.39) models.

### 4.3 Stability Analysis

| Model | Std | Stability Rank | Notes |
|-------|:---:|:--------------:|-------|
| Quantum MoE 4e | 0.014 | 1 (most stable) | Tightest range: 0.547–0.581 |
| Quantum SE | 0.016 | 2 | Consistent 0.537–0.577 |
| Classical MoE 2e | 0.017 | 3 | |
| Classical SE | 0.022 | 4 | Seed 2027 was low (0.557) |
| Classical MoE 4e | 0.023 | 5 | Seed 2024 was low (0.543) |
| Quantum MoE 2e | 0.026 | 6 (least stable) | High variance: 0.525–0.594 |

### 4.4 Parameter Efficiency (Quantum vs Classical)

| Quantum Model | Params | Mean AUC | Classical Equiv. | Params | Mean AUC | Q/C AUC | Param Ratio |
|---------------|:------:|:--------:|-----------------|:------:|:--------:|:-------:|:-----------:|
| Q SE | 13,648 | 0.560 | Classical SE | 134,849 | 0.590 | 94.9% | 9.9× fewer |
| Q MoE 2e | 26,771 | 0.557 | C MoE 2e | 269,827 | 0.579 | 96.2% | 10.1× fewer |
| Q MoE 4e | 30,373 | 0.567 | C MoE 4e | 516,485 | 0.576 | **98.5%** | 17.0× fewer |

Quantum MoE 4-expert achieves **98.5% of classical MoE 4-expert AUC with 17× fewer parameters** — the strongest parameter efficiency result.

---

## 5. Limitations

1. **Narrow performance band**: All 6 models cluster within 0.557–0.590 mean AUC (3.3-point spread). This is smaller than within-model seed variance (std=0.014–0.026). Architecture choice has a smaller effect than random initialization.

2. **ADHD classification from resting-state fMRI is inherently hard**. All models achieve ~0.52–0.62 test AUC per seed, consistent with published literature on ADHD neuroimaging phenotypes.

3. **Single train/val/test split**: All seeds use the same 70/15/15 data split (deterministic based on data loading). Only model initialization varies. Cross-validation would provide a more robust estimate.

---

## 6. Conclusions

1. **No architecture significantly outperforms any other** across 5 seeds (all pairwise p≥0.050). The ADHD classification task does not differentiate these architectures.

2. **Classical SE has the highest mean AUC** (0.590) but its advantage is not statistically significant over any other model (p=0.050–0.362).

3. **Classical > Quantum is borderline for single experts** (p=0.050, d=1.55). This advantage diminishes in MoE configurations (p=0.24 for 2e, p=0.58 for 4e), suggesting that brain-circuit-aware routing partially compensates for the quantum parameter deficit.

4. **Quantum MoE 4-expert is the most parameter-efficient model** — achieving 98.5% of classical MoE 4-expert AUC with 17× fewer parameters. This is the strongest case for practical quantum utility in this dataset.

5. **MoE routing provides no measurable benefit** over single-expert baselines for ADHD classification (all SE vs MoE comparisons p>0.05).

6. **Quantum MoE 4-expert is the most stable model** (std=0.014), suggesting that brain-circuit-aware routing with quantum experts provides consistent performance across initializations despite having only 30K parameters.

7. **All models are equally valid** for this task. The choice between architectures should be driven by secondary criteria: parameter efficiency (favors quantum), interpretability (favors circuit MoE), stability (favors Q MoE 4e), or simplicity (favors SE).

---

## 7. References

- Seed 2025 results: `docs/CircuitMoE_v5_Results.md`
- Multi-seed scripts: `scripts/multiseed/`
- Submission script: `scripts/submit_multiseed_v5.sh`
- Interpretability analysis: `docs/Interpretability_Heterogeneity_v5_Results.md`
- Per-expert saliency: `analyze_expert_saliency.py`, `docs/PerExpert_Saliency_Results.md`
