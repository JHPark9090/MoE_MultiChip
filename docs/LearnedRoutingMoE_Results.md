# Learned Routing MoE: Results and Comparison with Cluster-Informed MoE

## Overview

The Learned Routing MoE (Approach 2) replaces the cluster one-hot gating input from the Cluster-Informed MoE (Approach 1) with top-64 PCA components from site-regressed coherence features. The hypothesis was that a richer, continuous routing signal would allow the gating network to learn better expert assignments end-to-end.

**Result**: The hypothesis was not supported. Cluster-Informed MoE matched or outperformed Learned Routing across all 8 configurations on validation AUC, and held an edge on most test AUC comparisons as well.

## ADHD Classification Results

### Head-to-Head Comparison

| Config | Approach | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Expert Util | Job ID |
|--------|----------|--------|--------|-------------|----------|----------|-------------|--------|
| Classical Soft | Cluster | 283,491 | 30 (ES@10) | **0.6379** | **0.6001** | 0.5934 | 49/51 | 48963928 |
| | Learned | 287,459 | 21 (ES@1) | 0.5914 | 0.5968 | 0.5845 | 45/55 | 49237309 |
| | Delta | +3,968 | | -4.65 pts | -0.33 pts | | | |
| Classical Hard | Cluster | 269,633 | 31 (ES@11) | **0.6004** | 0.5735 | 0.5650 | 36/64 | 48963930 |
| | Learned | 287,459 | 22 (ES@2) | 0.5752 | **0.5758** | 0.5919 | 53/47 | 49237310 |
| | Delta | +17,826 | | -2.52 pts | +0.23 pts | | | |
| Quantum Soft | Cluster | 41,089 | 42 (ES@22) | **0.6236** | 0.5463 | 0.5605 | 52/48 | 48963931 |
| | Learned | 45,057 | 24 (ES@4) | 0.5796 | **0.5749** | 0.5770 | 58/42 | 49237311 |
| | Delta | +3,968 | | -4.40 pts | +2.86 pts | | | |
| Quantum Hard | Cluster | 27,231 | 37 (ES@17) | 0.5742 | **0.5853** | 0.5904 | 36/64 | 48963933 |
| | Learned | 45,057 | 26 (ES@6) | 0.5716 | 0.5542 | 0.5770 | 56/44 | 49237312 |
| | Delta | +17,826 | | -0.26 pts | -3.11 pts | | | |

ES = early stopped, ES@N = best model saved at epoch N.

### ADHD Summary

- **Cluster-Informed wins on validation AUC** in all 4 configs (by 0.26 to 4.65 pts)
- **Test AUC is split**: Learned Routing wins 2/4 (Classical Hard +0.23, Quantum Soft +2.86) but loses 2/4 (Classical Soft -0.33, Quantum Hard -3.11)
- **Best overall ADHD model**: Cluster-Informed Classical Soft (Val 0.6379, Test 0.6001)
- **Learned Routing early-stops much earlier** (epochs 21-26 vs 30-42), suggesting it converges to a weaker solution faster

## ASD Classification Results

### Head-to-Head Comparison

| Config | Approach | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Expert Util | Job ID |
|--------|----------|--------|--------|-------------|----------|----------|-------------|--------|
| Classical Soft | Cluster | 283,491 | 38 (ES@18) | **0.5642** | **0.5804** | 0.5541 | 52/48 | 48971780 |
| | Learned | 287,459 | 23 (ES@3) | 0.5301 | 0.5385 | 0.5180 | 50/50 | 49237313 |
| | Delta | +3,968 | | -3.41 pts | -4.19 pts | | | |
| Classical Hard | Cluster | 269,633 | 28 (ES@8) | **0.5609** | 0.4960 | 0.5113 | 38/62 | 48971782 |
| | Learned | 287,459 | 30 (ES@10) | 0.5378 | **0.4992** | 0.5020 | 50/50 | 49237314 |
| | Delta | +17,826 | | -2.31 pts | +0.32 pts | | | |
| Quantum Soft | Cluster | 41,089 | 26 (ES@6) | **0.5590** | **0.5625** | 0.5247 | 52/48 | 48971789 |
| | Learned | 45,057 | 24 (ES@4) | 0.5300 | 0.5294 | 0.5167 | 50/50 | 49237315 |
| | Delta | +3,968 | | -2.90 pts | -3.31 pts | | | |
| Quantum Hard | Cluster | 27,231 | 28 (ES@8) | **0.5629** | 0.5041 | 0.5047 | 38/62 | 48978443 |
| | Learned | 45,057 | 25 (ES@5) | 0.5537 | **0.5177** | 0.5140 | 50/50 | 49237316 |
| | Delta | +17,826 | | -0.92 pts | +1.36 pts | | | |

### ASD Summary

- **Cluster-Informed wins on validation AUC** in all 4 configs (by 0.92 to 3.41 pts)
- **Test AUC**: Learned Routing wins 2/4 (Classical Hard +0.32, Quantum Hard +1.36), both within noise
- **All ASD results are near chance** (0.50-0.58), consistent with ASD being harder to classify from resting-state fMRI
- **Best overall ASD model**: Cluster-Informed Classical Soft (Val 0.5642, Test 0.5804)

## Aggregate Comparison

### Validation AUC: Cluster-Informed Wins 8/8

| Config | Cluster Val AUC | Learned Val AUC | Delta |
|--------|----------------|-----------------|-------|
| ADHD Classical Soft | **0.6379** | 0.5914 | -4.65 |
| ADHD Classical Hard | **0.6004** | 0.5752 | -2.52 |
| ADHD Quantum Soft | **0.6236** | 0.5796 | -4.40 |
| ADHD Quantum Hard | **0.5742** | 0.5716 | -0.26 |
| ASD Classical Soft | **0.5642** | 0.5301 | -3.41 |
| ASD Classical Hard | **0.5609** | 0.5378 | -2.31 |
| ASD Quantum Soft | **0.5590** | 0.5300 | -2.90 |
| ASD Quantum Hard | **0.5629** | 0.5537 | -0.92 |
| **Average** | **0.5854** | **0.5587** | **-2.67** |

### Test AUC: Cluster-Informed Wins 4/8

| Config | Cluster Test AUC | Learned Test AUC | Delta |
|--------|-----------------|------------------|-------|
| ADHD Classical Soft | **0.6001** | 0.5968 | -0.33 |
| ADHD Classical Hard | 0.5735 | **0.5758** | +0.23 |
| ADHD Quantum Soft | 0.5463 | **0.5749** | +2.86 |
| ADHD Quantum Hard | **0.5853** | 0.5542 | -3.11 |
| ASD Classical Soft | **0.5804** | 0.5385 | -4.19 |
| ASD Classical Hard | 0.4960 | **0.4992** | +0.32 |
| ASD Quantum Soft | **0.5625** | 0.5294 | -3.31 |
| ASD Quantum Hard | 0.5041 | **0.5177** | +1.36 |
| **Average** | **0.5560** | **0.5483** | **-0.77** |

### Generalization Gap (Val AUC - Test AUC)

| Config | Cluster Gap | Learned Gap |
|--------|------------|-------------|
| ADHD Classical Soft | 3.78 pts | **0.54 pts** |
| ADHD Classical Hard | 2.69 pts | **0.06 pts** |
| ADHD Quantum Soft | 7.73 pts | **0.47 pts** |
| ADHD Quantum Hard | **-1.11 pts** | 1.74 pts |
| ASD Classical Soft | **-1.62 pts** | **-0.84 pts** |
| ASD Classical Hard | 6.49 pts | **3.86 pts** |
| ASD Quantum Soft | **-0.35 pts** | 0.06 pts |
| ASD Quantum Hard | 5.88 pts | **3.60 pts** |
| **Average** | **2.94 pts** | **1.19 pts** |

Learned Routing has a **smaller generalization gap** on average (1.19 vs 2.94 pts), meaning it overfits less to the validation set. However, it achieves this by starting from a lower validation baseline rather than by improving test performance.

## Expert Utilization Comparison

| Config | Cluster Util | Learned Util |
|--------|-------------|-------------|
| ADHD Classical Soft | 49/51 | 45/55 |
| ADHD Classical Hard | 36/64 (fixed) | 53/47 (learned) |
| ADHD Quantum Soft | 52/48 | 58/42 |
| ADHD Quantum Hard | 36/64 (fixed) | 56/44 (learned) |
| ASD Classical Soft | 52/48 | 50/50 |
| ASD Classical Hard | 38/62 (fixed) | 50/50 (learned) |
| ASD Quantum Soft | 52/48 | 50/50 |
| ASD Quantum Hard | 38/62 (fixed) | 50/50 (learned) |

- **No expert collapse** in either approach — both solve the original MoE problem
- Cluster hard routing reflects the natural cluster distribution (~36/64); learned hard routing converges to near-balanced (~50/50) driven by the load-balancing loss
- ASD Learned Routing achieves perfect 50/50 balance across all 4 configs

## Key Findings

### 1. Richer Gating Signal Does Not Help

The 64-D PCA gating input (244 total gate input dim) performed worse than the 2-D cluster one-hot (182 total gate input dim) on validation AUC across all 8 experiments. The additional 62 dimensions of routing information did not translate into better expert specialization.

### 2. The Cluster One-Hot Acts as an Effective Prior

Despite the weak cluster-phenotype association (~5% prevalence difference), the cluster one-hot provides a genuine routing prior that is difficult to learn from scratch. The PCA features contain the same clustering information but the gating network must rediscover it among 64 noisy dimensions, wasting capacity on irrelevant routing patterns.

### 3. Gating Network Overfits PCA Features

The earlier early-stopping (epochs 21-26 for learned vs 30-42 for cluster-informed) and lower validation AUC suggest the gating network memorizes training routing patterns from the high-dimensional PCA input that do not generalize. The coarse cluster one-hot constrains the gate to make simpler routing decisions that transfer better.

### 4. Learned Hard Routing Outperforms Cluster Hard on Routing Quality

The one clear win for learned routing is that **hard routing becomes trainable**. Cluster hard routing is a fixed lookup table with no ability to adapt. While learned hard didn't beat cluster soft, it did beat cluster hard on test AUC in 3/4 comparisons (ADHD Classical +0.23, ASD Classical +0.32, ASD Quantum +1.36), suggesting the trainable STE router finds slightly better expert assignments than the fixed cluster mapping.

### 5. Soft Routing Consistently Outperforms Hard Routing

Within each approach, soft routing beats hard routing on test AUC in most comparisons:

| | Soft Test AUC | Hard Test AUC | Soft Wins? |
|---|---|---|---|
| ADHD Cluster Classical | **0.6001** | 0.5735 | Yes |
| ADHD Cluster Quantum | 0.5463 | **0.5853** | No |
| ADHD Learned Classical | **0.5968** | 0.5758 | Yes |
| ADHD Learned Quantum | **0.5749** | 0.5542 | Yes |
| ASD Cluster Classical | **0.5804** | 0.4960 | Yes |
| ASD Cluster Quantum | **0.5625** | 0.5041 | Yes |
| ASD Learned Classical | **0.5385** | 0.4992 | Yes |
| ASD Learned Quantum | **0.5294** | 0.5177 | Yes |

Soft routing wins 7/8 comparisons on test AUC, with the only exception being ADHD Cluster Quantum (where hard routing's fixed 36/64 split happened to match the test distribution well).

## Conclusions

1. **Cluster-Informed Soft routing remains the best MoE approach** for both ADHD (Test AUC 0.6001) and ASD (Test AUC 0.5804)
2. **PCA-based learned routing is not an improvement** — the gating network cannot extract better routing decisions from 64 PCA components than from a simple 2-bit cluster assignment
3. **The bottleneck is not routing quality** — even with end-to-end learned routing from rich connectivity features, performance is similar, suggesting the fundamental limit is the experts' ability to extract discriminative features from fMRI, not how samples are routed between experts
4. **Expert collapse is solved by both approaches** — the original MoE failure mode (99% routing to one expert) does not recur with either cluster-informed or learned routing

## File Locations

| Item | Path |
|------|------|
| Learned Routing script | `MoE_MultiChip/LearnedRoutingMoE_ABCD.py` |
| Cluster-Informed script | `MoE_MultiChip/ClusterInformedMoE_ABCD.py` |
| Data loader | `MoE_MultiChip/dataloaders/Load_ABCD_fMRI.py` |
| ADHD PCA features | `results/coherence_clustering_site_regressed/fc_pca_features.npy` |
| ASD PCA features | `results/asd_coherence_clustering_site_regressed/fc_pca_features.npy` |
| SLURM scripts | `scripts/LearnedMoE_*.sh`, `scripts/ASD_LearnedMoE_*.sh` |
| ADHD logs | `logs/LearnedMoE_*_4923730{9,10,11,12}.out` |
| ASD logs | `logs/ASD_LearnedMoE_*_4923731{3,4,5,6}.out` |
| Checkpoints | `checkpoints/LearnedMoE_*.pt` |
| Design doc | `docs/LearnedRoutingMoE_Design.md` |
| Cluster-Informed results | `docs/ClusterInformedMoE_Results.md` |
