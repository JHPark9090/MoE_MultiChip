# Mixture of Experts (MoE) for ABCD fMRI — Results Summary

## Overview

This document summarizes all MoE experiments on the ABCD fMRI dataset across three phenotypes (ADHD, ASD, Sex) and two routing approaches (Cluster-Informed, Learned Routing). All models use HCP180 parcellation (180 ROIs, 363 timepoints) with site-regressed coherence clustering.

---

## 1. Experimental Setup

### Data

| Phenotype | Task | N Subjects | Label Distribution | Clustering k |
|-----------|------|------------|-------------------|-------------|
| ADHD | Binary classification | 4,458 | ADHD+: ~50% | 2 |
| ASD | Binary classification | 4,992 | ASD+: ~50% | 2 |
| Sex | Binary classification | 9,141 | F: 52.2%, M: 47.8% | 2 |

All datasets use 70/15/15 train/val/test splits with stratification, seed=2025.

### Clustering

Site-regressed coherence features (16,110 upper-triangle features from 180 ROIs) with PCA dimensionality reduction. Optimal k=2 recommended across all three phenotypes.

| Phenotype | PCA Components (95% var) | Silhouette (k=2) | Cluster-Phenotype Association |
|-----------|--------------------------|-------------------|-------------------------------|
| ADHD | 1,386 | 0.0475 | ~5% prevalence difference, p=0.008 |
| ASD | 1,480 | 0.0495 | ~5% prevalence difference, p=0.002 |
| Sex | 1,889 | 0.2850 | ~4.6% prevalence difference, p=1.9e-05 |

### Model Configurations

| Component | Classical | Quantum |
|-----------|-----------|---------|
| Expert type | TransformerEncoder (H=64, 2 layers, 4 heads) | QuantumTSTransformer (8Q, 2L, D=3) |
| Soft params | 283,491 | 41,089 |
| Hard params | 269,633 | 27,231–33,805 |

Shared hyperparameters: dropout=0.2, gating noise=0.1, balance loss alpha=0.1 (soft only), lr=1e-3, wd=1e-5, batch=32, patience=20, cosine LR scheduler.

### Routing Approaches

| Approach | Gating Input | Hard Routing | Gating Network |
|----------|-------------|--------------|----------------|
| **Cluster-Informed** | temporal_mean(x) + cluster_onehot → dim 182 | Deterministic lookup by cluster label (no gating network) | Soft only |
| **Learned Routing** | temporal_mean(x) + PCA_64 → dim 244 | Straight-through estimator (forward=hard, backward=soft) | Both soft and hard |

---

## 2. Results by Phenotype

### 2.1 ADHD Classification

#### Cluster-Informed MoE

| Config | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Test Loss |
|--------|--------|--------|-------------|----------|----------|-----------|
| **Classical Soft** | 283,491 | 30 | **0.6379** | **0.6001** | 0.5934 | 0.6810 |
| Classical Hard | 269,633 | 31 | 0.6004 | 0.5735 | 0.5650 | 0.7259 |
| Quantum Soft | 41,089 | 42 | 0.6236 | 0.5463 | 0.5605 | 0.7244 |
| Quantum Hard | 27,231 | 37 | 0.5742 | 0.5853 | 0.5904 | 0.6708 |

#### Learned Routing MoE

| Config | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Test Loss |
|--------|--------|--------|-------------|----------|----------|-----------|
| Classical Soft | 287,459 | 21 | 0.5914 | 0.5968 | 0.5845 | — |
| Classical Hard | 287,459 | 22 | 0.5752 | 0.5758 | 0.5919 | — |
| Quantum Soft | 45,057 | 24 | 0.5796 | 0.5749 | 0.5770 | — |
| Quantum Hard | 45,057 | 26 | 0.5716 | 0.5542 | 0.5770 | — |

**Best ADHD model**: Cluster-Informed Classical Soft (Test AUC **0.6001**)

---

### 2.2 ASD Classification

#### Cluster-Informed MoE

| Config | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Test Loss |
|--------|--------|--------|-------------|----------|----------|-----------|
| **Classical Soft** | 283,491 | 38 | **0.5642** | **0.5804** | 0.5541 | 0.7309 |
| Classical Hard | 269,633 | 28 | 0.5609 | 0.4960 | 0.5113 | 0.7113 |
| Quantum Soft | 41,089 | 26 | 0.5590 | 0.5625 | 0.5247 | 0.6904 |
| Quantum Hard | 33,805 | 28 | 0.5629 | 0.5041 | 0.5047 | 0.6935 |

#### Learned Routing MoE

| Config | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Test Loss |
|--------|--------|--------|-------------|----------|----------|-----------|
| Classical Soft | 287,459 | 23 | 0.5301 | 0.5385 | 0.5180 | — |
| Classical Hard | 287,459 | 30 | 0.5378 | 0.4992 | 0.5020 | — |
| Quantum Soft | 45,057 | 24 | 0.5300 | 0.5294 | 0.5167 | — |
| Quantum Hard | 45,057 | 25 | 0.5537 | 0.5177 | 0.5140 | — |

**Best ASD model**: Cluster-Informed Classical Soft (Test AUC **0.5804**)

---

### 2.3 Sex Classification

#### Cluster-Informed MoE

| Config | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Test Loss |
|--------|--------|--------|-------------|----------|----------|-----------|
| **Classical Soft** | 283,491 | 38 | **0.8379** | **0.8044** | **0.7318** | 0.7826 |
| Classical Hard | 269,633 | 34 | 0.7753 | 0.7716 | 0.6800 | 0.7434 |
| Quantum Soft | 41,089 | 69 | 0.7249 | 0.7267 | 0.6691 | 0.8022 |
| Quantum Hard | 27,231 | 54 | 0.6707 | 0.6296 | 0.5889 | 0.7050 |

**Best Sex model**: Cluster-Informed Classical Soft (Test AUC **0.8044**)

---

## 3. Cross-Phenotype Comparison

### Best Model per Phenotype (Cluster-Informed Classical Soft)

| Phenotype | Val AUC | Test AUC | Test Acc | N Subjects |
|-----------|---------|----------|----------|------------|
| **Sex** | **0.8379** | **0.8044** | **0.7318** | 9,141 |
| ADHD | 0.6379 | 0.6001 | 0.5934 | 4,458 |
| ASD | 0.5642 | 0.5804 | 0.5541 | 4,992 |

Sex classification is substantially easier than psychiatric diagnoses, consistent with known sex differences in resting-state functional connectivity.

### Difficulty Ranking

1. **Sex** (Test AUC 0.80): Strong signal in fMRI connectivity; well above chance across all configs
2. **ADHD** (Test AUC 0.60): Moderate signal; above chance but challenging
3. **ASD** (Test AUC 0.58): Weakest signal; near-chance for most configs

---

## 4. Routing Comparison: Cluster-Informed vs Learned Routing

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

**Conclusion**: Cluster-Informed routing outperforms Learned Routing on average. The cluster one-hot acts as an effective regularizing prior; the 64-D PCA gating signal leads to gating network overfitting.

---

## 5. Soft vs Hard Routing

### Test AUC: Soft Wins in Most Comparisons

| Phenotype | Config | Soft AUC | Hard AUC | Soft Wins? |
|-----------|--------|----------|----------|-----------|
| ADHD | Cluster Classical | **0.6001** | 0.5735 | Yes |
| ADHD | Cluster Quantum | 0.5463 | **0.5853** | No |
| ADHD | Learned Classical | **0.5968** | 0.5758 | Yes |
| ADHD | Learned Quantum | **0.5749** | 0.5542 | Yes |
| ASD | Cluster Classical | **0.5804** | 0.4960 | Yes |
| ASD | Cluster Quantum | **0.5625** | 0.5041 | Yes |
| ASD | Learned Classical | **0.5385** | 0.4992 | Yes |
| ASD | Learned Quantum | **0.5294** | 0.5177 | Yes |
| Sex | Cluster Classical | **0.8044** | 0.7716 | Yes |
| Sex | Cluster Quantum | **0.7267** | 0.6296 | Yes |

**Soft routing wins 9/10 comparisons** on test AUC. The weighted combination of expert outputs is consistently more robust than hard selection.

---

## 6. Classical vs Quantum

### Test AUC Comparison

| Phenotype | Approach | Classical Soft | Quantum Soft | Classical Wins? |
|-----------|----------|---------------|-------------|----------------|
| ADHD | Cluster | **0.6001** | 0.5463 | Yes |
| ADHD | Learned | **0.5968** | 0.5749 | Yes |
| ASD | Cluster | **0.5804** | 0.5625 | Yes |
| ASD | Learned | **0.5385** | 0.5294 | Yes |
| Sex | Cluster | **0.8044** | 0.7267 | Yes |

Classical experts outperform quantum in all 5 soft-routing comparisons, but with ~7x more parameters (283K vs 41K). The quantum models are significantly more parameter-efficient.

---

## 7. Expert Utilization

All configurations achieve balanced expert utilization — no expert collapse observed:

| Phenotype | Config | Expert 0 | Expert 1 |
|-----------|--------|----------|----------|
| ADHD | Cluster Soft | 49% | 51% |
| ADHD | Cluster Hard | 36% | 64% |
| ASD | Cluster Soft | 52% | 48% |
| ASD | Cluster Hard | 38% | 62% |
| Sex | Cluster Soft | 49% | 51% |
| Sex | Cluster Hard | 62% | 38% |
| Sex | Quantum Soft | 50% | 50% |
| Sex | Quantum Hard | 62% | 38% |

Hard routing reflects the natural cluster distribution; soft routing converges to near-balanced utilization driven by the load-balancing loss.

---

## 8. Key Findings

1. **Cluster-Informed Classical Soft is the best MoE configuration** across all three phenotypes, achieving the highest test AUC in every case.

2. **Sex classification is substantially easier** (0.80 AUC) than psychiatric diagnoses (ADHD 0.60, ASD 0.58), consistent with well-documented sex differences in resting-state functional connectivity.

3. **Soft routing consistently outperforms hard routing** (9/10 comparisons), as the weighted expert combination is more robust than discrete selection.

4. **Cluster-Informed routing outperforms Learned Routing** on average (-2.67 pts val AUC, -0.77 pts test AUC). The simple cluster one-hot acts as a regularizing prior that the gating network cannot improve upon with 64 PCA dimensions.

5. **Expert collapse is fully resolved** — both cluster-informed and learned routing maintain balanced expert utilization, unlike the original MoE that routed 99% of samples to a single expert.

6. **The performance bottleneck is not routing quality** — even with end-to-end learned routing from rich connectivity features, performance does not improve, suggesting the fundamental limit is the experts' ability to extract discriminative features from fMRI time series.

7. **Quantum experts are 7x more parameter-efficient** but underperform classical by ~3-10 pts AUC on average. The quantum-classical gap is smaller for ADHD/ASD (where overall performance is near chance) and larger for sex classification (where there is more signal to capture).

---

## 9. File Locations

| Item | Path |
|------|------|
| Cluster-Informed MoE script | `ClusterInformedMoE_ABCD.py` |
| Learned Routing MoE script | `LearnedRoutingMoE_ABCD.py` |
| Data loader | `dataloaders/Load_ABCD_fMRI.py` |
| Clustering pipeline | `abcd_fc_clustering.py` |
| ADHD clustering results | `results/coherence_clustering_site_regressed/` |
| ASD clustering results | `results/asd_coherence_clustering_site_regressed/` |
| Sex clustering results | `results/sex_coherence_clustering_site_regressed/` |
| SLURM scripts | `scripts/` |
| Checkpoints | `checkpoints/` |
| Logs | `logs/` |
| Learned Routing design doc | `docs/LearnedRoutingMoE_Design.md` |
| Learned Routing results | `docs/LearnedRoutingMoE_Results.md` |
