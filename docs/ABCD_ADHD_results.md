# ABCD fMRI ADHD Classification Results

## Dataset

- **Source**: ABCD resting-state fMRI, HCP180 parcellation
- **Target**: ADHD binary classification (`ADHD_label`)
- **Valid subjects**: 4,458
- **Total fMRI on disk**: 9,406 subjects (all three parcellations: HCP180, HCP360, Schaefer)
- **Input shape**: (N, 363 timepoints, 180 ROIs)
- **Split**: Train 3,120 / Val 669 / Test 669 (70/15/15, stratified)
- **Class balance**: Train [1776 non-ADHD, 1344 ADHD] (~57% / 43%)
- **Normalization**: per-ROI z-score (train-only stats), clipped to +/-10 std

### Subject Filtering Breakdown

| Stage | Count | Dropped | Reason |
|-------|-------|---------|--------|
| Phenotype CSV (total rows) | 11,877 | — | All subjects in `ABCD_phenotype_total.csv` |
| With non-null ADHD_label | 5,781 | -6,096 | Missing ADHD diagnosis |
| With HCP180 fMRI file on disk | 4,550 | -1,231 | No parcellated fMRI file available |
| With T >= 300 timepoints | **4,458** | -92 | Truncated scans (T=20..293), likely due to excessive head motion or early scan termination |

The `min_timepoints=300` QC filter (set in `Load_ABCD_fMRI.py`) removes 92 subjects
whose fMRI scans were severely truncated. Most have very few usable timepoints (median T~113,
smallest T=20 out of expected T=363). Including them would require heavy zero-padding
(up to 94% padding) and introduce noisy/incomplete data. Remaining subjects are center-cropped
or zero-padded to 363 timepoints as needed.

> **Note (2026-02-15)**: Previous results (jobs 48941280 and 48941281) were run on only
> 2,606 subjects due to missing fMRI files on disk. The missing 3,934 subjects have since
> been restored from `/global/cfs/cdirs/m4750/7.ROI/`, bringing the total fMRI files to
> 9,406 and ADHD-labeled subjects with fMRI to 4,458. All previous results have been
> invalidated and deleted. The jobs below are fresh runs on the full dataset.

## Models

### Classical MoE (Mixture of Experts)

- **Job ID**: 48949632
- **Architecture**: `DistributedEEGMoE` — 4 Transformer experts, 64 hidden dim, 2 layers, 4 heads, halo=2
- **Training**: Adam (lr=1e-3, wd=1e-5), cosine LR schedule, batch_size=32, grad_clip=1.0
- **Regularization**: dropout=0.1, balance_loss_alpha=0.01, patience=20

### QTS (Quantum Time-series Transformer)

- **Job ID**: 48949635
- **Architecture**: `QuantumTSTransformer` — 8 qubits, 2 ansatz layers (sim14), degree 3 (QSVT)
- **Training**: Adam (lr=1e-3, wd=1e-5), batch_size=16, patience=20

### Quantum MoE (Quantum Distributed EEG MoE)

- **Job ID**: 48949930
- **Architecture**: `QuantumDistributedEEGMoE` — 4 QTS experts (8Q/2L/D3), 64 hidden dim, halo=2, input-based classical gating
- **Training**: Adam (lr=1e-3, wd=1e-5), cosine LR schedule, batch_size=32, grad_clip=1.0
- **Regularization**: dropout=0.1, balance_loss_alpha=0.01, patience=20
- **Docs**: See `docs/Quantum_Distributed_EEG_MoE.md` for full architecture description

### FC Clustering (Subject-Subtype Analysis)

- **Job ID**: 48949637
- **Purpose**: Identify neural subtypes for MoE expert selection
- **Method**: FC matrices (Fisher-z) -> PCA (95% variance) -> KMeans/Spectral/Agglomerative (k=2..8)
- **Output**: `results/fc_clustering/` — cluster assignments, metrics, visualizations

## Results

| Model | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Time/Epoch | Job ID |
|-------|--------|--------|-------------|----------|----------|------------|--------|
| Classical MoE | 508,197 | 33 (ES@13) | 0.6213 | 0.5802 | 0.5516 | ~1-2s | 48949632 |
| Standalone QTS | 12,008 | 34 (ES@14) | 0.5964 | 0.5648 | 0.5874 | ~1m 9s | 48949635 |
| Quantum MoE | 34,657 | 38 (ES@18) | 0.5607 | 0.5605 | 0.5755 | ~3m 5s | 48949930 |

ES = early stopped, ES@N = best model saved at epoch N.

### Classical MoE Analysis (Job 48949632)

**Training dynamics**:
- Rapid overfitting: train AUC reached 0.99 by epoch 33 while val AUC peaked at 0.6213 (epoch 13)
- Val loss diverged sharply after epoch 10 (0.69 → 2.01 by epoch 32)
- The 508K parameter count is excessive for the 3,120-sample training set (ratio ~163 params/sample)

**Expert utilization**:
- Significant expert collapse over training: E0 dropped from 16% to 1%, E3 dominated up to 64%
- By late training, E1+E2+E3 handled ~99% of routing, E0 was nearly unused
- Load-balancing loss (alpha=0.01) was insufficient to prevent collapse

**Epoch-by-epoch expert utilization**:
```
Epoch  1: E0:0.16 | E1:0.22 | E2:0.25 | E3:0.37  (relatively balanced)
Epoch 13: E0:0.20 | E1:0.14 | E2:0.10 | E3:0.56  (best model, E3 dominant)
Epoch 33: E0:0.01 | E1:0.43 | E2:0.33 | E3:0.23  (E0 collapsed)
```

### Standalone QTS Analysis (Job 48949635)

**Training dynamics**:
- Slower overfitting than Classical MoE due to much smaller capacity (12K vs 508K params)
- Train AUC reached 0.80 by epoch 34 while val AUC peaked at 0.5964 (epoch 14)
- Val loss remained more stable (0.67 → 0.78), less severe divergence than MoE
- No learning rate scheduler used (unlike MoE's cosine schedule)

**Comparison to Classical MoE**:
- QTS achieved comparable test AUC (0.5648 vs 0.5802) with 42x fewer parameters
- QTS had higher test accuracy (0.5874 vs 0.5516), suggesting better calibration
- QTS generalized slightly better relative to its val AUC (test/val ratio: 0.95 vs 0.93)

### Quantum MoE Analysis (Job 48949930)

**Training dynamics**:
- Very slow initial learning: accuracy was stuck at 56.92% (majority class) for the first 13 epochs
- First sign of learning at epoch 14 (train AUC 0.5375, acc 56.96%)
- Best Val AUC 0.5607 reached at epoch 18; train AUC was only 0.5317 at that point
- After epoch 26, train AUC rose rapidly (0.60 → 0.77 by epoch 37) while val AUC collapsed
- Early stopped at epoch 38 (patience=20 from epoch 18)

**Expert utilization — severe collapse**:
- Started balanced: E0:0.22, E1:0.25, E2:0.26, E3:0.27 (epoch 1)
- E2 gradually dominated as training progressed
- Complete collapse by epoch 38: E0:0.00, E1:0.01, E2:0.99, E3:0.00
- The model effectively degenerated into a single-expert system

**Epoch-by-epoch expert utilization**:
```
Epoch  1: E0:0.22 | E1:0.25 | E2:0.26 | E3:0.27  (balanced)
Epoch 18: E0:0.26 | E1:0.31 | E2:0.25 | E3:0.19  (best model, still reasonable)
Epoch 26: E0:0.17 | E1:0.23 | E2:0.48 | E3:0.12  (E2 starting to dominate)
Epoch 33: E0:0.00 | E1:0.09 | E2:0.91 | E3:0.00  (near-total collapse)
Epoch 38: E0:0.00 | E1:0.01 | E2:0.99 | E3:0.00  (single expert)
```

**Comparison to other models**:
- Test AUC (0.5605) is comparable to QTS (0.5648) but slightly lower than Classical MoE (0.5802)
- 34.7K params — 14.7x fewer than Classical MoE (508K), 2.9x more than QTS (12K)
- Expert collapse was even more severe than Classical MoE (99% vs 64% max expert share)
- Load-balancing loss (alpha=0.01) completely failed to prevent collapse

### Cross-Model Comparison

| Metric | Classical MoE | Standalone QTS | Quantum MoE |
|--------|--------------|----------------|-------------|
| **Test AUC** | **0.5802** | 0.5648 | 0.5605 |
| **Test Acc** | 0.5516 | **0.5874** | 0.5755 |
| Params | 508,197 | **12,008** | 34,657 |
| AUC per 1K params | 0.00114 | **0.0470** | 0.0162 |
| Expert collapse | E0→1% | N/A | E2→99% |
| Time/Epoch | ~1-2s | ~1m 9s | ~3m 5s |

**Key takeaways**:
1. All three models perform near-chance on ABCD ADHD (AUC 0.56-0.58), suggesting the task is inherently difficult with resting-state fMRI alone
2. **Standalone QTS** is the most parameter-efficient: best AUC/param ratio by far
3. **Expert collapse is a critical problem** for both MoE variants, especially Quantum MoE where a single expert (E2, ROIs 90-135) captured 99% of routing
4. **Spatial decomposition hurts** on this task: breaking 180 ROIs into 4 groups destroys long-range functional connectivity patterns important for ADHD classification
5. The balance_loss_alpha=0.01 is insufficient — need 0.05-0.1 or alternative strategies (e.g., hard expert dropout, capacity constraints)

## Previous Results (INVALIDATED)

The following results were obtained on a **partial dataset (2,606 / 4,550 subjects)** and
are no longer valid. They are kept here for reference only.

| Model | Params | Epochs | Best Val AUC | Test AUC | Test Acc | N subjects | Job ID |
|-------|--------|--------|-------------|----------|----------|------------|--------|
| MoE   | 508K   | 59     | 0.6057      | 0.5553   | 0.5396   | 2,606      | 48941280 |
| QTS   | 12K    | 63     | 0.5928      | 0.5862   | 0.5754   | 2,606      | 48941281 |

Key observations from partial-dataset runs (may or may not hold on full dataset):
- Both models showed overfitting (train AUC >> val AUC)
- MoE had expert collapse (E2 and E3 dominated, E0 and E1 underutilized)
- QTS generalized slightly better than MoE despite being 42x smaller

## Potential Improvements

1. **Expert balancing** (critical): Increase `balance-loss-alpha` to 0.05-0.1 — both MoE variants suffered severe expert collapse (Classical: E0→1%, Quantum: E2→99%). Consider hard expert dropout or capacity constraints
2. **Regularization**: Increase dropout (0.2-0.3), weight decay (1e-4), or add label smoothing — all three models overfit
3. **Phenotypic features**: Include age, sex, motion parameters via `phenotypes_to_include` — resting-state fMRI alone yields near-chance ADHD classification
4. **Subject-aware MoE**: Use FC/coherence clustering results to inform expert specialization (route subjects by neural subtype instead of spatial channel partition)
5. **Reduced model capacity**: For Classical MoE, try fewer experts (2-3), smaller hidden dim (32), or fewer layers (1) — 508K params is excessive for 3,120 training samples
6. **All-channel quantum processing**: Quantum MoE's spatial decomposition hurts performance. Consider experts that each process all 180 ROIs but with different circuit configurations, or fewer experts (2) with larger channel groups
7. **Data augmentation**: Time-series jittering, temporal cropping, or mixup
8. **LR scheduler for QTS**: The standalone QTS did not use a cosine LR schedule; adding one may improve convergence

## Reproducibility

```bash
# Classical MoE
sbatch scripts/MoE_ABCD_ADHD.sh
# or manually:
python DistributedEEGMoE.py \
    --dataset=ABCD_fMRI --parcel-type=HCP180 --target-phenotype=ADHD_label \
    --num-experts=4 --expert-hidden-dim=64 --expert-layers=2 --nhead=4 \
    --halo-size=2 --num-classes=2 --batch-size=32 --lr=1e-3 \
    --sample-size=0 --seed=2025

# Standalone QTS
sbatch scripts/QTS_ABCD_ADHD.sh
# or manually:
python QTSTransformer_PhysioNet_EEG.py \
    --dataset=ABCD_fMRI --parcel-type=HCP180 --target-phenotype=ADHD_label \
    --n-qubits=8 --n-layers=2 --degree=3 --batch-size=16 --lr=1e-3 \
    --sample-size=0 --seed=2025

# Quantum MoE
sbatch scripts/QuantumMoE_ABCD_ADHD.sh
# or manually:
python QuantumDistributedEEGMoE.py \
    --dataset=ABCD_fMRI --parcel-type=HCP180 --target-phenotype=ADHD_label \
    --num-experts=4 --expert-hidden-dim=64 --n-qubits=8 --n-ansatz-layers=2 \
    --degree=3 --halo-size=2 --num-classes=2 --batch-size=32 --lr=1e-3 \
    --sample-size=0 --seed=2025

# FC Clustering
sbatch scripts/abcd_fc_clustering.sh
# or manually:
python abcd_fc_clustering.py \
    --data-root=/pscratch/sd/j/junghoon/ABCD \
    --output-dir=./results/fc_clustering \
    --seed=2025 --max-k=8 --use-fisher-z
```

## Data Provenance

| Item | Path |
|------|------|
| Phenotype CSV | `/pscratch/sd/j/junghoon/ABCD/ABCD_phenotype_total.csv` |
| fMRI data | `/pscratch/sd/j/junghoon/ABCD/sub-*/` (9,406 subjects, 3 parcellations each) |
| Source backup | `/global/cfs/cdirs/m4750/7.ROI/` (original parcellated data) |
| Logs | `MoE_MultiChip/logs/` |
| Checkpoints | `MoE_MultiChip/checkpoints/` |
| Clustering output | `MoE_MultiChip/results/fc_clustering/` |
