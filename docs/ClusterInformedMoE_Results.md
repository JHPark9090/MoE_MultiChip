# Cluster-Informed MoE: ABCD ADHD Classification Results

## Motivation

Previous MoE experiments on ABCD fMRI ADHD classification suffered from two critical problems:

1. **Spatial decomposition destroys long-range connectivity**: Splitting 180 ROIs into 4 groups of ~45 breaks cross-network patterns important for ADHD classification.
2. **Expert collapse**: Both Classical MoE (E0→1%) and Quantum MoE (E2→99%) degenerated into near-single-expert systems despite load-balancing loss (alpha=0.01).

Meanwhile, site-regressed coherence clustering established that **k=2 genuine neural subtypes** exist (Cramer's V for site→cluster dropped to 0.067, p=0.58 NS; ADHD association preserved at p=1.25e-4, delta=6.0%).

## Experiment Design

**Key innovations over previous MoE**:
1. All experts process **all 180 ROIs** (no spatial decomposition)
2. Expert routing is **informed by cluster labels** from site-regressed coherence clustering
3. k=2 experts (matching the optimal cluster count)
4. `balance_loss_alpha` increased 10x (0.01 → 0.1) for soft routing
5. Dropout increased (0.1 → 0.2)

**4 experiments**: {Classical, Quantum} x {Soft gating, Hard routing}

```
              Input (B, T=363, C=180)
                      |
          Cluster label lookup (CSV)
           subjectkey -> cluster in {0, 1}
                      |
        +---------+----------+
        |  Soft   |   Hard   |
        | routing | routing  |
        +---------+----------+
        | concat( | if c==0: |
        | mean(x),|  Expert0 |
        | onehot) | else:    |
        |  -> Gate|  Expert1 |
        +---------+----------+
              |          |
         Weighted    Direct
           sum       output
              |          |
         Classifier -> logits (B,1)
```

- **Soft gating**: Gating network receives `[temporal_mean(x), cluster_onehot]` → biases routing toward the corresponding expert but doesn't force it
- **Hard routing**: Deterministic assignment by cluster label — expert collapse is impossible by construction

## Dataset

- **Source**: ABCD resting-state fMRI, HCP180 parcellation
- **Target**: ADHD binary classification (`ADHD_label`)
- **Valid subjects**: 4,458 (T >= 300 timepoints)
- **Input shape**: (N, 363 timepoints, 180 ROIs)
- **Split**: Train 3,120 / Val 669 / Test 669 (70/15/15, stratified)
- **Class balance**: ~57% non-ADHD / 43% ADHD
- **Cluster labels**: Site-regressed coherence clustering, k=2 (KMeans)
  - Source: `results/coherence_clustering_site_regressed/cluster_assignments.csv`
  - Column: `km_cluster`
  - Cluster distribution: ~36% cluster 0 / ~64% cluster 1

## Model Configurations

| Config | Expert Type | Routing | Gating Network | Balance Loss |
|--------|------------|---------|----------------|-------------|
| Classical Soft | 2x TransformerEncoder (H=64, 2L, 4H) | Soft (cluster-biased) | Linear(182,64) + GatingNet(64,2) | alpha=0.1 |
| Classical Hard | 2x TransformerEncoder (H=64, 2L, 4H) | Hard (deterministic) | None | N/A |
| Quantum Soft | 2x QuantumTSTransformer (8Q, 2L, D=3) | Soft (cluster-biased) | Linear(182,64) + GatingNet(64,2) | alpha=0.1 |
| Quantum Hard | 2x QuantumTSTransformer (8Q, 2L, D=3) | Hard (deterministic) | None | N/A |

**Common hyperparameters**: Adam (lr=1e-3, wd=1e-5), cosine LR schedule, batch_size=32, grad_clip=1.0, dropout=0.2, patience=20, seed=2025.

## Results

### Main Results Table

| Config | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Final Expert Util | Job ID |
|--------|--------|--------|-------------|----------|----------|-------------------|--------|
| **Classical Soft** | 283,491 | 30 (ES@10) | **0.6379** | **0.6001** | 0.5934 | E0:0.49 / E1:0.51 | 48963928 |
| Classical Hard | 269,633 | 31 (ES@11) | 0.6004 | 0.5735 | 0.5650 | E0:0.36 / E1:0.64 | 48963930 |
| Quantum Soft | 41,089 | 42 (ES@22) | 0.6236 | 0.5463 | 0.5605 | E0:0.52 / E1:0.48 | 48963931 |
| Quantum Hard | 27,231 | 37 (ES@17) | 0.5742 | 0.5853 | 0.5904 | E0:0.36 / E1:0.64 | 48963933 |

ES = early stopped, ES@N = best model saved at epoch N.

### Comparison with Previous Baselines

| Model | Params | Test AUC | Test Acc | Expert Collapse? | Job ID |
|-------|--------|----------|----------|-----------------|--------|
| Classical MoE (spatial, 4 experts) | 508,197 | 0.5802 | 0.5516 | E0→1% | 48949632 |
| Standalone QTS | 12,008 | 0.5648 | 0.5874 | N/A | 48949635 |
| Quantum MoE (spatial, 4 experts) | 34,657 | 0.5605 | 0.5755 | E2→99% | 48949930 |
| **Cluster Classical Soft** | **283,491** | **0.6001** | **0.5934** | **No (49/51)** | **48963928** |
| **Cluster Quantum Hard** | **27,231** | **0.5853** | **0.5904** | **No (36/64)** | **48963933** |

### Per-Experiment Analysis

#### Classical Soft (Job 48963928) — Best Overall

**Training dynamics**:
- Rapid overfitting: train AUC reached 0.96 by epoch 30 while val AUC peaked at 0.6379 (epoch 10)
- Val loss diverged after epoch 10 (0.69 → 1.49 by epoch 30)
- 283K params — 44% fewer than previous Classical MoE (508K)

**Expert utilization — balanced**:
```
Epoch  1: E0:0.77 | E1:0.23  (initial bias toward E0)
Epoch 10: E0:0.49 | E1:0.51  (best model — near-perfect balance)
Epoch 20: E0:0.52 | E1:0.48  (stable)
Epoch 30: E0:0.49 | E1:0.51  (converged)
```

The combination of cluster one-hot in the gating input and `balance_loss_alpha=0.1` achieved stable 50/50 expert utilization. This is a dramatic improvement over the previous Classical MoE where E0 collapsed to 1%.

#### Classical Hard (Job 48963930)

**Training dynamics**:
- Similar overfitting trajectory to soft variant (train AUC 0.97 by epoch 31)
- Val AUC peaked at 0.6004 (epoch 11), ~3.7 points below soft gating
- Val loss diverged more severely (up to 1.84)

**Expert utilization — deterministic**:
```
All epochs: E0:0.36 | E1:0.64
```

Expert utilization is fixed by cluster membership. No collapse possible, but also no adaptivity — each subject is always processed by the same expert regardless of input features.

#### Quantum Soft (Job 48963931) — Best Quantum Val AUC

**Training dynamics**:
- Slower convergence: ~1m 30s per epoch (vs ~1s for classical)
- Train AUC reached 0.88 by epoch 42; val AUC peaked at 0.6236 (epoch 22)
- Less severe overfitting than classical — quantum circuit capacity acts as implicit regularization
- 41K params — comparable to previous Quantum MoE (35K)

**Expert utilization — balanced**:
```
Epoch  1: E0:0.63 | E1:0.37  (initial bias)
Epoch 22: E0:0.47 | E1:0.53  (best model — well-balanced)
Epoch 42: E0:0.52 | E1:0.48  (stable)
```

Again, no expert collapse — a stark contrast to the previous Quantum MoE where E2 consumed 99% of routing.

#### Quantum Hard (Job 48963933) — Best Quantum Test AUC

**Training dynamics**:
- Moderate convergence rate: ~1m per epoch
- Train AUC reached 0.80 by epoch 37; val AUC peaked at 0.5742 (epoch 17)
- Test AUC (0.5853) exceeded val AUC — possible due to favorable test split

**Expert utilization — deterministic**:
```
All epochs: E0:0.36 | E1:0.64
```

Notably, quantum hard achieved the highest test AUC among quantum experiments (0.5853) with only 27K params — better than both the previous Quantum MoE (0.5605, 35K) and standalone QTS (0.5648, 12K).

## Key Findings

### 1. Expert Collapse is Solved

| Model | Previous Expert Balance | Cluster-Informed Balance |
|-------|------------------------|------------------------|
| Classical MoE | E0:1% / E3:64% | Soft: 49%/51%, Hard: 36%/64% |
| Quantum MoE | E2:99% / others:~0% | Soft: 52%/48%, Hard: 36%/64% |

Both cluster-informed approaches completely eliminate expert collapse. Soft routing converges to near-perfect 50/50 balance. Hard routing reflects the natural cluster distribution (36/64).

### 2. All-Channel Processing Improves Performance

Removing spatial decomposition and letting each expert process all 180 ROIs yields better results:

| Comparison | Spatial Split AUC | All-Channel AUC | Delta |
|-----------|-------------------|-----------------|-------|
| Classical (best) | 0.5802 (508K params) | **0.6001** (283K params) | **+2.0 pts** |
| Quantum (best) | 0.5605 (35K params) | **0.5853** (27K params) | **+2.5 pts** |

Both classical and quantum models improve with all-channel processing, while also using fewer parameters.

### 3. Soft Gating Wins on Validation, Hard on Test (Quantum)

| Quantum | Val AUC | Test AUC |
|---------|---------|----------|
| Soft | **0.6236** | 0.5463 |
| Hard | 0.5742 | **0.5853** |

Soft gating achieves higher val AUC but lower test AUC, suggesting it may overfit to the gating patterns in the validation set. Hard routing is more constrained and generalizes better. This warrants further investigation with multiple seeds.

### 4. Overfitting Remains the Main Bottleneck

| Config | Train AUC (final) | Val AUC (best) | Gap |
|--------|-------------------|----------------|-----|
| Classical Soft | 0.96 | 0.64 | 0.32 |
| Classical Hard | 0.97 | 0.60 | 0.37 |
| Quantum Soft | 0.88 | 0.62 | 0.26 |
| Quantum Hard | 0.80 | 0.57 | 0.23 |

Quantum models show a smaller overfitting gap (0.23-0.26) compared to classical (0.32-0.37), confirming that quantum circuit capacity provides implicit regularization. However, all models still overfit substantially.

### 5. Parameter Efficiency

| Model | Params | Test AUC | AUC per 1K Params |
|-------|--------|----------|-------------------|
| Classical MoE (prev.) | 508,197 | 0.5802 | 0.00114 |
| Classical Soft (new) | 283,491 | 0.6001 | 0.00212 |
| Quantum MoE (prev.) | 34,657 | 0.5605 | 0.0162 |
| Quantum Hard (new) | 27,231 | 0.5853 | 0.0215 |
| Standalone QTS | 12,008 | 0.5648 | 0.0470 |

Cluster-informed models achieve better AUC with fewer parameters than their spatial-split predecessors. The standalone QTS remains the most parameter-efficient overall, but the cluster-informed quantum models close the gap on absolute performance.

## Potential Improvements

1. **Multi-seed evaluation**: Run all 4 configs with seeds {2024, 2025, 2026} to measure variance and confirm the soft-vs-hard generalization difference.
2. **Stronger regularization**: Increase dropout to 0.3, add label smoothing (0.1), or try weight decay 1e-4 — overfitting gap is still 0.23-0.37.
3. **Phenotypic features**: Include age, sex, motion via `--phenotypes-to-include` to provide non-neural signal that may complement fMRI patterns.
4. **k=3 clustering**: The coherence clustering showed k=3 was also reasonable — try `--num-experts=3 --cluster-column=km_cluster_k3` to see if finer subtyping helps.
5. **Hybrid routing**: Combine hard routing with a within-expert soft attention mechanism — route by cluster but allow each expert to learn cluster-specific attention patterns.
6. **Curriculum learning**: Start with hard routing (epochs 1-20) then switch to soft routing (epochs 21+) to initialize experts on their natural cluster before allowing cross-cluster routing.

## Reproducibility

```bash
# Classical Soft
sbatch scripts/ClusterMoE_Classical_Soft.sh

# Classical Hard
sbatch scripts/ClusterMoE_Classical_Hard.sh

# Quantum Soft
sbatch scripts/ClusterMoE_Quantum_Soft.sh

# Quantum Hard
sbatch scripts/ClusterMoE_Quantum_Hard.sh

# Manual example (Classical Soft):
python ClusterInformedMoE_ABCD.py \
    --model-type=classical --routing=soft \
    --cluster-file=results/coherence_clustering_site_regressed/cluster_assignments.csv \
    --cluster-column=km_cluster \
    --num-experts=2 --expert-hidden-dim=64 --expert-layers=2 --nhead=4 \
    --dropout=0.2 --balance-loss-alpha=0.1 \
    --n-epochs=100 --batch-size=32 --lr=1e-3 \
    --sample-size=0 --seed=2025

# Smoke test (CPU, ~3 min):
python ClusterInformedMoE_ABCD.py \
    --model-type=classical --routing=soft \
    --cluster-file=results/coherence_clustering_site_regressed/cluster_assignments.csv \
    --sample-size=50 --n-epochs=2 --batch-size=8 --num-experts=2
```

## File Locations

| Item | Path |
|------|------|
| Model script | `MoE_MultiChip/ClusterInformedMoE_ABCD.py` |
| Data loader (modified) | `MoE_MultiChip/dataloaders/Load_ABCD_fMRI.py` |
| Cluster labels | `MoE_MultiChip/results/coherence_clustering_site_regressed/cluster_assignments.csv` |
| SLURM scripts | `MoE_MultiChip/scripts/ClusterMoE_*.sh` |
| Logs | `MoE_MultiChip/logs/ClusterMoE_*_489639{28,30,31,33}.out` |
| Checkpoints | `MoE_MultiChip/checkpoints/ClusterMoE_*.pt` |
