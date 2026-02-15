# PhysioNet EEG Motor Imagery Classification — Model Comparison

## Dataset

- **Source**: PhysioNet EEG Motor Movement/Imagery Dataset
- **Task**: 3-class motor imagery classification (T0: rest, T1: left fist, T2: right fist) → binary (T1 vs T2)
- **Channels**: 64 EEG electrodes
- **Sampling**: 160 Hz, downsampled to 16 Hz (51 timesteps per trial)
- **Subjects**: 109 total — Train 76 / Val 16 / Test 17
- **Trials**: Train 3,256 / Val 684 / Test 743 (after bad epoch rejection)
- **Normalization**: Per-channel z-score (train mean/std)
- **Seed**: 2025

## Models

### 1. Classical MoE (Mixture of Experts)

- **Job ID**: 48920246
- **Architecture**: `DistributedEEGMoE` — 4 classical Transformer experts
  - Expert hidden dim: 64, 2 layers, 4 attention heads
  - Halo size: 2 (channel overlap between experts)
  - Expert channel ranges: [(0,16), (16,32), (32,48), (48,64)]
  - Expert input dim (with halo): 20
- **Training**: Adam (lr=1e-3, wd=1e-5), cosine LR schedule, batch_size=32, grad_clip=1.0
- **Regularization**: dropout=0.1, balance_loss_alpha=0.01, patience=20

### 2. Standalone QTS (Quantum Time-Series Transformer)

- **Job ID**: 48922909
- **Architecture**: `QuantumTSTransformer` — single QTS processing all 64 channels
  - 8 qubits, 2 ansatz layers (sim14), degree 3 (QSVT)
  - Feature dim: 64 (all channels), output dim: 1
- **Training**: Adam (lr=1e-3, wd=1e-5), no LR schedule, batch_size=16, patience=20

### 3. Quantum MoE (Quantum Distributed EEG MoE)

- **Job ID**: 48949928
- **Architecture**: `QuantumDistributedEEGMoE` — 4 QTS experts with classical gating
  - Each expert: 8 qubits, 2 ansatz layers (sim14), degree 3 (QSVT)
  - Expert hidden dim: 64, halo size: 2
  - Expert channel ranges: [(0,16), (16,32), (32,48), (48,64)]
  - Expert input dim (with halo): 20
  - Gating: input-based (temporal mean → Linear → GatingNetwork)
- **Training**: Adam (lr=1e-3, wd=1e-5), cosine LR schedule, batch_size=32, grad_clip=1.0
- **Regularization**: dropout=0.1, balance_loss_alpha=0.01, patience=20

## Results Summary

| Model | Params | Epochs | Best Val AUC | Test AUC | Test Acc | Time/Epoch | Job ID |
|-------|--------|--------|-------------|----------|----------|------------|--------|
| Classical MoE | 420,901 | 26 (ES@6) | 0.8666 | 0.8265 | 0.7295 | ~1-2s | 48920246 |
| Standalone QTS | 4,272 | 77 (ES@57) | 0.8241 | 0.8241 | 0.7281 | ~29s | 48922909 |
| Quantum MoE | 18,561 | 30 (ES@10) | 0.8034 | 0.7868 | 0.7201 | ~2m 6s | 48949928 |

ES = early stopped, ES@N = best model saved at epoch N.

## Detailed Analysis

### Classical MoE (Job 48920246)

**Training dynamics**:
- Fast convergence: best Val AUC 0.8666 reached by epoch 6
- Significant overfitting: train AUC reached ~0.99 while val AUC plateaued at ~0.87
- Val loss diverged after epoch 6, early stopped at epoch 26 (patience=20)
- The 420K parameter count is large relative to the 3,256-sample training set (~129 params/sample)

**Expert utilization**:
- Reasonably balanced early on but showed gradual expert preference over training
- Load-balancing loss (alpha=0.01) partially effective

**Strengths**: Highest test AUC (0.8265), fastest training (~1-2s/epoch)
**Weaknesses**: Massive overparameterization (420K params for 3K samples), overfitting

### Standalone QTS (Job 48922909)

**Training dynamics**:
- Slow, steady convergence: best Val AUC 0.8241 reached at epoch 57
- More gradual overfitting due to tiny model size (4,272 params)
- No LR scheduler used (flat learning rate throughout)
- Trained for 77 epochs before early stopping

**Strengths**: Highest parameter efficiency (0.8241 AUC with only 4,272 params = 99x fewer than Classical MoE)
**Weaknesses**: Slowest convergence (77 epochs), no spatial decomposition

**Parameter efficiency**: Achieved 99.7% of Classical MoE's test AUC with 98.9% fewer parameters.

### Quantum MoE (Job 48949928)

**Training dynamics**:
- Moderate convergence: best Val AUC 0.8034 reached at epoch 10
- Early stopping triggered at epoch 30 (patience=20 from epoch 10)
- Train AUC reached 0.93 by epoch 30 — moderate overfitting gap

**Expert utilization**:
- Started relatively balanced: [E0:0.20, E1:0.22, E2:0.30, E3:0.28] (epoch 1)
- Showed gradual specialization: E0 and E2 grew dominant, E1 shrank
- By epoch 30: [E0:0.38, E1:0.11, E2:0.23, E3:0.28]
- E1 experienced partial collapse (0.22 → 0.11), suggesting its channel group (16-32) may be less informative or redundant

**Strengths**: Moderate parameter count (18.5K), spatial decomposition with quantum processing
**Weaknesses**: Lowest test AUC of the three, slowest per-epoch time (~2m 6s)

## Cross-Model Comparison

### Parameter Efficiency

| Model | Params | Test AUC | AUC per 1K params |
|-------|--------|----------|--------------------|
| Classical MoE | 420,901 | 0.8265 | 0.00196 |
| Standalone QTS | 4,272 | 0.8241 | 0.1930 |
| Quantum MoE | 18,561 | 0.7868 | 0.0424 |

The **Standalone QTS** is the most parameter-efficient model by a wide margin (98x better AUC/param ratio than Classical MoE).

### Compute Cost

| Model | Time/Epoch | Total Training Time | Backend |
|-------|------------|---------------------|---------|
| Classical MoE | ~1-2s | ~40s (26 epochs) | GPU (Transformer) |
| Standalone QTS | ~29s | ~37m (77 epochs) | GPU (quantum sim) |
| Quantum MoE | ~2m 6s | ~63m (30 epochs) | GPU (4x quantum sim) |

Quantum simulation overhead is ~60-120x compared to classical Transformer inference per epoch.

### Generalization Gap (Train AUC - Test AUC)

| Model | Best Train AUC | Test AUC | Gap |
|-------|---------------|----------|-----|
| Classical MoE | ~0.99 | 0.8265 | ~0.16 |
| Standalone QTS | ~0.80 | 0.8241 | ~0.00 |
| Quantum MoE | ~0.93 | 0.7868 | ~0.14 |

The **Standalone QTS** shows essentially zero generalization gap, suggesting its small capacity acts as an implicit regularizer. Both MoE variants overfit more due to their larger capacity.

## Key Takeaways

1. **Classical MoE leads in raw performance** (AUC 0.8265) but uses 420K parameters — severe overparameterization for a 3,256-sample dataset.

2. **Standalone QTS is remarkably parameter-efficient**: achieves 99.7% of Classical MoE's AUC with 98.9% fewer parameters (4,272 vs 420,901). Zero generalization gap.

3. **Quantum MoE underperforms both baselines** on PhysioNet EEG (AUC 0.7868). Possible reasons:
   - **Channel splitting hurts**: Dividing 64 channels into 4 groups of 16+halo may destroy inter-regional correlations that the standalone QTS can capture by processing all 64 channels together.
   - **Expert collapse**: E1 (channels 16-32) dropped to 11% utilization, wasting capacity.
   - **Gating overhead**: The classical gating network adds parameters and complexity without clear benefit when experts process non-overlapping spatial regions.

4. **Spatial decomposition may not suit EEG**: EEG channels have strong inter-regional correlations (e.g., motor imagery involves bilateral cortical activation). Splitting channels into non-overlapping groups (even with halo=2) breaks these long-range dependencies.

5. **For PhysioNet EEG, simpler is better**: The standalone QTS with all 64 channels achieves the best parameter-performance tradeoff.

## Reproducibility

```bash
cd /pscratch/sd/j/junghoon/MoE_MultiChip

# Classical MoE
sbatch scripts/MoE_PhysioNet_EEG.sh

# Standalone QTS
sbatch scripts/QTS_PhysioNet_EEG.sh

# Quantum MoE
sbatch scripts/QuantumMoE_PhysioNet_EEG.sh
```

## ABCD fMRI ADHD Results (Separate Dataset)

Results on the ABCD fMRI ADHD classification task are tracked in [`ABCD_ADHD_results.md`](ABCD_ADHD_results.md). The Quantum MoE ABCD run (job 48949930) is still in progress.

| Model | Params | Test AUC | Test Acc | Job ID |
|-------|--------|----------|----------|--------|
| Classical MoE | 508,197 | 0.5802 | 0.5516 | 48949632 |
| Standalone QTS | 12,008 | 0.5648 | 0.5874 | 48949635 |
| Quantum MoE | ~35K | — (running) | — | 48949930 |

Note: ABCD ADHD classification is a much harder task (near-chance AUC ~0.55-0.62) compared to PhysioNet motor imagery (AUC ~0.78-0.83).
