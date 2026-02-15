# Quantum Time-Series Transformer Baseline — Job 48922909

**Date**: 2026-02-14
**SLURM Job**: 48922909
**Status**: COMPLETED
**Wall Time**: 38m 59s (early stopped at epoch 77/100, ~29s/epoch)
**Device**: NVIDIA A100 80GB (CUDA)
**Commit**: `0749ec3` (branch `main`)

## Final Results

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Loss** | 0.4468 | 0.5287 (best) | 0.5426 |
| **Accuracy** | 78.16% | 75.00% (best) | **72.81%** |
| **AUC** | 0.8710 | 0.8241 (best) | **0.8241** |

- **Best Validation AUC**: 0.8241 (epoch 57)
- **Early stopping**: Triggered at epoch 77 (patience 20, no improvement since epoch 57)

## Comparison: MoE vs Quantum Transformer

| Metric | MoE (Job 48920246) | QTS (Job 48922909) | Delta |
|--------|-------------------|-------------------|-------|
| **Test Accuracy** | **72.95%** | 72.81% | +0.14% |
| **Test AUC** | **0.8265** | 0.8241 | +0.0024 |
| **Test Loss** | **0.5342** | 0.5426 | -0.0084 |
| Best Val AUC | **0.8666** | 0.8241 | +0.0425 |
| Best Val Epoch | 6 | 57 | -51 epochs |
| Early Stop Epoch | 26 | 77 | -51 epochs |
| Trainable Params | 420,901 | 4,272 | 98.5x more |
| Wall Time | 1m 42s | 38m 59s | 23x faster |
| Time/Epoch | ~1s | ~29s | 29x faster |

### Key Takeaways

1. **Near-identical test performance**: MoE and QTS achieve essentially the same test accuracy (72.95% vs 72.81%) and AUC (0.8265 vs 0.8241), within noise margins.
2. **MoE converges much faster**: Best val AUC at epoch 6 vs epoch 57. The MoE's 420K parameters learn the task rapidly.
3. **MoE overfits more**: The large train-val gap in MoE (train AUC 0.94 vs val 0.87) vs QTS (train AUC 0.87 vs val 0.82) shows the MoE has excess capacity.
4. **QTS is far more parameter-efficient**: 4,272 parameters vs 420,901 — the quantum model achieves comparable results with **~99% fewer parameters**.
5. **QTS is slower per epoch**: 29s vs 1s due to quantum circuit simulation overhead, despite having far fewer parameters.

## Model Configuration

| Parameter | Value |
|-----------|-------|
| n_qubits | 8 |
| n_layers (ansatz) | 2 |
| degree (QSVT polynomial) | 3 |
| dropout | 0.1 |
| batch_size | 32 |
| lr | 1e-3 |
| weight_decay | 1e-5 |
| patience | 20 |
| seed | 2025 |
| **Trainable parameters** | **4,272** |

### Architecture

```
Input (B, C=64, T=51) → permute → (B, T=51, C=64)
  → + Sinusoidal PE (Vaswani et al., 2017)
  → Linear(64, n_rots=64) + Sigmoid * 2π → [0, 2π]
  → Classical LCU+QSVT simulation (degree-3 polynomial)
  → QFF sim14 circuit → measure PauliX/Y/Z on 8 qubits → 24 features
  → Linear(24, 1) → BCEWithLogitsLoss
```

## Dataset

Same split as MoE experiment for fair comparison:

| Split | Subjects | Trials |
|-------|----------|--------|
| Train | 76 | 3,256 |
| Val | 16 | 684 |
| Test | 17 | 743 |

- PhysioNet Motor Imagery EEG: 64 channels, 51 timesteps @ 16 Hz
- Binary classification: left vs right motor imagery
- Subject-level split, per-channel z-score normalization

## Training Progression (Key Epochs)

| Epoch | Train Loss | Train Acc | Train AUC | Val Loss | Val Acc | Val AUC |
|-------|-----------|-----------|-----------|----------|---------|---------|
| 1 | 0.6925 | 0.5101 | 0.5243 | 0.6898 | 0.5570 | 0.5854 |
| 5 | 0.6889 | 0.5504 | 0.5763 | 0.6874 | 0.5643 | 0.5959 |
| 10 | 0.6827 | 0.5599 | 0.6007 | 0.6728 | 0.5614 | 0.6554 |
| 15 | 0.6375 | 0.6517 | 0.7094 | 0.6118 | 0.6784 | 0.7531 |
| 20 | 0.5820 | 0.6895 | 0.7638 | 0.5469 | 0.7281 | 0.8046 |
| 25 | 0.5550 | 0.7002 | 0.7843 | 0.5354 | 0.7485 | 0.8133 |
| 30 | 0.5326 | 0.7251 | 0.8051 | 0.5504 | 0.7222 | 0.8048 |
| 40 | 0.4901 | 0.7439 | 0.8389 | 0.5738 | 0.7164 | 0.8058 |
| 50 | 0.4747 | 0.7650 | 0.8529 | 0.5383 | 0.7500 | 0.8190 |
| **57** | **0.4718** | **0.7598** | **0.8534** | **0.5287** | **0.7500** | **0.8241** |
| 60 | 0.4546 | 0.7687 | 0.8650 | 0.5809 | 0.7237 | 0.8127 |
| 70 | 0.4560 | 0.7715 | 0.8650 | 0.5512 | 0.7383 | 0.8187 |
| 77 | 0.4468 | 0.7816 | 0.8710 | 0.5982 | 0.7178 | 0.7980 |

## Overfitting Analysis

| Metric | At Best Val (Epoch 57) | At Early Stop (Epoch 77) | Gap Growth |
|--------|----------------------|--------------------------|------------|
| Train Loss | 0.4718 | 0.4468 | -0.0250 |
| Val Loss | 0.5287 | 0.5982 | +0.0695 |
| Train-Val Loss Gap | 0.0569 | 0.1514 | +0.0945 |
| Train AUC | 0.8534 | 0.8710 | +0.0176 |
| Val AUC | 0.8241 | 0.7980 | -0.0261 |

Moderate overfitting. Gap growth much smaller than MoE (0.09 vs 0.29), consistent with fewer parameters.

## Artifacts

| File | Path |
|------|------|
| Checkpoint (best val) | `checkpoints/QTS_PhysioNet_48922909.pt` |
| Training CSV log | `checkpoints/training_logs_48922909.csv` |
| SLURM stdout | `logs/QTS_EEG_48922909.out` |
| SLURM stderr | `logs/QTS_EEG_48922909.err` |

## Suggested Next Steps

1. **Increase quantum model capacity**: Try n_qubits=12, n_layers=3, degree=4 to close any remaining gap
2. **Regularize MoE**: Reduce MoE capacity to match QTS generalization (hidden_dim=32, more dropout)
3. **Hybrid MoE+Quantum**: Replace MoE experts with quantum circuits for parameter-efficient spatial experts
4. **Statistical significance**: Run both models with multiple seeds (2024, 2025, 2026) to determine if differences are significant
