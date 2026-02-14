# MoE_MultiChip Results — Job 48920246

**Date**: 2026-02-14
**SLURM Job**: 48920246
**Status**: COMPLETED
**Wall Time**: 1m 42s (early stopped at epoch 26/100)
**Device**: NVIDIA A100 80GB (CUDA)
**Commit**: `98866ea` (branch `main`)

## Final Results

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Loss** | 0.3043 | 0.4923 (best) | 0.5342 |
| **Accuracy** | 86.58% | 78.65% (best) | **72.95%** |
| **AUC** | 0.9429 | 0.8666 (best) | **0.8265** |

- **Best Validation AUC**: 0.8666 (epoch 6)
- **Early stopping**: Triggered at epoch 26 (patience = 20, no improvement since epoch 6)

## Model Configuration

| Parameter | Value |
|-----------|-------|
| num_experts | 4 |
| expert_hidden_dim | 64 |
| expert_layers | 2 |
| nhead | 4 |
| halo_size | 2 |
| num_classes | 2 |
| dropout | 0.1 |
| gating_noise_std | 0.1 |
| balance_loss_alpha | 0.01 |
| batch_size | 32 |
| lr | 1e-3 |
| weight_decay | 1e-5 |
| lr_scheduler | cosine |
| grad_clip | 1.0 |
| seed | 2025 |
| **Trainable parameters** | **420,901** |

## Dataset

| Split | Subjects | Trials |
|-------|----------|--------|
| Train | 76 | 3,256 |
| Val | 16 | 684 |
| Test | 17 | 743 |

- PhysioNet Motor Imagery EEG: 64 channels, 51 timesteps @ 16 Hz
- Binary classification: left vs right motor imagery
- Subject-level split (no data leakage)
- Per-channel z-score normalization (train mean=-0.0000, std=0.9997)

## Epoch-by-Epoch Training Log

| Epoch | Train Loss | Train Acc | Train AUC | Val Loss | Val Acc | Val AUC | Bal Loss | E0 | E1 | E2 | E3 |
|-------|-----------|-----------|-----------|----------|---------|---------|----------|------|------|------|------|
| 1 | 0.6766 | 0.5743 | 0.6074 | 0.6052 | 0.7061 | 0.7541 | 1.1011 | 0.10 | 0.44 | 0.37 | 0.10 |
| 2 | 0.6073 | 0.6701 | 0.7337 | 0.5873 | 0.7164 | 0.7856 | 1.0689 | 0.02 | 0.38 | 0.45 | 0.15 |
| 3 | 0.5676 | 0.7101 | 0.7782 | 0.5508 | 0.7383 | 0.8080 | 1.0737 | 0.04 | 0.28 | 0.53 | 0.15 |
| 4 | 0.5452 | 0.7282 | 0.7991 | 0.5282 | 0.7588 | 0.8207 | 1.0434 | 0.13 | 0.28 | 0.42 | 0.17 |
| 5 | 0.5139 | 0.7485 | 0.8253 | 0.5143 | 0.7427 | 0.8411 | 1.0562 | 0.30 | 0.21 | 0.38 | 0.11 |
| **6** | **0.4871** | **0.7669** | **0.8461** | **0.4923** | **0.7865** | **0.8666** | **1.1114** | **0.40** | **0.16** | **0.29** | **0.15** |
| 7 | 0.4779 | 0.7657 | 0.8519 | 0.4931 | 0.7558 | 0.8458 | 1.1139 | 0.30 | 0.20 | 0.35 | 0.16 |
| 8 | 0.4619 | 0.7813 | 0.8624 | 0.4827 | 0.7646 | 0.8534 | 1.0576 | 0.29 | 0.12 | 0.43 | 0.16 |
| 9 | 0.4559 | 0.7850 | 0.8670 | 0.5033 | 0.7807 | 0.8626 | 1.0415 | 0.26 | 0.16 | 0.42 | 0.16 |
| 10 | 0.4467 | 0.7826 | 0.8728 | 0.4732 | 0.7749 | 0.8597 | 1.0560 | 0.28 | 0.17 | 0.36 | 0.19 |
| 11 | 0.4274 | 0.8010 | 0.8846 | 0.5279 | 0.7690 | 0.8552 | 1.0421 | 0.20 | 0.21 | 0.43 | 0.16 |
| 12 | 0.4227 | 0.7991 | 0.8867 | 0.4697 | 0.7792 | 0.8587 | 1.0440 | 0.26 | 0.18 | 0.42 | 0.14 |
| 13 | 0.3987 | 0.8166 | 0.9003 | 0.5011 | 0.7822 | 0.8505 | 1.0528 | 0.25 | 0.12 | 0.48 | 0.15 |
| 14 | 0.4075 | 0.8120 | 0.8955 | 0.5025 | 0.7763 | 0.8664 | 1.0506 | 0.21 | 0.13 | 0.45 | 0.21 |
| 15 | 0.4030 | 0.8163 | 0.8982 | 0.4961 | 0.7705 | 0.8552 | 1.0395 | 0.26 | 0.18 | 0.34 | 0.22 |
| 16 | 0.3936 | 0.8194 | 0.9036 | 0.4828 | 0.7851 | 0.8604 | 1.0516 | 0.35 | 0.16 | 0.34 | 0.15 |
| 17 | 0.3704 | 0.8342 | 0.9144 | 0.5105 | 0.7675 | 0.8484 | 1.0564 | 0.31 | 0.16 | 0.38 | 0.15 |
| 18 | 0.3632 | 0.8354 | 0.9187 | 0.5210 | 0.7705 | 0.8555 | 1.0293 | 0.25 | 0.24 | 0.29 | 0.21 |
| 19 | 0.3619 | 0.8369 | 0.9184 | 0.6437 | 0.7237 | 0.8413 | 1.0333 | 0.21 | 0.28 | 0.32 | 0.18 |
| 20 | 0.3579 | 0.8412 | 0.9207 | 0.5360 | 0.7705 | 0.8469 | 1.0310 | 0.21 | 0.23 | 0.36 | 0.20 |
| 21 | 0.3378 | 0.8504 | 0.9295 | 0.5515 | 0.7661 | 0.8499 | 1.0288 | 0.19 | 0.20 | 0.41 | 0.20 |
| 22 | 0.3313 | 0.8443 | 0.9319 | 0.5901 | 0.7515 | 0.8474 | 1.0507 | 0.23 | 0.25 | 0.36 | 0.17 |
| 23 | 0.3175 | 0.8639 | 0.9380 | 0.6669 | 0.7325 | 0.8449 | 1.0302 | 0.20 | 0.25 | 0.33 | 0.22 |
| 24 | 0.3159 | 0.8581 | 0.9395 | 0.6101 | 0.7734 | 0.8380 | 1.0298 | 0.22 | 0.21 | 0.36 | 0.22 |
| 25 | 0.3166 | 0.8606 | 0.9385 | 0.5593 | 0.7588 | 0.8468 | 1.0304 | 0.22 | 0.25 | 0.32 | 0.21 |
| 26 | 0.3043 | 0.8658 | 0.9429 | 0.6020 | 0.7573 | 0.8410 | 1.0378 | 0.19 | 0.29 | 0.28 | 0.23 |

## Expert Utilization Analysis

**Expert channel assignments** (with halo_size=2, effective input 20 channels each):
- **E0**: Channels 0–17 (frontal left)
- **E1**: Channels 14–33 (frontal right / central)
- **E2**: Channels 30–49 (central / parietal)
- **E3**: Channels 46–63 (parietal / occipital)

**Average utilization across all epochs**:
| Expert | Mean Util | Min | Max | Interpretation |
|--------|-----------|-----|-----|----------------|
| E0 | 0.22 | 0.02 | 0.40 | Variable; high early, stabilized mid-range |
| E1 | 0.22 | 0.12 | 0.44 | Initially dominant, declined |
| E2 | 0.37 | 0.28 | 0.53 | Consistently most utilized |
| E3 | 0.18 | 0.10 | 0.23 | Consistently least utilized |

**Key observations**:
- E2 (central/parietal channels) is the most important expert, consistent with motor cortex location for motor imagery tasks
- Load balancing improved over training: initial imbalance (0.02–0.53 range) narrowed to (0.19–0.28) by epoch 26
- Ideal uniform utilization = 0.25 per expert; the gating network converges toward this but retains task-relevant bias

## Overfitting Analysis

| Metric | At Best Val (Epoch 6) | At Early Stop (Epoch 26) | Gap Growth |
|--------|----------------------|--------------------------|------------|
| Train Loss | 0.4871 | 0.3043 | -0.1828 |
| Val Loss | 0.4923 | 0.6020 | +0.1097 |
| Train-Val Loss Gap | 0.0052 | 0.2977 | +0.2925 |
| Train AUC | 0.8461 | 0.9429 | +0.0968 |
| Val AUC | 0.8666 | 0.8410 | -0.0256 |

Clear overfitting after epoch 6. Train loss kept decreasing while val loss increased. The generalization gap widened from 0.005 to 0.30 in loss.

## Artifacts

| File | Path |
|------|------|
| Checkpoint (best val) | `checkpoints/MoE_EEG_48920246.pt` |
| Training CSV log | `checkpoints/training_logs_48920246.csv` |
| SLURM stdout | `logs/MoE_EEG_48920246.out` |
| SLURM stderr | `logs/MoE_EEG_48920246.err` |

## Suggested Next Steps

1. **Regularization** — Increase dropout (0.2–0.3), increase weight_decay (1e-4), or add label smoothing to combat overfitting
2. **Data augmentation** — Time-shift, Gaussian noise, channel dropout on EEG trials
3. **Reduce model capacity** — Try expert_hidden_dim=32 or expert_layers=1 to reduce overfitting
4. **Expert sweep** — Compare num_experts=2,4,8 to find optimal spatial decomposition
5. **Halo sweep** — Compare halo_size=0,2,4 to quantify spatial overlap benefit
6. **Baseline comparison** — Run monolithic Transformer and quantum baselines on same split for fair comparison
