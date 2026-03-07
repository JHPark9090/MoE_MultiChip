# Circuit MoE v2 — Quantum Fix Results

**Date**: 2026-03-07
**Status**: Complete. All v2 quantum jobs finished.

## Summary

Circuit MoE v2 addresses 4 identified bottlenecks in quantum MoE training that caused the v1 quantum 4-expert model to fail (0.5139 AUC, near chance). The fixes restored functional training, improving 4-expert test AUC by +6.25 points and 2-expert by +1.68 points.

## The 4 Fixes

1. **poly_coeffs = [0, 1, 0, 0]** — Starts with a single LCU round so output depends on input. Random init activated all QSVT iterations, causing gradient vanishing through deep unitary chains.

2. **Uniform mix_coeffs = 1/sqrt(T)** — Equal real-valued weight to all timesteps. Random complex coefficients created destructive interference at initialization.

3. **Expert gradient scaling (x K)** — Compensates 1/K gradient attenuation from soft gating. In 4-expert MoE, each expert receives only ~0.25x the gradient of a single-expert model. A backward hook scales gradients by K without affecting forward pass.

4. **Temporal average pooling (363 -> 36)** — Reduces LCU mix_coeffs from 363 to 36 parameters and QNode batch from B*363 to B*36. Provides ~10x training speedup per epoch.

## Results

### v1 vs v2 Quantum Circuit MoE

| Model | v1 Test AUC | v2 Test AUC | Improvement | v1 Best Val | v2 Best Val |
|-------|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Q 4-expert (adhd_3) | 0.5139 | **0.5764** | **+6.25 pts** | 0.5393 | 0.6042 |
| Q 2-expert (adhd_2) | 0.5615 | **0.5783** | **+1.68 pts** | 0.5968 | 0.5987 |

### Full ADHD Classification Leaderboard

| Model | Params | Test AUC | Test Acc | Type |
|-------|--------|:--------:|:--------:|------|
| Classical SE | 134,849 | **0.6193** | 59.5% | classical |
| Classical Circuit MoE 4-expert | 516,485 | 0.6167 | 62.8% | classical |
| Cluster MoE Soft | 283,491 | 0.6001 | 59.3% | classical |
| Classical Circuit MoE 2-expert | 269,827 | 0.5987 | 58.7% | classical |
| Learned MoE Soft | 287,459 | 0.5968 | 58.5% | classical |
| NAME Classical | 289,906 | 0.5800 | 57.0% | classical |
| **Quantum Circuit MoE v2 2-expert** | **26,771** | **0.5783** | **59.3%** | quantum |
| Quantum SE | 13,648 | 0.5769 | 58.6% | quantum |
| **Quantum Circuit MoE v2 4-expert** | **30,373** | **0.5764** | **57.9%** | quantum |
| Quantum Circuit MoE v1 2-expert | 27,425 | 0.5615 | 58.0% | quantum |
| Quantum Circuit MoE v1 4-expert | 30,373 | 0.5139 | 56.9% | quantum |

### Training Details

#### v2 Quantum 4-Expert (Job 49767122)

| Metric | Value |
|--------|-------|
| Config | adhd_3 (DMN, Executive, Salience, SensoriMotor) |
| Parameters | 30,373 |
| Best Val AUC | 0.6042 (epoch 13) |
| Test AUC | 0.5764 |
| Test Accuracy | 57.9% |
| Epochs trained | 33 / 100 (early stopping, patience=20) |
| Time per epoch | ~2-3 min |
| Temporal pooling | 363 -> 36 (factor 10) |

Training dynamics:
- v1 was stuck at majority-class prediction for 28 epochs (acc = 0.5692, AUC ~0.50)
- v2 begins learning from epoch 1: val AUC 0.5718 at epoch 1, reaches 0.6042 by epoch 13
- Train AUC reaches 0.79 by epoch 33 — clear overfitting, consistent with classical models
- Expert utilization balanced: DMN ~0.24, Executive ~0.25, Salience ~0.27, SensoriMotor ~0.24

#### v2 Quantum 2-Expert (Job 49767123)

| Metric | Value |
|--------|-------|
| Config | adhd_2 (Internal, External) |
| Parameters | 26,771 |
| Best Val AUC | 0.5987 (epoch 22) |
| Test AUC | 0.5783 |
| Test Accuracy | 59.3% |
| Epochs trained | 42 / 100 (early stopping, patience=20) |
| Time per epoch | ~1 min |
| Temporal pooling | 363 -> 36 (factor 10) |

Training dynamics:
- v1 was slow to learn, stuck at 0.5692 acc for first 10 epochs
- v2 begins differentiating by epoch 2 (val AUC 0.5578)
- Train AUC reaches 0.78 by epoch 42 (v1 only reached 0.60 by epoch 36)
- Expert utilization balanced: Internal ~0.50, External ~0.50

## Analysis

### Finding 1: The v2 Fixes Restored Functional Quantum MoE Training

The 4-expert model was completely non-functional in v1 (0.5139 test AUC = chance level for 57/43 class balance). The v2 fixes restored it to 0.5764, comparable to the quantum SE baseline. This validates the bottleneck diagnosis: the failure was due to initialization and gradient flow problems, not a fundamental limitation of quantum MoE.

### Finding 2: Classical-Quantum Gap Analysis

| Configuration | Classical AUC | Quantum AUC | Gap |
|---------------|:------------:|:-----------:|:---:|
| Single Expert | 0.6193 | 0.5769 | 4.2 pts |
| Circuit MoE 4-expert | 0.6167 | 0.5764 | **4.0 pts** |
| Circuit MoE 2-expert | 0.5987 | 0.5783 | **2.0 pts** |

The classical-quantum gap for Circuit MoE 4-expert (4.0 pts) is comparable to the SE gap (4.2 pts). Circuit specialization did not close the gap — the quantum bottleneck persists within each expert regardless of how many ROIs it processes. This is consistent with the quantum bottleneck being intrinsic to the QSVT/LCU architecture's expressivity, not the input compression ratio.

The 2-expert gap (2.0 pts) is notably smaller, but this reflects the classical 2-expert model underperforming (0.5987) rather than the quantum model excelling.

### Finding 3: Quantum Performance Ceiling at ~0.58

All quantum models cluster tightly between 0.5764 and 0.5783 test AUC:

- Quantum SE: 0.5769
- Quantum Circuit MoE v2 4-expert: 0.5764
- Quantum Circuit MoE v2 2-expert: 0.5783

This ~0.58 ceiling appears to be the limit of the QSVT/LCU quantum architecture for ADHD classification on ABCD fMRI, regardless of routing strategy. The bottleneck is the quantum expert itself, not the MoE framework around it.

### Finding 4: Which Fix Mattered Most?

The 4-expert model benefited far more than the 2-expert (+6.25 vs +1.68 pts), suggesting:

- **Fix 3 (gradient scaling)** had the largest impact on the 4-expert model, since 1/4 = 0.25 attenuation is much worse than 1/2 = 0.50
- **Fix 1 (poly_coeffs init)** was critical for both — it determines whether the QSVT chain produces input-dependent output at all
- **Fix 4 (temporal pooling)** enabled more epochs within the same time budget and reduced the parameter space of mix_coeffs
- **Fix 2 (uniform mix_coeffs)** prevented destructive interference at initialization

An ablation study isolating each fix would clarify individual contributions but was not run in this round.

## Implications for Quantum MoE Contributions

### What Can Be Claimed

1. **Diagnostic contribution**: Identified and resolved 4 specific bottlenecks in quantum MoE training (poly_coeffs init, mix_coeffs init, gradient attenuation, temporal complexity). These are general insights for any quantum MoE system.

2. **Circuit specialization works for quantum**: The v2 4-expert model (0.5764) matches quantum SE (0.5769), demonstrating that neuroscience-guided circuit decomposition preserves diagnostic signal through the quantum pipeline.

3. **Training methodology**: The fixes (gradient scaling, initialization strategy, temporal pooling) constitute a practical training recipe for quantum MoE systems that may generalize beyond this specific task.

### What Cannot Be Claimed

1. **No quantum advantage**: All quantum models (0.576-0.578) underperform all classical models (0.580-0.619). The classical-quantum gap remains.

2. **Circuit MoE does not close the classical-quantum gap**: The 4-expert gap (4.0 pts) matches the SE gap (4.2 pts), refuting the compression bottleneck hypothesis for this architecture.

3. **No MoE-specific quantum benefit**: Quantum MoE does not outperform quantum SE. The routing overhead adds complexity without improving performance.

## Configuration

All v2 experiments used identical hyperparameters to v1, plus the 4 fixes:

| Parameter | Value |
|-----------|-------|
| Dataset | ABCD fMRI, ADHD_label, N=4,458 |
| Split | Train: 3,120 / Val: 669 / Test: 669 |
| expert_hidden_dim | 64 |
| n_qubits | 8 |
| n_ansatz_layers | 2 |
| degree | 3 |
| pool_factor | 10 (363 -> 36 timesteps) |
| grad_scale | True |
| poly_coeffs init | [0, 1, 0, 0] |
| mix_coeffs init | uniform(1/sqrt(36)) |
| Dropout | 0.2 |
| Gating noise | 0.1 |
| Balance loss alpha | 0.1 |
| Learning rate | 1e-3 (cosine schedule) |
| Weight decay | 1e-5 |
| Batch size | 32 |
| Patience | 20 |
| Seed | 2025 |

## Files

| File | Description |
|------|-------------|
| `models/CircuitMoE_v2.py` | Fixed quantum Circuit MoE model |
| `CircuitMoE_v2_ABCD.py` | Training script for v2 |
| `scripts/ADHD_CircuitMoE_v2_Quantum_3expert.sh` | SLURM script, 4-expert |
| `scripts/ADHD_CircuitMoE_v2_Quantum_2expert.sh` | SLURM script, 2-expert |
| `logs/CircuitMoE_v2_Q3_ADHD_49767122.out` | 4-expert training log |
| `logs/CircuitMoE_v2_Q2_ADHD_49767123.out` | 2-expert training log |

## Job IDs

| Job | SLURM ID | Status |
|-----|----------|--------|
| v2 Quantum 4-expert | 49767122 | Complete |
| v2 Quantum 2-expert | 49767123 | Complete |
| v1 Quantum 4-expert | 49755381 | Complete |
| v1 Quantum 2-expert | 49755382 | Complete |
| Classical 4-expert | 49731003 | Complete |
| Classical 2-expert | 49731010 | Complete |
