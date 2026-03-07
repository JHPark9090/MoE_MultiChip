# Quantum Circuit MoE — Results, Bottleneck Analysis, and Solutions

**Date**: 2026-03-07
**Status**: Classical and Quantum Circuit MoE results complete. Fixes designed, ready to implement.

## 1. Complete Results

### All Models — ADHD Classification (ABCD fMRI, N=4,458)

| Model | Params | Test AUC | Test Acc |
|-------|--------|:---:|:---:|
| **Classical SE** | **134,849** | **0.6193** | **59.5%** |
| Classical Circuit MoE 4-expert | 516,485 | 0.6167 | 62.8% |
| Cluster MoE Soft | 283,491 | 0.6001 | 59.3% |
| Classical Circuit MoE 2-expert | 269,827 | 0.5987 | 58.7% |
| Learned MoE Soft | 287,459 | 0.5968 | 58.5% |
| NAME Classical | 289,906 | 0.5800 | 57.0% |
| Quantum SE | 13,648 | 0.5769 | 58.6% |
| Quantum Circuit MoE 2-expert | 27,425 | 0.5615 | 58.0% |
| **Quantum Circuit MoE 4-expert** | **31,681** | **0.5139** | **59.2%** |

### Classical vs. Quantum Gap

| Model | Classical AUC | Quantum AUC | Gap | Compression |
|-------|:---:|:---:|:---:|:---:|
| All-channel SE | 0.6193 | 0.5769 | 4.2 pts | 180 to 64 (2.8:1) |
| Circuit MoE 2-expert | 0.5987 | 0.5615 | 3.7 pts | 84-96 to 64 (~1.4:1) |
| Circuit MoE 4-expert | 0.6167 | 0.5139 | **10.3 pts** | 29-55 to 64 (expansion) |

### Training Dynamics

| Model | Epochs to first learning | Best Val AUC | Final Test AUC |
|-------|:---:|:---:|:---:|
| Classical 4-expert | 1 | 0.6152 (epoch 9) | 0.6167 |
| Classical 2-expert | 1 | 0.6389 (epoch 11) | 0.5987 |
| Quantum 2-expert | ~7 | 0.5968 (epoch 16) | 0.5615 |
| **Quantum 4-expert** | **~28** | 0.5794 (epoch 35) | 0.5139 |

The quantum 4-expert model was stuck predicting the majority class (accuracy = 56.92%, AUC ~ 0.50) for the first 28 epochs. Classical 4-expert started learning immediately.

## 2. Honest Assessment of Quantum MoE Contributions

The `docs/QuantumMoE_Contributions.md` document proposed 4 contributions. Here is their status after evaluating all results:

### Contribution 1: Domain-Informed Partitioning Addressing Multi-Chip Limitations

**Verdict: Methodologically sound. Empirically a lateral move.**

The architectural argument is valid — domain-informed splitting is objectively more principled than index-based splitting for brain data, uses fewer partitions (4 vs 272), and recovers between-circuit information through learned gating. Classical 4-expert Circuit MoE (0.6167) matches the SE baseline (0.6193), confirming that the circuit decomposition preserves diagnostic information.

However, it does not improve over SE. This falls in the "neutral" outcome from the plan document: "demonstrates circuit-specialized processing without performance loss."

**Claimable as**: a methodological framework contribution. **Not claimable as**: a performance improvement.

### Contribution 2: First Quantum MoE Framework for Neuroimaging

**Verdict: The framework works mechanically, but results do not demonstrate practical value.**

Soft gating, load balancing, and expert specialization all function correctly with quantum circuits — this is verified. However:

- Quantum Circuit MoE 4-expert: **0.5139 test AUC** — barely above chance, far worse than Quantum SE (0.5769)
- Quantum Circuit MoE 2-expert: **0.5615 test AUC** — also worse than Quantum SE (0.5769)
- The quantum 4-expert model was stuck near random for 28 epochs

Quantum experts perform *worse* in the MoE framework than as a single expert. The practical significance of the contribution is undermined.

**Claimable as**: a framework novelty ("first to do X"). **Weakened by**: degraded quantum performance within the framework.

### Contribution 3: Parameter Efficiency (10x Fewer Params, ~93% Performance)

**Verdict: Holds for single-expert and 2-expert settings. Broken for 4-expert.**

| Setting | Classical | Quantum | Ratio | Q/C Performance |
|---------|:---:|:---:|:---:|:---:|
| SE | 134,849 | 13,648 | 10x | 93% |
| Circuit MoE 2-expert | 269,827 | 27,425 | 10x | 94% |
| **Circuit MoE 4-expert** | **516,485** | **31,681** | **16x** | **83%** |

The 4-expert quantum result (0.5139) breaks the "93% at 10x fewer params" narrative. With 4 separate quantum circuits, the optimization problem becomes too hard, and parameter efficiency becomes meaningless if the model cannot learn.

**Claimable for**: single-expert and 2-expert configurations. **Not claimable for**: 4-expert Circuit MoE.

### Contribution 4: Circuit Specialization Reduces the Quantum Bottleneck

**Verdict: Rejected by the data.**

The hypothesis was: if pre-quantum compression is the bottleneck for quantum experts, reducing compression (180 to 64 down to 29-55 to 64) should narrow the classical-quantum gap.

| Model | C-Q Gap | Compression |
|-------|:---:|:---:|
| SE (baseline) | 4.2 pts | 2.8:1 |
| Circuit MoE 2-expert | 3.7 pts | ~1.4:1 |
| **Circuit MoE 4-expert** | **10.3 pts** | expansion |

The 4-expert model, with the most compression reduction, shows the largest classical-quantum gap (10.3 pts) — more than double the SE gap. The opposite of the prediction. The compression bottleneck was not the limiting factor; the real bottleneck is optimization difficulty.

**This hypothesis is falsified.**

### What Can Be Honestly Claimed

1. **Classical Circuit MoE with neuroscience-guided partitioning matches SE performance** (0.6167 vs 0.6193) — domain-informed circuit decomposition preserves diagnostic signal while enabling interpretable expert specialization.
2. **Quantum parameter efficiency holds for single-expert models** — 10x fewer parameters at 93% performance across 3 phenotypes.
3. **The MoE framework is mechanically compatible with quantum circuits** — an engineering result, not a performance result.
4. **The 4-circuit decomposition is better than the 2-circuit decomposition** — supported by Cortese 2012 and Feng 2024.

### What Cannot Be Claimed

1. Quantum MoE improves over single-expert quantum — it is worse (0.51-0.56 vs 0.58).
2. Circuit specialization specifically benefits quantum experts — it hurts them (10.3 pt gap vs 4.2 pt gap).
3. The compression bottleneck hypothesis — falsified.
4. Any quantum advantage or quantum-specific benefit from MoE — no evidence.

## 3. Bottleneck Analysis

Four distinct bottlenecks were identified by tracing gradient magnitudes through the quantum 4-expert Circuit MoE and comparing to the Quantum SE baseline.

### Bottleneck 1: Gradient Attenuation Through Soft Gating

**Mechanism**: In the MoE forward pass, the gradient to each expert is multiplied by its gating weight:

```
dL/dh_i = dL/d(weighted) * gate_weight_i
```

With 4 experts and near-uniform gating (gate_weight ~ 0.25), each expert receives only ~25% of the gradient signal.

**Measured gradient comparison (SE vs MoE Expert 0)**:

| Parameter | SE grad norm | MoE grad norm | Attenuation |
|-----------|:---:|:---:|:---:|
| `feature_projection.weight` | 0.9167 | 0.1373 | 6.7x |
| `poly_coeffs` (QSVT) | 0.000927 | 0.000214 | 4.3x |
| `mix_coeffs` (LCU) | 0.01978 | 0.005367 | 3.7x |
| `qff_params` | 0.008937 | 0.003389 | 2.6x |
| `output_ff.weight` | 0.04531 | 0.01158 | 3.9x |

**Why classical experts survive**: TransformerEncoder has LayerNorm (stabilizes gradients), residual connections (provides gradient shortcuts), and multi-head attention (diverse gradient signals). These architectural features make classical experts robust to gradient dilution. Classical 4-expert started learning at epoch 1 despite the same 4x gating attenuation.

**Impact**: Directly explains the "stuck at random" period:
- 4-expert quantum (25% gradient): stuck for 28 epochs
- 2-expert quantum (50% gradient): stuck for ~7 epochs
- SE quantum (100% gradient): no stuck period

### Bottleneck 2: QSVT Polynomial Parameters Have Near-Zero Gradients

**Mechanism**: The `poly_coeffs` (4 parameters controlling the QSVT polynomial) sit deep in the gradient chain:

```
loss -> classifier -> weighted_sum (*0.25) -> output_ff -> QFF measurement
-> state normalization -> QSVT polynomial -> LCU (363 states) -> sim14 circuit
-> feature_projection
```

Each stage can reduce gradient magnitude. With random initialization `poly_coeffs = [0.65, 0.63, 0.74, 0.61]`, all 3 QSVT polynomial iterations are active simultaneously, creating a deep, interacting gradient path.

**Measured poly_coeffs gradients per expert**:

| Expert | poly_coeffs grad norm |
|--------|:---:|
| Salience (29 ROIs) | **0.000015** (effectively zero) |
| DMN (55 ROIs) | 0.000214 |
| Executive (50 ROIs) | 0.000462 |
| SensoriMotor (46 ROIs) | 0.001157 |

At lr=1e-3, the Salience expert's poly_coeffs update per step is ~1.5e-8 — the QSVT polynomial is frozen. Without meaningful QSVT updates, the quantum circuit applies a random transformation to the input states.

**Root cause**: Random initialization activates all polynomial terms at once. The gradient must flow backward through 3 nested LCU iterations, each involving 363 quantum state evolutions through the sim14 circuit. This compounds vanishing gradients.

### Bottleneck 3: Cold Start — Uniform Experts + Uniform Gating

**Mechanism**: At initialization:
- All 4 experts produce nearly identical output statistics (mean ~ 0, std ~ 0.12, norm ~ 1.0)
- Gate weights are near-maximum-entropy (1.3838 vs 1.3863 max)
- The weighted combination produces logits ~ 0.03 (sigmoid ~ 0.508)

This creates a vicious cycle:
1. Experts produce similar outputs -> classifier cannot differentiate classes -> loss gradients are small
2. Small loss gradients * 0.25 gating -> expert updates are tiny -> experts stay similar
3. Similar expert outputs -> gating has no reason to differentiate -> stays uniform
4. Return to step 1

Classical experts break this cycle within 1-2 epochs due to their smooth optimization landscape. Quantum experts with 8 qubits and 2 ansatz layers cannot.

### Bottleneck 4: 363 Independent LCU Coefficients Per Expert

**Mechanism**: Each expert has `mix_coeffs` of shape `(363,)` — one complex coefficient per fMRI timestep. The LCU linearly combines 363 evolved quantum states:

```python
lcs = einsum('bti,bt->bi', evolved_states, coeffs)  # sum over 363 timesteps
```

**Computational cost**: The QNode processes `B * 363 = 11,616` state vectors per QSVT iteration. With 4 experts and 3 QSVT iterations each, this is 12 QNode calls at batch size 11,616 per training step — approximately 30 seconds per batch (vs ~1s for classical).

**Optimization cost**: 4 experts * 363 complex LCU coefficients = 1,452 complex parameters for temporal aggregation alone. Each set must independently discover which timesteps are informative for its brain circuit, while receiving only ~25% of the gradient signal.

| Metric | Value |
|--------|-------|
| QNode batch size per expert | 11,616 (B=32 * T=363) |
| QNode calls per training step | 12 (4 experts * 3 QSVT iterations) + 4 QFF = 16 |
| mix_coeffs total | 1,452 complex params |
| Training time per epoch | ~3 minutes |
| Classical epoch time | ~2 seconds |

## 4. Solutions

### Solution 1: Expert Gradient Scaling Hook

**Addresses**: Bottleneck 1 (gradient attenuation through gating)

Register a backward hook on each expert's output that compensates for the gating attenuation:

```python
# In CircuitMoE.forward():
h = expert(x_subset)
h.register_hook(lambda grad, n=num_experts: grad * n)
expert_outputs.append(h)
```

Forward pass is unchanged — gating still produces a properly weighted combination. But each expert receives gradient as if it were the sole contributor: `gate_weight_i * num_experts = 0.25 * 4 = 1.0`.

**Measured effect**: output_ff gradient improves 2.2-3.4x across all experts, matching SE-level gradients.

### Solution 2: QSVT `[0, 1, 0, 0]` Initialization

**Addresses**: Bottleneck 2 (poly_coeffs near-zero gradients)

Initialize the QSVT polynomial as the simplest non-trivial case — a single LCU round:

```python
with torch.no_grad():
    expert.poly_coeffs.data = torch.tensor([0.0, 1.0, 0.0, 0.0])
```

With `[0, 1, 0, 0]`:
- `P(U) = 0 * I + 1 * U + 0 * U^2 + 0 * U^3 = U` (single LCU application)
- Only one round of sim14 + LCU mixing is active
- Gradients for `poly_coeffs[0], [2], [3]` represent a clear signal: "should I add the identity term / higher-order polynomial terms?"
- The model starts simple and learns to add complexity

**Why not `[1, 0, 0, 0]`?** This produces `P(U) = I` — the output equals the base state `|0...0>` regardless of input. The LCU, feature_projection, and mix_coeffs all receive zero gradient because they are never used.

**Measured effect**:

| Expert | Original grad | Fixed grad | Improvement |
|--------|:---:|:---:|:---:|
| DMN | 0.000214 | 1.068 | 4,990x |
| Executive | 0.000462 | 0.254 | 550x |
| Salience | 0.000015 | 1.492 | 99,466x |
| SensoriMotor | 0.001157 | 0.540 | 467x |

The QSVT polynomial goes from being effectively frozen to having the strongest gradient of any parameter.

### Solution 3: Uniform `mix_coeffs` Initialization

**Addresses**: Bottleneck 3 (cold start)

Initialize the LCU temporal mixing coefficients as uniform (equal weight per timestep):

```python
with torch.no_grad():
    expert.mix_coeffs.data = torch.ones(n_timesteps, dtype=torch.complex64) / n_timesteps
```

Random complex coefficients can cause destructive interference among LCU-combined states, producing near-zero output norms. Uniform initialization makes the LCU start as a simple temporal average, then learn selective temporal weighting.

Combined with Solutions 1 and 2, this breaks the cold start cycle: experts produce diverse, input-dependent outputs from the start (different ROI inputs + different feature_projection weights), and gradients are strong enough for rapid differentiation.

### Solution 4: Classical Temporal Pooling Before Quantum Processing

**Addresses**: Bottleneck 4 (363 timesteps)

Add average pooling to reduce timesteps before the quantum circuit:

```python
# In CircuitMoE.__init__():
self.pool_factor = 10
effective_timesteps = time_points // self.pool_factor  # 363 // 10 = 36

# Create experts with reduced timesteps
QuantumTSTransformer(n_timesteps=effective_timesteps, ...)

# In CircuitMoE.forward():
x_pooled = F.avg_pool1d(x_subset.permute(0, 2, 1), kernel_size=10, stride=10)
h = expert(x_pooled)  # (B, n_rois, 36) instead of (B, n_rois, 363)
```

**Neurophysiological justification**: ABCD fMRI has TR=0.8s, so 363 timesteps = 290s. Pooling by factor 10 yields bins of 8 seconds. ADHD-relevant brain network dynamics operate at timescales of 10-30 seconds (Koirala 2024), so 8-second bins preserve relevant temporal structure while removing high-frequency noise.

**Measured effect**:

| Metric | Before (363) | After (36) | Improvement |
|--------|:---:|:---:|:---:|
| QNode batch size | 11,616 | 1,152 | 10x smaller |
| mix_coeffs params | 363 complex | 36 complex | 10x fewer |
| Forward time (1 expert) | 4.27s | 0.29s | 15x faster |
| Backward time (1 expert) | 3.15s | 0.26s | 12x faster |
| 4-expert batch time | ~30s | ~2.2s | 14x faster |
| Estimated epoch time | ~3 min | ~13s | ~14x faster |

### Combined Effect

All four fixes applied together:

| Parameter | Original MoE | All Fixes | Improvement |
|-----------|:---:|:---:|:---:|
| poly_coeffs (DMN) | 0.000214 | 1.068 | 4,990x |
| poly_coeffs (Salience) | 0.000015 | 1.492 | 99,466x |
| output_ff (avg) | 0.011 | 0.036 | 3.3x |
| qff_params (avg) | 0.003 | 0.007 | 2.5x |
| Training speed | ~3 min/epoch | ~13s/epoch | 14x |
| mix_coeffs per expert | 363 complex | 36 complex | 10x fewer |

### Implementation Summary

| Fix | Lines Changed | Location |
|-----|:---:|----------|
| `poly_coeffs = [0, 1, 0, 0]` | 2 | `CircuitMoE.__init__` |
| `mix_coeffs = uniform` | 2 | `CircuitMoE.__init__` |
| `h.register_hook(grad * K)` | 1 | `CircuitMoE.forward` |
| `avg_pool1d(kernel=10)` | 3 | `CircuitMoE.__init__` + `forward` |

Total: ~8 lines of new code in `models/CircuitMoE.py`.

## 5. Classical Dimension Reduction Analysis

### Background

All variational quantum models in this project use a single `nn.Linear` layer to project input features into quantum rotation angles before the sim14 ansatz. With 8 qubits and 2 ansatz layers, this produces `n_rots = 4 × 8 × 2 = 64` angles. There is no hidden layer or nonlinearity before the sigmoid — the linear projection is the sole classical bottleneck.

### Per-Model Compression Comparison

In SE, Cluster MoE, and Learned MoE, every expert sees all 180 ROIs:

| Model | Expert Input | Projection | Compression Ratio |
|-------|:---:|-----------|:---:|
| Single Expert | 180 | `Linear(180, 64)` | 2.81:1 |
| Cluster MoE (each expert) | 180 | `Linear(180, 64)` | 2.81:1 |
| Learned MoE (each expert) | 180 | `Linear(180, 64)` | 2.81:1 |
| Circuit MoE — DMN | 55 | `Linear(55, 64)` | 0.86:1 (expansion) |
| Circuit MoE — Executive | 50 | `Linear(50, 64)` | 0.78:1 (expansion) |
| Circuit MoE — Salience | 29 | `Linear(29, 64)` | 0.45:1 (expansion) |
| Circuit MoE — SensoriMotor | 46 | `Linear(46, 64)` | 0.72:1 (expansion) |
| Circuit MoE — Internal (2-expert) | 84 | `Linear(84, 64)` | 1.31:1 |
| Circuit MoE — External (2-expert) | 96 | `Linear(96, 64)` | 1.50:1 |

Circuit MoE is the only architecture that reduces the per-expert compression. All other quantum models use `Linear(180, 64)` regardless of routing or gating.

### Structural Advantages of Circuit MoE

1. **Per-expert compression is eliminated (4-expert) or reduced (2-expert).** In the 4-expert config, every expert expands its input (ratios < 1:1) — the linear projection is a feature transform, not a lossy bottleneck.

2. **Total information bandwidth is K× higher.** The 4-expert model maps 180 ROIs into 4 × 64 = 256 total rotation angles, vs 64 in SE. Collective quantum processing capacity is 4× that of SE.

3. **Simpler per-expert optimization.** Each expert's `Linear(n_rois, 64)` only encodes within-circuit correlations among functionally related ROIs, rather than all 180 cross-circuit relationships.

### Empirical Results (v2 with quantum fixes)

| Model | Quantum AUC | Classical AUC | C-Q Gap | Compression |
|-------|:---:|:---:|:---:|:---:|
| SE | 0.5769 | 0.6193 | 4.2 pts | 2.81:1 |
| Circuit MoE 4-expert | 0.5764 | 0.6167 | 4.0 pts | <1:1 (expansion) |
| Circuit MoE 2-expert | 0.5783 | 0.5987 | 2.0 pts | ~1.4:1 |

All quantum models plateau at ~0.577. The reduced compression does not yield a measurable quantum AUC improvement, suggesting the bottleneck has shifted from classical compression to quantum expressivity or the ADHD classification ceiling itself.

### Relationship to Contribution 4 (Bottleneck Hypothesis)

Contribution 4 hypothesized that reducing pre-quantum compression would narrow the classical-quantum gap. The v1 results (before quantum fixes) rejected this — the 4-expert model had a 10.3 pt gap. After v2 fixes, the gap narrowed to 4.0 pts, showing that the v1 gap was an **optimization** bottleneck (vanishing gradients, cold start), not a compression bottleneck. The structural advantage of reduced compression is real but does not dominate performance in the current regime.

## 6. Next Steps

1. Implement the 4 fixes in a new model file (`models/CircuitMoE_v2.py`)
2. Submit quantum 4-expert and 2-expert jobs with fixes
3. Compare against:
   - Quantum SE baseline: 0.5769 AUC (target to match or beat)
   - Classical Circuit MoE 4-expert: 0.6167 AUC (measure new C-Q gap)
4. If the C-Q gap narrows significantly (< 4.2 pts), the optimization bottleneck hypothesis is confirmed
5. Multi-seed runs for statistical significance

## References

1. Park, J., et al. (2025). Multi-Chip Ensembles. *arXiv:2505.08782v2*.
2. Nigg, J. T., et al. (2020). ADHD Heterogeneity. *Biol. Psychiatry: CNNI*, 5(8), 726-737.
3. Koirala, S., et al. (2024). Neurobiology of ADHD. *Nat. Rev. Neurosci.*, 25(12), 759-775.
4. Feng, A., et al. (2024). ADHD biotypes. *EClinicalMedicine*, 77, 102876.
5. Cortese, S., et al. (2012). Systems neuroscience of ADHD. *Am. J. Psychiatry*, 169(10), 1038-1055.
6. Fedus, W., et al. (2022). Switch Transformers. *JMLR*, 23(120), 1-39.
