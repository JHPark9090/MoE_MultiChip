# Quantum Distributed EEG Mixture of Experts

## 1. Overview

The **Quantum Distributed EEG MoE** (`QuantumDistributedEEGMoE`) is a hybrid
classical-quantum architecture that combines the spatial partitioning strategy
of the classical Distributed EEG MoE with the quantum feature-extraction
capabilities of the Quantum Time-Series Transformer (QTS v2.5). The result is a
model where a classical module handles input distribution and expert routing,
while multiple independent `QuantumTSTransformer` instances serve as quantum
experts for feature extraction.

### Motivation

| Model | Experts | Feature Extractor | Gating Source | Params (ABCD) |
|-------|---------|-------------------|---------------|---------------|
| Classical MoE | 4 TransformerEncoders | Self-attention (2 layers, 4 heads) | Post-hoc global token fusion | ~508K |
| Standalone QTS | 1 quantum circuit | sim14 ansatz + QSVT/LCU | N/A (no routing) | ~12K |
| **Quantum MoE** | **4 QTS circuits** | **sim14 ansatz + QSVT/LCU** | **Input-based classical summary** | **~35K** |

The Quantum MoE tests whether distributing quantum processing across
spatially-partitioned channel subsets (with classical routing) can match or
exceed the classical MoE at a fraction of the parameter count, while retaining
the expressivity advantages of quantum circuits.

---

## 2. Architecture

```
Input x: (B, T, C)
          │
          ├──────────────────────────────────────────────┐
          │                                              │
          ▼                                              ▼
┌─────────────────────────────┐            ┌──────────────────────────┐
│  Classical Input Summary    │            │  Channel Splitting       │
│                             │            │  with Halo Overlap       │
│  x.mean(dim=1)  → (B, C)   │            │                          │
│  Linear(C, H)   → (B, H)   │            │  C channels → K groups   │
│  ReLU                       │            │  ± halo_size overlap     │
└─────────────┬───────────────┘            │  zero-pad to uniform    │
              │                            │  width C_local           │
              ▼                            └──┬───┬───┬───┬──────────┘
┌─────────────────────────────┐               │   │   │   │
│  Gating Network             │               ▼   ▼   ▼   ▼
│                             │            (B,T,C_local) per expert
│  Linear(H, H/2) + ReLU     │               │   │   │   │
│  Linear(H/2, K)             │               │   │   │   │
│  + noise (training only)    │          permute to (B, C_local, T)
│  softmax → (B, K)           │               │   │   │   │
└─────────────┬───────────────┘               ▼   ▼   ▼   ▼
              │                     ┌─────┐┌─────┐┌─────┐┌─────┐
              │                     │QTS_0││QTS_1││QTS_2││QTS_3│
              │                     │     ││     ││     ││     │
              │                     │(B,H)││(B,H)││(B,H)││(B,H)│
              │                     └──┬──┘└──┬──┘└──┬──┘└──┬──┘
              │                        │      │      │      │
              │                        └──┬───┴──┬───┘      │
              │                           │      │          │
              │                     torch.stack → (B, K, H) │
              │                           │                 │
              ▼                           ▼                 │
┌──────────────────────────────────────────────────────┐    │
│  Classical Aggregator                                │    │
│                                                      │◄───┘
│  weighted = (gate_weights · expert_stack).sum(dim=1) │
│  logits = Linear(H, 1)     → (B,)                   │
└──────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Forward Pass

The following traces a single forward pass through `QuantumDistributedEEGMoE`
using the default ABCD fMRI configuration: `C=180` channels, `T=363`
timesteps, `K=4` experts, `H=64` hidden dim, `n_qubits=8`, `n_ansatz_layers=2`,
`degree=3`, `halo_size=2`.

### Step 0: Input Preparation (in training loop)

The data arrives from the DataLoader as `(B, C, T)` — i.e. `(B, 180, 363)`.
The training loop (imported from `DistributedEEGMoE.py`) permutes it before
passing to the model:

```python
data = data.permute(0, 2, 1)  # (B, 180, 363) → (B, 363, 180)
logits, gate_weights = model(data)
```

The model receives `x: (B, T=363, C=180)`.

### Step 1: Classical Input Summary and Gating

**Purpose**: Compute per-sample expert routing weights from a compressed
representation of the full input, *before* any expert sees the data. This is
the standard MoE approach (gating from input, not from expert outputs).

```python
# Temporal mean pooling — collapse time dimension
summary_input = x.mean(dim=1)              # (B, 363, 180) → (B, 180)

# Project to hidden dimension
summary = self.input_summary(summary_input)  # Linear(180, 64) + ReLU → (B, 64)

# Gating network produces K soft routing weights
gate_weights = self.gate(summary)            # (B, 64) → (B, 4)
```

**GatingNetwork internals**:

```
(B, 64) → Linear(64, 32) → ReLU → Linear(32, 4) → [+ noise σ=0.1 if training] → softmax → (B, 4)
```

The noise injection during training encourages exploration of different expert
combinations, preventing the gate from collapsing to always pick one expert.

### Step 2: Channel Splitting with Halo Overlap

**Purpose**: Divide the `C=180` input channels into `K=4` non-overlapping
spatial groups, each extended by a symmetric halo for spatial continuity.

**Range computation** (`_compute_expert_ranges`):

```
C=180, K=4 → base=45, remainder=0
Expert 0: channels [0,   45)
Expert 1: channels [45,  90)
Expert 2: channels [90,  135)
Expert 3: channels [135, 180)
```

**Halo extension** (`_split_with_halo`):

With `halo_size=2`, each expert's slice is extended by 2 channels on each side.
The uniform width is `max_base + 2*halo = 45 + 4 = 49`.

```
Expert 0: want [0-2, 45+2) = [-2, 47) → clamp to [0, 47), pad 2 left  → 49 channels
Expert 1: want [45-2, 90+2) = [43, 92)                                 → 49 channels
Expert 2: want [90-2, 135+2) = [88, 137)                               → 49 channels
Expert 3: want [135-2, 180+2) = [133, 182) → clamp to [133, 180), pad 2 right → 49 channels
```

Result: a list of `K=4` tensors, each `(B, 363, 49)`.

### Step 3: Quantum Expert Processing

Each expert receives its channel subset and processes it through a full
`QuantumTSTransformer`. The chunk is first permuted to match QTS's expected
input format:

```python
chunk = expert_inputs[k].permute(0, 2, 1)  # (B, 363, 49) → (B, 49, 363)
h = qts(chunk)                              # → (B, 64)
```

Inside each `QuantumTSTransformer.forward()`, the following happens:

#### Step 3a: Permute and Add Positional Encoding

```python
x = x.permute(0, 2, 1)                     # (B, 49, 363) → (B, 363, 49)
x = x + self.pe[:, :363]                    # Add sinusoidal PE (Vaswani et al.)
```

The positional encoding is a fixed (non-trainable) sinusoidal pattern of shape
`(1, 363, 49)` following the standard Transformer formulation:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

This injects temporal ordering information so the quantum circuit can
distinguish early vs. late timesteps.

#### Step 3b: Feature Projection to Rotation Angles

```python
x = self.feature_projection(self.dropout(x))   # Linear(49, 64) → (B, 363, 64)
timestep_params = self.rot_sigm(x) * (2 * π)   # Sigmoid * 2π → [0, 2π]
```

Each of the 363 timesteps is projected from 49 channel values to 64 rotation
angles. These angles will parameterize the sim14 quantum ansatz.

The number of rotation angles per timestep is:

```
n_rots = 4 × n_qubits × n_ansatz_layers = 4 × 8 × 2 = 64
```

#### Step 3c: Initialize Quantum Base State

```python
base_states = torch.zeros(B, 2^8, dtype=complex64)  # (B, 256)
base_states[:, 0] = 1.0                              # |00000000⟩
```

All qubits start in the computational basis state `|0⟩^⊗8`.

#### Step 3d: QSVT Polynomial State Preparation

This is the core quantum computation. It classically simulates the **Quantum
Singular Value Transformation (QSVT)** protocol, which applies a polynomial
transformation of a Linear Combination of Unitaries (LCU) to the base state.

**Conceptually**:

```
|ψ_out⟩ = p(A) |0⟩^⊗n
```

where:
- `A = Σ_{t=1}^{T} α_t U_t` is a Linear Combination of Unitaries (LCU)
- `U_t` is the sim14 ansatz parameterized by timestep `t`'s rotation angles
- `α_t` are trainable complex mixing coefficients (`mix_coeffs`, shape `(T,)`)
- `p(x) = Σ_{j=0}^{D} c_j x^j` is a polynomial of degree `D=3`
- `c_j` are trainable real polynomial coefficients (`poly_coeffs`, shape `(D+1,)`)

**Implementation** (`evaluate_polynomial_state_pl`):

```python
acc = c_0 · |ψ_base⟩
working = |ψ_base⟩

for j = 1..D:
    working = A · working     # apply_unitaries_pl
    acc = acc + c_j · working

return acc / ‖[c_0, ..., c_D]‖₁
```

Each application of `A` (via `apply_unitaries_pl`) works as follows:

1. **Flatten**: `(B, T, n_rots)` → `(B·T, n_rots)` — treat each timestep as
   an independent circuit execution
2. **Repeat base states**: `(B, 256)` → `(B·T, 256)` — each timestep evolves
   the same input state
3. **Execute QNode once**: Apply `StatePrep(|ψ⟩)` then `sim14_circuit(params)`
   to the entire `B·T` batch in one vectorized call
4. **Reshape**: `(B·T, 256)` → `(B, T, 256)` — separate timesteps again
5. **LCU contraction**: `einsum('bti, bt → bi')` — weighted sum over timesteps
   using the complex mixing coefficients → `(B, 256)`

This procedure is repeated `D` times (polynomial degree), each time feeding the
previous output as the new base state.

**The sim14 Ansatz** (Sim et al., 2019):

Each layer of the sim14 circuit has four sub-layers:

```
Layer l:
  1. RY(θ_i) on each qubit i          ← n_qubits params
  2. CRX(θ_i) ring: i → (i+1) mod n   ← n_qubits params (reversed order)
  3. RY(θ_i) on each qubit i          ← n_qubits params
  4. CRX(θ_i) counter-ring: i → (i-1) ← n_qubits params

Total per layer: 4 × n_qubits parameters
```

The ring + counter-ring CRX pattern creates bidirectional entanglement,
enabling the circuit to capture correlations in both "directions" of the qubit
register.

#### Step 3e: Normalize the Polynomial State

```python
norm = ‖ψ_out‖₂
ψ_normalized = ψ_out / (norm + 1e-9)   # (B, 256) → (B, 256)
```

#### Step 3f: Quantum Feature Function (QFF)

A final single-layer sim14 circuit with trainable parameters (`qff_params`,
32 values for 8 qubits) is applied, followed by measurement of Pauli
expectation values:

```python
exps = qff_qnode_expval(ψ_normalized, qff_params)
```

**QNode internals**:

```
StatePrep(ψ_normalized)
sim14_circuit(qff_params, layers=1)   # 1 layer, 32 params
Measure: ⟨X_0⟩, ⟨X_1⟩, ..., ⟨X_7⟩,
         ⟨Y_0⟩, ⟨Y_1⟩, ..., ⟨Y_7⟩,
         ⟨Z_0⟩, ⟨Z_1⟩, ..., ⟨Z_7⟩
→ 3 × 8 = 24 real expectation values
```

Result: `(B, 24)` — a 24-dimensional classical feature vector extracted from the
quantum state.

#### Step 3g: Output Feed-Forward

```python
output = self.output_ff(exps)   # Linear(24, 64) → (B, 64)
```

Because `output_dim=expert_hidden_dim=64`, each QTS expert directly produces a
feature vector in the shared hidden space. No wrapper or adapter is needed.

### Step 4: Expert Feature Aggregation

```python
expert_stack = torch.stack([h_0, h_1, h_2, h_3], dim=1)  # (B, 4, 64)
```

### Step 5: Gated Weighted Combination

```python
# gate_weights: (B, 4) from Step 1
weighted = (gate_weights.unsqueeze(-1) * expert_stack).sum(dim=1)
# (B, 4, 1) × (B, 4, 64) → sum over dim=1 → (B, 64)
```

Each sample's final representation is a convex combination of the four expert
outputs, weighted by the gating network's softmax probabilities.

### Step 6: Classification

```python
logits = self.classifier(weighted)   # Linear(64, 1) → (B, 1)
logits = logits.squeeze(-1)          # (B,) for binary classification
```

For binary classification, BCEWithLogitsLoss is applied to the raw logits.

**Output**: `(logits, gate_weights)` — same interface as the classical
`DistributedEEGMoE`, so the training loop works unchanged.

---

## 4. Training Procedure

The training functions (`train`, `validate`, `test`) are imported directly
from `DistributedEEGMoE.py`. The model interface is identical:
`forward(x) → (logits, gate_weights)`.

### Loss Function

```
L_total = L_task + α · L_balance
```

- **Task loss** (`L_task`): `BCEWithLogitsLoss` for binary classification
- **Balance loss** (`L_balance`): Switch Transformer load-balancing auxiliary
  loss, encouraging uniform expert utilization
- **α** (`--balance-loss-alpha`): weight for the balance term (default: 0.01)

### Load-Balancing Loss (Switch Transformer)

```
L_balance = K · Σ_{i=1}^{K} f_i · P_i
```

Where:
- `f_i` = fraction of batch samples where expert `i` has the highest gate weight
  (hard argmax assignment)
- `P_i` = mean softmax gate probability for expert `i` across the batch
- `K` = number of experts

This loss is minimized when all experts are equally utilized (`f_i = P_i = 1/K`).

### Optimization

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| LR scheduler | Cosine annealing over `n_epochs` |
| Gradient clipping | Max norm 1.0 |
| Early stopping | Patience 20 epochs (on Val AUC) |

### Checkpointing

Best model (highest validation AUC) is saved to
`checkpoints/QMoE_EEG_{job_id}.pt` containing:

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `epoch`, `best_val_auc`, `patience_counter`
- `args` (full hyperparameter dict)

---

## 5. Component Comparison: Classical MoE vs. Quantum MoE

### Expert Architecture

| Aspect | Classical MoE Expert | Quantum MoE Expert |
|--------|---------------------|--------------------|
| Module | `EEGExpert` (nn.TransformerEncoder) | `QuantumTSTransformer` |
| Feature extraction | Self-attention (2 layers, 4 heads) | sim14 ansatz + QSVT/LCU |
| Input | (B, T, C_local) + global token | (B, C_local, T) |
| Output | (B, H) + updated global token | (B, H) |
| Params per expert (ABCD) | ~127K | ~5.2K |
| Temporal modeling | Learned positional embedding | Sinusoidal PE + LCU mixing |

### Gating Strategy

| Aspect | Classical MoE | Quantum MoE |
|--------|--------------|-------------|
| Gating input | Fused global tokens (post-hoc, after expert execution) | Temporal mean of raw input (pre-expert) |
| Inter-expert communication | Global memory token passed through experts | None needed (gating is input-based) |
| Computation order | Experts first → fuse tokens → gate → weight | Gate from input → experts → weight |

The classical MoE uses a **post-hoc** gating strategy: a learnable global token
is passed through each expert, the updated tokens are fused, and the fused
representation drives gating. This creates a dependency between expert execution
and gating.

The Quantum MoE uses a **pre-hoc** gating strategy: gating weights are computed
from a classical summary of the input *before* any expert runs. This is simpler,
more standard in the MoE literature, and avoids adding a global token mechanism
to the QTS architecture.

### Parameter Count (ABCD fMRI, C=180, T=363)

| Component | Classical MoE | Quantum MoE |
|-----------|--------------|-------------|
| Expert 0 | ~127K | ~5,200 |
| Expert 1 | ~127K | ~5,200 |
| Expert 2 | ~127K | ~5,200 |
| Expert 3 | ~127K | ~5,200 |
| **4 experts total** | **~508K** | **~20,800** |
| Global token | 64 | — |
| Input summary | — | 11,584 |
| Gating network | 2,212 | 2,212 |
| Classifier | 65 | 65 |
| **Total** | **~510K** | **~35K** |

The Quantum MoE is **~15x smaller** than the classical MoE.

### Per-Expert Parameter Breakdown (8Q, 2L, D=3, C_local=49, H=64, T=363)

| Component | Params | Description |
|-----------|--------|-------------|
| `feature_projection` | 3,200 | Linear(49, 64): project channels → rotation angles |
| `output_ff` | 1,600 | Linear(24, 64): Pauli expectation values → hidden dim |
| `poly_coeffs` | 4 | QSVT polynomial coefficients (degree+1) |
| `mix_coeffs` | 363 | Complex LCU mixing coefficients (one per timestep) |
| `qff_params` | 32 | QFF sim14 circuit parameters (4 × 8 × 1) |
| **Total** | **5,199** | |

---

## 6. Comparison with Standalone QTS

The standalone `QTSTransformer` (run via `QTSTransformer_PhysioNet_EEG.py`)
processes *all* channels at once with a single quantum circuit.

| Aspect | Standalone QTS | Quantum MoE (per expert) |
|--------|---------------|-------------------------|
| Input channels | All C | C/K + 2×halo (subset) |
| feature_dim | C (e.g., 180) | C_local (e.g., 49) |
| output_dim | 1 (scalar logit) | H=64 (feature vector) |
| Gating | None | Classical GatingNetwork |
| Specialization | Global features | Region-specific features |

By setting `output_dim=expert_hidden_dim` instead of `output_dim=1`, each QTS
expert produces a feature vector rather than a classification score. The
classifier head then operates on the gated combination of all expert features.

---

## 7. Data Flow Summary

```
DataLoader → (B, C, T)
    │
    ▼ permute in train loop
(B, T, C)
    │
    ├── mean(dim=1) → (B, C) → Linear+ReLU → (B, H) → GatingNet → (B, K)
    │
    ├── split_with_halo → K × (B, T, C_local)
    │       │
    │       ▼ permute per chunk
    │   K × (B, C_local, T)
    │       │
    │       ▼ QTS.forward() each
    │   K × (B, T, C_local) → +PE → Linear → σ·2π → QSVT(LCU(sim14))
    │       → normalize → QFF(sim14, PauliXYZ) → Linear → (B, H)
    │       │
    │       ▼ stack
    │   (B, K, H)
    │
    ▼ weighted sum
(B, H) → Linear(H, 1) → (B,) logits
```

---

## 8. Usage

### Command-Line Arguments

#### MoE Architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--num-experts` | 4 | Number of QTS expert circuits |
| `--expert-hidden-dim` | 64 | Output dimension per expert / classifier input dim |
| `--halo-size` | 2 | Channel overlap on each side of expert partition |
| `--num-classes` | 2 | 2=binary (BCE), >2=multiclass (CE) |
| `--dropout` | 0.1 | Dropout rate |
| `--gating-noise-std` | 0.1 | Gaussian noise std on gate logits during training |
| `--balance-loss-alpha` | 0.01 | Weight for load-balancing auxiliary loss |

#### Quantum Circuit
| Argument | Default | Description |
|----------|---------|-------------|
| `--n-qubits` | 8 | Qubits per expert circuit |
| `--n-ansatz-layers` | 2 | sim14 VQC depth per expert |
| `--degree` | 3 | QSVT polynomial degree |

#### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--n-epochs` | 100 | Maximum training epochs |
| `--batch-size` | 32 | Mini-batch size |
| `--lr` | 1e-3 | Learning rate |
| `--wd` | 1e-5 | Weight decay |
| `--patience` | 20 | Early-stopping patience (epochs) |
| `--grad-clip` | 1.0 | Max gradient norm (0=disabled) |
| `--lr-scheduler` | cosine | LR schedule: none, cosine, step |

#### Data
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | PhysioNet_EEG | PhysioNet_EEG or ABCD_fMRI |
| `--parcel-type` | HCP180 | fMRI parcellation (ABCD only) |
| `--target-phenotype` | ADHD_label | Target column (ABCD only) |
| `--sampling-freq` | 16 | EEG resample frequency in Hz |
| `--sample-size` | 50 | Subjects (PhysioNet: 1-109, ABCD: 0=all) |

#### Experiment
| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | 2025 | Random seed |
| `--job-id` | QMoE_EEG | Identifier for checkpoints and logs |
| `--resume` | false | Resume from checkpoint |
| `--wandb` | false | Enable Weights & Biases logging |
| `--base-path` | ./checkpoints | Checkpoint and log directory |

### Quick Smoke Test (CPU, ~1-2 min)

```bash
python QuantumDistributedEEGMoE.py \
    --dataset=ABCD_fMRI --parcel-type=HCP180 --target-phenotype=ADHD_label \
    --num-experts=2 --expert-hidden-dim=32 --n-qubits=4 --n-ansatz-layers=1 \
    --degree=2 --halo-size=2 --num-classes=2 --batch-size=8 --n-epochs=2 \
    --sample-size=50 --seed=2025 --lr-scheduler=none
```

### Full SLURM Runs

```bash
# ABCD fMRI ADHD classification (all ~9400 subjects)
sbatch scripts/QuantumMoE_ABCD_ADHD.sh

# PhysioNet Motor Imagery EEG (109 subjects)
sbatch scripts/QuantumMoE_PhysioNet_EEG.sh
```

---

## 9. File Structure

```
MoE_MultiChip/
├── QuantumDistributedEEGMoE.py          # Quantum MoE model + main()
├── DistributedEEGMoE.py                 # Classical MoE (imports: GatingNetwork,
│                                        #   load_balancing_loss, train/validate/test,
│                                        #   set_all_seeds, epoch_time, transpose_fmri_loaders)
├── QTSTransformer_PhysioNet_EEG.py      # Standalone QTS training script
├── models/
│   └── QTSTransformer_v2_5.py           # QuantumTSTransformer class (sim14 + QSVT/LCU)
├── dataloaders/
│   ├── Load_PhysioNet_EEG.py            # PhysioNet Motor Imagery EEG loader
│   └── Load_ABCD_fMRI.py               # ABCD fMRI loader
├── scripts/
│   ├── QuantumMoE_ABCD_ADHD.sh          # SLURM: Quantum MoE on ABCD
│   ├── QuantumMoE_PhysioNet_EEG.sh      # SLURM: Quantum MoE on PhysioNet
│   ├── MoE_ABCD_ADHD.sh                 # SLURM: Classical MoE on ABCD
│   ├── MoE_PhysioNet_EEG.sh             # SLURM: Classical MoE on PhysioNet
│   ├── QTS_ABCD_ADHD.sh                 # SLURM: Standalone QTS on ABCD
│   └── QTS_PhysioNet_EEG.sh             # SLURM: Standalone QTS on PhysioNet
├── docs/
│   └── Quantum_Distributed_EEG_MoE.md   # This document
└── checkpoints/                         # Saved model checkpoints
```

---

## 10. Key Design Decisions

1. **QTS used as-is**: No modifications to `QTSTransformer_v2_5.py`. Setting
   `output_dim=hidden_dim` naturally makes the existing `output_ff` layer
   produce H-dimensional feature vectors instead of scalar logits.

2. **Input-based gating**: The gating network receives a classical summary of
   the raw input (temporal mean → linear projection), rather than relying on
   post-hoc global token fusion. This is more standard in the MoE literature
   (cf. Switch Transformer, GShard) and avoids architectural coupling between
   gating and expert execution.

3. **No global token**: The classical MoE passes a shared learnable token
   through each expert for inter-expert communication. The Quantum MoE omits
   this — the GatingNetwork provides inter-expert coordination, and each QTS
   expert independently specializes on its channel partition.

4. **Reuse over copy**: Training functions, the gating network, load-balancing
   loss, and all utilities are imported from `DistributedEEGMoE.py`. Only the
   channel-splitting helpers (`_compute_expert_ranges`, `_split_with_halo`)
   are copied into the new class — they are trivial index arithmetic that
   does not warrant import indirection.

5. **Matched experimental settings**: Both SLURM scripts match the
   hyperparameters used by the classical MoE and standalone QTS baselines
   (seed, learning rate, batch size, epochs, patience, data splits) to enable
   fair comparison. Only expert-internal architecture differs.
