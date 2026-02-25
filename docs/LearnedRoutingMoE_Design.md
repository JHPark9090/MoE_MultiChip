# Learned Routing MoE: Design and Rationale

## Motivation

The Cluster-Informed MoE (Approach 1) showed that all-channel expert processing and cluster-based routing completely solved expert collapse. However, routing quality was bottlenecked by **weak cluster-phenotype association**: site-regressed coherence clusters showed only ~5-6% ADHD prevalence difference between clusters. Hard routing performed at chance; soft routing was slightly better but still limited by the low-dimensional (2-D one-hot) gating signal.

The Learned Routing MoE (Approach 2) addresses this by replacing the cluster one-hot gating input with **top-64 PCA components** from site-regressed coherence features. This gives the gating network a rich, continuous representation of each subject's functional connectivity to learn routing end-to-end.

## Architecture

### Overview

```
         Input (B, T=363, C=180)
                    |
         +----------+----------+
         |                     |
    temporal_mean(x)     PCA features
      (B, 180)           (B, 64)
         |                     |
         +------> concat <-----+
                 (B, 244)
                    |
              input_summary
            Linear(244, 64) + ReLU
                    |
              GatingNetwork
            Linear(64, 32) + ReLU
            Linear(32, 2)  + Softmax
                    |
          +---------+---------+
          |  Soft   |   Hard  |
          | routing | routing |
          +---------+---------+
          | weights |  STE:   |
          | direct  | argmax  |
          |         | + STE   |
          +---------+---------+
                    |
         Expert 0       Expert 1
        (all 180)      (all 180)
                    |
              Weighted sum
                    |
              Classifier
            Linear(64, 1) -> logits
```

### PCA Features

The PCA features are derived from site-regressed functional coherence matrices:

1. Coherence matrices computed per subject from fMRI time series
2. Site effects regressed out to isolate genuine neural connectivity
3. PCA applied to the flattened upper-triangle of coherence matrices
4. Top 64 components retained (from 1,386 total for ADHD, 1,480 for ASD)
5. Z-scored using train-only statistics in the data loader

These features capture subject-level connectivity patterns that are informative for routing without being confounded by acquisition site.

### Gating Input Comparison

| | Cluster-Informed | Learned Routing |
|---|---|---|
| Temporal summary | `mean(x)` over time → (B, 180) | Same |
| Routing signal | Cluster one-hot → (B, 2) | PCA features → (B, 64) |
| Gate input dim | 182 | **244** |
| Information content | 1 bit (cluster 0 or 1) | 64 continuous dimensions |

The gating network itself is identical in structure (`input_summary` → `GatingNetwork`), just with a wider input layer.

## Soft vs Hard Routing

Both routing modes use the **same gating network** — the difference is how the gate output is applied to expert outputs.

### Soft Routing

```python
soft_weights = gate(summary)                     # (B, K), continuous [0, 1]
output = sum(soft_weights.unsqueeze(-1) * expert_stack, dim=1)  # weighted blend
```

- Gate weights are continuous probabilities
- Output is a weighted average of all expert outputs
- Gradients flow directly through the soft weights
- Load-balancing loss encourages balanced utilization

### Hard Routing

```python
soft_weights = gate(summary)                     # (B, K), continuous
hard_idx = argmax(soft_weights)                  # pick winner (non-differentiable)
hard_weights = one_hot(hard_idx)                 # (B, K), binary

# Straight-through estimator
gate_weights = hard_weights - soft_weights.detach() + soft_weights
output = sum(gate_weights.unsqueeze(-1) * expert_stack, dim=1)
```

- Forward pass: only the winning expert's output contributes (hard selection)
- Backward pass: gradients flow through `soft_weights` as if it were soft routing
- The STE trick makes `argmax` differentiable: `hard - soft.detach() + soft` has the same forward value as `hard` but the same gradient as `soft`
- Load-balancing loss still applies (the gating network is trainable)

### Key Properties

| Property | Soft | Hard |
|---|---|---|
| Forward computation | All experts blended | Single expert selected |
| Backward computation | Direct through soft weights | Through soft weights via STE |
| Expert specialization | Implicit (shared responsibility) | Explicit (winner-take-all) |
| Gradient signal | Smooth | Approximate (STE) |
| Load-balancing loss | Yes | Yes |
| All experts run | Yes | Yes (same compute cost) |

## Learned Routing vs Cluster-Informed Routing

### Soft Routing Comparison

| | Cluster-Informed Soft | Learned Routing Soft |
|---|---|---|
| Gating input | `concat(mean(x), cluster_onehot)` → 182-D | `concat(mean(x), pca_features)` → 244-D |
| Routing signal | 1 bit of cluster identity | 64 continuous connectivity dimensions |
| Can learn new routing patterns | Limited by cluster quality | Yes, end-to-end |
| Gating network | Same architecture | Same architecture (wider input) |
| Expert combination | Weighted average | Weighted average |

The cluster one-hot provides a strong prior ("this subject is cluster 0, bias toward expert 0") but the signal is coarse. PCA features let the gating network discover finer-grained routing patterns from the full connectivity spectrum.

### Hard Routing Comparison

This is where the two approaches differ most fundamentally:

| | Cluster-Informed Hard | Learned Routing Hard |
|---|---|---|
| Gating network | **None** — no learnable router | **Yes** — trainable gating network |
| Routing decision | Deterministic lookup: `cluster_label → expert` | Learned: `gating_network(pca) → argmax → expert` |
| Experts run per sample | Only the assigned expert | All experts (STE requires all outputs) |
| Gradient to router | N/A (no parameters) | Through STE |
| Can adapt during training | No — fixed from epoch 1 to 100 | Yes — routing improves as gating learns |
| Expert collapse risk | Impossible (fixed by clusters) | Possible but mitigated by load-balancing loss |
| Routing quality depends on | Cluster-phenotype association | End-to-end learning from PCA features |

**In summary**: Cluster-informed hard routing is a **fixed assignment table** — each subject always goes to the same expert based on a pre-computed cluster label. Learned hard routing is a **trainable discrete router** — it starts random and learns which expert should handle each subject based on 64 connectivity features, with the straight-through estimator enabling gradient flow through the discrete selection.

### Implications for the Soft→Hard Transition

The meaning of "switching from soft to hard" is fundamentally different between the two approaches:

- **Cluster-Informed**: soft→hard = removing the gating network entirely, replacing it with a lookup table. This removes learnable parameters and makes routing fixed.
- **Learned Routing**: soft→hard = keeping the same gating network but discretizing its output. The router still learns; only the combination rule changes from weighted-average to winner-take-all.

## Parameter Count Comparison

| Config | Cluster-Informed | Learned Routing | Delta |
|--------|-----------------|-----------------|-------|
| Classical Soft | 283,491 | 287,459 | +3,968 (+1.4%) |
| Classical Hard | 269,633 | 287,459 | +17,826 (+6.6%) |
| Quantum Soft | 41,089 | ~45,057 | +3,968 (+9.7%) |
| Quantum Hard | 27,231 | ~45,057 | +17,826 (+65.5%) |

The parameter increase for soft routing is minimal (+1.4% classical) — just the wider input layer (244 vs 182 inputs to `input_summary`).

The increase for hard routing is larger in relative terms because cluster-informed hard has **no gating network at all** (no `input_summary`, no `GatingNetwork`), while learned hard has the full gating network. This is the cost of making hard routing trainable.

## Experiment Matrix

### ADHD (target: `ADHD_label`)

| Script | Config | Time | PCA Source |
|--------|--------|------|------------|
| `LearnedMoE_Classical_Soft.sh` | Classical, Soft | 2h | `coherence_clustering_site_regressed/fc_pca_features.npy` |
| `LearnedMoE_Classical_Hard.sh` | Classical, Hard | 2h | Same |
| `LearnedMoE_Quantum_Soft.sh` | Quantum (8q/d3), Soft | 12h | Same |
| `LearnedMoE_Quantum_Hard.sh` | Quantum (8q/d3), Hard | 12h | Same |

### ASD (target: `ASD_label`)

| Script | Config | Time | PCA Source |
|--------|--------|------|------------|
| `ASD_LearnedMoE_Classical_Soft.sh` | Classical, Soft | 2h | `asd_coherence_clustering_site_regressed/fc_pca_features.npy` |
| `ASD_LearnedMoE_Classical_Hard.sh` | Classical, Hard | 2h | Same |
| `ASD_LearnedMoE_Quantum_Soft.sh` | Quantum (8q/d3), Soft | 12h | Same |
| `ASD_LearnedMoE_Quantum_Hard.sh` | Quantum (8q/d3), Hard | 12h | Same |

### Shared Hyperparameters

All experiments use identical hyperparameters to the cluster-informed experiments for fair comparison:

- `num_experts=2`, `expert_hidden_dim=64`, `dropout=0.2`
- `gating_noise_std=0.1`, `balance_loss_alpha=0.1`
- `n_epochs=100`, `batch_size=32`, `lr=1e-3`, `wd=1e-5`
- `patience=20`, `grad_clip=1.0`, `lr_scheduler=cosine`
- `seed=2025`, `pca_components=64`
- Classical: `expert_layers=2`, `nhead=4`
- Quantum: `n_qubits=8`, `n_ansatz_layers=2`, `degree=3`

## What to Expect

### Hypothesis

The learned routing MoE should outperform the cluster-informed MoE because:

1. **Richer routing signal**: 64 PCA components provide orders of magnitude more information than a 1-bit cluster label for routing decisions.
2. **End-to-end optimization**: The gating network can discover routing patterns that optimize the classification objective, rather than being constrained by pre-computed clusters.
3. **Hard routing becomes meaningful**: Cluster-informed hard routing was limited by the 5% prevalence difference between clusters. Learned hard routing can discover more discriminative routing boundaries in the 64-D PCA space.

### Risks

1. **Overfitting the gate**: With 64 PCA input dimensions, the gating network has more capacity to overfit. The load-balancing loss and exploration noise should mitigate this.
2. **Expert collapse via PCA**: If PCA features don't provide useful routing signal, the gate may learn to ignore them and route uniformly, effectively reducing to a single-expert model.
3. **Compute cost for hard routing**: Unlike cluster-informed hard (which only runs one expert per sample), learned hard runs all experts for every sample (needed for STE gradient flow), so it has the same compute cost as soft routing.

## Reproducibility

```bash
# Submit all ADHD jobs
sbatch scripts/LearnedMoE_Classical_Soft.sh
sbatch scripts/LearnedMoE_Classical_Hard.sh
sbatch scripts/LearnedMoE_Quantum_Soft.sh
sbatch scripts/LearnedMoE_Quantum_Hard.sh

# Submit all ASD jobs
sbatch scripts/ASD_LearnedMoE_Classical_Soft.sh
sbatch scripts/ASD_LearnedMoE_Classical_Hard.sh
sbatch scripts/ASD_LearnedMoE_Quantum_Soft.sh
sbatch scripts/ASD_LearnedMoE_Quantum_Hard.sh

# Smoke test (CPU, ~3 min):
python LearnedRoutingMoE_ABCD.py \
    --model-type=classical --routing=soft \
    --pca-file=results/coherence_clustering_site_regressed/fc_pca_features.npy \
    --cluster-csv=results/coherence_clustering_site_regressed/cluster_assignments.csv \
    --sample-size=50 --n-epochs=2 --batch-size=8
```

## File Locations

| Item | Path |
|------|------|
| Model script | `MoE_MultiChip/LearnedRoutingMoE_ABCD.py` |
| Data loader | `MoE_MultiChip/dataloaders/Load_ABCD_fMRI.py` |
| ADHD PCA features | `MoE_MultiChip/results/coherence_clustering_site_regressed/fc_pca_features.npy` |
| ADHD cluster CSV | `MoE_MultiChip/results/coherence_clustering_site_regressed/cluster_assignments.csv` |
| ASD PCA features | `MoE_MultiChip/results/asd_coherence_clustering_site_regressed/fc_pca_features.npy` |
| ASD cluster CSV | `MoE_MultiChip/results/asd_coherence_clustering_site_regressed/cluster_assignments.csv` |
| SLURM scripts | `MoE_MultiChip/scripts/LearnedMoE_*.sh`, `MoE_MultiChip/scripts/ASD_LearnedMoE_*.sh` |
| Logs | `MoE_MultiChip/logs/` |
| Checkpoints | `MoE_MultiChip/checkpoints/` |
