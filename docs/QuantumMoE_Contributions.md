# Novel Contributions of Quantum Mixture of Experts

> **DEPRECATION NOTICE (2026-03-07):** The circuit expert ROI counts in this document (DMN 55, Executive 50, Salience 29, SensoriMotor 46) were based on an **incorrect Yeo-17 network mapping**. The corrected mapping (via volumetric Glasser-to-Yeo17 overlap) yields different circuit compositions (DMN 41, Executive 52, Salience 44, SensoriMotor 43). The conceptual contributions (domain-informed partitioning, quantum MoE framework, parameter efficiency) remain valid, but specific ROI assignments and associated claims need updating with v5 results.

**Date**: 2026-03-07
**Status**: All experiments complete. 10Q 4-expert achieves 0.6022 AUC (97.7% of classical, 1.5 pt gap).

## Context: From Multi-Chip Ensembles to Quantum MoE

This work extends our prior Multi-Chip Ensemble framework (Park et al., 2025) to spatio-temporal biomedical data. The Multi-Chip paper demonstrated that partitioning high-dimensional computations across ensembles of smaller, independently operating quantum chips overcomes three fundamental challenges of variational quantum circuits: scalability, trainability (barren plateaus), and noise resilience. However, that approach was designed for static, spatially independent data (images, tabular features) and faces critical limitations when applied to spatio-temporal data such as EEG and fMRI.

> Park, J., et al. (2025). Addressing the Current Challenges of Quantum Machine Learning through Multi-Chip Ensembles. *arXiv:2505.08782v2*.

## Contribution 1: Addressing Multi-Chip Limitations for Spatio-Temporal Data

### The Problem

Multi-Chip Ensembles split input features across K independent quantum chips by index order (e.g., features 1-12 to chip 1, 13-24 to chip 2). This works for images where spatial locality provides a reasonable heuristic, but creates three problems for brain time-series data:

1. **Arbitrary splitting destroys functional dependencies.** In fMRI, brain regions are not independent pixels — they form functional circuits with correlated activity. Splitting ROIs by index assigns functionally related regions (e.g., medial prefrontal cortex and posterior cingulate cortex, both DMN) to different chips, severing the within-network correlations that carry the most diagnostic information. There is no principled ordering of brain ROIs that preserves all relevant dependencies through contiguous index ranges.

2. **No mechanism to recover between-chip information.** Multi-Chip outputs are classically aggregated (averaged or concatenated). Once entanglement between two ROIs is prevented by chip boundaries, the quantum circuits cannot capture their joint statistics. The classical aggregation layer has limited capacity to compensate.

3. **Excessive partitioning is computationally expensive.** Processing 180 fMRI ROIs x 363 timepoints (65,340 features) with 12-qubit chips would require ~5,445 chips. Even with the 3,264-dimensional PhysioNet EEG data, the Multi-Chip paper used 272 chips, requiring 272 forward/backward passes per sample.

### The Solution: Domain-Informed Quantum Circuit Partitioning

Quantum MoE replaces arbitrary index-based splitting with **neuroscience-guided circuit specialization**. Each quantum expert processes a functionally coherent subset of brain ROIs defined by the ADHD neuroscience literature:

| Expert | Brain Circuit | ROIs | Literature Basis |
|--------|--------------|------|-----------------|
| DMN | Default mode network | 55 | Nigg 2020, Feng 2024, Koirala 2024 |
| Executive | Frontoparietal control + dorsal attention | 50 | Nigg 2020, Cortese 2012 |
| Salience | Salience/ventral attention + limbic | 29 | Nigg 2020, Pan 2026 |
| SensoriMotor | Visual + somatomotor | 46 | Cortese 2012, Feng 2024 |

This addresses each limitation:

**Within-circuit dependencies are preserved.** ROIs that belong to the same functional brain circuit are processed together by the same quantum expert. The DMN expert maintains entanglement between all DMN regions (mPFC, PCC, angular gyrus, temporal pole), preserving the within-network correlations. Feng et al. (2024) found that default mode and cognitive control regions exhibited the greatest involvement in distinguishing ADHD biotypes, highlighting the importance of preserving within-circuit connectivity structure.

**Between-circuit information is recovered through learned gating.** Unlike Multi-Chip's simple averaging, the MoE gating network takes the full 180-ROI temporal mean as input and learns soft routing weights over experts. This classical gating network captures cross-circuit relationships (e.g., DMN-frontoparietal anti-correlation) while quantum experts specialize in within-circuit temporal dynamics. This is a principled division of labor: quantum circuits handle within-circuit entangled dynamics, classical gating handles between-circuit routing.

**Far fewer partitions with higher expert capacity.** 4 circuit experts instead of 272 chips. Each expert is a full Quantum Time-Series Transformer (QSVT + sim14 ansatz with Pauli X/Y/Z measurements) rather than a simple VQC with angle embedding. Training requires 4 quantum forward passes per sample instead of 272.

### Comparison

| Property | Multi-Chip Ensemble | Quantum Circuit MoE |
|----------|:---:|:---:|
| Splitting criterion | Index-based (arbitrary) | Neuroscience-guided (brain circuits) |
| Within-group dependencies | Preserved | Preserved |
| Between-group dependencies | Lost | Partially recovered (classical gating) |
| Number of partitions | Many (e.g., 272) | Few (2 or 4) |
| Expert capacity | Simple VQC | Quantum Transformer (QSVT) |
| Training cost per sample | K quantum forward passes | K quantum forward passes (K=4 vs K=272) |
| Domain knowledge required | None | Yes (circuit definitions from literature) |
| Aggregation | Average/concatenate | Learned soft gating + load balancing |

### Limitation

Between-circuit quantum entanglement is still lost — the DMN expert's qubits cannot entangle with the Executive expert's qubits. If classification requires quantum correlations spanning multiple brain circuits simultaneously, MoE cannot capture this. However, the neuroscience literature suggests within-circuit dynamics carry substantial diagnostic signal (Feng 2024: default mode and cognitive control regions show the greatest involvement in biotype discrimination; Cortese 2012: ADHD dysfunction maps to identifiable network-level systems). The classical gating network provides a complementary mechanism for integrating cross-circuit information.

## Contribution 2: First Quantum MoE Framework for Neuroimaging

This work provides the first demonstration that quantum variational circuits can function as specialized experts within a Mixture of Experts architecture for neuroimaging classification.

The Quantum MoE framework was developed in two stages:

**Stage 1: Cluster-Informed Quantum MoE** (all-channel experts). Demonstrated that quantum circuits are compatible with soft gating and load balancing within an MoE framework. Specifically:

- **Quantum experts are compatible with soft gating.** The gating network produces continuous weights over quantum experts, and gradients flow through both the gating network and quantum circuits via backpropagation (PennyLane's `diff_method="backprop"` on `default.qubit`). This is not trivial — quantum measurement collapse and shot noise could in principle interfere with smooth gradient-based routing, but statevector simulation avoids this.

- **Load-balancing loss works with quantum experts.** The Switch Transformer auxiliary loss (Fedus et al., 2022) prevents expert collapse (all weight on one expert) for quantum experts just as it does for classical Transformers. This ensures all quantum circuits are utilized during training.

However, in the Cluster-Informed MoE, all quantum experts process the same full 180-ROI input — they differ only in their learned parameters, not their inputs. Expert "specialization" is implicit (different parameters) rather than explicit (different inputs).

**Stage 2: Circuit-Specialized Quantum MoE** (circuit-specific experts). Adds true expert specialization on top of the Stage 1 framework:

- **Each quantum expert processes a different, functionally coherent ROI subset.** The DMN expert sees only DMN ROIs (55 channels), the Salience expert sees only salience/limbic ROIs (29 channels), etc. This is explicit architectural specialization, not just learned parameter divergence.

- **Different QSVT dynamics per circuit.** Each quantum expert learns different QSVT polynomial coefficients, LCU mixing coefficients, and sim14 ansatz parameters — specializing in the temporal dynamics of its assigned brain circuit. The DMN expert may learn a different polynomial transformation than the Salience expert, reflecting the different frequency characteristics and temporal autocorrelation structures of these circuits.

Soft gating and load balancing (from Stage 1) remain unchanged in Stage 2.

## Contribution 3: Parameter Efficiency of Quantum Experts

Quantum experts achieve 93-97.7% of classical expert performance with 10-15x fewer parameters:

| Model | Classical Params | Quantum Params | Ratio | Q/C Performance |
|-------|:---:|:---:|:---:|:---:|
| SE (8Q) | 134,849 | 13,648 | 10x | 93% (0.5769/0.6193) |
| SE (10Q) | 134,849 | 16,936 | 8x | 94% (0.5800/0.6193) |
| Circuit MoE 4-expert (8Q) | 516,485 | 30,373 | 17x | 93% (0.5764/0.6167) |
| **Circuit MoE 4-expert (10Q)** | **516,485** | **34,885** | **15x** | **97.7% (0.6022/0.6167)** |
| Circuit MoE 2-expert (8Q) | 269,827 | 26,771 | 10x | 97% (0.5783/0.5987) |

The 10Q 4-expert result (0.6022) achieves 97.7% of classical performance with 15x fewer parameters, the strongest parameter efficiency result in this project. This improves over the initial 8Q finding where 4-expert MoE was broken (0.5139, 83% of classical).

This efficiency arises from the exponential dimensionality of Hilbert space: a 10-qubit circuit operates in a 1024-dimensional complex vector space (2^10) with only 80 rotation parameters per ansatz layer. A classical network would require a 1024x1024 weight matrix (~1M parameters) to represent a general linear transformation in the same space.

For near-term quantum hardware with limited qubit counts and circuit depths, parameter efficiency is not merely an academic point — it determines whether the model can be executed at all. A quantum MoE with 4 experts of ~8.7K parameters each (~35K total) is within reach of current superconducting processors, whereas a classical MoE with 4 experts of ~129K parameters each (~516K) has no quantum analog.

## Contribution 4: Circuit Specialization Reduces the Quantum Bottleneck

### The Bottleneck

In all prior experiments (SE, Cluster MoE, Learned MoE), both classical and quantum experts compress the full 180-ROI input through a single linear projection:

```
Linear(180, 64) → 64 rotation angles → 8-qubit quantum circuit
```

This 180-to-64 compression (2.8:1 ratio) is a severe classical bottleneck that occurs *before* any quantum processing. The quantum circuit never sees the full input — it operates on a lossy classical summary. This may explain why quantum and classical experts show similar performance ceilings: both are limited by the same pre-quantum information loss.

### How Circuit Specialization Helps

In the Circuit MoE, each quantum expert processes only its assigned circuit's ROIs:

| Expert | ROIs | Projection | Compression Ratio |
|--------|------|-----------|:---:|
| DMN | 55 | Linear(55, 64) | 0.86:1 (expansion) |
| Executive | 50 | Linear(50, 64) | 0.78:1 (expansion) |
| Salience | 29 | Linear(29, 64) | 0.45:1 (expansion) |
| SensoriMotor | 46 | Linear(46, 64) | 0.72:1 (expansion) |
| All-channel (prior) | 180 | Linear(180, 64) | 2.8:1 (compression) |

For the Salience expert with 29 ROIs, the projection is `Linear(29, 64)` — an **expansion** rather than compression. No information is lost before the quantum circuit. Even for the DMN expert (55 ROIs), the compression is minimal compared to 180→64.

This means circuit specialization could specifically benefit quantum experts more than classical experts, because quantum circuits are more sensitive to input information loss (they operate in a lower-dimensional Hilbert space and cannot compensate with additional classical layers).

**This is a testable hypothesis.** If Circuit MoE narrows the classical-quantum performance gap compared to all-channel models, it would provide evidence that the pre-quantum compression bottleneck was the limiting factor for quantum experts.

### Current Evidence

| Model | Classical AUC | Quantum AUC (8Q) | Gap (8Q) | Quantum AUC (10Q) | Gap (10Q) |
|-------|:---:|:---:|:---:|:---:|:---:|
| All-channel SE | 0.6193 | 0.5769 | 4.2 pts | 0.5800 | 3.9 pts |
| Circuit MoE 4-expert | 0.6167 | 0.5764 | 4.0 pts | **0.6022** | **1.5 pts** |
| Circuit MoE 2-expert | 0.5987 | 0.5783 | 2.0 pts | 0.5674 | 3.1 pts |

At 8Q, the gap is similar across configurations (~4 pts for SE and 4-expert), suggesting compression was not the dominant bottleneck. At 10Q, the 4-expert model narrows its gap to **1.5 pts** — 2.6x smaller than the SE gap (3.9 pts). This supports the hypothesis: circuit specialization reduces the quantum bottleneck, but only when the quantum feature space is wide enough (10Q = 80 angles) to exploit the reduced compression.

The 2-expert model regresses at 10Q, likely because its experts (84-96 ROIs to 80 angles) move from mild compression to near-parity or slight expansion — the transition is less dramatic than the 4-expert's pure expansion.

## Contribution 5: Reduced Reliance on Classical Dimension Reduction

### The Classical Dimension Reduction Problem in Variational Quantum Circuits

Variational quantum circuits require a fixed number of rotation angle inputs, determined by qubit count and ansatz depth. For the sim14 ansatz with 8 qubits and 2 layers, this is `4 × 8 × 2 = 64` rotation angles. When real-world data exceeds this dimensionality, a classical projection layer must compress the input before quantum processing:

```
Raw input (feature_dim) → nn.Linear(feature_dim, n_rots) → Sigmoid × 2π → rotation angles → quantum circuit
```

This `nn.Linear` is the **only** classical layer between the raw data and the quantum circuit — there is no hidden layer or nonlinearity before the sigmoid. Any information discarded by this projection is irrecoverable by the quantum circuit, regardless of its expressivity.

### Comparison: All Models

In SE, Cluster MoE, and Learned MoE, every expert sees all 180 ROIs and uses the same compression:

| Model | Expert Input | Projection | Compression Ratio |
|-------|:---:|-----------|:---:|
| Single Expert | 180 ROIs | `Linear(180, 64)` | 2.81:1 |
| Cluster MoE (each expert) | 180 ROIs | `Linear(180, 64)` | 2.81:1 |
| Learned MoE (each expert) | 180 ROIs | `Linear(180, 64)` | 2.81:1 |
| **Circuit MoE — DMN** | **55 ROIs** | **`Linear(55, 64)`** | **0.86:1 (expansion)** |
| **Circuit MoE — Executive** | **50 ROIs** | **`Linear(50, 64)`** | **0.78:1 (expansion)** |
| **Circuit MoE — Salience** | **29 ROIs** | **`Linear(29, 64)`** | **0.45:1 (expansion)** |
| **Circuit MoE — SensoriMotor** | **46 ROIs** | **`Linear(46, 64)`** | **0.72:1 (expansion)** |

Circuit MoE is the **only** architecture that reduces the per-expert input dimensionality. All other quantum models (SE, Cluster MoE, Learned MoE) use the identical `Linear(180, 64)` compression regardless of their routing or gating mechanisms.

### Three Structural Advantages

**1. Per-expert compression is eliminated (4-expert) or greatly reduced (2-expert).**

In the 4-expert configuration, every expert *expands* rather than compresses — all compression ratios are below 1:1. The `Linear(n_rois, 64)` is a learned feature transform (basis change), not a lossy bottleneck. In the 2-expert configuration, the compression is mild: Internal `Linear(84, 64)` = 1.31:1, External `Linear(96, 64)` = 1.50:1 — both far less severe than the 2.81:1 of all-channel models.

**2. Total information bandwidth is K× higher.**

Across all 4 experts, the model maps 180 ROIs into 4 × 64 = 256 rotation angles, compared to 64 in the Single Expert. The collective quantum processing capacity is 4× that of SE, despite each expert using an identical 8-qubit circuit.

**3. Each expert's projection is a simpler learning problem.**

The DMN expert's `Linear(55, 64)` only needs to represent within-DMN correlations among 55 functionally related ROIs. The Single Expert's `Linear(180, 64)` must simultaneously encode within-circuit and cross-circuit relationships for all 180 ROIs into 64 dimensions — a harder optimization target.

### Caveat: Gating Re-Compresses to 64D

Expert outputs are combined via soft gating: `weighted = Σ gate_weight_i × expert_output_i → (B, 64)`. The final representation is 64-dimensional. However, this is a *task-optimized selection* from a 256D pool (4 × 64D expert outputs), not a universal compression of 180D raw input.

### Empirical Evidence

| Model | 8Q AUC | 10Q AUC | Compression (8Q) | Compression (10Q) |
|-------|:---:|:---:|:---:|:---:|
| Q SE | 0.5769 | 0.5800 | 2.81:1 | 2.25:1 |
| Q Circuit MoE 2-expert | 0.5783 | 0.5674 | ~1.4:1 | ~1.1:1 |
| Q Circuit MoE 4-expert | 0.5764 | **0.6022** | <1:1 (expansion) | <1:1 (expansion) |

At 8Q, all quantum models plateaued at ~0.577, suggesting the reduced compression alone was insufficient. At 10Q, the 4-expert model breaks through to 0.6022 — demonstrating that reduced compression **plus** sufficient quantum feature space width yields a measurable performance gain.

**This contribution is both structural and empirically validated.** The architectural fact (less-compressed inputs, more total rotation angles) is complemented by the 10Q result showing the 4-expert model disproportionately benefits from additional qubits (+2.58 pts vs +0.31 pts for SE). This is exactly the predicted outcome of reduced reliance on classical dimension reduction: when the quantum feature space expands, experts that are already in expansion mode (29-55 ROIs to 80 angles) gain more than experts that remain compression-limited (180 ROIs to 80 angles).

## Summary of Novel Contributions

| # | Contribution | Type | Status |
|---|-------------|------|--------|
| 1 | Domain-informed quantum circuit partitioning for spatio-temporal data, addressing Multi-Chip Ensemble limitations | Methodological | Implemented |
| 2 | First quantum MoE framework for neuroimaging (soft gating, load balancing, expert specialization with quantum circuits) | Framework | Implemented |
| 3 | 10-15x parameter efficiency of quantum vs classical experts (93-97.7% performance) | Empirical | Confirmed — 10Q 4-expert achieves 97.7% at 15x fewer params |
| 4 | Circuit specialization reduces pre-quantum compression bottleneck | Hypothesis | Partially supported — 10Q 4-expert narrows gap to 1.5 pts (vs 3.9 pts for SE) |
| 5 | Reduced reliance on classical dimension reduction via circuit-specialized input splitting | Structural + Empirical | Confirmed — 10Q 4-expert breaks 8Q ceiling (+2.58 pts vs +0.31 pts for SE) |

## References

1. Park, J., et al. (2025). Addressing the Current Challenges of Quantum Machine Learning through Multi-Chip Ensembles. *arXiv:2505.08782v2*.
2. Nigg, J. T., et al. (2020). Toward a Revised Nosology for ADHD Heterogeneity. *Biol. Psychiatry: CNNI*, 5(8), 726-737.
3. Koirala, S., et al. (2024). Neurobiology of ADHD. *Nat. Rev. Neurosci.*, 25(12), 759-775.
4. Feng, A., et al. (2024). Functional imaging derived ADHD biotypes. *EClinicalMedicine*, 77, 102876.
5. Pan, N., et al. (2026). Mapping ADHD Heterogeneity and Biotypes. *JAMA Psychiatry*.
6. Cortese, S., et al. (2012). Toward systems neuroscience of ADHD: a meta-analysis of 55 fMRI studies. *Am. J. Psychiatry*, 169(10), 1038-1055.
7. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*, 23(120), 1-39.
