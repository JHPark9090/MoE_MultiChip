# Per-Expert Gradient Saliency Analysis — Methodology, Significance, and Interpretation Guide

**Date**: 2026-03-08
**Script**: `analyze_expert_saliency.py`
**Models**: Classical & Quantum Circuit MoE (2-expert and 4-expert configurations)
**Dataset**: ABCD resting-state fMRI, ADHD binary classification (N=4,458)

---

## 1. What Is Per-Expert Saliency Analysis?

### 1.1 The Core Idea

In a Circuit MoE (Mixture of Experts) model, each expert processes a **different subset of brain ROIs** corresponding to a specific neuroscience-informed circuit. For example, in the 4-expert configuration:

| Expert | Brain Circuit | Yeo-17 Networks | ROIs |
|--------|--------------|-----------------|:----:|
| Expert 0 | Default Mode Network (DMN) | DefaultA, DefaultB, DefaultC, TempPar | 41 |
| Expert 1 | Executive Control | ContA, ContB, ContC, DorsAttnA, DorsAttnB | 52 |
| Expert 2 | Salience | SalVentAttnA, SalVentAttnB, Limbic_TempPole, Limbic_OFC | 44 |
| Expert 3 | SensoriMotor | VisCent, VisPeri, SomMotA, SomMotB | 43 |

After training, each expert has learned its own internal representation of its assigned circuit. **Per-expert saliency analysis asks: within a given expert's circuit, which specific sub-networks and ROIs did the expert learn to prioritize for ADHD classification?**

### 1.2 How It Differs from Model-Level Saliency

Our previous interpretability analysis (`analyze_circuit_moe.py`) computed **model-level saliency**:

```
Model-level:  ∂(logits) / ∂(input)
```

This gradient flows through **all experts** and the **gating network** simultaneously. The result is a composite signal that conflates:
- What each expert learned
- How the gating network weights the experts
- Interactions between the gating and expert pathways

**Per-expert saliency** isolates each expert's contribution:

```
Per-expert:   ∂(expert_k output) / ∂(expert_k input ROIs)
```

This gradient flows through **only expert k's weights**, revealing what that specific expert — and only that expert — learned to attend to within its assigned brain circuit.

### 1.3 Why This Decomposition Matters

Consider a concrete example. Suppose the model-level analysis shows that the DMN circuit has high saliency overall. This could mean:
1. The DMN expert learned strong, informative features (expert contribution), or
2. The gating network upweights the DMN expert (routing contribution), or
3. Both

Per-expert analysis disentangles these. If the DMN expert's per-expert saliency is high for DefaultC but low for TempPar, we know the expert itself learned to prioritize the posterior cingulate cortex / precuneus region (DefaultC) over the temporal-parietal junction (TempPar) — **regardless of how much the gating network weights this expert**.

---

## 2. Scientific Significance

### 2.1 For Neuroscience

#### 2.1.1 Intra-Circuit Feature Hierarchies

Traditional neuroimaging analysis treats brain networks as monolithic entities (e.g., "the DMN is altered in ADHD"). Per-expert saliency reveals the **internal hierarchy within each network**: which sub-networks and individual ROIs within a circuit are most informative for ADHD classification.

This is a fundamentally new analytical lens. Rather than asking "is the DMN involved in ADHD?" (yes — this is well-established), we ask: **"within the DMN, does the model find DefaultA (medial prefrontal), DefaultB (angular gyrus), or DefaultC (posterior cingulate) most informative, and in what direction?"**

This directly addresses the heterogeneity of large-scale brain network involvement in ADHD. The DMN is not a single entity — it has distinct subsystems with different functional roles (Andrews-Hanna et al., 2010). Per-expert saliency maps these subsystem-level contributions to ADHD classification, providing finer-grained insight than network-level analyses.

#### 2.1.2 Cross-Expert Comparison: Classical vs. Quantum

We run the same analysis on both classical and quantum experts trained on identical brain circuits. If classical and quantum experts prioritize the **same** sub-networks and ROIs, this suggests the feature hierarchy is driven by the data (robust biological signal). If they prioritize **different** features, this reveals architecture-dependent biases in what the model learns — an important finding for interpretable AI in neuroimaging.

#### 2.1.3 Circuit Granularity Effects

By comparing 4-expert (fine-grained: DMN, Executive, Salience, SensoriMotor) vs. 2-expert (coarse: Internal, External) configurations, we can assess whether finer circuit decomposition changes what features each expert learns. The Internal expert (DMN + Executive) might prioritize different features than the DMN expert alone, revealing interactions between sub-circuits.

### 2.2 For Psychiatry

#### 2.2.1 Biomarker Discovery at Sub-Network Resolution

Current ADHD biomarker research operates at the level of broad brain networks (Cortese et al., 2012; Castellanos & Proal, 2012). Per-expert saliency can identify **specific ROIs within specific circuits** that are most ADHD-discriminative, advancing biomarker precision from "DMN dysfunction" to, e.g., "area p32 in DefaultB shows the strongest ADHD-associated signal within the DMN circuit."

These ROI-level findings, if replicated across seeds and architectures, can inform targeted interventions (e.g., neurofeedback, TMS) by identifying specific cortical targets within broader dysfunctional circuits.

#### 2.2.2 Signed Saliency and Directional Hypotheses

The signed saliency tells us the **direction** of each ROI's influence:

- **+ADHD**: Increasing this ROI's temporal activity pushes the expert's output toward the ADHD prediction. This suggests the ROI exhibits hyperactivation or dysregulated dynamics in ADHD.
- **-ADHD**: Increasing this ROI's temporal activity pushes the expert's output away from ADHD. This suggests the ROI exhibits hypoactivation or reduced dynamics in ADHD.

These directional findings generate testable hypotheses for task-based and pharmacological fMRI studies. For example, if area FEF (frontal eye field) in the Executive circuit shows strong +ADHD saliency, this predicts that ADHD subjects should show hyperactivation of FEF during attentional tasks — a prediction that can be tested independently.

#### 2.2.3 ADHD+ vs. ADHD- Differential Saliency

For each network and ROI, we statistically compare saliency between ADHD+ and ADHD- subjects using Welch's t-test. A significant difference means the expert's sensitivity to that brain region **differs between diagnostic groups** — the expert is not just using the region for classification, but using it **differently** depending on the subject's ADHD status.

This goes beyond standard group-difference neuroimaging (which compares raw brain activity) to compare **how a trained ADHD-classification model processes brain activity** differently for the two groups.

### 2.3 For AI / Machine Learning

#### 2.3.1 MoE Interpretability Methodology

This analysis establishes a general methodology for interpreting Mixture of Experts models in scientific domains. The per-expert saliency decomposition can be applied to any MoE where experts process different input partitions — not just brain circuits, but multi-modal medical data, multi-sensor IoT, or multi-source document analysis.

#### 2.3.2 Quantum vs. Classical Feature Learning

Comparing per-expert saliency between quantum and classical experts provides a novel lens on quantum machine learning. If quantum experts (with 10-17x fewer parameters) identify the same critical ROIs as classical experts, this strengthens the case for quantum parameter efficiency. If they identify different ROIs, it suggests quantum circuits explore a different feature space — a finding relevant to quantum advantage research.

---

## 3. Novel Contributions and Potential Breakthroughs

### 3.1 First Intra-Circuit Feature Hierarchy for ADHD

To our knowledge, no prior work has decomposed brain circuit contributions to ADHD classification at the sub-network level using per-expert gradient analysis. Existing MoE interpretability work (Shazeer et al., 2017; Fedus et al., 2022) focuses on NLP/CV domains. Applying per-expert saliency decomposition to neuroscience-informed brain circuits is a methodological contribution.

### 3.2 Data-Driven Sub-Network Ranking within Neuroscience Circuits

The result of our analysis is an empirical ranking of sub-networks within each neuroscience-defined circuit, ordered by their relevance to ADHD classification. For example, within the Salience circuit:

> SalVentAttnA > SalVentAttnB > Limbic_OFC > Limbic_TempPole

This ranking is **data-driven** (learned from 4,458 subjects' resting-state fMRI) rather than hypothesis-driven, potentially revealing sub-network importance orderings that differ from prior theoretical expectations.

### 3.3 Cross-Architecture Robustness of Neural Biomarkers

If the same ROIs emerge as top-saliency across classical 4e, classical 2e, quantum 4e, and quantum 2e models, these represent **architecture-robust biomarkers** — brain regions whose ADHD-relevance is not an artifact of a specific model architecture. This cross-validation across architectures provides stronger evidence than any single model can.

### 3.4 Linking Expert Specialization to ADHD Subtypes

Combined with our heterogeneity analysis (gradient-based clustering that identifies 3 ADHD subtypes: SalVentAttnA, Limbic_OFC, Limbic_TempPole), per-expert saliency can reveal which experts are most informative for which subtypes. This bridges the MoE architecture with psychiatric nosology.

---

## 4. How the Analysis Works — Step by Step

### Step 1: Load the Trained Model and Data

```
Input:  Trained Circuit MoE checkpoint (.pt file)
        ABCD fMRI test set (669 subjects, 180 ROIs, 363 timepoints)
```

The checkpoint contains the trained model weights, architecture configuration, and the circuit-to-ROI mapping. The data is loaded with the same preprocessing as training (transpose to B×C×T format, same train/val/test split).

### Step 2: For Each Expert, Compute Per-Expert Gradient Saliency

For each test subject and each expert k:

1. **Create a fresh input tensor** that requires gradient tracking:
   ```python
   x_input = data.detach().clone()     # (B, T, 180)
   x_input.requires_grad_(True)
   ```

2. **Extract this expert's ROI subset**:
   ```python
   x_subset = x_input[:, :, roi_idx_k]  # (B, T, n_rois_k)
   ```
   For example, if expert k is the DMN expert, `roi_idx_k` contains the 41 DMN ROI indices.

3. **Forward through this expert only** (not through gating or other experts):
   ```python
   # Classical expert:
   h_k = expert_k(x_subset)            # (B, T, n_rois_k) → (B, H=64)

   # Quantum expert (v2 with temporal pooling):
   x_q = avg_pool1d(x_subset, pool_factor=10)  # (B, n_rois_k, T_pooled)
   h_k = expert_k(x_q)                 # (B, H)
   ```

4. **Backpropagate from expert output to input**:
   ```python
   h_k.sum().backward()
   grad = x_input.grad                 # (B, T, 180)
   ```
   Using `sum()` ensures each sample in the batch contributes independently.

5. **Extract gradients at this expert's ROIs only**:
   ```python
   grad_expert = grad[:, :, roi_idx_k]  # (B, T, n_rois_k)
   ```

6. **Temporal averaging** to get per-ROI saliency:
   ```python
   abs_saliency = grad_expert.abs().mean(dim=1)   # (B, n_rois_k) — magnitude
   signed_saliency = grad_expert.mean(dim=1)      # (B, n_rois_k) — direction
   ```

**Key insight**: Because the gradient is computed from `h_k` (expert k's output) rather than from the model's final logits, the gradient flows **only through expert k's weights**. The gating network, other experts, and the classifier head are not in the computational graph. This isolates expert k's learned feature priorities.

### Step 3: Aggregate to Network Level

For each expert, group its ROIs by Yeo-17 sub-network membership. For example, the DMN expert's 41 ROIs belong to DefaultA, DefaultB, DefaultC, and TempPar. For each sub-network:

1. **Identify local ROI indices** belonging to this network within the expert
2. **Average saliency** across ROIs in the network → one value per subject per network
3. **Compute group statistics**:
   - Mean absolute saliency (overall importance)
   - Mean signed saliency (direction: +ADHD or -ADHD)
   - Welch's t-test comparing ADHD+ vs. ADHD- subjects

This produces a table like:

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- p |
|---------|:------:|:------------:|:------:|:---------:|:----------------:|
| DefaultC | 8 | 0.000142 | +0.000089 | +ADHD | 0.023* |
| DefaultA | 16 | 0.000118 | +0.000045 | +ADHD | 0.187 |
| DefaultB | 12 | 0.000095 | -0.000032 | -ADHD | 0.412 |
| TempPar | 5 | 0.000078 | +0.000011 | +ADHD | 0.891 |

*(Example values — actual results will come from the analysis.)*

### Step 4: Aggregate to ROI Level

For each individual ROI within each expert:

1. **Extract per-subject saliency** at that ROI
2. **Compute**: mean absolute saliency, mean signed saliency, direction
3. **Welch's t-test**: ADHD+ vs. ADHD- comparison
4. **Annotate** with anatomical labels (from `hcp_mmp1_labels.py`): region name, long name, lobe, Yeo-17 network

This produces a ranked list of ROIs per expert, e.g.:

| Rank | ROI | Region | Network | Abs Saliency | Direction | p-value |
|:----:|:---:|--------|---------|:------------:|:---------:|:-------:|
| 1 | 23 | POS2 (Parieto-occipital Sulcus) | DefaultA | 0.000312 | +ADHD | 0.018* |
| 2 | 8 | PCV (Precuneus Visual) | DefaultC | 0.000289 | +ADHD | 0.034* |
| 3 | 31 | p32 (Area p32) | DefaultB | 0.000256 | -ADHD | 0.201 |

*(Example values.)*

### Step 5: Cross-Expert Comparison

Compare across all experts:

1. **Dominant network per expert**: Which sub-network does each expert prioritize most (by absolute saliency)?
2. **Intra-circuit feature hierarchy**: Rank all sub-networks within each expert, producing a hierarchy like:
   ```
   DMN expert:        DefaultC > DefaultA > DefaultB > TempPar
   Executive expert:  ContA > DorsAttnB > DorsAttnA > ContB > ContC
   Salience expert:   SalVentAttnB > SalVentAttnA > Limbic_OFC > Limbic_TempPole
   SensoriMotor:      SomMotA > VisPeri > SomMotB > VisCent
   ```

3. **Significant findings summary**: All networks and ROIs reaching p<0.05 in the ADHD+ vs. ADHD- comparison, organized by expert.

### Step 6: Generate Outputs

The analysis produces three output files per model:

| Output | Content | Purpose |
|--------|---------|---------|
| `expert_saliency_report.md` | Full markdown report with tables | Human-readable results |
| `expert_saliency_results.json` | All statistics in JSON format | Programmatic downstream analysis |
| `expert_saliency_per_subject.npz` | Per-subject saliency matrices (N × n_rois_k) | Cross-seed stability, clustering, heterogeneity |

---

## 5. How to Interpret the Results

### 5.1 Reading the Network-Level Table

Each expert produces a table of intra-circuit sub-network saliency. Here's how to interpret each column:

| Column | Meaning |
|--------|---------|
| **Abs Saliency** | How much this sub-network's ROIs influence the expert's output, regardless of direction. Higher = more important for the expert's representation. |
| **Signed Saliency** | The net direction of influence. Positive (+) = increasing activity in these ROIs pushes expert output toward ADHD. Negative (-) = toward healthy. |
| **Direction** | Summary of signed saliency: "+ADHD" or "-ADHD". |
| **ADHD+ vs ADHD- p** | Whether the expert processes this sub-network **differently** for ADHD+ vs. ADHD- subjects. Significant p means the expert's sensitivity to this region is diagnosis-dependent. |

**Key interpretation patterns:**

- **High abs saliency + significant p-value**: The expert strongly relies on this sub-network AND uses it differently for ADHD+ vs. ADHD-. This is the strongest evidence for a diagnosis-relevant feature.
- **High abs saliency + non-significant p-value**: The expert strongly relies on this sub-network but processes it similarly for both groups. This may reflect general brain structure rather than ADHD-specific signal.
- **Low abs saliency + significant p-value**: The expert weakly relies on this sub-network overall, but what little it uses differs between groups. Worth noting but lower confidence.

### 5.2 Reading the ROI-Level Table

The ROI-level table provides the finest-grained view. The top ROIs by absolute saliency are the specific brain regions most critical to this expert's representation.

**What a top-ranked ROI means**: The expert's output is most sensitive to changes in this ROI's temporal activity. If you perturbed this ROI's signal, the expert's representation would change the most.

**The +ADHD / -ADHD direction**:
- A +ADHD ROI with high saliency suggests the expert learned that temporal dynamics in this region are positively associated with ADHD — consistent with hyperactivation, dysregulated connectivity, or altered temporal dynamics in ADHD.
- A -ADHD ROI with high saliency suggests the expert learned that activity in this region is associated with healthy control — consistent with hypoactivation or reduced functional connectivity in ADHD.

### 5.3 Reading the Cross-Expert Comparison

The cross-expert comparison reveals the overall architecture of learned feature priorities:

1. **Do experts match expectations?** If the DMN expert's top sub-network is DefaultA (medial prefrontal) and the Salience expert's top sub-network is SalVentAttnA (anterior insula), the model has learned feature priorities consistent with neuroscience literature on these circuits' hub regions.

2. **Are there surprises?** If the Executive expert's top sub-network is DorsAttnB rather than ContA, this suggests the model found dorsal attention features more discriminative than frontoparietal control features for ADHD — a potentially novel finding.

3. **Classical vs. Quantum agreement**: If the intra-circuit hierarchy is the same for classical and quantum experts (e.g., both rank DefaultC > DefaultA > DefaultB within DMN), the hierarchy is architecture-robust. If they differ, it suggests different model classes extract different information from the same circuit.

### 5.4 Caveats and Limitations

1. **Gradient saliency is a first-order approximation**: It captures local sensitivity around the test data distribution. It does not capture nonlinear interactions or counterfactual effects. Interpret as "what the model is most sensitive to," not "what causes ADHD."

2. **Signed saliency direction depends on the learned representation**, not directly on biological activity direction. A +ADHD direction means the expert learned to associate higher values at this ROI with ADHD, which could reflect hyperactivation, increased variability, or other temporal features.

3. **Multiple comparisons**: With 41-52 ROIs per expert and 4 experts, there are 160+ ROI-level tests. Apply Bonferroni or FDR correction when interpreting p-values. The report uses uncorrected p-values; significant findings should be confirmed after correction.

4. **Single seed (seed=2025)**: Initial analysis runs on seed=2025 checkpoints only. Cross-seed stability should be verified by running on additional seed checkpoints (2024, 2026, 2027, 2028).

5. **Temporal averaging**: Saliency is averaged across all 363 timepoints. Time-resolved saliency could reveal temporal dynamics (e.g., certain ROIs are more salient early vs. late in the scan), but this is not captured in the current analysis.

---

## 6. What We Run

### 6.1 Models Analyzed

All 4 Circuit MoE v5 checkpoints (seed=2025):

| Model | Checkpoint | Config | Experts |
|-------|-----------|--------|:-------:|
| Classical 4e | `CircuitMoE_classical_adhd_3_49777876.pt` | adhd_3 | DMN, Exec, Sal, SM |
| Classical 2e | `CircuitMoE_classical_adhd_2_49777878.pt` | adhd_2 | Internal, External |
| Quantum 4e | `CircuitMoE_quantum_adhd_3_49777879.pt` | adhd_3 | DMN, Exec, Sal, SM |
| Quantum 2e | `CircuitMoE_quantum_adhd_2_49777880.pt` | adhd_2 | Internal, External |

### 6.2 Analysis Output Structure

```
analysis/
├── expert_saliency_v5_classical_adhd_3/
│   ├── expert_saliency_report.md       # Detailed report
│   ├── expert_saliency_results.json    # Machine-readable results
│   └── expert_saliency_per_subject.npz # Per-subject data
├── expert_saliency_v5_classical_adhd_2/
│   └── ...
├── expert_saliency_v5_quantum_8q_d3_adhd_3/
│   └── ...
└── expert_saliency_v5_quantum_8q_d3_adhd_2/
    └── ...
```

### 6.3 Key Questions the Analysis Will Answer

1. **Within each brain circuit, which sub-networks are most ADHD-relevant?** (Intra-circuit hierarchy)
2. **Which specific ROIs drive each expert's ADHD classification?** (ROI-level biomarker candidates)
3. **Do classical and quantum experts learn the same feature priorities?** (Architecture robustness)
4. **Does the 4-expert vs. 2-expert decomposition change what features are learned?** (Granularity effects)
5. **Are there sub-networks or ROIs with significant ADHD+ vs. ADHD- saliency differences?** (Diagnosis-dependent feature processing)
6. **Do the per-expert findings align with or contradict existing ADHD neuroimaging literature?** (Neuroscience validation)

---

## 7. Relation to Prior Analyses

### 7.1 Model-Level Interpretability (existing)

Our existing analysis (`analyze_circuit_moe.py`) found:
- DefaultC consistently +ADHD across all 4 models (high-confidence finding)
- SalVentAttnB consistently +ADHD (salience disruption)
- Limbic_OFC consistently -ADHD (OFC hypoactivation)
- ContB consistently -ADHD (executive dysfunction)

Per-expert analysis will **decompose** these model-level findings. For example, does the DefaultC +ADHD signal come primarily from the DMN expert, or is it also reflected in the Internal expert (which includes DMN)?

### 7.2 Heterogeneity Analysis (existing)

Our heterogeneity analysis identified 3 ADHD subtypes:
1. SalVentAttnA subtype (~35%): insular/salience processing deficits
2. Limbic_OFC subtype (~10-33%): reward/motivational deficits
3. Limbic_TempPole subtype (~2-8%): social/emotional deficits

Per-expert saliency can be cross-referenced with these subtypes: which experts show the highest saliency for ROIs associated with each subtype? This links expert specialization to patient heterogeneity.

### 7.3 Multi-Seed Robustness (ongoing)

Once per-expert saliency is computed for multiple seeds, we can assess **saliency stability**: do the same ROIs emerge as top-saliency across seeds? Stable ROIs are stronger biomarker candidates than seed-dependent ones.

---

## 8. Expected Impact

### 8.1 For the Paper

Per-expert saliency analysis provides:
- A novel interpretability methodology for circuit-specialized MoE models in neuroimaging
- Sub-network-resolution ADHD biomarker candidates, advancing beyond network-level findings
- Cross-architecture validation (classical vs. quantum) of learned feature priorities
- Direct link between MoE expert specialization and established ADHD neuroscience

### 8.2 For the Field

If validated across seeds and architectures, this methodology establishes:
- A general framework for interpreting domain-specialized MoE models
- Evidence for (or against) quantum parameter efficiency in neuroscience feature extraction
- ROI-level targets for follow-up neurofeedback or stimulation studies

---

## 9. References

- Andrews-Hanna, J. R., et al. (2010). Functional-anatomic fractionation of the brain's default network. *Neuron*, 65(4), 550-562.
- Castellanos, F. X., & Proal, E. (2012). Large-scale brain systems in ADHD: beyond the prefrontal-striatal model. *Trends in Cognitive Sciences*, 16(1), 17-26.
- Cortese, S., et al. (2012). Toward systems neuroscience of ADHD: a meta-analysis of 55 fMRI studies. *American Journal of Psychiatry*, 169(10), 1038-1055.
- Fedus, W., et al. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR*, 23(120), 1-39.
- Feng, C., et al. (2024). Neuroimaging-based ADHD subtypes. *EClinicalMedicine*.
- Nigg, J. T., et al. (2020). Working memory and vigilance as multivariate endophenotypes related to ADHD. *Biological Psychiatry: CNNI*, 5(7), 673-681.
- Pan, N., et al. (2026). Circuit-specific dysfunction in ADHD. *JAMA Psychiatry*.
- Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR 2017*.
