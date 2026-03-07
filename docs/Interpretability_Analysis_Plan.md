# Circuit MoE Interpretability Analysis Plan

**Date**: 2026-03-07
**Status**: Analysis script implemented (`analyze_circuit_moe.py`), not yet run.

## Goal

Extract scientific interpretations from trained Circuit MoE models at three levels of granularity:

1. **Circuit level** — Which brain circuits are differentially engaged for ADHD+ vs ADHD- subjects?
2. **Network level** — Within each circuit, which Yeo-17 sub-networks contribute most?
3. **ROI level** — Which specific brain regions are most diagnostic?

## Three Analysis Methods

### Method 1: Gate Weight Analysis (Circuit Level)

**What it measures**: The gating network produces soft weights over experts for each input subject. By comparing mean gate weights between ADHD+ and ADHD- subjects, we can identify which circuit experts are preferentially engaged for each class.

**How it works**:
- Run the test set through the trained model
- Record gate weights (B, K) for each subject
- Split by class: ADHD+ (label=1) vs ADHD- (label=0)
- Compute mean, std per expert per class
- Welch's t-test for statistical significance

**Interpretation**: If the DMN expert receives significantly higher gate weight for ADHD+ subjects (p < 0.05), this suggests DMN circuit activity is differentially involved in ADHD classification, consistent with Nigg (2020) and Feng (2024).

**Limitations**: The Switch Transformer load-balancing loss (alpha=0.1) encourages uniform routing, which may suppress class-dependent routing differences. Any differences that survive this regularization are therefore conservative estimates.

### Method 2: Gradient Saliency (All Levels)

**What it measures**: The gradient of the model output with respect to the input, dOutput/dInput, quantifies how much each input feature (ROI x timepoint) influences the prediction.

**How it works**:
- Enable gradient tracking on the input tensor (B, T, 180)
- Forward pass through the model
- Backward pass from the logit output
- Saliency = mean absolute gradient over the time dimension -> (B, 180)
- Average saliency across subjects, split by class
- Aggregate by circuit, network, and ROI

**Three levels of aggregation**:

| Level | Aggregation | Example output |
|-------|------------|----------------|
| Circuit | Mean saliency across all ROIs in each circuit | "DMN circuit: 0.0045, Executive: 0.0038" |
| Network | Mean saliency across ROIs in each Yeo-17 network | "DefaultA: 0.0052, ContA: 0.0041" |
| ROI | Per-ROI saliency | "ROI 112 (DefaultA, DMN): 0.0068" |

**Interpretation**: Higher saliency = the model is more sensitive to that region's signal. Differences between ADHD+ and ADHD- saliency reveal which regions are class-discriminative.

### Method 3: Input Projection Weight Analysis (Network + ROI Level)

**What it measures**: The first linear layer in each expert maps ROIs to the hidden representation. The magnitude of learned weights reveals which ROIs (and by extension, networks) each expert learns to attend to.

**How it works**:
- Extract weight matrix W from each expert's first layer:
  - Classical: `CircuitExpert.input_projection.weight` (hidden_dim, n_rois)
  - Quantum: `QuantumTSTransformer.feature_projection.weight` (n_rots, n_rois)
- Compute L2 norm of each column -> per-ROI weight magnitude
- Group by Yeo-17 network within each circuit

**Interpretation**: If the DMN expert (which receives DefaultA/B/C + TempPar ROIs) assigns higher weight norms to DefaultA than DefaultC, this suggests DefaultA is more informative for the expert's representation. This is complementary to gradient saliency — weights show what the model learned to attend to, while gradients show what matters for a specific prediction.

## Circuit Definitions

### 4-Expert Configuration (adhd_3)

| Expert | Circuit | Yeo-17 Networks | ROIs | Neurobiological Basis |
|--------|---------|----------------|:----:|----------------------|
| 0 | DMN | DefaultA (21), DefaultB (16), DefaultC (10), TempPar (8) | 55 | DMN suppression failure (Nigg 2020); default mode regions show greatest biotype involvement (Feng 2024) |
| 1 | Executive | ContA (13), ContB (8), ContC (4), DorsAttnA (13), DorsAttnB (12) | 50 | Top-down executive dysfunction, frontoparietal hypoactivation (Cortese 2012) |
| 2 | Salience | SalVentAttnA (15), SalVentAttnB (6), Limbic_TempPole (4), Limbic_OFC (4) | 29 | Emotional dysregulation, salience network dysfunction (Nigg 2020) |
| 3 | SensoriMotor | VisCent (10), VisPeri (12), SomMotA (16), SomMotB (8) | 46 | Somatomotor hyperactivation in children (Cortese 2012, Feng 2024) |

### 2-Expert Configuration (adhd_2)

| Expert | Circuit | Yeo-17 Networks | ROIs |
|--------|---------|----------------|:----:|
| 0 | Internal | DefaultA/B/C, TempPar, Limbic_TempPole, Limbic_OFC, SalVentAttnA/B | 84 |
| 1 | External | ContA/B/C, DorsAttnA/B, VisCent, VisPeri, SomMotA/B | 96 |

## Models to Analyze

| Model | Checkpoint | Test AUC |
|-------|-----------|:--------:|
| Classical 4-expert | `CircuitMoE_classical_adhd_3_49731003.pt` | 0.6167 |
| Classical 2-expert | `CircuitMoE_classical_adhd_2_49731010.pt` | 0.5987 |
| Quantum v2 4-expert | `CircuitMoE_quantum_adhd_3_49767122.pt` | 0.5764 |
| Quantum v2 2-expert | `CircuitMoE_quantum_adhd_2_49767123.pt` | 0.5783 |

## Example Scientific Claims (if supported by data)

### Circuit-Level Claims

- "ADHD+ subjects show increased routing to the DMN expert (gate weight = X vs Y, p < 0.05), consistent with DMN suppression failure as a hallmark of ADHD (Nigg 2020, Koirala 2024)."
- "The Salience/Affective expert shows higher saliency for ADHD+ than ADHD- subjects (diff = +Z), supporting emotional dysregulation as a key ADHD dimension (Nigg 2020)."
- "The SensoriMotor expert shows the smallest class-dependent routing difference, consistent with sensorimotor involvement being more prominent in childhood ADHD samples (Cortese 2012) than in the ABCD cohort (ages 9-10)."

### Network-Level Claims

- "Within the DMN expert, DefaultA (posterior medial cortex, angular gyrus) contributes more to ADHD classification than DefaultC (medial prefrontal), as shown by both higher gradient saliency and input projection weights."
- "Within the Executive expert, the Dorsal Attention networks (DorsAttnA/B) show higher saliency than Frontoparietal Control networks (ContA/B/C), suggesting attentional orienting deficits are more diagnostic than executive control per se."

### ROI-Level Claims

- "ROI 112 (left posterior cingulate cortex, DefaultA) shows the highest gradient saliency among all 180 ROIs, consistent with the PCC's role as a hub of the DMN and its established involvement in ADHD (Castellanos & Proal, 2012)."
- "The top 5 most salient ROIs span 3 different circuits (2 DMN, 2 Executive, 1 Salience), supporting the distributed nature of ADHD-related dysfunction."

### Cross-Model Comparisons

- "Classical and quantum models show concordant circuit-level routing patterns (Spearman rho = X), suggesting the interpretability findings are robust to the choice of expert architecture."
- "The 4-expert model provides finer-grained interpretation than the 2-expert model: the DMN and Salience circuits (merged into 'Internal' in the 2-expert config) show distinct routing patterns when separated."

## Implementation

### Script: `analyze_circuit_moe.py`

```
Usage:
    python analyze_circuit_moe.py \
        --checkpoint=checkpoints/CircuitMoE_classical_adhd_3_49731003.pt \
        --output-dir=analysis/classical_adhd_3
```

**Outputs per model**:
- `interpretability_report.md` — formatted markdown with all results
- `results.json` — raw numerical results for further analysis

### SLURM: `scripts/run_interpretability_analysis.sh`

Runs all 4 models sequentially. Estimated time: < 30 min total (inference only, no training).

## Caveats

1. **Correlation, not causation**: Gradient saliency shows which inputs the model uses, not which brain regions cause ADHD. The model may exploit confounds (e.g., motion artifacts correlating with ADHD diagnosis).

2. **Model performance ceiling**: All models achieve only 0.58-0.62 AUC. Interpretations from a model that is barely above chance should be treated as exploratory, not confirmatory.

3. **Load balancing suppresses routing differences**: The auxiliary loss encourages uniform routing, so any class-dependent gate weight differences are conservative.

4. **Parcellation limitations**: HCP-MMP1 180 ROIs cover cortex only. Subcortical structures (caudate, pallidum, nucleus accumbens) implicated in ADHD (Pan 2026, Nigg 2020) are absent.

5. **Single seed**: All models trained with seed=2025. Interpretability findings should ideally be replicated across multiple seeds.

## References

1. Nigg et al. (2020) Biol. Psychiatry: CNNI — ADHD heterogeneity dimensions
2. Koirala et al. (2024) Nat. Rev. Neurosci. — ADHD neurobiology review
3. Feng et al. (2024) EClinicalMedicine — ADHD biotypes via ABCD dataset
4. Pan et al. (2026) JAMA Psychiatry — ADHD biotypes via morphometry
5. Cortese et al. (2012) Am. J. Psychiatry — Meta-analysis of 55 fMRI studies
6. Castellanos & Proal (2012) Trends Cogn. Sci. — Large-scale brain systems in ADHD
