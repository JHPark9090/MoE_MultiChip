# Interpretability Visualization Guide

**Date**: 2026-03-08
**Script**: `visualize_interpretability.py`
**Data source**: `analyze_circuit_moe.py` output (`results.json` per model)
**Mapping**: Corrected Yeo-17 (v5, bilateral volumetric overlap)

## Overview

The Circuit MoE interpretability analysis asks: **what has the model learned about ADHD?** For each trained model, we examine three levels of learned representations — circuit-level gate routing, network-level gradient saliency, and ROI-level gradient saliency — to understand which brain regions and networks the model associates with ADHD prediction. Six visualizations are generated per model.

**Output directories**: `analysis/v5_{model_type}_{config}/figures/`

| Model | Directory |
|-------|-----------|
| Classical 4-expert | `analysis/v5_classical_adhd_3/figures/` |
| Classical 2-expert | `analysis/v5_classical_adhd_2/figures/` |
| Quantum 4-expert | `analysis/v5_quantum_8q_d3_adhd_3/figures/` |
| Quantum 2-expert | `analysis/v5_quantum_8q_d3_adhd_2/figures/` |

---

## Figure 1: Gate Weights Comparison

**File**: `gate_weights_comparison.pdf/png`

### What It Shows

A grouped bar chart comparing mean gate weights for ADHD+ (red) vs ADHD- (blue) subjects, per expert circuit. Error bars show standard deviation. Significance stars appear above each pair:
- `***` p < 0.001, `**` p < 0.01, `*` p < 0.05, `n.s.` not significant

A dashed horizontal line marks the uniform routing baseline (1/N_experts).

### How It Is Calculated

1. For each test subject, extract the gate weight vector from the MoE gating network. This is the softmax output that determines how much of the subject's data flows to each expert. For a 4-expert model: `[w_DMN, w_Executive, w_Salience, w_SensoriMotor]` summing to 1.0. For a 2-expert model: `[w_Internal, w_External]`.

2. Split subjects into ADHD+ (diagnosed) and ADHD- (control) groups.

3. Compute mean and standard deviation of each expert's gate weight within each group.

4. Run an independent-samples t-test between ADHD+ and ADHD- gate weights for each expert to assess statistical significance.

### How to Interpret

- **Bars above the dashed uniform line**: The model routes more data to that expert than expected. If one group (ADHD+ or ADHD-) is significantly higher, the model has learned class-conditional routing.
- **Significant stars (*)**: The model routes ADHD+ and ADHD- subjects differently through that expert. This is the strongest evidence that the gating mechanism captures diagnostic-relevant information.
- **`n.s.` on all experts**: Load balancing loss (auxiliary loss alpha=0.1) has suppressed class-conditional routing differences. The model still learns discriminative features inside each expert, but the gating itself is not class-informative.
- **Small error bars**: Subjects within each group are routed similarly — homogeneous routing strategy.

### v5 Scientific Findings

- **Only the quantum 2-expert model shows significant gate differentiation** (p=0.038): ADHD+ routes more to Internal (DMN+Executive), ADHD- more to External (Salience+SensoriMotor).
- All other models show `n.s.` across all experts, confirming that load balancing suppresses gate-level class signal in most configurations.
- The quantum 2-expert finding is the only statistically significant circuit-level interpretability result.

---

## Figure 2: Circuit-Level Saliency

**File**: `circuit_saliency.pdf/png`

### What It Shows

Two side-by-side bar charts:
- **Left panel**: Absolute saliency per expert circuit (how much each circuit contributes to the prediction, regardless of direction).
- **Right panel**: Signed saliency per expert circuit with +ADHD (red) or -ADHD (blue) labels indicating direction.

### How It Is Calculated

1. **Gradient saliency** is computed via backpropagation: for each test subject, compute the gradient of the model's ADHD prediction logit with respect to each input ROI's fMRI signal.

2. **Absolute saliency** per circuit: average the absolute gradient values across all ROIs assigned to that circuit, then average across all test subjects.

3. **Signed saliency** per circuit: average the raw (signed) gradient values across all ROIs assigned to that circuit, then average across all test subjects. A positive value means increasing activity in that circuit's ROIs pushes the prediction toward ADHD (+ADHD); negative means it pushes toward healthy control (-ADHD).

### How to Interpret

- **Left panel (absolute)**: Taller bars = the model pays more attention to that circuit overall. This indicates which brain circuit groups are most informative for the ADHD/non-ADHD distinction, regardless of direction.
- **Right panel (signed)**: Red (+ADHD) bars = increasing activity in this circuit pushes prediction toward ADHD. Blue (-ADHD) bars = increasing activity pushes prediction away from ADHD.
- **All bars the same height (left)**: The model distributes attention roughly equally across circuits.
- **Opposite signs between circuits**: The model has learned a differential pattern — some circuits are ADHD-associated, others are protective.

### v5 Scientific Findings

- In the classical 4-expert model, Executive has the highest absolute and signed saliency (+ADHD direction).
- In quantum models, the pattern differs: Executive shows -ADHD direction and SensoriMotor shows +ADHD, suggesting quantum circuits learn different feature extraction strategies than classical ones.
- Saliency magnitudes differ by ~1000× across models, so only within-model comparisons are valid.

---

## Figure 3: Network-Level Saliency

**File**: `network_saliency.pdf/png`

### What It Shows

Two horizontal bar charts:
- **Left panel**: All 17 Yeo networks ranked by absolute saliency (bottom to top).
- **Right panel**: The same 17 networks showing signed saliency with +ADHD (red) and -ADHD (blue) bars, in the same order as the left panel.

### How It Is Calculated

1. For each of the 17 Yeo networks, compute the mean gradient saliency across all ROIs belonging to that network (using the corrected bilateral Yeo-17 mapping).

2. **Absolute**: mean of |gradient| across ROIs in the network, averaged across subjects.

3. **Signed**: mean of raw gradient across ROIs in the network, averaged across subjects. Positive = +ADHD, negative = -ADHD.

4. Networks are sorted by absolute saliency for readability (highest at top).

### How to Interpret

- **Top of left panel**: The networks the model considers most important for ADHD prediction. These are the brain networks that, when perturbed, most change the model's output.
- **Right panel, red bars (+)**: Networks where increased activity is associated with ADHD prediction. Example: DefaultC (+ADHD) means higher default mode C activity → model predicts ADHD.
- **Right panel, blue bars (-)**: Networks where increased activity is associated with healthy control prediction. Example: Limbic_OFC (-ADHD) means higher OFC activity → model predicts non-ADHD.
- **Look for consistency across models**: Networks that show the same sign (+ or -) across all 4 models are the most reliable findings.

### v5 Scientific Findings

**Consistently +ADHD across all 4 models**:
- **DefaultC** (medial prefrontal / posterior cingulate) — aligns with DMN hyperactivation in ADHD
- **SalVentAttnB** (anterior insula) — aligns with salience network disruption in ADHD

**Consistently -ADHD across all 4 models**:
- **Limbic_OFC** (orbitofrontal cortex) — aligns with OFC hypoactivation / reward processing deficits
- **ContB** (frontoparietal control) — aligns with executive dysfunction

These cross-model consensus findings are the highest-confidence interpretability results (Tier 1).

---

## Figure 4: Top ROI Saliency Rankings

**File**: `roi_saliency_top20.pdf/png`

### What It Shows

Horizontal bar chart of the top 20 individual brain ROIs ranked by absolute saliency (highest at top). Each bar is colored red (+ADHD) or blue (-ADHD) based on the signed gradient direction. ROI labels include both the region name and its Yeo-17 network assignment in parentheses.

### How It Is Calculated

1. For each of the 180 bilateral ROIs (HCP-MMP1 parcellation), compute:
   - **Absolute saliency**: mean |gradient| across all test subjects
   - **Signed saliency**: mean raw gradient across all test subjects

2. Rank ROIs by absolute saliency and display the top 20.

3. Color each bar by sign: red if signed saliency > 0 (+ADHD), blue if < 0 (-ADHD).

### How to Interpret

- **Red bars at top**: These brain regions are the most important for the ADHD prediction AND increasing their activity pushes toward ADHD.
- **Blue bars at top**: These regions are highly important but increasing their activity pushes AWAY from ADHD (i.e., high activity in these regions is a marker of healthy control).
- **ROI labels in parentheses**: The Yeo-17 network assignment helps contextualize individual ROIs. Clusters of ROIs from the same network at the top of the list reinforce network-level findings.
- **Mix of red and blue**: The model has learned a complex pattern — some ROIs contribute positively and others negatively.

### v5 Scientific Findings

Top ROIs vary across models, but common themes include:
- Executive/Control network ROIs frequently appear as +ADHD (AIP, RSC, STSdp)
- SomMotA ROIs frequently appear as -ADHD (area 5m)
- OFC-related ROIs (s32, OFC) frequently appear as -ADHD, especially in quantum models

---

## Figure 5: Signed ROI Rankings

**File**: `signed_roi_rankings.pdf/png`

### What It Shows

Two side-by-side panels:
- **Left**: Top 10 ROIs with the most positive signed saliency (+ADHD direction). All bars are red.
- **Right**: Top 10 ROIs with the most negative signed saliency (-ADHD direction). All bars are blue.

### How It Is Calculated

1. Rank all 180 ROIs by signed saliency (raw gradient averaged across subjects).
2. The top 10 most positive form the +ADHD panel; the top 10 most negative form the -ADHD panel.
3. Unlike Figure 4 (which ranks by absolute value), this figure explicitly separates the two directions.

### How to Interpret

- **Left panel (+ADHD)**: These are the brain regions most strongly driving the model's ADHD prediction. If a subject has high activity in these regions, the model is more likely to predict ADHD. Regions from the same network appearing together (e.g., multiple DefaultA ROIs) reinforce that network's role.
- **Right panel (-ADHD)**: These regions push against the ADHD prediction. High activity in these regions is associated with healthy control. These can be thought of as "protective" regions.
- **Compare across models**: +ADHD ROIs that appear in multiple models are the most robust ADHD biomarker candidates.

### v5 Scientific Findings

The +ADHD and -ADHD ROI lists differ across models (architecture-dependent), which is why ROI-level conclusions are generally Tier 2-3. The more robust findings come from aggregating ROIs to the network level (Figure 3), where cross-model consensus emerges.

---

## Figure 6: Expert Input Weight Heatmap

**File**: `input_weight_heatmap.pdf/png`

### What It Shows

A heatmap with experts on the Y-axis and 17 Yeo networks on the X-axis. Color intensity (yellow-orange-red scale) represents the mean absolute weight magnitude of the learned input projection connecting each network's ROIs to each expert. Black rectangles highlight the maximum weight per expert.

### How It Is Calculated

1. Each expert has a learned `Linear(n_roi, hidden_dim)` layer that projects its assigned ROIs into a hidden representation. The weight matrix dimensions differ per expert (e.g., DMN expert: 41×64, Executive: 52×64).

2. For each ROI assigned to an expert, compute the L2 norm of its weight vector (across the 64 hidden dimensions): `||W[roi, :]||_2`. This measures how much the expert "attends" to that ROI.

3. Group ROIs by Yeo-17 network and compute the mean weight norm per network within each expert.

4. Display as a heatmap. Each row (expert) sums only over ROIs actually assigned to that expert, so cells outside an expert's assigned networks will be zero or very small.

### How to Interpret

- **Bright cells along expected diagonals**: The expert has learned to attend most to its assigned networks. For example, the DMN expert should have the brightest cells at DefaultA/B/C and TempPar. This confirms the Yeo-17 mapping is coherent and the experts respect their neuroscience-guided assignments.
- **Off-diagonal bright cells**: The expert also attends to networks outside its assignment. This could indicate cross-network information flow or mapping imprecision.
- **Black rectangle position**: Shows which network each expert weighs most heavily. If this aligns with the expert's assignment, the model has learned the expected structure.
- **Uniform row**: The expert does not distinguish between its assigned networks — it treats all ROIs equally.

### v5 Scientific Findings

**All 8 experts (4 per model × 2 models with 4 experts)** correctly learn highest weights for their assigned network group. This is a **Tier 1 finding** (replicated across all 4 models) that serves as an internal consistency check validating the corrected Yeo-17 bilateral mapping.

For example, in the classical 4-expert model:
- DMN expert: highest weight at DefaultA (0.728)
- Executive expert: highest weight at ContA (0.656)
- Salience expert: highest weight at Limbic_OFC (0.720)
- SensoriMotor expert: highest weight at SomMotA (0.739)

---

## Summary: From Visualization to Scientific Claims

| Figure | Primary Question | Key Metric | v5 Strongest Finding |
|--------|-----------------|------------|---------------------|
| Gate Weights | Does the model route ADHD+ differently? | p-value | Only quantum 2-expert significant (p=0.038) |
| Circuit Saliency | Which brain circuit groups matter most? | Signed saliency | Classical: Executive +ADHD; Quantum: different pattern |
| Network Saliency | Which Yeo-17 networks drive ADHD prediction? | Signed saliency | DefaultC, SalVentAttnB consistently +ADHD; Limbic_OFC, ContB consistently -ADHD |
| ROI Saliency Top 20 | Which individual ROIs are most salient? | Absolute saliency | Model-dependent; useful for detailed exploration |
| Signed ROI Rankings | Which ROIs push toward/away from ADHD? | Signed saliency | Architecture-dependent; lower confidence |
| Input Weight Heatmap | Does the model respect neuroscience assignments? | Weight norms | All experts correctly prioritize assigned networks |

### How the Visualizations Build a Narrative

1. **Gate weights** (Figure 1) test whether the gating mechanism itself distinguishes ADHD+ from ADHD-. In most models, load balancing suppresses this signal.
2. **Circuit saliency** (Figure 2) looks inside the experts via gradient analysis to find circuit-level directional associations.
3. **Network saliency** (Figure 3) disaggregates circuits into 17 functional networks, revealing which specific networks drive the ADHD association. This is where cross-model consensus emerges.
4. **ROI saliency** (Figures 4-5) provides the finest resolution — individual brain regions — for hypothesis generation and literature comparison.
5. **Input weight heatmap** (Figure 6) serves as a validation check that the model has learned the intended neuroscience structure.

The overall narrative: **The model's ADHD signal resides primarily at the network and ROI level, not in gating.** Gate-level signal is suppressed by load balancing in most models. The networks most consistently associated with ADHD are DefaultC (+ADHD), SalVentAttnB (+ADHD), Limbic_OFC (-ADHD), and ContB (-ADHD) — all neuroscientifically plausible.

## Cross-References

- Full numerical results: `docs/Interpretability_Heterogeneity_v5_Results.md` (Sections 2-5)
- Heterogeneity visualization guide: `docs/Heterogeneity_Visualization_Guide.md`
- Confidence framework: `docs/Interpretability_Heterogeneity_v5_Results.md` (Section 12)
- Terminology (ADHD+ vs +ADHD): `docs/Interpretability_Heterogeneity_v5_Results.md` (Terminology section)
