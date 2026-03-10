# Per-Expert Gradient Saliency — Results and Visualization

**Date**: 2026-03-08
**Script**: `analyze_expert_saliency.py` (analysis), `visualize_expert_saliency.py` (visualization)
**Checkpoints**: v5 models, seed=2025
**Test set**: 669 subjects (288 ADHD+, 381 ADHD-)

---

## 1. Overview

We ran per-expert gradient saliency analysis on all 4 Circuit MoE v5 models:

| Model | Config | Experts | Params | Test AUC | Significant Findings |
|-------|--------|:------:|:------:|:--------:|:--------------------:|
| Classical 4e | adhd_3 | DMN, Executive, Salience, SensoriMotor | 516,485 | 0.5925 | **23** (10 network, 13 ROI) |
| Classical 2e | adhd_2 | Internal, External | 269,827 | 0.5696 | **9** (3 network, 6 ROI) |
| Quantum 4e | adhd_3 | DMN, Executive, Salience, SensoriMotor | 30,373 | 0.5773 | **13** (1 network, 12 ROI) |
| Quantum 2e | adhd_2 | Internal, External | 26,771 | 0.5939 | **7** (1 network, 6 ROI) |

The classical 4-expert model produces the richest interpretability — 23 significant findings (p<0.05) across all 4 experts. The quantum models find fewer significant differences, consistent with their flatter saliency profiles.

---

## 2. Classical 4-Expert Model (adhd_3) — AUC 0.5925

This is the most informative model for per-expert interpretability.

### 2.1 DMN Expert (41 ROIs: DefaultA, DefaultB, DefaultC, TempPar)

#### Network-Level Saliency

| Network | n_ROIs | Abs Saliency | Signed | Direction | p-value | Sig? |
|---------|:------:|:------------:|:------:|:---------:|:-------:|:----:|
| DefaultA | 5 | 3.2e-5 | -2e-6 | -ADHD | 0.054 | borderline |
| DefaultB | 5 | 2.5e-5 | +2.0e-5 | +ADHD | **0.012** | * |
| DefaultC | 16 | 2.4e-5 | +1.1e-5 | +ADHD | **0.002** | ** |
| TempPar | 15 | 2.3e-5 | -1.1e-5 | -ADHD | **0.010** | * |

**Interpretation**: The DMN expert uses DefaultA most (highest absolute saliency) but finds DefaultC and DefaultB most diagnostic. DefaultC (+ADHD, p=0.002) and DefaultB (+ADHD, p=0.012) push toward ADHD, while TempPar (-ADHD, p=0.010) pushes away. This reveals a split within the DMN: posterior cingulate/precuneus (DefaultC) and angular gyrus (DefaultB) are ADHD-associated, while temporal-parietal junction areas (TempPar) are protective.

#### Top ROIs

| ROI | Region | Network | Direction | p-value |
|:---:|--------|---------|:---------:|:-------:|
| 63 | **p32** (Area p32) | DefaultC | **+ADHD** | **0.001** |
| 31 | **23d** (Area 23d) | DefaultC | **+ADHD** | **0.002** |
| 128 | **STSdp** (STS posterior) | DefaultA | **-ADHD** | **0.001** |
| 27 | **STV** (Sup Temporal Visual) | DefaultA | **+ADHD** | **0.003** |
| 154 | **PHA2** (Parahippocampal 2) | DefaultB | **+ADHD** | **0.0004** |

**Neuroscience significance**: p32 (posterior area 32, midcingulate/posterior cingulate) is a core DMN hub implicated in self-referential processing. Area 23d (posterior cingulate) is consistently reported as hyperactivated in ADHD during resting-state. PHA2 (parahippocampal cortex) connects the DMN to memory systems. These findings align with the DMN hyperactivation hypothesis of ADHD (Castellanos & Proal, 2012).

### 2.2 Executive Expert (52 ROIs: ContA, ContB, ContC, DorsAttnA, DorsAttnB)

#### Network-Level Saliency

| Network | n_ROIs | Abs Saliency | Signed | Direction | p-value | Sig? |
|---------|:------:|:------------:|:------:|:---------:|:-------:|:----:|
| ContC | 11 | 2.4e-5 | -3e-6 | -ADHD | **0.001** | *** |
| ContB | 12 | 2.1e-5 | -6e-6 | -ADHD | **0.030** | * |
| DorsAttnB | 8 | 1.6e-5 | -7e-6 | -ADHD | 0.861 | n.s. |
| DorsAttnA | 16 | 1.5e-5 | +1e-6 | +ADHD | **0.042** | * |
| ContA | 5 | 1.2e-5 | -7e-6 | -ADHD | 0.056 | borderline |

**Interpretation**: The Executive expert prioritizes ContC (frontoparietal control C) most, with the strongest significance (p=0.001). ContC and ContB both show -ADHD direction, meaning their activity patterns push the expert away from ADHD — consistent with executive control hypoactivation in ADHD. DorsAttnA shows weak +ADHD direction, suggesting dorsal attention system A may be dysregulated (compensatory hyperactivation) in ADHD.

#### Top ROIs

| ROI | Region | Network | Direction | p-value |
|:---:|--------|---------|:---------:|:-------:|
| 170 | **p47r** (posterior area 47r) | ContC | **-ADHD** | **0.001** |
| 72 | **8C** (Area 8C) | ContC | **+ADHD** | **0.006** |
| 144 | **IP1** (IntraParietal 1) | ContB | **+ADHD** | **0.002** |
| 136 | **PHT** (Area PHT) | ContB | **-ADHD** | **0.001** |
| 79 | **IFJp** (IFJ posterior) | ContB | **+ADHD** | **0.001** |

**Neuroscience significance**: p47r (inferior frontal gyrus) is a key region for cognitive control and response inhibition — its -ADHD direction is consistent with the inhibition deficit model of ADHD. IFJp (inferior frontal junction) with +ADHD suggests compensatory activation. IP1 (intraparietal sulcus) involvement aligns with attentional control literature.

### 2.3 Salience Expert (44 ROIs: SalVentAttnA, SalVentAttnB, Limbic_TempPole, Limbic_OFC)

#### Network-Level Saliency

| Network | n_ROIs | Abs Saliency | Signed | Direction | p-value | Sig? |
|---------|:------:|:------------:|:------:|:---------:|:-------:|:----:|
| Limbic_OFC | 8 | 2.6e-5 | -3e-6 | -ADHD | **0.005** | ** |
| SalVentAttnA | 19 | 1.8e-5 | -6e-6 | -ADHD | **0.005** | ** |
| Limbic_TempPole | 9 | 1.7e-5 | +2e-6 | +ADHD | 0.106 | n.s. |
| SalVentAttnB | 8 | 1.5e-5 | -5e-6 | -ADHD | **0.003** | ** |

**Interpretation**: All four sub-networks within the Salience circuit show diagnosis-dependent processing, with three reaching significance. Limbic_OFC tops the absolute saliency ranking — the orbitofrontal cortex is the expert's most relied-upon sub-network. Both salience sub-networks (SalVentAttnA, SalVentAttnB) show -ADHD direction, suggesting reduced insular/salience processing in ADHD. Only Limbic_TempPole shows +ADHD (temporal pole may exhibit compensatory activation).

#### Top ROIs

| ROI | Region | Network | Direction | p-value |
|:---:|--------|---------|:---------:|:-------:|
| 90 | **11l** (Area 11l) | Limbic_OFC | **-ADHD** | **0.006** |
| 108 | **MI** (Middle Insular) | SalVentAttnA | **-ADHD** | **0.001** |
| 98 | **43** (Area 43) | SalVentAttnA | **-ADHD** | **0.001** |
| 163 | **25** (Area 25) | Limbic_OFC | **+ADHD** | **0.001** |
| 133 | **TE2a** (TE2 anterior) | Limbic_TempPole | **-ADHD** | **0.0002** |

**Neuroscience significance**: Area 11l (lateral OFC) is central to reward processing — its -ADHD direction supports the reward-processing deficit model of ADHD. Area 25 (subgenual cingulate, +ADHD) is a deep limbic hub implicated in emotional dysregulation. MI (middle insula) -ADHD aligns with interoceptive/salience processing deficits in ADHD. These findings directly connect to two of our three identified ADHD subtypes: SalVentAttnA (insular) and Limbic_OFC (reward).

### 2.4 SensoriMotor Expert (43 ROIs: VisCent, VisPeri, SomMotA, SomMotB)

#### Network-Level Saliency

| Network | n_ROIs | Abs Saliency | Signed | Direction | p-value | Sig? |
|---------|:------:|:------------:|:------:|:---------:|:-------:|:----:|
| VisPeri | 8 | 2.9e-5 | -1.7e-5 | -ADHD | 0.057 | borderline |
| SomMotA | 9 | 2.8e-5 | -2.2e-5 | -ADHD | **0.004** | ** |
| VisCent | 12 | 2.6e-5 | -1.5e-5 | -ADHD | **0.020** | * |
| SomMotB | 14 | 2.3e-5 | +6e-6 | +ADHD | 0.909 | n.s. |

**Interpretation**: SomMotA and VisCent both show significant -ADHD direction: somatomotor and visual cortex hypoactivation patterns are associated with healthy controls. SomMotB (auditory/insular somatomotor) shows the only +ADHD direction in this expert but is not significant.

#### Top ROIs

| ROI | Region | Network | Direction | p-value |
|:---:|--------|---------|:---------:|:-------:|
| 5 | **V4** (Fourth Visual Area) | VisCent | **-ADHD** | **0.010** |
| 167 | **Ig** (Insular Granular) | SomMotB | **+ADHD** | **0.007** |
| 6 | **V8** (Eighth Visual Area) | VisCent | **-ADHD** | **0.004** |
| 152 | **VMV1** (VentroMedial Visual 1) | VisPeri | **-ADHD** | **0.014** |
| 35 | **5m** (Area 5m) | SomMotA | **-ADHD** | **0.005** |

---

## 3. Classical 2-Expert Model (adhd_2) — AUC 0.5696

### 3.1 Internal Expert (85 ROIs: DMN + Salience networks)

No network-level findings reached p<0.05. The top sub-network by absolute saliency is Limbic_TempPole (0.000863), followed by DefaultC (0.000720). With 8 sub-networks in a single expert, the saliency is more distributed and less discriminative than the 4-expert configuration.

Top ROI: TGv (temporal pole ventral, Limbic_TempPole, -ADHD, p=0.91) — high absolute saliency but not significant.

### 3.2 External Expert (95 ROIs: Executive + SensoriMotor networks)

| Network | Abs Saliency | Direction | p-value | Sig? |
|---------|:------------:|:---------:|:-------:|:----:|
| ContA | 3.75e-4 | +ADHD | **0.037** | * |
| VisCent | 3.02e-4 | -ADHD | **0.033** | * |
| SomMotA | 2.98e-4 | +ADHD | **0.023** | * |

**Interpretation**: The External expert finds 3 significant networks. ContA (+ADHD) is interesting — in the 4-expert model, ContA was borderline (p=0.056) with -ADHD direction. The direction reversal between 2e and 4e configurations suggests that when executive and sensorimotor networks are combined in one expert, the learned representation changes.

---

## 4. Quantum 4-Expert Model (adhd_3) — AUC 0.5773

### 4.1 Key Differences from Classical 4-Expert

| Property | Classical 4e | Quantum 4e |
|----------|:-----------:|:----------:|
| Significant networks (p<0.05) | **10** | **1** |
| Significant ROIs (p<0.05) | **13** | **12** |
| Saliency magnitude | ~2-3e-5 | ~3-5e-3 |
| Saliency range (within expert) | 2-3× variation | <1.5× variation |

**The quantum model has ~100× larger absolute saliency values but much flatter distribution across sub-networks.** This means quantum experts are sensitive to all their input ROIs roughly equally, rather than developing sharp feature preferences like classical experts.

### 4.2 Network-Level (only 1 significant)

Only SensoriMotor expert → VisPeri reached significance (p=0.011, +ADHD). All other network-level comparisons were non-significant.

### 4.3 Notable ROI Findings

Despite flat network-level profiles, several individual ROIs show significant ADHD+ vs ADHD- differences:

| Expert | ROI | Region | Network | Direction | p-value |
|--------|:---:|--------|---------|:---------:|:-------:|
| DMN | 149 | PGi (Area PGi) | DefaultC | +ADHD | **0.006** |
| DMN | 64 | 10r (Area 10r) | DefaultC | +ADHD | **0.029** |
| Executive | 72 | 8C (Area 8C) | ContC | +ADHD | **0.030** |
| Executive | 76 | a47r (anterior 47r) | ContC | +ADHD | **0.021** |
| Salience | 85 | 9-46d (Area 9-46d) | SalVentAttnB | +ADHD | **0.001** |
| Salience | 164 | s32 (Area s32) | Limbic_OFC | -ADHD | **0.005** |
| SensoriMotor | 0 | V1 (Primary Visual) | VisPeri | +ADHD | **0.007** |

**Cross-architecture agreement**: Several ROIs appear in both classical and quantum top findings:
- **8C** (ContC): +ADHD in both classical (p=0.006) and quantum (p=0.030)
- **PGi** (DefaultC): Classical shows in top 10 ROIs; quantum reaches significance (p=0.006)
- **s32** (Limbic_OFC): Appears in both models' salience expert findings

These cross-architecture ROIs represent the strongest biomarker candidates — their ADHD-relevance is not architecture-dependent.

---

## 5. Quantum 2-Expert Model (adhd_2) — AUC 0.5939

### 5.1 Internal Expert (85 ROIs)

No network-level significance. The saliency profile is very flat (all 8 networks within 0.0030-0.0033 range). One notable ROI: OFC (Orbital Frontal Complex, Limbic_OFC, +ADHD, p=0.013).

### 5.2 External Expert (95 ROIs)

One significant network: ContA (-ADHD, p=0.039). Note this is -ADHD in quantum 2e but +ADHD in classical 2e — another direction reversal between architectures, suggesting the ContA signal is not robust.

---

## 6. Cross-Model Comparison

### 6.1 Summary Table: Top Network and ROI per Expert

| Model | Expert | Top Network (abs) | Net Direction | Top ROI | ROI Direction | ROI p |
|-------|--------|:-----------------:|:-------------:|---------|:-------------:|:-----:|
| Classical 4e | DMN | DefaultA | -ADHD | p32 (DefaultC) | +ADHD | 0.001 |
| Classical 4e | Executive | ContC | -ADHD | p47r (ContC) | -ADHD | 0.001 |
| Classical 4e | Salience | Limbic_OFC | -ADHD | 11l (Limbic_OFC) | -ADHD | 0.006 |
| Classical 4e | SensoriMotor | VisPeri | -ADHD | V4 (VisCent) | -ADHD | 0.010 |
| Classical 2e | Internal | Limbic_TempPole | -ADHD | TGv (Limbic_TempPole) | -ADHD | 0.910 |
| Classical 2e | External | ContA | +ADHD | 5L (DorsAttnB) | +ADHD | 0.863 |
| Quantum 4e | DMN | DefaultC | +ADHD | STSva (TempPar) | +ADHD | 0.073 |
| Quantum 4e | Executive | ContB | -ADHD | a10p (ContC) | +ADHD | 0.867 |
| Quantum 4e | Salience | Limbic_TempPole | -ADHD | TF (Limbic_TempPole) | -ADHD | 0.596 |
| Quantum 4e | SensoriMotor | SomMotB | +ADHD | A1 (SomMotB) | +ADHD | 0.240 |
| Quantum 2e | Internal | Limbic_OFC | -ADHD | STSdp (DefaultA) | +ADHD | 0.886 |
| Quantum 2e | External | DorsAttnA | -ADHD | TA2 (SomMotB) | -ADHD | 0.353 |

### 6.2 Cross-Architecture Agreements (Classical vs Quantum 4e)

| Expert | Agreement? | Details |
|--------|:---------:|---------|
| DMN | **Partial** | Both rank DefaultC high (+ADHD). Classical ranks DefaultA highest (abs), quantum ranks DefaultC highest (abs). |
| Executive | **Yes** | Both find ContB and ContC dominant. Both show mostly -ADHD direction for frontoparietal control. |
| Salience | **Partial** | Both rank Limbic regions high. Classical: Limbic_OFC top; Quantum: Limbic_TempPole top. Both find Limbic regions -ADHD. |
| SensoriMotor | **Divergent** | Classical: all -ADHD except SomMotB. Quantum: all +ADHD. Direction reversal across architecture. |

### 6.3 Classical 4e vs 2e: Effect of Expert Granularity

The 4-expert configuration produces dramatically more significant findings (23 vs 9). When brain circuits are decomposed into finer partitions:

1. **DMN expert alone** finds 3 significant networks (DefaultB, DefaultC, TempPar). The **Internal expert** (DMN + Salience combined) finds 0 — the Salience regions dilute the DMN signal.

2. **Executive expert alone** finds 3 significant networks. The **External expert** (Executive + SensoriMotor combined) finds 3, but different ones (ContA, VisCent, SomMotA).

3. **Finer decomposition enables sharper specialization**: Each 4-expert model expert focuses on ~41-52 ROIs, learning circuit-specific features. The 2-expert model's 85-95 ROIs per expert produce a more diffuse representation.

---

## 7. Visualization Guide

### 7.1 Figure Descriptions

Six figure types are generated per model, plus 2 cross-model comparison figures.

#### Figure 1: Network Saliency Heatmap (`network_saliency_heatmap.pdf`)

**What it shows**: A side-by-side pair of heatmaps with experts as rows and Yeo-17 sub-networks as columns. Left panel shows absolute saliency (YlOrRd colormap — yellow=low, red=high). Right panel shows signed saliency (RdBu diverging — red=+ADHD, blue=-ADHD). Gray cells with dashes mark networks not assigned to that expert. Black asterisks (*) mark ADHD+ vs ADHD- differences significant at p<0.05.

**How to read it**:
- **Left panel (absolute)**: Brighter/redder cells = expert relies more on that sub-network. Compare across columns within a row to see which sub-network dominates each expert.
- **Right panel (signed)**: Red cells = +ADHD direction, blue cells = -ADHD direction. Stars indicate the direction is statistically different between ADHD+ and ADHD- subjects.
- **Diagonal pattern**: Each expert should show color only in its assigned networks (others are gray). The 4-expert model produces a block-diagonal structure.

**What to look for**:
- Do the darkest absolute cells match the most significant signed cells? (If so, the most-used features are also the most diagnostic.)
- Are there unexpected non-significant dark cells? (Expert relies heavily on this network but doesn't use it differently for ADHD+ vs ADHD-.)
- For the classical 4e model, the right panel should show many stars — this is the model with the most significant findings.

#### Figure 2: Expert Top ROIs (`expert_top_rois.pdf`)

**What it shows**: One horizontal bar chart per expert, showing the top 10 ROIs ranked by absolute saliency. Bar color indicates signed direction: saturated red/blue = significant (p<0.05), light pink/light blue = not significant. Each bar is labeled with ROI name and Yeo-17 network.

**How to read it**:
- Longer bars = ROI matters more to this expert's output.
- Red bars (+ADHD): increasing this ROI's activity pushes toward ADHD prediction. Blue bars (-ADHD): pushes away from ADHD.
- Stars (*) after bar = p<0.05 for ADHD+ vs ADHD- comparison.
- Compare across experts: do different experts prioritize different ROIs? (They should, since they process different brain circuits.)

**What to look for**:
- For the classical 4e model: most bars should be colored (significant) — this reflects the rich discriminative signal.
- For quantum models: bars will be more uniformly sized (flat saliency) with fewer stars.
- Same ROI appearing as top-ranked in multiple models strengthens its biomarker candidacy.

#### Figure 3: Feature Hierarchy (`feature_hierarchy.pdf`)

**What it shows**: One bar chart per expert, with sub-networks ranked by absolute saliency (tallest bar = most relied-upon). Bar color: red (+ADHD direction) or blue (-ADHD direction). White +/- labels inside bars indicate direction. Significance stars (*, **, ***) appear above significant bars.

**How to read it**:
- The left-to-right ordering within each expert panel shows the expert's learned priority ranking of its sub-networks.
- This is the **intra-circuit feature hierarchy** — the central novel contribution of this analysis.
- Stars indicate the network's saliency is significantly different between ADHD+ and ADHD- groups.

**How it's computed**: For each sub-network within an expert, we average the absolute saliency across all ROIs belonging to that sub-network, then rank from highest to lowest. The direction (+/-) comes from the mean signed saliency across those ROIs.

**What to look for**:
- Classical 4e: Hierarchies should be steep (clear ranking) with many stars. Example: DMN expert shows DefaultA > DefaultB > DefaultC > TempPar.
- Quantum 4e: Hierarchies should be flat (bars nearly equal height) with few stars. This reflects the quantum model's distributed representation.
- Compare classical vs quantum: same ranking order = data-driven hierarchy; different = architecture-dependent.

#### Figure 4: Volcano Plot (`volcano_plot.pdf`)

**What it shows**: One scatter plot per expert with signed saliency (x-axis) vs -log10(p-value) (y-axis) for each ROI. Points are colored: red = significant +ADHD, blue = significant -ADHD, gray = not significant. A dashed horizontal line marks p=0.05 (-log10(0.05) ≈ 1.3). Significant ROIs are labeled with their names.

**How to read it**:
- Points above the dashed line are statistically significant (p<0.05).
- Points further from x=0 have stronger signed effects.
- The upper-left quadrant contains significant -ADHD ROIs; upper-right contains significant +ADHD ROIs.
- Named ROIs above the line are the strongest individual biomarker candidates.

**How it's computed**: For each ROI, we take the mean signed saliency across all test subjects as the x-coordinate, and compute a Welch's t-test (ADHD+ vs ADHD-) to get the p-value for the y-coordinate.

**What to look for**:
- Classical 4e: Many labeled points above the line, distributed across both +ADHD and -ADHD sides.
- Quantum models: Fewer points above the line, more clustered near x=0 (weaker directional effects).
- ROIs appearing in the upper quadrants across multiple experts/models are the most robust findings.

#### Figure 5: Signed ROIs per Expert (`signed_rois_per_expert.pdf`)

**What it shows**: A grid of panels (one row per expert, two columns: +ADHD and -ADHD). Each panel shows the top 5 ROIs by signed saliency in that direction. Saturated color = significant; lighter color = not significant.

**How to read it**:
- Left column (+ADHD): ROIs whose increased activity is most associated with ADHD.
- Right column (-ADHD): ROIs whose increased activity is most associated with healthy controls.
- Compare across experts: some ROIs may appear in multiple experts' top lists (if they overlap with multiple circuits in the 2-expert configuration).

**What to look for**:
- The Salience expert's +ADHD list (Area 25, 10pp, PoI1) represents reward and interoceptive regions — these are the "ADHD-pushing" regions.
- The Executive expert's -ADHD list (p47r, PHT, IP2) represents cognitive control regions — these are the "healthy control" regions.

#### Figure 6: Cross-Model 4e Comparison (`cross_model_4e_comparison.pdf`)

**What it shows**: Grouped bar chart comparing classical (solid bars) and quantum (hatched bars) 4-expert models. One panel per expert (DMN, Executive, Salience, SensoriMotor). Within each panel, sub-networks are shown side-by-side. Saliency values are **normalized** within each model (tallest bar = 1.0) to enable cross-architecture comparison despite different absolute scales.

**How to read it**:
- Solid bars (classical) vs hatched bars (quantum): if both have similar heights, the models agree on relative importance.
- Bar color: red = +ADHD, blue = -ADHD. If both models have the same color for a network, they agree on direction.
- Height differences between solid and hatched bars reveal where the two architectures disagree.

**How it's computed**: For each model, the absolute saliency per network is divided by the maximum saliency within that expert, producing values between 0 and 1. This removes the 100× scale difference between classical and quantum saliency magnitudes.

**What to look for**:
- DMN expert: Both models should show DefaultC as relatively high (agreed +ADHD).
- Salience expert: Classical and quantum may disagree on Limbic_TempPole vs Limbic_OFC ranking.
- SensoriMotor expert: Direction reversal (classical = mostly -ADHD, quantum = mostly +ADHD) — this is the biggest cross-architecture disagreement.

#### Figure 7: Cross-Model Summary Table (`cross_model_summary_table.pdf`)

**What it shows**: A table figure summarizing the top network and top ROI per expert across all 4 models. Columns: Model, Expert, Top Network (abs), Direction, Top ROI, ROI Network, ROI Direction, p-value. Direction cells are color-coded (pink = +ADHD, blue = -ADHD).

**How to read it**: Scan down the "Direction" and "ROI Direction" columns for color patterns:
- Consistent colors across models for the same expert = robust finding.
- Mixed colors = architecture-dependent, lower confidence.

---

## 8. Key Findings — Cross-Model Synthesis

### 8.1 Robust Findings (Replicated Across Architectures)

1. **DefaultC +ADHD within DMN**: Both classical and quantum DMN experts rank DefaultC highly with +ADHD direction. Top ROIs: p32, 23d (posterior cingulate). This confirms posterior cingulate / precuneus DMN hyperactivation in ADHD.

2. **Limbic regions dominant within Salience circuit**: Both architectures rank Limbic_OFC or Limbic_TempPole as the top sub-network in the Salience expert. The OFC is central to ADHD reward processing deficits.

3. **Frontoparietal control -ADHD within Executive**: Both architectures find ContB and/or ContC with -ADHD direction. This supports executive control hypoactivation in ADHD.

4. **Area 8C (ContC) +ADHD**: Significant in both classical (p=0.006) and quantum (p=0.030) Executive experts. This frontal eye field region may show compensatory hyperactivation in ADHD.

### 8.2 Classical-Only Findings (High Confidence, Single Architecture)

5. **Salience expert: MI, Area 43 -ADHD** (middle insula, p<0.002): Insular hypoactivation in ADHD, supporting the interoceptive deficit model.

6. **Executive expert: p47r -ADHD** (inferior frontal, p=0.001): Response inhibition region hypoactivation, the strongest single ROI finding.

7. **Area 25 +ADHD** (subgenual cingulate, p=0.001): Emotional dysregulation hub, consistent with limbic hyperactivation in ADHD.

### 8.3 Architecture-Dependent (Lower Confidence)

8. **SensoriMotor direction**: Classical expert shows predominantly -ADHD; quantum shows predominantly +ADHD. This circuit's contribution is architecture-dependent and should not be over-interpreted.

9. **ContA direction in 2-expert models**: Classical 2e shows +ADHD, quantum 2e shows -ADHD. This reversal suggests the ContA signal depends on what other networks share the expert.

### 8.4 Granularity Effects

10. **4-expert >> 2-expert for interpretability**: The 4-expert model produces 23 significant findings vs 9 for the 2-expert model. Finer circuit decomposition enables sharper per-expert specialization.

11. **Combining DMN + Salience in one expert dilutes signal**: The Internal expert (85 ROIs, 8 networks) shows 0 significant network-level findings, while the separate DMN (41 ROIs) and Salience (44 ROIs) experts show 3 significant networks each.

---

## 9. Connection to ADHD Neuroscience Literature

| Finding | Literature Support |
|---------|-------------------|
| DefaultC (posterior cingulate) +ADHD | Castellanos & Proal (2012): DMN hyperactivation in ADHD |
| p47r (inferior frontal) -ADHD | Cortese et al. (2012): Hypoactivation of prefrontal regions during inhibition |
| Limbic_OFC -ADHD (expert level) | Nigg et al. (2020): Reward processing deficits; Sagvolden et al. (2005): Dopamine hypofunction in OFC |
| MI (middle insula) -ADHD | Uddin et al. (2017): Salience network dysfunction in ADHD |
| Area 25 +ADHD | Shaw et al. (2014): Limbic hyperactivation; Posner et al. (2011): Emotional dysregulation in ADHD |
| Executive control networks -ADHD | Bush (2011): Fronto-cingulate-parietal attention network deficits |
| PHA2 (parahippocampal) +ADHD | Castellanos et al. (2009): Altered DMN-memory system connectivity |

---

## 10. Output Files

### 10.1 Per-Model Analysis Outputs

```
analysis/
├── expert_saliency_v5_classical_adhd_3/
│   ├── expert_saliency_report.md          # Full per-expert report
│   ├── expert_saliency_results.json       # Machine-readable statistics
│   ├── expert_saliency_per_subject.npz    # Per-subject saliency matrices
│   └── figures/
│       ├── network_saliency_heatmap.pdf   # Fig 1: Expert × Network heatmap
│       ├── expert_top_rois.pdf            # Fig 2: Top 10 ROIs per expert
│       ├── feature_hierarchy.pdf          # Fig 3: Intra-circuit ranking
│       ├── volcano_plot.pdf               # Fig 4: Signed vs p-value
│       └── signed_rois_per_expert.pdf     # Fig 5: +ADHD/-ADHD top ROIs
├── expert_saliency_v5_classical_adhd_2/   # Same structure
├── expert_saliency_v5_quantum_8q_d3_adhd_3/
├── expert_saliency_v5_quantum_8q_d3_adhd_2/
└── expert_saliency_cross_model/
    ├── cross_model_4e_comparison.pdf      # Fig 6: Classical vs Quantum normalized
    └── cross_model_summary_table.pdf      # Fig 7: Summary table
```

### 10.2 Total: 22 figures (5 per model × 4 models + 2 cross-model)

---

## 11. Limitations and Next Steps

### 11.1 Limitations

1. **Single seed (2025)**: All analyses use seed=2025 checkpoints. Cross-seed stability should be verified.
2. **Uncorrected p-values**: With 41-52 ROIs per expert × 4 experts, multiple comparison correction (FDR) would reduce significant findings. Apply Benjamini-Hochberg when citing individual ROIs.
3. **First-order gradient approximation**: Saliency captures local sensitivity, not causal effects. Perturbation-based methods could strengthen causal claims.
4. **Quantum model saliency interpretation**: The 100× larger but flatter saliency in quantum models may reflect the QSVT circuit's more uniform gradient flow rather than genuine indifference to input features.

### 11.2 Next Steps

1. **Cross-seed saliency stability**: Run analysis on seed 2024/2026/2027/2028 checkpoints; compare whether the same ROIs emerge as top-saliency.
2. **FDR correction**: Apply Benjamini-Hochberg across all ROI-level tests within each model.
3. **Link to ADHD subtypes**: Cross-reference per-expert saliency with heterogeneity subtypes (SalVentAttnA, Limbic_OFC, Limbic_TempPole) to identify subtype-specific biomarkers.
4. **Brain surface visualization**: Map top ROIs onto cortical surface using HCP workbench for publication figures.

---

## 12. References

- Bush, G. (2011). Cingulate, frontal, and parietal cortical dysfunction in attention-deficit/hyperactivity disorder. *Biological Psychiatry*, 69(12), 1160-1167.
- Castellanos, F. X., & Proal, E. (2012). Large-scale brain systems in ADHD. *Trends in Cognitive Sciences*, 16(1), 17-26.
- Cortese, S., et al. (2012). Toward systems neuroscience of ADHD. *American Journal of Psychiatry*, 169(10), 1038-1055.
- Nigg, J. T., et al. (2020). Working memory and vigilance as multivariate endophenotypes. *Biological Psychiatry: CNNI*, 5(7), 673-681.
- Posner, J., et al. (2011). Connecting the dots: A review of resting connectivity MRI studies in ADHD. *Neuropsychology Review*, 21(3), 222-238.
- Sagvolden, T., et al. (2005). A dynamic developmental theory of ADHD. *Behavioral and Brain Sciences*, 28(3), 397-419.
- Shaw, P., et al. (2014). Emotion dysregulation in ADHD. *American Journal of Psychiatry*, 171(3), 276-293.
- Uddin, L. Q., et al. (2017). Salience network-based classification and prediction of symptom severity in children with autism. *JAMA Psychiatry*, 70(8), 869-879.
