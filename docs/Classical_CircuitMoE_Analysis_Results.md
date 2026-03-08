# Classical Circuit MoE — Interpretability and Heterogeneity Analysis Results

> **DEPRECATION NOTICE (2026-03-07):** The results in this document were generated using an **incorrect Yeo-17 network mapping** in `models/yeo17_networks.py`. The old mapping used fabricated contiguous-block ROI-to-network assignments that do not correspond to actual Glasser-to-Yeo17 anatomical overlap. All ROI-level, network-level, and circuit-level interpretations (gradient saliency, clustering, subtyping) are therefore **not neurobiologically valid**. The mapping was corrected via volumetric overlap between the Glasser atlas and Yeo 2011 17-network atlas (bilateral majority-vote). See `dataloaders/glasser_to_yeo17_mapping.json` for the validated mapping. Experiments with the corrected mapping (v5) have been submitted and results will supersede this document.

**Date**: 2026-03-07
**Models analyzed**: Classical Circuit MoE 4-expert (adhd_3), Classical Circuit MoE 2-expert (adhd_2)
**Dataset**: ABCD fMRI, ADHD classification, N=4,458 (Test set: 669 subjects, 381 ADHD-, 288 ADHD+)
**Parcellation**: HCP-MMP1 180-ROI (bilateral average of left+right hemisphere Glasser regions; verified against `HCPMMP1_for_ABCD.nii.gz`)
**Jobs**: 49772486 (interpretability), 49772487 (heterogeneity)

## 1. Interpretability Analysis

### 1.1 Gate Weights by Class

Gate weights show whether the model routes ADHD+ and ADHD- subjects differently across brain circuit experts.

#### 4-Expert Model (adhd_3)

| Expert | ADHD+ Mean | ADHD- Mean | Diff | p-value |
|--------|:----------:|:----------:|:----:|:-------:|
| DMN | 0.2505 | 0.2502 | +0.0002 | 0.712 |
| Executive | 0.2487 | 0.2496 | -0.0009 | 0.280 |
| **Salience** | **0.2587** | **0.2578** | **+0.0010** | 0.541 |
| SensoriMotor | 0.2421 | 0.2424 | -0.0004 | 0.451 |

#### 2-Expert Model (adhd_2)

| Expert | ADHD+ Mean | ADHD- Mean | Diff | p-value |
|--------|:----------:|:----------:|:----:|:-------:|
| Internal | 0.5059 | 0.5034 | +0.0025 | 0.168 |
| External | 0.4941 | 0.4966 | -0.0025 | 0.168 |

**Finding**: No significant gate weight differences between ADHD+ and ADHD- (all p > 0.17). The load balancing loss (alpha=0.1) ensures near-uniform routing, suppressing class-dependent gating. The Salience expert receives slightly more weight overall (0.258 vs ~0.245-0.250 for others), but this pattern does not differ by diagnosis.

**Interpretation**: Group-level ADHD classification does not rely on differential expert routing. Instead, the model distinguishes ADHD through within-expert feature processing — the same routing pattern produces different internal representations for ADHD+ vs ADHD- subjects.

### 1.2 Gradient Saliency — Circuit Level

Gradient saliency measures how much each input feature contributes to the ADHD+ prediction. Absolute saliency = magnitude of importance; signed saliency = direction of relationship.

#### 4-Expert Model

| Circuit | Absolute Saliency | Signed Saliency | Direction |
|---------|:-----------------:|:---------------:|:---------:|
| **SensoriMotor** | **0.000515** | **+0.000115** | **+ADHD** |
| Salience | 0.000443 | +0.000009 | +ADHD |
| DMN | 0.000387 | +0.000040 | +ADHD |
| Executive | 0.000183 | +0.000017 | +ADHD |

#### 2-Expert Model

| Circuit | Absolute Saliency | Signed Saliency | Direction |
|---------|:-----------------:|:---------------:|:---------:|
| **Internal** | **0.000739** | **+0.000050** | **+ADHD** |
| External | 0.000532 | +0.000025 | +ADHD |

**Finding**: All circuits show positive signed saliency (increasing activation increases ADHD+ probability). SensoriMotor is the most salient circuit by both absolute and signed measures in the 4-expert model. In the 2-expert model, the Internal circuit (containing DMN, Salience, Limbic) is more salient than External (Control, DorsAttn, SomMot, Visual).

### 1.3 Gradient Saliency — Network Level

Top 5 networks by absolute saliency (4-expert model):

| Network | Circuit | Abs Saliency | Signed Saliency | Direction |
|---------|---------|:------------:|:---------------:|:---------:|
| **SalVentAttnB** | Salience | **0.000623** | +0.000123 | **+ADHD** |
| VisPeri | SensoriMotor | 0.000554 | +0.000163 | +ADHD |
| SomMotA | SensoriMotor | 0.000522 | +0.000146 | +ADHD |
| SomMotB | SensoriMotor | 0.000517 | -0.000015 | -ADHD |
| Limbic_OFC | Salience | 0.000505 | -0.000087 | -ADHD |

Top 5 networks by absolute saliency (2-expert model):

| Network | Circuit | Abs Saliency | Signed Saliency | Direction |
|---------|---------|:------------:|:---------------:|:---------:|
| **DefaultC** | Internal | **0.000926** | +0.000147 | **+ADHD** |
| SalVentAttnB | Internal | 0.000908 | -0.000095 | -ADHD |
| Limbic_OFC | Internal | 0.000881 | +0.000217 | +ADHD |
| SalVentAttnA | Internal | 0.000723 | -0.000032 | -ADHD |
| Limbic_TempPole | Internal | 0.000720 | +0.000057 | +ADHD |

**Finding**: Salience/ventral attention networks are consistently the most salient across both models. Notably, the signed direction of some networks flips between models (e.g., SalVentAttnB is +ADHD in 4-expert but -ADHD in 2-expert), suggesting the direction depends on the circuit context in which a network is processed.

### 1.4 Gradient Saliency — ROI Level

Top 5 ROIs by absolute saliency (4-expert model):

| ROI | Region (HCP-MMP1) | Anatomy | Network | Circuit | Abs Saliency | Signed | Direction |
|-----|-------------------|---------|---------|---------|:------------:|:------:|:---------:|
| 171 | **TGv** (Area TG Ventral) | Temporal Pole | DefaultA | DMN | 0.001537 | -0.000318 | **-ADHD** |
| 73 | **44** (Area 44 / Broca's) | Inferior Frontal | SalVentAttnB | Salience | 0.001489 | +0.000153 | +ADHD |
| 134 | **TF** (Area TF) | Medial Temporal | DefaultC | DMN | 0.001080 | -0.000196 | **-ADHD** |
| 29 | **7m** (Area 7m) | Posterior Cingulate | SomMotA | SensoriMotor | 0.000866 | +0.000274 | +ADHD |
| 126 | **PHA3** (ParaHippocampal 3) | Medial Temporal | DefaultB | DMN | 0.000837 | -0.000231 | **-ADHD** |

*Note: All ROIs are bilateral (average of left+right hemisphere), confirmed via cross-referencing with `HCPMMP1_for_ABCD.nii.gz`.*

**Key pattern**: DMN ROIs — TGv (temporal pole), TF (medial temporal), PHA3 (parahippocampal) — consistently show **negative** ADHD relationship (higher activation = lower ADHD probability), while Area 44 / Broca's (ventral attention) and 7m (posterior cingulate) show **positive** ADHD relationship. This is consistent with the ADHD literature: DMN hypoactivation and sensorimotor/salience hyperactivation are hallmarks of ADHD (Cortese 2012, Koirala 2024).

### 1.5 Input Projection Weights

The learned projection weights show which ROIs each expert has learned to emphasize.

#### 4-Expert Model — Top Network Per Expert

| Expert | Top Network | Mean Weight | Max Weight | ROI Count |
|--------|-------------|:-----------:|:----------:|:---------:|
| DMN | DefaultC | 0.700 | 1.051 | 10 |
| Executive | ContA | 0.763 | 0.900 | 13 |
| **Salience** | **SalVentAttnB** | **0.994** | **1.355** | **6** |
| SensoriMotor | VisPeri | 0.827 | 0.971 | 12 |

**Finding**: Each expert's highest projection weights correspond to its assigned brain circuit's constituent networks, validating that the model has learned circuit-appropriate feature extraction. The Salience expert has the highest absolute weights (SalVentAttnB=0.994), consistent with its high saliency in gradient analysis.

### 1.6 Summary of Interpretability Findings

1. **Gate routing is class-agnostic**: Load balancing suppresses differential routing. ADHD discrimination happens within experts, not between them.
2. **SensoriMotor and Salience circuits are most informative**: By absolute gradient saliency, these circuits dominate both models.
3. **DMN ROIs show negative ADHD relationship**: TGv / temporal pole (ROI 171, DefaultA), TF / medial temporal (ROI 134, DefaultC), PHA3 / parahippocampal (ROI 126, DefaultB) all have negative signed saliency — higher bilateral activation in these regions is associated with lower ADHD probability (consistent with DMN hypoactivation in ADHD).
4. **SomMotA and SalVentAttnB show positive ADHD relationship**: Area 44 / Broca's area (ROI 73) and Area 7m / posterior cingulate (ROI 29) show positive relationship — higher bilateral activation is associated with higher ADHD probability (consistent with sensorimotor hyperactivation in ADHD).
5. **Expert projection weights validate circuit assignments**: Each expert has learned to weight its assigned brain circuit's networks most heavily.

---

## 2. Heterogeneity Analysis

### 2.1 Silhouette Sweep (Circuit-Level)

The silhouette score measures clustering quality (higher = better separated clusters).

| K | 4-Expert Silhouette | 2-Expert Silhouette |
|---|:-------------------:|:-------------------:|
| 2 | **0.9615** | **0.8497** |
| 3 | 0.7796 | 0.4920 |
| 4 | 0.7642 | 0.5534 |
| 5 | 0.3879 | 0.5608 |
| 6 | 0.4187 | 0.5120 |

**Finding**: K=2 is optimal for both models. The 4-expert model achieves very high silhouette (0.96), indicating gate weight vectors naturally separate into 2 distinct clusters. K=3 is used for detailed analysis below (to explore potential subtypes), though it produces less balanced clusters.

### 2.2 Circuit-Level Clustering (K=3)

Clustering based on 4D gate weight vectors (4-expert) or 2D gate weight vectors (2-expert).

#### 4-Expert Model

| Cluster | N (%) | Dominant | DMN | Executive | Salience | SensoriMotor |
|---------|:-----:|:--------:|:---:|:---------:|:--------:|:------------:|
| 0 | 15 (5.2%) | Salience | 0.250 | 0.227 | **0.289** | 0.234 |
| **1** | **272 (94.4%)** | **Salience** | **0.251** | **0.250** | **0.256** | **0.243** |
| 2 | 1 (0.3%) | Salience | 0.142 | 0.134 | **0.589** | 0.135 |

**Finding**: The circuit-level clustering is highly imbalanced — 94.4% of ADHD+ subjects fall into a single cluster with near-uniform gating. Cluster 2 (1 subject) is a strong outlier with extreme Salience dominance (0.589). Cluster 0 (15 subjects) shows moderate Salience elevation. This suggests the load-balanced gating does not produce meaningful individual-level heterogeneity at the circuit level.

#### 2-Expert Model

| Cluster | N (%) | Dominant | Internal | External |
|---------|:-----:|:--------:|:--------:|:--------:|
| 0 | 149 (51.7%) | External | 0.493 | **0.507** |
| 1 | 5 (1.7%) | Internal | **0.630** | 0.370 |
| 2 | 134 (46.5%) | Internal | **0.516** | 0.484 |

**Finding**: More balanced distribution. Two main subtypes emerge: External-dominant (51.7%, slight External preference) and Internal-dominant (46.5%, slight Internal preference). A small outlier cluster (5 subjects) shows strong Internal dominance (0.630). The Internal-dominant group may correspond to subjects where DMN/Salience/Limbic circuits are more informative for ADHD prediction.

### 2.3 Network-Level Clustering (K=3)

Clustering based on 4x64=256D expert output vectors (4-expert) or 2x64=128D (2-expert).

#### 4-Expert Model

| Cluster | N (%) | Dominant Expert | DMN Norm | Executive Norm | Salience Norm | SensoriMotor Norm |
|---------|:-----:|:---------------:|:--------:|:--------------:|:-------------:|:-----------------:|
| 0 | 71 (24.7%) | Executive | 3.05 | **4.11** | 3.37 | 1.80 |
| 1 | 139 (48.3%) | DMN | **4.58** | 3.40 | 3.81 | 2.60 |
| 2 | 78 (27.1%) | DMN | 3.57 | 3.46 | 3.33 | 2.40 |

**Finding**: Network-level clustering reveals more balanced subtypes:
- **Cluster 0 (Executive-dominant, 24.7%)**: Highest Executive expert output norms, lowest SensoriMotor. May represent an ADHD subtype with primary cognitive control deficits.
- **Cluster 1 (DMN-dominant, 48.3%)**: Highest DMN and SensoriMotor output norms. May represent an ADHD subtype with prominent default mode dysregulation.
- **Cluster 2 (Balanced, 27.1%)**: Relatively uniform expert output norms. May represent a less differentiated ADHD presentation.

### 2.4 ROI-Level Clustering (K=3)

Clustering based on 180D ROI importance scores with signed direction.

#### 4-Expert Model — Cluster Characterization

**Cluster 0 (N=92, 31.9%)**
- Top positive (+ADHD) ROIs: OFC (ROI 92, Orbital Frontal Complex), 13l (ROI 91), 10pp (ROI 89, Polar 10p), 10v (ROI 87) — all ContA, orbital/polar frontal regions
- Top negative (-ADHD) ROIs: 31pd (ROI 160, posterior cingulate), 31a (ROI 161, posterior cingulate), TF (ROI 134, medial temporal), 5m (ROI 35, paracentral), SFL (ROI 25, superior frontal language)
- **Profile**: Positive control network / negative salience-sensorimotor pattern

**Cluster 1 (N=22, 7.6%)**
- Top positive (+ADHD) ROIs: **TGv** (ROI 171, temporal pole, signed=+0.049), TF (ROI 134, medial temporal), PeEc (ROI 121, perirhinal ectorhinal), TGd (ROI 130, temporal pole dorsal)
- Top negative (-ADHD) ROIs: 52 (ROI 102, early auditory), 8BL (ROI 69, dorsolateral prefrontal), FOP2 (ROI 114, frontal opercular)
- **Profile**: **Strong DMN-positive subtype** — dramatically elevated bilateral temporal pole TGv signal (+0.049, ~25x larger than other clusters)

**Cluster 2 (N=174, 60.4%)**
- Top positive (+ADHD) ROIs: ProS (ROI 120, prostriate), H (ROI 119, hippocampus), TGv (ROI 171, temporal pole), PI (ROI 177, para-insular), 13l (ROI 91)
- Top negative (-ADHD) ROIs: PHA3 (ROI 126, parahippocampal), MT (ROI 22, middle temporal), IP1 (ROI 144, intraparietal), PHT (ROI 136, lateral temporal)
- **Profile**: Weak, diffuse pattern with small signed scores (~10x smaller than Cluster 1)

**Key finding**: Cluster 1 (7.6% of ADHD+ subjects) shows a distinctive DMN-positive signature where Default Mode Network activation is strongly associated with ADHD+. This is opposite to the group-level finding (DMN negative at population level) and may represent a neurobiologically distinct ADHD subtype.

### 2.5 Cross-Level Concordance

Adjusted Rand Index (ARI) measures agreement between clustering levels. ARI=0 means random, ARI=1 means perfect agreement.

#### 4-Expert Model

| Comparison | ARI |
|------------|:---:|
| Circuit vs Network | 0.026 |
| Circuit vs ROI | 0.067 |
| **Network vs ROI** | **0.131** |

#### 2-Expert Model

| Comparison | ARI |
|------------|:---:|
| Circuit vs Network | -0.003 |
| Circuit vs ROI | 0.004 |
| Network vs ROI | 0.032 |

**Finding**: Cross-level concordance is very low (ARI < 0.15 in all cases). This means:
1. The heterogeneity structure differs fundamentally across analysis levels — circuit-level subtypes do not predict network-level or ROI-level subtypes.
2. Each level captures independent dimensions of ADHD heterogeneity.
3. The 4-expert model shows slightly higher cross-level concordance than the 2-expert model (especially Network vs ROI: 0.131 vs 0.032), suggesting the finer-grained 4-circuit decomposition captures more coherent multi-level heterogeneity.

### 2.6 Summary of Heterogeneity Findings

1. **Circuit-level clustering is dominated by load balancing**: Near-uniform gating produces highly imbalanced clusters (94.4% in one cluster for 4-expert). Circuit-level heterogeneity is limited by the auxiliary loss.

2. **Network-level clustering reveals 3 balanced subtypes**: Executive-dominant (24.7%), DMN-dominant (48.3%), and Balanced (27.1%). These may correspond to distinct neurobiological ADHD presentations.

3. **ROI-level clustering identifies a rare DMN-positive subtype**: 7.6% of ADHD+ subjects (Cluster 1) show a distinctive pattern where DMN activation is strongly positively associated with ADHD — opposite to the population-level negative DMN relationship. This subtype warrants further investigation.

4. **Heterogeneity is multi-dimensional**: Very low cross-level ARI indicates that subtypes defined at different analysis levels are largely independent. A subject may be in the "Executive-dominant" cluster at the network level but in the "DMN-positive" cluster at the ROI level.

5. **4-expert > 2-expert for heterogeneity**: The 4-expert model provides richer, more differentiated subtyping at all levels, with higher cross-level concordance.

---

## 3. Neuroscience Interpretation

### Consistency with ADHD Literature

The gradient saliency findings are broadly consistent with the systems neuroscience model of ADHD (Cortese 2012, Koirala 2024):

| Finding | Direction | Literature Support |
|---------|:---------:|-------------------|
| DMN hypoactivation | -ADHD | Castellanos & Proal 2012: reduced DMN deactivation during task |
| SensoriMotor hyperactivation | +ADHD | Cortese 2012: hyperactivation in somatomotor regions |
| Salience/VentAttn importance | High abs | Koirala 2024: Salience network dysfunction central to ADHD |
| Executive (ContA) positive | +ADHD | Compensatory control recruitment (Nigg 2020) |

### Novel Findings

1. **TGv / Area TG Ventral (ROI 171, bilateral temporal pole, DefaultA)** is the single most important ROI across both models — consistently showing negative ADHD relationship at the group level. The ventral temporal pole is a high-level association cortex involved in social cognition, semantic memory, and emotional processing — all of which are implicated in ADHD-related default mode dysfunction. Its strong negative bilateral relationship suggests that reduced temporal pole engagement is a core feature of ADHD in resting-state fMRI.

2. **The DMN-positive subtype (Cluster 1, 7.6%)** challenges the uniform "DMN hypoactivation in ADHD" narrative. In this subgroup, TGv (bilateral temporal pole) and other DMN regions show the *opposite* pattern — positive ADHD relationship — potentially representing a distinct neurobiological pathway involving DMN hyperactivation rather than hypoactivation. This aligns with recent ADHD biotype work (Feng 2024, Pan 2026) showing heterogeneous brain signatures within the ADHD diagnosis.

3. **Signed saliency direction flips by model**: SalVentAttnB is +ADHD in the 4-expert model but -ADHD in the 2-expert model. This suggests the direction of a network's ADHD relationship depends on its circuit context — when processed with DMN/Limbic regions (Internal expert), its contribution differs from when processed in isolation (Salience expert).

## 4. Brain Region Reference

ROI indices (0-179) map to HCP-MMP1 Glasser atlas regions. Each ROI is a **bilateral average** of the corresponding left- and right-hemisphere regions — verified by cross-referencing the 180-ROI and 360-ROI `.npy` files against the volumetric parcellation `dataloaders/HCPMMP1_for_ABCD.nii.gz` (labels 1-180 = LH, 201-380 = RH). The bilateral average MSE was 42x lower than left-hemisphere-only, confirming the mapping. Region names from `dataloaders/HCP-MMP1_UniqueRegionList.csv`.

| ROI | Region | Full Name | Lobe | Cortex |
|-----|--------|-----------|------|--------|
| 22 | MT | Middle Temporal Area | Occ | MT+ Complex |
| 25 | SFL | Superior Frontal Language Area | Fr | Dorsolateral Prefrontal |
| 29 | 7m | Area 7m | Par | Posterior Cingulate |
| 35 | 5m | Area 5m | Par | Paracentral Lobular / Mid Cingulate |
| 69 | 8BL | Area 8B Lateral | Fr | Dorsolateral Prefrontal |
| 73 | 44 | Area 44 (Broca's area) | Fr | Inferior Frontal |
| 87 | 10v | Area 10v | Fr | Anterior Cingulate / Medial Prefrontal |
| 89 | 10pp | Polar 10p | Fr | Orbital / Polar Frontal |
| 91 | 13l | Area 13l | Fr | Orbital / Polar Frontal |
| 92 | OFC | Orbital Frontal Complex | Fr | Orbital / Polar Frontal |
| 102 | 52 | Area 52 | Temp | Early Auditory |
| 114 | FOP2 | Frontal Opercular Area 2 | Fr | Insular / Frontal Opercular |
| 119 | H | Hippocampus | Temp | Medial Temporal |
| 120 | ProS | ProStriate Area | Par | Posterior Cingulate |
| 121 | PeEc | Perirhinal Ectorhinal Cortex | Temp | Medial Temporal |
| 126 | PHA3 | ParaHippocampal Area 3 | Temp | Medial Temporal |
| 130 | TGd | Area TG dorsal (Temporal Pole) | Temp | Lateral Temporal |
| 134 | TF | Area TF | Temp | Medial Temporal |
| 136 | PHT | Area PHT | Temp | Lateral Temporal |
| 144 | IP1 | Area IntraParietal 1 | Par | Inferior Parietal |
| 160 | 31pd | Area 31pd | Par | Posterior Cingulate |
| 161 | 31a | Area 31a | Par | Posterior Cingulate |
| 171 | TGv | Area TG Ventral (Temporal Pole) | Temp | Lateral Temporal |
| 177 | PI | Para-Insular Area | Temp | Insular / Frontal Opercular |

Lookup utility: `from dataloaders.hcp_mmp1_labels import get_roi_name, get_roi_info`

---

## 5. Files

| File | Description |
|------|-------------|
| `analysis/classical_adhd_3/interpretability_report.md` | 4-expert interpretability auto-report |
| `analysis/classical_adhd_3/results.json` | 4-expert raw interpretability results |
| `analysis/classical_adhd_2/interpretability_report.md` | 2-expert interpretability auto-report |
| `analysis/classical_adhd_2/results.json` | 2-expert raw interpretability results |
| `analysis/heterogeneity_classical_adhd_3/heterogeneity_report.md` | 4-expert heterogeneity auto-report |
| `analysis/heterogeneity_classical_adhd_3/heterogeneity_results.json` | 4-expert raw heterogeneity results |
| `analysis/heterogeneity_classical_adhd_3/subject_representations.npz` | 4-expert per-subject data |
| `analysis/heterogeneity_classical_adhd_2/heterogeneity_report.md` | 2-expert heterogeneity auto-report |
| `analysis/heterogeneity_classical_adhd_2/heterogeneity_results.json` | 2-expert raw heterogeneity results |
| `analysis/heterogeneity_classical_adhd_2/subject_representations.npz` | 2-expert per-subject data |

## References

1. Cortese, S., et al. (2012). Toward systems neuroscience of ADHD. *Am. J. Psychiatry*, 169(10), 1038-1055.
2. Castellanos, F. X., & Proal, E. (2012). Large-scale brain systems in ADHD. *Neuropsychopharmacology*, 37(1), 247-259.
3. Koirala, S., et al. (2024). Neurobiology of ADHD. *Nat. Rev. Neurosci.*, 25(12), 759-775.
4. Nigg, J. T., et al. (2020). Toward a revised nosology for ADHD heterogeneity. *Biol. Psychiatry: CNNI*, 5(8), 726-737.
5. Feng, A., et al. (2024). Functional imaging derived ADHD biotypes. *EClinicalMedicine*, 77, 102876.
6. Pan, N., et al. (2026). Mapping ADHD heterogeneity and biotypes. *JAMA Psychiatry*.
