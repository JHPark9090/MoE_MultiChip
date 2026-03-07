# ADHD Heterogeneity Analysis via Circuit MoE Learned Representations

**Date**: 2026-03-07
**Status**: Analysis script implemented (`analyze_heterogeneity.py`), not yet run.

## Motivation

ADHD is a heterogeneous disorder — patients sharing the same diagnosis exhibit markedly different symptom profiles, neurobiological substrates, and treatment responses (Nigg 2020, Koirala 2024). Recent large-scale neuroimaging studies have identified distinct ADHD biotypes based on functional connectivity (Feng 2024) and brain morphometry (Pan 2026), suggesting that ADHD is not a single entity but a collection of neurobiologically distinguishable subtypes.

The Circuit MoE architecture provides a natural framework for data-driven ADHD subtyping because its internal representations are organized by brain circuit. Rather than applying post-hoc dimensionality reduction (PCA, UMAP) to raw fMRI data, we can extract the model's own **learned, task-optimized representations** at multiple levels of neural organization and cluster ADHD+ subjects within these spaces. This yields biologically interpretable subtypes defined by how the model processes each subject's brain data through its circuit-specialized experts.

## Single Expert Models: What They Can and Cannot Do

### SE Can Produce Per-Subject Heterogeneity Analyses

A single expert model (SE) is not limited to group-level analyses. Several standard interpretability methods can produce per-subject representations suitable for clustering:

| Method | Output per Subject | Dimensionality | How It Works |
|--------|-------------------|:-:|--------------|
| Gradient saliency | Per-ROI sensitivity | (N, 180) | `dOutput/dInput`, averaged over time |
| Attention weights | Per-timestep attention | (N, T, T) per head | From `nn.MultiheadAttention` inside the TransformerEncoder, accessed via `need_weights=True` |
| Hidden representation | Transformer output | (N, 64) | Output of `AllChannelExpert.forward()`, the 64D vector after temporal mean pooling |
| Input projection weights × input | Per-ROI importance | (N, 180) | `||W[:, roi]||_2 × |x[:, roi]|` from `Linear(180, 64)` |
| SHAP values | Per-ROI attribution | (N, 180) | Model-agnostic Shapley value estimation (e.g., KernelSHAP, DeepSHAP) |

Any of these can be used to cluster ADHD+ subjects and identify potential subtypes. Data-driven subtyping of neuroimaging data using multivariate methods (e.g., CCA-based brain-behaviour covariation analysis; Mihalik et al., 2019) and model-agnostic attribution methods (e.g., SHAP) are established approaches in classical ML for identifying heterogeneous subgroups.

### SE Can Approximate Circuit-Level Interpretation Post-Hoc

A researcher using SE could approximate Circuit MoE's interpretability by:

1. **Post-hoc circuit grouping**: Compute SE gradient saliency per ROI → group ROIs by Yeo-17 network assignment → aggregate into circuit-level scores (DMN, Executive, Salience, SensoriMotor). This produces a 4D circuit profile per subject, superficially similar to Circuit MoE's gate weights.

2. **SHAP interaction values**: Use pairwise SHAP interactions to measure which ROI pairs contribute jointly → infer circuit-level co-activation structure.

3. **Concept-based explanations (TCAV)**: Define "circuit concepts" (e.g., mean DMN ROI activation) and test whether the model's predictions are sensitive to these concepts using Testing with Concept Activation Vectors (Kim et al., 2018).

### What Circuit MoE Provides That SE Cannot Match

The fundamental distinction is not that SE *cannot* do heterogeneity analysis — it can. The distinction is **where the circuit structure comes from**:

- In **Circuit MoE**, circuit structure is **architectural** — built into the model before training. Each expert processes only its assigned circuit's ROIs, so every internal representation is automatically circuit-attributed.
- In **SE**, circuit structure is **post-hoc** — imposed by the analyst after training. The model was not designed to distinguish circuits, so post-hoc groupings are interpretive overlays on a model that may organize information differently.

This difference has three concrete consequences:

**1. Architectural attribution vs post-hoc labeling.**

When Circuit MoE's DMN expert produces a high activation norm for a subject, this is a direct architectural fact: the DMN expert, which receives only DMN ROIs (55 regions including mPFC, PCC, angular gyrus, temporal pole), processed those ROIs and produced a strong output. The circuit attribution is causal — the DMN expert physically cannot access Executive or Salience ROIs.

When SE's post-hoc analysis shows high gradient saliency for DMN ROIs, the interpretation is less certain. The model's `Linear(180, 64)` mixes all 180 ROIs into every hidden dimension. A DMN ROI may have high saliency not because of its DMN function, but because of its learned correlation with Executive ROIs in the 64D hidden space. The post-hoc "DMN saliency" label is an aggregation convenience, not a reflection of how the model actually processes information.

**2. Multi-level hierarchy vs single-level representations.**

Circuit MoE provides three naturally nested representation levels:

```
Level 1 — Circuit (4D):   Gate weights — which circuits matter for this subject
Level 2 — Network (256D): Expert outputs — what each circuit learned about this subject
Level 3 — ROI (180D):     Input projection activations — which ROIs drive each circuit
```

These levels are architecturally linked: Level 1 weights come from the gating network, Level 2 vectors come from the experts that Level 1 routes to, and Level 3 scores come from the input projections within each Level 2 expert. Cross-level concordance (e.g., Adjusted Rand Index between Level 1 and Level 2 cluster assignments) measures whether the heterogeneity structure is consistent across spatial scales — a multi-scale analysis that is unique to the MoE architecture.

SE provides per-subject representations (64D hidden, 180D saliency, attention weights), but these are independent outputs of different interpretability methods, not architecturally nested levels. There is no principled way to assess cross-level concordance because the "levels" are not structurally related.

**3. Learned circuit routing vs passive sensitivity.**

Circuit MoE's gate weights represent an **active routing decision**: the gating network learned, during ADHD classification training, to assign different weights to different circuit experts for different subjects. This routing is part of the model's forward pass — it directly affects the prediction. If the model learns that DMN features are more informative for Subject A than Subject B, this is reflected in Subject A receiving a higher DMN gate weight.

SE's gradient saliency is a **passive sensitivity measure**: it describes how the output changes if the input is perturbed, but the model did not make an explicit "decision" to attend to specific circuits. The saliency map is a property of the learned function, not a routing mechanism. This distinction matters for heterogeneity analysis: Circuit MoE's subtypes are defined by how the model *chooses* to process each subject, while SE's subtypes are defined by what the model *happens* to be sensitive to.

### Summary: Efficiency and Directness, Not Exclusivity

The advantage of Circuit MoE for heterogeneity analysis is **efficiency and directness**, not exclusivity:

| Aspect | Circuit MoE | SE + Post-Hoc Analysis |
|--------|-------------|------------------------|
| Circuit-level profiles | Automatic (gate weights) | Manual (group saliency by Yeo-17) |
| Circuit attribution | Architectural (guaranteed) | Interpretive (approximate) |
| Multi-level hierarchy | Built-in (3 nested levels) | Constructed (independent methods) |
| Cross-level concordance | Principled (ARI across levels) | Ad hoc (no structural link between methods) |
| Computational cost | Single forward pass | Multiple methods (SHAP, saliency, attention) |
| Reproducibility | Deterministic (model outputs) | Method-dependent (SHAP sampling, gradient noise) |
| Additional analysis design | None required | Requires choosing post-hoc methods and aggregation strategies |

Circuit MoE provides circuit-attributed, multi-level subtyping as a **natural byproduct** of its architecture. SE requires bespoke interpretability pipelines to approximate the same structure, and the results depend on the chosen post-hoc method rather than the model's own learned organization.

## Level 1: Circuit-Level Heterogeneity (4D Gate Weights)

### What It Captures

The gating network produces a K-dimensional soft weight vector for each subject:

```
gate_input = temporal_mean(fMRI)         # (B, 180)
gate_weights = softmax(MLP(gate_input))  # (B, K=4)
```

Each subject's gate weight vector is a **circuit fingerprint** — a 4-dimensional profile showing how much the model routes that subject's brain data through each circuit expert:

```
Subject A: [DMN=0.35, Executive=0.20, Salience=0.30, SensoriMotor=0.15]
Subject B: [DMN=0.15, Executive=0.40, Salience=0.10, SensoriMotor=0.35]
```

Subject A is "DMN/Salience-dominant" and Subject B is "Executive/SensoriMotor-dominant." These profiles directly reflect how the model's ADHD classifier weighs different brain circuits for each individual.

### What SE Can Do Instead

SE has no gating network, so there is no direct equivalent. The closest approximation is to compute gradient saliency per ROI, then aggregate by circuit:

```
saliency = |dOutput/dInput|.mean(dim=time)   # (B, 180) — per-subject, per-ROI
dmn_score = saliency[:, dmn_roi_indices].mean(dim=1)      # (B,)
exec_score = saliency[:, exec_roi_indices].mean(dim=1)    # (B,)
sal_score = saliency[:, sal_roi_indices].mean(dim=1)      # (B,)
sm_score = saliency[:, sm_roi_indices].mean(dim=1)        # (B,)
circuit_profile = stack([dmn, exec, sal, sm])              # (B, 4)
```

This produces a 4D circuit profile per subject, but it is a post-hoc aggregation of gradient saliency — not a learned routing decision. The SE model's `Linear(180, 64)` does not respect circuit boundaries, so the per-circuit saliency scores may reflect cross-circuit weight interactions rather than genuine circuit-specific sensitivity.

### Analysis Method

1. **Extract** gate weights for all ADHD+ subjects in the test set → (N_adhd, 4)
2. **Cluster** using k-means (K=2 to 6, select optimal K by silhouette score)
3. **Characterize** each cluster by its mean gate weight profile and dominant circuit
4. **Test** whether gate weights differ significantly across clusters (one-way ANOVA per circuit)

### Neurobiological Interpretation

If we find, for example, 3 clusters:

- **Cluster 0** (DMN-dominant): subjects whose ADHD classification relies primarily on default mode network features → consistent with DMN suppression failure subtype (Nigg 2020, Castellanos & Proal 2012)
- **Cluster 1** (Executive-dominant): classification driven by frontoparietal/dorsal attention → consistent with executive dysfunction subtype (Cortese 2012)
- **Cluster 2** (Salience-dominant): classification driven by salience/limbic features → consistent with emotional dysregulation subtype (Nigg 2020)

These would align with the heterogeneity proposals reviewed by Nigg et al. (2020), which include executive dysfunction, emotional/anger dysregulation, and reward response/delay aversion as distinct neurobiological pathways to ADHD — each associated with different neural circuits and temperament systems.

### Limitations

- The Switch Transformer load-balancing loss (alpha=0.1) encourages uniform routing, which compresses cross-subject gate weight variability. Any clustering structure that survives this regularization is conservative.
- With only 4 dimensions, clustering may identify only coarse subtypes. Fine-grained heterogeneity requires higher-dimensional representations (Levels 2 and 3).

## Level 2: Network-Level Heterogeneity (K×H Expert Outputs)

### What It Captures

Before gating combines the expert outputs, each expert produces an independent H-dimensional (H=64) hidden vector per subject:

```
expert_outputs = [expert_0(x_DMN), expert_1(x_Exec), expert_2(x_Sal), expert_3(x_SM)]
# Each: (B, 64)
# Concatenated: (B, 4×64 = 256)
```

This 256-dimensional representation is richer than the 4D gate weights. While gate weights capture *how much* each circuit is used, expert output vectors capture *what* each circuit learned about the subject. Two subjects may have identical gate weights (e.g., both DMN-dominant) but differ in their DMN expert output vectors — meaning the model detected different patterns within their DMN activity.

### What SE Can Do Instead

SE produces a single 64D hidden vector per subject (after temporal mean pooling of the Transformer output). This vector can be used for clustering, but it is an undifferentiated mixture of all 180 ROIs' contributions. The `Linear(180, 64)` projection entangles all ROIs into every hidden dimension — there is no way to decompose it post-hoc into "the DMN component" and "the Executive component."

A researcher could attempt to recover circuit structure from SE's 64D representation by:
- Training a linear probe to predict circuit-level features from the 64D vector
- Applying PCA and inspecting whether principal components align with circuit boundaries

However, there is no guarantee that SE's learned representation organizes information by circuit. The model may have learned a different factorization that does not correspond to neuroscience-defined circuits. Circuit MoE's expert outputs are **architecturally separated** — the DMN expert's 64D vector contains only information from DMN ROIs, by construction.

### Why Learned Representations Are Better Than Post-Hoc Dimensionality Reduction

A common alternative is to apply PCA or UMAP to the raw fMRI data (N_subjects × 180 ROIs × 363 timepoints) and cluster in the reduced space. This approach has two disadvantages:

1. **PCA/UMAP is unsupervised and task-agnostic.** It finds directions of maximum variance in the data, which may not correspond to ADHD-relevant dimensions. The expert output vectors are **task-optimized** — each expert learned to extract features specifically for ADHD classification from its assigned circuit's ROIs.

2. **PCA/UMAP is post-hoc and model-independent.** The dimensionality reduction has no relationship to the classification model. Expert outputs are the model's own internal representation — clustering in this space finds subtypes defined by how the ADHD classifier actually processes each subject's brain data.

### Analysis Method

1. **Extract** per-subject expert output vectors → (N_adhd, K, H) = (N_adhd, 4, 64)
2. **Flatten** to (N_adhd, 256) for joint clustering, or analyze per-expert (N_adhd, 64)
3. **Cluster** using k-means on the 256D space
4. **Characterize** each cluster by per-expert activation norms — which experts produce the strongest signals for each subtype
5. **Cross-level concordance**: compute Adjusted Rand Index (ARI) between Level 1 and Level 2 cluster assignments to assess whether gate-weight-based and expert-output-based subtypes agree

### Neurobiological Interpretation

Expert output norms reveal which circuits produce the most distinctive representations for each subtype:

- If Cluster 0 has high DMN expert norm and low SensoriMotor expert norm, this subtype is characterized by strong DMN-derived features — the model found rich, discriminative temporal patterns in their DMN activity
- If two clusters have similar gate weights but different expert output vectors, the heterogeneity lies *within* the circuit representations, not in circuit routing — a finer-grained distinction than Level 1 can capture

## Level 3: ROI-Level Heterogeneity (180D Importance Scores)

### What It Captures

For each subject, we compute a 180-dimensional importance score vector that measures how much each brain ROI contributes to the model's internal representation:

```
For each expert i:
    W_i = expert_i.input_projection.weight    # (H, n_rois_i)
    w_norms_i = L2_norm(W_i, axis=0)          # (n_rois_i,) — learned weight magnitude per ROI
    x_magnitude = mean(|x_subset|, dim=time)  # (B, n_rois_i) — input activation per ROI
    roi_importance_i = x_magnitude × w_norms_i  # (B, n_rois_i) — subject-specific ROI importance

    # Map back to global ROI indices
    roi_scores[:, global_indices] = roi_importance_i
```

This produces a (B, 180) matrix where `roi_scores[subject, roi]` quantifies how much that ROI contributes to the model's representation for that specific subject. The score combines two factors:

- **Learned weight norms** (`w_norms`): what the model learned to attend to during training (static across subjects)
- **Input magnitude** (`x_magnitude`): how active that ROI is for this specific subject (varies across subjects)

Their product captures subject-specific, model-informed ROI importance.

### What SE Can Do Instead

SE can produce per-subject, per-ROI importance through several methods:

1. **Gradient saliency**: `|dOutput/dInput|` averaged over time → (N, 180). Measures output sensitivity to each ROI's input. Available for both classical and quantum SE.

2. **Input projection importance**: Same `||W[:, roi]||_2 × |x[:, roi]|` approach using SE's `Linear(180, 64)`. The weight norms reflect learned per-ROI importance, and input magnitude provides subject specificity. Structurally identical to Circuit MoE's method, but uses one projection covering all 180 ROIs instead of circuit-specific projections.

3. **SHAP values**: Model-agnostic attribution. KernelSHAP or DeepSHAP can produce per-subject, per-ROI Shapley values. More principled than gradient saliency (satisfies completeness, symmetry, null player axioms) but computationally expensive.

4. **Attention weights**: For classical SE only. The `TransformerEncoderLayer` contains `nn.MultiheadAttention`, which can return attention weight matrices via `need_weights=True`. These are (N, nhead, T, T) — per-subject temporal attention patterns. These can be analyzed per-head or averaged, but they capture temporal (not spatial/ROI) importance.

All of these produce per-subject 180D vectors (or higher-dimensional attention tensors) suitable for clustering. The key limitation is that SE's per-ROI scores are **circuit-unattributed**: if ROI 112 (PCC, part of DMN) has a high score, we know the model is sensitive to it, but we do not know whether this sensitivity arises from its DMN membership, its connectivity to Executive regions, or some other factor. In Circuit MoE, ROI 112's importance score comes specifically from the DMN expert's `Linear(55, 64)`, because the DMN expert is the only expert that receives ROI 112 — the circuit attribution is architectural.

### Analysis Method

1. **Extract** per-subject ROI importance scores → (N_adhd, 180)
2. **Cluster** using k-means on the 180D space
3. **Characterize** each cluster by:
   - Top 10 ROIs by mean importance score
   - Network-level aggregation (mean score per Yeo-17 network)
4. **Cross-level concordance**: ARI with Level 1 and Level 2 clusters

### Neurobiological Interpretation

ROI-level clustering can reveal spatially specific subtypes:

- **Cluster with high PCC/angular gyrus importance**: posterior DMN subtype, consistent with Castellanos & Proal (2012) who identified PCC as a hub of ADHD-related DMN dysfunction
- **Cluster with high DLPFC/FEF importance**: executive control subtype, consistent with frontoparietal hypoactivation in ADHD (Cortese 2012)
- **Cluster with high insula/ACC importance**: salience processing subtype, consistent with anterior insula's role in ADHD's attentional deficits

This level provides the spatial specificity needed to connect model-derived subtypes to the neuroanatomy literature.

## Cross-Level Integration

The three levels form a hierarchy:

```
Level 1 — Circuit (4D):   Which brain circuits matter most for this subject?
Level 2 — Network (256D): What did each circuit expert learn about this subject?
Level 3 — ROI (180D):     Which specific brain regions drive each circuit's representation?
```

By computing Adjusted Rand Index (ARI) across levels, we can assess whether the subtype structure is consistent:

- **High ARI (circuit vs network)**: subtypes defined by routing weights agree with subtypes defined by expert output content → coherent heterogeneity structure
- **Low ARI (circuit vs ROI)**: gate-weight-based subtypes and ROI-importance-based subtypes disagree → the heterogeneity has different structure at different spatial scales, suggesting multi-scale ADHD subtypes

### Why SE Cannot Provide Cross-Level Analysis

SE can produce per-subject representations at different granularities (64D hidden vector, 180D saliency, T×T attention matrices), but these come from **independent interpretability methods**, not architecturally nested levels. There is no structural link between SE's 64D hidden vector and its 180D gradient saliency — they are separate computations that happen to use the same model. Computing ARI between clusters derived from these two representations would measure agreement between two arbitrary analysis methods, not consistency across spatial scales of the model's internal organization.

In Circuit MoE, the three levels are architecturally linked: Level 1 gate weights determine how much each Level 2 expert output contributes to the prediction, and Level 3 input projection activations feed into each Level 2 expert. Cross-level concordance measures whether the model's own hierarchical organization produces consistent subtype structure — a question that only makes sense when the hierarchy is built into the model.

## Silhouette Sweep for Optimal Cluster Count

Rather than fixing K a priori, we sweep K from 2 to 6 at the circuit level (4D, cheapest) and select the K with the highest silhouette score. This data-driven approach avoids assuming the number of ADHD subtypes.

The silhouette sweep is informative in itself:
- **Monotonically decreasing silhouette**: no clear cluster structure → ADHD heterogeneity may be continuous rather than categorical
- **Clear peak at K=3**: three distinct subtypes → potentially maps to the neurobiological pathways reviewed by Nigg (2020): executive dysfunction, emotional dysregulation, and reward/delay aversion
- **Peak at K=2**: binary structure → possibly internalizing vs externalizing ADHD, or the Internal/External circuit split from the 2-expert configuration

## Implementation

### Script: `analyze_heterogeneity.py`

```
Usage:
    python analyze_heterogeneity.py \
        --checkpoint=checkpoints/CircuitMoE_classical_adhd_3_49731003.pt \
        --output-dir=analysis/heterogeneity_classical_adhd_3 \
        --n-clusters=3
```

**Outputs per model:**
- `heterogeneity_report.md` — formatted markdown with cluster profiles and cross-level concordance
- `heterogeneity_results.json` — raw numerical results (cluster assignments, silhouette scores, ARI values)
- `subject_representations.npz` — per-subject arrays (gate weights, expert outputs, ROI scores, labels, cluster IDs) for downstream analysis

### SLURM: `scripts/run_heterogeneity_analysis.sh`

Runs all 4 trained models (classical/quantum × 4-expert/2-expert). Configurable cluster count via `N_CLUSTERS` environment variable (default: 3).

## Models to Analyze

| Model | Checkpoint | Test AUC | Experts |
|-------|-----------|:--------:|:-------:|
| Classical 4-expert | `CircuitMoE_classical_adhd_3_49731003.pt` | 0.6167 | DMN, Executive, Salience, SensoriMotor |
| Classical 2-expert | `CircuitMoE_classical_adhd_2_49731010.pt` | 0.5987 | Internal, External |
| Quantum v2 4-expert | `CircuitMoE_quantum_adhd_3_49767122.pt` | 0.5764 | DMN, Executive, Salience, SensoriMotor |
| Quantum v2 2-expert | `CircuitMoE_quantum_adhd_2_49767123.pt` | 0.5783 | Internal, External |

The 4-expert models are preferred for heterogeneity analysis because they provide finer-grained circuit decomposition (4 circuits vs 2).

## Caveats

1. **Model performance ceiling.** All models achieve 0.58-0.62 AUC. Subtype structure from a model that is barely above chance should be treated as exploratory, not confirmatory. The model may not have learned sufficient ADHD-relevant features for meaningful subtyping.

2. **Load balancing suppresses routing differences.** The auxiliary loss encourages uniform gate weights, compressing cross-subject variability in the circuit-level representation. This biases against finding distinct subtypes at Level 1.

3. **Single seed.** All models trained with seed=2025. Cluster stability across seeds is not validated. Ideally, the same subtypes should emerge across multiple training runs.

4. **Cortical ROIs only.** HCP-MMP1 covers 180 cortical ROIs. Subcortical structures (caudate, pallidum, nucleus accumbens) implicated in ADHD (Pan 2026, Nigg 2020) are absent from the parcellation.

5. **Cluster count is a modeling choice.** K-means assumes K spherical clusters. The true ADHD heterogeneity structure may be non-spherical, hierarchical, or continuous. Silhouette scores guide but do not determine the optimal K.

6. **Correlation, not causation.** Model-derived subtypes reflect how the classifier processes brain data, not the causal neurobiological mechanisms of ADHD subtypes.

## References

1. Nigg, J. T., et al. (2020). Toward a Revised Nosology for ADHD Heterogeneity. *Biol. Psychiatry: CNNI*, 5(8), 726-737.
2. Koirala, S., et al. (2024). Neurobiology of ADHD. *Nat. Rev. Neurosci.*, 25(12), 759-775.
3. Feng, A., et al. (2024). Functional imaging derived ADHD biotypes. *EClinicalMedicine*, 77, 102876.
4. Pan, N., et al. (2026). Mapping ADHD Heterogeneity and Biotypes. *JAMA Psychiatry*.
5. Cortese, S., et al. (2012). Toward systems neuroscience of ADHD: a meta-analysis of 55 fMRI studies. *Am. J. Psychiatry*, 169(10), 1038-1055.
6. Castellanos, F. X. & Proal, E. (2012). Large-scale brain systems in ADHD: beyond the prefrontal-striatal model. *Trends Cogn. Sci.*, 16(1), 17-26.
7. Kim, B., et al. (2018). Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV). *ICML*.
8. Mihalik, A., et al. (2019). Brain-behaviour modes of covariation in healthy and clinically depressed young people. *Scientific Reports*, 9, 11536.
