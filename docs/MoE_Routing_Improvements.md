# Improving Quantum MoE Routing for ADHD/ASD Classification

## Problem Statement

The current Cluster-Informed MoE pipeline uses unsupervised coherence-based clustering to route subjects to quantum/classical experts. However, the resulting clusters show weak association with diagnostic labels:

- **ASD**: Cluster 0 prevalence 52.1% vs Cluster 1 prevalence 47.4% (chi-squared p=0.001, but small effect size)
- **ADHD**: Similar pattern — statistically significant but clinically marginal separation

Because the routing signal is nearly orthogonal to the prediction target, the MoE experts cannot specialize for different clinical presentations. This limits performance, particularly for hard routing (which performs at chance) and explains why soft gating only marginally outperforms a single-expert baseline.

This document outlines four strategies to improve routing quality and MoE performance.

---

## 1. Supervised or Semi-Supervised Clustering

### Current Limitation

KMeans optimizes intra-cluster compactness in connectivity space, completely ignoring whether clusters are useful for ADHD/ASD prediction. The resulting clusters reflect scanner differences, age/sex demographics, or global connectivity strength — not diagnostic-relevant variation.

### Approach

Modify the clustering objective to jointly optimize for connectivity similarity *and* phenotype separation.

**Constrained KMeans**: Add must-link/cannot-link constraints. Pairs of subjects with *different* ADHD labels get a soft penalty for being in the same cluster. The cluster boundaries shift to better separate ADHD+ from ADHD- subjects while still respecting the connectivity manifold. This doesn't force perfect separation (that would just be a classifier), but biases clusters toward diagnostically relevant boundaries.

**Semi-supervised spectral clustering**: Build the affinity graph from connectivity features, then modify edge weights using label information. Edges between same-label subjects get boosted, cross-label edges get dampened. The spectral embedding then captures both connectivity structure and diagnostic signal. This is particularly natural since spectral clustering is already part of the pipeline.

### Impact on Quantum MoE

The experts would receive subject subgroups that are both neurally coherent *and* diagnostically distinct. Expert 0 might get a subgroup with high ASD prevalence and a specific connectivity profile, while Expert 1 gets a low-prevalence subgroup with a different profile. Each expert can then specialize — one learning subtle markers in a high-risk connectivity pattern, the other handling a different phenotypic presentation. Hard routing would actually become viable because the routing signal would carry diagnostic information.

### Risk

Overfitting. If the clustering uses train-set labels, the routing becomes a data-leaky feature. This must be done carefully — either cluster on the full dataset without using labels at test time, or use cross-validated clustering where subjects in the test fold are assigned to clusters via nearest-centroid without label access.

---

## 2. Learned Routing Entirely

### Current Limitation

The gating network receives cluster ID (a single integer) as its primary routing signal. Even in soft mode, the gating network's input dimensionality is severely bottlenecked — it's making routing decisions from a 1-dimensional discrete variable rather than the rich 16,110-dimensional connectivity profile.

### Approach

Remove the pre-computed cluster assignments entirely. Instead, the gating network takes the subject's connectivity features (or their PCA embedding) as input and learns to route directly:

```
Input fMRI → PCA features (1480-d) → Gating Network → expert weights
                                   → Expert 0 (quantum circuit)
                                   → Expert 1 (quantum circuit)
                                   → Weighted combination → prediction
```

The gating network would be a small MLP (e.g., 1480 → 64 → num_experts) trained jointly with the expert networks via backpropagation. The routing emerges from the data rather than being imposed. The balance loss (already implemented in `ClusterInformedMoE_ABCD.py`) prevents expert collapse where all subjects get routed to one expert.

### Impact on Quantum MoE

The system discovers its own "subtypes" optimized for classification accuracy rather than connectivity compactness. Different quantum experts would learn different circuit parameterizations tuned to different subpopulations — this is the core MoE value proposition. The routing would naturally correlate with diagnostic-relevant features because the entire system is trained end-to-end with the classification loss.

For quantum experts specifically, this is powerful because different quantum circuits have different inductive biases. One expert's circuit might learn to capture long-range connectivity patterns relevant to one ASD presentation, while another captures local circuit abnormalities relevant to a different presentation.

### Risk

Training instability. End-to-end MoE training is harder than fixed routing — experts can collapse (all load on one expert), oscillate (routing flips between experts across epochs), or converge to a trivial solution. The balance loss, gating noise, and careful learning rate tuning (already in the codebase) mitigate this. Starting with a pre-trained gating network initialized from the unsupervised cluster assignments could provide a warm start.

---

## 3. Multi-Task Auxiliary Loss

### Current Limitation

The model has a single objective (ADHD/ASD classification), but the cluster structure carries useful regularization information that isn't being leveraged during training. The cluster assignments are used only for routing, not as a learning signal.

### Approach

Add a secondary prediction head that predicts the cluster assignment alongside the primary diagnostic classification:

```
Total Loss = L_classification(ADHD/ASD) + λ * L_cluster(predict cluster ID) + α * L_balance
```

The shared feature backbone (the part before the experts) is trained to extract representations useful for *both* tasks. The cluster prediction head acts as a regularizer — it forces the learned features to encode connectivity subtype information, preventing the model from collapsing to trivial features. The hyperparameter λ controls the trade-off.

### Why This Helps

The ADHD/ASD labels are noisy (clinical diagnosis has inter-rater variability, comorbidities, spectrum effects). The cluster assignments, while unsupervised, are a clean signal derived from objective neural measurements. By jointly training on both, the model learns a feature space that respects neural subtype structure while optimizing for diagnosis. This is essentially a form of **auxiliary task regularization**.

### Impact on Quantum MoE

The quantum circuit parameters are regularized to produce representations that capture meaningful neural variation, not just overfit to noisy diagnostic labels. This is especially important for quantum models with limited parameters (27-41K) — every parameter needs to count, and the auxiliary loss provides additional gradient signal that helps the quantum circuits learn richer representations. For ASD particularly, where the primary signal is weak (AUC ~0.56), the auxiliary signal could prevent the quantum experts from converging to trivial solutions.

### Risk

If λ is too large, the model prioritizes cluster prediction over diagnosis and test AUC drops. Cross-validate λ on a held-out set. A schedule where λ starts high (learn connectivity structure first) and decays (shift focus to diagnosis) often works well.

---

## 4. Different Clustering Targets

### Current Limitation

Clustering uses whole-brain coherence (all 16,110 ROI pairs). Most of these edges are irrelevant to ADHD/ASD — they capture visual cortex connectivity, motor cortex connectivity, and other systems unrelated to the disorders. The diagnostic signal drowns in irrelevant connectivity variation.

### Approach

Instead of clustering on all connectivity features, restrict to edges known to be associated with ADHD/ASD from prior neuroimaging literature:

- **ADHD-relevant networks**: Fronto-parietal (executive control), default mode network (DMN), ventral attention network. ADHD is consistently associated with DMN-frontoparietal hypo-connectivity and intra-DMN hyper-connectivity. Select ROI pairs within and between these networks (~500-2000 edges instead of 16,110).

- **ASD-relevant networks**: Social brain network (medial prefrontal, temporoparietal junction, superior temporal sulcus), salience network (anterior insula, dorsal anterior cingulate), and long-range under-connectivity patterns. ASD literature consistently shows reduced long-range functional connectivity and altered local connectivity.

With the HCP180 parcellation, the relevant networks can be mapped to specific ROI indices and only the relevant edges extracted before clustering. The clusters would then reflect *disorder-relevant* connectivity subtypes.

### Impact on Quantum MoE

Clusters derived from disorder-relevant edges would naturally show stronger associations with diagnostic labels. The routing would send subjects to experts based on their pattern of dysfunction in clinically relevant circuits rather than their whole-brain connectivity profile. Each quantum expert could specialize in a different *clinical presentation* — for example, one expert for ASD subjects with primarily social-network disruption, another for those with primarily sensory-processing disruption.

An additional benefit for the quantum models: fewer input features (500-2000 vs 16,110) means the PCA dimensionality is lower, and the quantum circuit's limited expressivity is focused on the most relevant features. The information-to-noise ratio improves substantially.

### Risk

Bias toward prior knowledge. If the selected edges don't capture the full picture, signal is lost. A hybrid approach works well — use literature-guided edges for clustering but give the MoE experts access to all connectivity features for classification.

---

## Practical Recommendations

### Highest Impact (Recommended First)

**Approach 2 (Learned Routing)** — requires minimal new code. Replace the cluster-ID input to the gating network with the PCA features already being computed. The balance loss and gating noise are already implemented. This directly addresses the core bottleneck (uninformative routing) while keeping the quantum expert architecture unchanged.

### Quickest to Test

**Approach 4 (Different Clustering Targets)** — simplest to implement. Add a feature mask before clustering to filter the coherence matrix to DMN + frontoparietal ROI pairs only, re-run clustering, and compare the cluster-phenotype chi-squared effect size.

### Summary Table

| Approach | Code Effort | Expected Impact | Risk |
|----------|-------------|-----------------|------|
| 1. Supervised clustering | Medium | Medium-High | Label leakage if not cross-validated |
| 2. Learned routing | Low-Medium | High | Training instability (mitigated by existing balance loss) |
| 3. Multi-task auxiliary loss | Medium | Medium | λ tuning required |
| 4. Literature-guided clustering | Low | Medium | Prior knowledge bias |

---

## Current Baseline Results (for reference)

### ASD MoE (coherence clustering, site-regressed, k=2)

| Config | Test AUC | Test Acc | Params |
|--------|----------|----------|--------|
| Classical + Soft | 0.5804 | 55.4% | 283K |
| Classical + Hard | 0.4960 | 51.1% | 270K |
| Quantum + Soft (q8, d3) | 0.5625 | 52.5% | 41K |
| Quantum + Hard (q8, d3) | 0.5000 | 50.2% | 27K |
| Quantum + Hard (q10, d2) | 0.5041 | 50.5% | 34K |

### Key Observations

- Soft gating consistently outperforms hard routing across all configurations
- Hard routing performs at chance — the cluster-based routing signal is too weak
- Quantum models achieve competitive AUC with 7x fewer parameters than classical
- ASD is a harder target than ADHD (AUC ~0.55-0.58 vs ~0.60-0.62)
