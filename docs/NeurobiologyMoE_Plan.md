# Neurobiology-Based Circuit-Specialized MoE — Implementation Plan

**Date**: 2026-03-06
**Status**: In Progress (classical jobs submitted)

## Motivation

Prior experiments show that (1) the feature extraction bottleneck hypothesis was not validated by NAME (0.58 AUC < 0.62 SE baseline), and (2) current whole-brain clustering produces clusters only weakly associated with ADHD (6% rate difference). The literature identifies specific brain circuits implicated in ADHD, suggesting that **circuit-specialized experts** — where each expert processes only its assigned brain circuit's ROIs — may capture ADHD-relevant features better than all-channel experts.

## Literature Foundation

### Nigg et al. (2020) — Revised Nosology for ADHD Heterogeneity

Proposes ADHD has **parallel, non-overlapping neurobiological dimensions**:
- **Emotional dysregulation** (irritable vs. surgent temperament profiles) — maps to limbic, salience, and ventral attention circuits
- **Cognitive dysfunction** (top-down executive vs. bottom-up arousal) — maps to frontoparietal and cingulo-opercular circuits
- **Reward processing deficits** — mapped to nucleus accumbens resting-state FC

Key insight: ADHD subtypes cut across multiple dimensions, with **parallel profiles based on cognition, emotion, and specific neural signatures**. The emotional dysregulation subprofile is most reproducible.

> Nigg, J. T., Karalunas, S. L., Feczko, E., & Fair, D. A. (2020). Toward a Revised Nosology for Attention-Deficit/Hyperactivity Disorder Heterogeneity. *Biological Psychiatry: Cognitive Neuroscience and Neuroimaging*, 5(8), 726-737. https://doi.org/10.1016/j.bpsc.2020.02.005

### Koirala et al. (2024) — Neurobiology of ADHD: Historical Challenges and Emerging Frontiers

Comprehensive review arguing for a **brain-wide, "global" view** of ADHD rather than single-circuit localizationism. ADHD involves distributed dysfunction across multiple networks simultaneously, but with particular involvement of:
- Default mode network (DMN)
- Frontoparietal control network
- Attention networks (dorsal and ventral)
- Salience network

Emphasizes that the ABCD dataset (which we use) is central to modern ADHD neuroscience and that heterogeneity is a core challenge requiring dimensional approaches.

> Koirala, S., Grimsrud, G., Mooney, M. A., et al. (2024). Neurobiology of attention-deficit hyperactivity disorder: historical challenges and emerging frontiers. *Nature Reviews Neuroscience*, 25(12), 759-775. https://doi.org/10.1038/s41583-024-00869-z

### Feng et al. (2024) — ADHD Biotypes via Deep Clustering (GCN-BSD)

Directly relevant: uses the **same ABCD dataset** to identify **2 ADHD biotypes** via graph convolution network-based deep clustering on functional network connectivity (FNC):
- **Biotype 1** (mild, typical): positive within-DMN connectivity, SM-CC-CB connectivity
- **Biotype 2** (severe, atypical): widespread abnormalities, cerebellum-fusiform connectivity, lower fluid intelligence, more hyperactivity/impulsivity
- Top discriminative connections: **Default Mode and Cognitive Control networks**, followed by visual-sensorimotor
- Cross-site replication: r = 0.75-0.76 (ABCD → Peking University)
- Biotype 1 responds better to methylphenidate; Biotype 2 to atomoxetine
- Explicitly states: "ADHD-related dysfunctions are not only involved in higher-level cognitive-behavioral functions...but also in **sensorimotor processes, including SM**"

> Feng, A., Zhi, D., Feng, Y., et al. (2024). Functional imaging derived ADHD biotypes based on deep clustering: a study on personalized medication therapy guidance. *EClinicalMedicine*, 77, 102876. https://doi.org/10.1016/j.eclinm.2024.102876

### Pan et al. (2026) — ADHD Biotypes via Morphometric Similarity Networks

Identifies **3 ADHD biotypes** via normative modeling of brain morphometry topology (HYDRA algorithm):
- **Biotype 1**: Pallidum deviations, most severe symptomatology across all domains
- **Biotype 2**: Anterior cingulate cortex + pallidum, predominantly hyperactivity/impulsivity
- **Biotype 3**: Superior frontal gyrus alterations, marked inattention profile
- Key regions: orbitofrontal cortex, caudate, hippocampus, inferior frontal gyrus
- Discovery: 446 ADHD + 708 controls; Validation: 554 ADHD + 123 controls

> Pan, N., Long, Y., Qin, K., et al. (2026). Mapping ADHD Heterogeneity and Biotypes by Topological Deviations in Morphometric Similarity Networks. *JAMA Psychiatry*. https://doi.org/10.1001/jamapsychiatry.2026.0001

### Cortese et al. (2012) — Systems Neuroscience Meta-Analysis of 55 fMRI Studies

The definitive meta-analysis establishing that ADHD involves dysfunction beyond higher-order cognitive circuits:
- In children: significant **hyperactivation in somatomotor networks** (26% of hyperactivated voxels in comorbidity-free samples), plus ventral attention and default networks
- In adults: **hyperactivation in visual and dorsal attention** networks; somatomotor hyperactivation decreases with age (consistent with motoric hyperactivity decreasing clinically)
- Explicitly hypothesized and confirmed: "ADHD-related dysfunctions in networks involved not only in higher-level cognitive/behavioral functions, such as the frontoparietal, dorsal attention, and default networks, but also in **sensorimotor processes, including somatomotor and visual networks**"

> Cortese, S., Kelly, C., Chabernaud, C., Proal, E., Di Martino, A., Milham, M. P., & Castellanos, F. X. (2012). Toward systems neuroscience of ADHD: a meta-analysis of 55 fMRI studies. *American Journal of Psychiatry*, 169(10), 1038-1055. https://doi.org/10.1176/appi.ajp.2012.11101521

## Important Conceptual Clarification: Biotypes vs. Circuit Experts

The literature on biotypes (Feng: 2 biotypes; Pan: 3 biotypes) and the literature on ADHD-implicated circuits (Nigg, Koirala, Cortese) address **different aspects of ADHD heterogeneity**:

- **Biotypes** = subtypes of ADHD patients identified by unsupervised clustering. Different patients have different neural profiles.
- **Circuit experts** = brain regions/networks implicated in ADHD. Different brain circuits serve different functions (self-regulation, executive control, salience/emotion, sensorimotor).

**The number of biotypes does not dictate the number of circuit experts.** Feng finding 2 patient biotypes does not mean we should use 2 brain circuits. Pan finding 3 biotypes does not mean we should use 3 circuits. The number of circuit experts is driven by **how many distinct ADHD-relevant brain circuits exist**, as identified by the neuroscience literature.

Based on the literature, **4 ADHD-relevant circuit systems** are well-established:

1. **DMN** — Nigg (2020), Koirala (2024), Feng (2024): DMN suppression failure, within-DMN connectivity differentiates biotypes
2. **Executive/Frontoparietal** — Nigg (2020), Koirala (2024), Cortese (2012): top-down executive dysfunction, frontoparietal hypoactivation
3. **Salience/Affective** — Nigg (2020), Koirala (2024): emotional dysregulation, salience network dysfunction
4. **Sensorimotor/Visual** — Cortese (2012), Feng (2024): somatomotor hyperactivation in children (26% of hyperactivated voxels), sensorimotor processes contribute to biotype discrimination

We test two configurations: **4-expert** (one per circuit) and **2-expert** (coarser Internal/External grouping).

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Number of experts | Test 4-expert and 2-expert configs | 4 experts: one per ADHD-relevant circuit (DMN, Executive, Salience, SensoriMotor). 2 experts: coarser Internal/External split. |
| Expert input | Each expert sees different ROIs | Circuit specialization aligns with Nigg's parallel dimensions |
| Gating | Learned soft routing (no clustering) | Gating network learns to weight experts from the full 180-ROI temporal mean. Simpler than two-stage clustering pipeline. |
| Parcellation | HCP180 (cortical only) | Acceptable; subcortical regions (pallidum, caudate from Pan) not available but cortical circuits are well-represented |

## Architecture

### Circuit Definitions

Based on the Yeo 17-network mapping in `models/yeo17_networks.py`:

#### 4-Expert Configuration (`adhd_3`)

| Expert | Circuit | Yeo17 Networks | ROIs | Neurobiological Basis |
|--------|---------|---------------|------|----------------------|
| 0 | DMN | DefaultA, DefaultB, DefaultC, TempPar | 55 | Nigg: self-regulation failures; Feng: within-DMN connectivity differentiates biotypes; Koirala: DMN suppression failure |
| 1 | Executive Control | ContA, ContB, ContC, DorsAttnA, DorsAttnB | 50 | Nigg: top-down executive dysfunction; Feng: Cognitive Control network is top discriminator; Pan: Biotype 3 (inattention) maps to frontal; Cortese: frontoparietal hypoactivation |
| 2 | Salience / Affective | SalVentAttnA, SalVentAttnB, Limbic_TempPole, Limbic_OFC | 29 | Nigg: emotional dysregulation dimension; Pan: Biotype 2 (hyperactivity) involves anterior cingulate (salience hub) |
| 3 | Sensorimotor / Visual | VisCent, VisPeri, SomMotA, SomMotB | 46 | Cortese: somatomotor hyperactivation in children with ADHD (26% of hyperactivated voxels); Feng: SM-CC-CB and VI-SM connectivity patterns differentiate biotypes |

All 180 ROIs are covered across the 4 experts — no ROIs are discarded.

#### 2-Expert Configuration (`adhd_2`)

| Expert | Circuit | Yeo17 Networks | ROIs |
|--------|---------|---------------|------|
| 0 | Internal / DMN + Limbic | DefaultA/B/C, TempPar, Limbic_TempPole, Limbic_OFC, SalVentAttnA/B | 84 |
| 1 | External / Control + Sensorimotor | ContA/B/C, DorsAttnA/B, VisCent, VisPeri, SomMotA/SomMotB | 96 |

This coarser split groups self-referential/affective networks (Internal) vs. task-positive/executive/sensorimotor networks (External).

All 180 ROIs are covered across the 2 experts — no ROIs are discarded.

### Model Architecture

```
Input: (B, T=363, C=180)

CIRCUIT-SPECIALIZED MoE:
  For each expert k (k = num_circuit_experts):
    ROI subset = circuit_rois[k]
    x_k = x[:, :, circuit_rois[k]]              # (B, T, n_k)
    h_k = Expert_k(x_k)                          # (B, H)
      Expert_k architecture:
        Linear(n_k, hidden_dim) + Dropout
        + Learnable positional embedding (T, hidden_dim)
        TransformerEncoder(d=hidden_dim, nhead=4, layers=2)
        mean temporal pooling → (B, hidden_dim)

  GATING:
    gate_input = temporal_mean(x)                 # (B, 180)
    gate_weights = GatingNetwork(gate_input)      # (B, num_experts)

  COMBINATION:
    h = sum(gate_weights[:, k] * h_k)             # (B, hidden_dim)
    + load_balancing_loss (Switch Transformer)

  CLASSIFIER:
    logits = Linear(hidden_dim, 1)                # (B,)
```

### Quantum Variant

Same architecture but each circuit expert is a `QuantumTSTransformer`:

```
Expert_k (quantum):
  Linear(n_k, n_rots) + Sigmoid * 2pi
  QSVT polynomial (degree=3) via sim14 circuit
  Measure PauliX/Y/Z on 8 qubits → 24 values
  Linear(24, hidden_dim) → (B, hidden_dim)
```

### Gating Network

```python
class CircuitGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, noise_std=0.1):
        self.fc1 = Linear(input_dim, 64)
        self.fc2 = Linear(64, num_experts)
        self.noise_std = noise_std

    def forward(self, x):
        x = ReLU(self.fc1(x))
        logits = self.fc2(x)
        if self.training:
            logits += randn_like(logits) * self.noise_std
        return softmax(logits, dim=-1)
```

### Parameter Counts (from smoke test)

| Config | Params | Details |
|--------|--------|---------|
| 4-expert Classical (H=64) | 516,485 | DMN:55, Exec:50, Sal:29, SM:46 ROIs |
| 2-expert Classical (H=64) | 269,827 | Internal:84, External:96 ROIs |
| SE Classical Baseline | 134,849 | Single expert, 180 ROIs |

## Implementation Plan

### Files Created

| File | Description | Status |
|------|-------------|--------|
| `models/yeo17_networks.py` | Yeo17 mapping + ADHD circuit definitions | Done |
| `models/CircuitMoE.py` | CircuitExpert, CircuitGatingNetwork, CircuitMoE | Done |
| `CircuitMoE_ABCD.py` | Training script | Done |
| `abcd_neuro_clustering.py` | Circuit-specific FC clustering (optional, not used in current experiments) | Done |
| `scripts/ADHD_CircuitMoE_Classical_3expert.sh` | 4-expert classical SLURM | Done |
| `scripts/ADHD_CircuitMoE_Classical_2expert.sh` | 2-expert classical SLURM | Done |
| `scripts/ADHD_CircuitMoE_Quantum_3expert.sh` | 4-expert quantum SLURM | Done |
| `scripts/ADHD_CircuitMoE_Quantum_2expert.sh` | 2-expert quantum SLURM | Done |

### Execution Order

1. ~~Add circuit definitions to `models/yeo17_networks.py`~~ Done
2. ~~Create model `models/CircuitMoE.py`~~ Done
3. ~~Create training script `CircuitMoE_ABCD.py`~~ Done
4. ~~Create SLURM scripts~~ Done
5. ~~Submit ADHD classical as canary~~ Done (jobs 49731003, 49731010)
6. **Evaluate** against baselines:
   - SE Classical: 0.6193 AUC (target to beat)
   - Cluster MoE Soft: 0.6001 AUC
   - NAME Classical: 0.5800 AUC

## Expected Outcomes

**Optimistic**: Circuit-specialized experts capture ADHD-relevant features better than all-channel experts. Test AUC > 0.65.

**Neutral**: Classification performance similar to SE baseline (~0.62). Would still be scientifically interesting — demonstrates circuit-specialized processing without performance loss despite each expert seeing fewer ROIs.

**Pessimistic**: ADHD classification from resting-state fMRI has a hard ceiling near 0.62 regardless of architecture. Even with circuit-informed design, performance doesn't improve. This matches the Koirala et al. argument that ADHD involves distributed, global dysfunction rather than circuit-specific signatures.

## Verification

1. **Circuit coverage check**: All 180 ROIs assigned to exactly one circuit (verified)
2. **Smoke test**: Both configs produce correct output shapes (verified)
3. **Parameter count**: 4-expert 517K, 2-expert 270K (verified)
4. **Compare against baselines** after jobs complete

## References

1. Nigg, J. T., Karalunas, S. L., Feczko, E., & Fair, D. A. (2020). Toward a Revised Nosology for Attention-Deficit/Hyperactivity Disorder Heterogeneity. *Biological Psychiatry: Cognitive Neuroscience and Neuroimaging*, 5(8), 726-737. https://doi.org/10.1016/j.bpsc.2020.02.005

2. Koirala, S., Grimsrud, G., Mooney, M. A., Larsen, B., Feczko, E., Elison, J. T., Nelson, S. M., Nigg, J. T., Tervo-Clemmens, B., & Fair, D. A. (2024). Neurobiology of attention-deficit hyperactivity disorder: historical challenges and emerging frontiers. *Nature Reviews Neuroscience*, 25(12), 759-775. https://doi.org/10.1038/s41583-024-00869-z

3. Feng, A., Zhi, D., Feng, Y., Jiang, R., Fu, Z., Xu, M., Zhao, M., Yu, S., Stevens, M., Sun, L., Calhoun, V., & Sui, J. (2024). Functional imaging derived ADHD biotypes based on deep clustering: a study on personalized medication therapy guidance. *EClinicalMedicine*, 77, 102876. https://doi.org/10.1016/j.eclinm.2024.102876

4. Pan, N., Long, Y., Qin, K., Pope, I. Z., Chen, Q., Zhu, Z., Cao, Y., Li, L., Singh, M. K., McNamara, R. K., DelBello, M. P., Chen, Y., Fornito, A., & Gong, Q. (2026). Mapping ADHD Heterogeneity and Biotypes by Topological Deviations in Morphometric Similarity Networks. *JAMA Psychiatry*. https://doi.org/10.1001/jamapsychiatry.2026.0001

5. Cortese, S., Kelly, C., Chabernaud, C., Proal, E., Di Martino, A., Milham, M. P., & Castellanos, F. X. (2012). Toward systems neuroscience of ADHD: a meta-analysis of 55 fMRI studies. *American Journal of Psychiatry*, 169(10), 1038-1055. https://doi.org/10.1176/appi.ajp.2012.11101521
