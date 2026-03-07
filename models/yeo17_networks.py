"""
Static mapping from HCP-MMP1 180-ROI parcellation to Yeo 17 networks.

The ABCD HCP180 parcellation has 180 bilateral ROIs (average of left+right
hemisphere Glasser regions). Each ROI is assigned to one of 17 Yeo networks
via volumetric overlap between the Glasser atlas (HCPMMP1_for_ABCD.nii.gz)
and the Yeo 2011 17-network atlas (FreeSurfer MNI152 1mm), using bilateral
majority-vote (both LH label i+1 and RH label i+201 voxels combined).

Validation: The Yeo17 atlas was resampled to Glasser space (nearest-neighbor)
and for each Glasser region, the Yeo network with the most overlapping voxels
(across both hemispheres) was assigned. Full per-ROI report with confidence
scores saved in dataloaders/glasser_to_yeo17_mapping.json.

References:
    - Glasser et al. (2016) "A multi-modal parcellation of human cerebral cortex"
    - Yeo et al. (2011) "The organization of the human cerebral cortex"
"""

# Validated Yeo 17-network mapping for HCP-MMP1 180 bilateral ROIs (0-indexed).
# Generated via volumetric overlap (bilateral majority-vote).
# Each key is a Yeo 17 network name, values are lists of ROI indices.
YEO17_HCP180 = {
    "VisCent": [4, 5, 6, 15, 18, 19, 20, 21, 153, 155, 157, 162],
    "VisPeri": [0, 2, 3, 12, 120, 141, 152, 159],
    "SomMotA": [7, 8, 35, 39, 50, 51, 52, 53, 54],
    "SomMotB": [11, 23, 99, 100, 101, 102, 103, 106, 114, 123, 167, 172, 173, 174],
    "DorsAttnA": [1, 16, 17, 22, 45, 47, 48, 49, 137, 139, 140, 142, 145, 151, 156, 158],
    "DorsAttnB": [9, 38, 41, 44, 46, 55, 95, 115],
    "SalVentAttnA": [24, 36, 37, 40, 42, 43, 56, 57, 59, 98, 104, 105, 108, 112, 113, 146, 147, 166, 177],
    "SalVentAttnB": [58, 83, 85, 107, 110, 111, 168, 178],
    "Limbic_TempPole": [109, 117, 119, 121, 130, 133, 134, 135, 171],
    "Limbic_OFC": [87, 89, 90, 91, 92, 163, 164, 165],
    "ContA": [13, 14, 26, 28, 161],
    "ContB": [10, 77, 78, 79, 80, 81, 82, 94, 116, 136, 143, 144],
    "ContC": [62, 66, 72, 76, 84, 88, 96, 132, 148, 169, 170],
    "DefaultA": [27, 122, 124, 128, 138],
    "DefaultB": [30, 118, 125, 126, 154],
    "DefaultC": [29, 31, 32, 33, 34, 60, 61, 63, 64, 67, 71, 97, 149, 150, 160, 179],
    "TempPar": [25, 65, 68, 69, 70, 73, 74, 75, 86, 93, 127, 129, 131, 175, 176],
}


def get_network_indices():
    """Return ordered list of (network_name, roi_indices) tuples."""
    return list(YEO17_HCP180.items())


def get_network_sizes():
    """Return dict of network name -> number of ROIs."""
    return {name: len(indices) for name, indices in YEO17_HCP180.items()}


def get_fc_network_pairs():
    """Return list of (i, j) index pairs for network-level FC summarization.

    Produces 17 within-network pairs (i, i) and C(17,2)=136 between-network
    pairs (i, j) for i < j. Total: 153 pairs.

    Returns:
        list of (int, int): Network index pairs.
    """
    n_nets = len(YEO17_HCP180)
    pairs = []
    # Within-network (diagonal)
    for i in range(n_nets):
        pairs.append((i, i))
    # Between-network (upper triangle)
    for i in range(n_nets):
        for j in range(i + 1, n_nets):
            pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# ADHD Circuit Groupings (for neurobiology-based MoE)
# ---------------------------------------------------------------------------
# Based on: Nigg et al. (2020), Feng et al. (2024), Pan et al. (2026)

ADHD_CIRCUITS_3 = {
    "DMN": ["DefaultA", "DefaultB", "DefaultC", "TempPar"],
    "Executive": ["ContA", "ContB", "ContC", "DorsAttnA", "DorsAttnB"],
    "Salience": ["SalVentAttnA", "SalVentAttnB", "Limbic_TempPole", "Limbic_OFC"],
    "SensoriMotor": ["VisCent", "VisPeri", "SomMotA", "SomMotB"],
}

ADHD_CIRCUITS_2 = {
    "Internal": ["DefaultA", "DefaultB", "DefaultC", "TempPar",
                  "Limbic_TempPole", "Limbic_OFC", "SalVentAttnA", "SalVentAttnB"],
    "External": ["ContA", "ContB", "ContC", "DorsAttnA", "DorsAttnB",
                  "VisCent", "VisPeri", "SomMotA", "SomMotB"],
}


# ---------------------------------------------------------------------------
# Arbitrary (contiguous index) splits — baselines for ablation
# ---------------------------------------------------------------------------
# Same number of ROIs per expert as the neuroscience configs, but assigned
# by contiguous index order instead of by brain circuit.

def _compute_arbitrary_splits():
    """Compute arbitrary splits matching neuroscience config sizes."""
    # 4-expert: match DMN, Executive, Salience, SensoriMotor sizes
    sizes_4 = []
    for circuit_name, networks in ADHD_CIRCUITS_3.items():
        n = sum(len(YEO17_HCP180[net]) for net in networks)
        sizes_4.append((circuit_name, n))

    splits_4 = {}
    start = 0
    for i, (_, size) in enumerate(sizes_4):
        name = f"Block_{chr(65 + i)}"
        splits_4[name] = list(range(start, start + size))
        start += size
    # Assign any remaining ROIs to last block
    if start < 180:
        last_key = list(splits_4.keys())[-1]
        splits_4[last_key].extend(range(start, 180))

    # 2-expert: match Internal, External sizes
    sizes_2 = []
    for circuit_name, networks in ADHD_CIRCUITS_2.items():
        n = sum(len(YEO17_HCP180[net]) for net in networks)
        sizes_2.append((circuit_name, n))

    splits_2 = {}
    start = 0
    for i, (_, size) in enumerate(sizes_2):
        name = f"Block_{chr(65 + i)}"
        splits_2[name] = list(range(start, start + size))
        start += size
    if start < 180:
        last_key = list(splits_2.keys())[-1]
        splits_2[last_key].extend(range(start, 180))

    return splits_4, splits_2


ARBITRARY_SPLIT_4, ARBITRARY_SPLIT_2 = _compute_arbitrary_splits()


def get_circuit_config(name):
    """Return a circuit configuration by name.

    Args:
        name: "adhd_3" (4 experts: DMN, Executive, Salience, SensoriMotor)
              "adhd_2" (2 experts: Internal, External)
              "arbitrary_4" (4 experts: contiguous index blocks, same sizes as adhd_3)
              "arbitrary_2" (2 experts: contiguous index blocks, same sizes as adhd_2)

    Returns:
        dict: circuit_name -> list of Yeo17 network names OR list of ROI indices
    """
    configs = {
        "adhd_3": ADHD_CIRCUITS_3,
        "adhd_2": ADHD_CIRCUITS_2,
        "arbitrary_4": ARBITRARY_SPLIT_4,
        "arbitrary_2": ARBITRARY_SPLIT_2,
    }
    if name not in configs:
        raise ValueError(f"Unknown circuit config: {name!r}. Choose from {list(configs)}")
    return configs[name]


def get_circuit_roi_indices(circuit_config):
    """Convert circuit config to ROI index lists.

    Args:
        circuit_config: dict of circuit_name -> list of Yeo17 network names
                        OR dict of circuit_name -> list of ROI indices (arbitrary splits)

    Returns:
        list of (circuit_name, roi_indices) tuples, ordered consistently
    """
    result = []
    for circuit_name, values in circuit_config.items():
        if isinstance(values[0], str):
            # Neuroscience config: values are network names
            roi_indices = []
            for net_name in values:
                roi_indices.extend(YEO17_HCP180[net_name])
            roi_indices.sort()
        else:
            # Arbitrary config: values are already ROI indices
            roi_indices = sorted(values)
        result.append((circuit_name, roi_indices))
    return result


def get_circuit_fc_edge_indices(circuit_config, n_rois=180):
    """Get upper-triangle FC edge indices restricted to circuit-relevant ROIs.

    Returns only edges where BOTH ROIs belong to circuits in the config
    (within-circuit + between-circuit edges).

    Args:
        circuit_config: dict of circuit_name -> list of Yeo17 network names
        n_rois: total number of ROIs

    Returns:
        np.ndarray: 1D array of indices into the full upper-triangle vector
    """
    import numpy as np

    # Collect all circuit ROIs
    circuit_rois = set()
    for network_names in circuit_config.values():
        for net_name in network_names:
            circuit_rois.update(YEO17_HCP180[net_name])

    # Full upper triangle indices
    triu_r, triu_c = np.triu_indices(n_rois, k=1)

    # Mask: both ROIs must be in circuit ROIs
    mask = np.array([(r in circuit_rois and c in circuit_rois)
                     for r, c in zip(triu_r, triu_c)])
    return np.where(mask)[0]


def verify_coverage(n_rois=180):
    """Verify that all ROI indices 0..n_rois-1 are covered exactly once."""
    all_indices = []
    for indices in YEO17_HCP180.values():
        all_indices.extend(indices)
    all_indices_set = set(all_indices)
    expected = set(range(n_rois))
    missing = expected - all_indices_set
    duplicate = len(all_indices) - len(all_indices_set)
    extra = all_indices_set - expected
    assert not missing, f"Missing ROIs: {sorted(missing)}"
    assert duplicate == 0, f"Duplicate ROIs found ({duplicate} duplicates)"
    assert not extra, f"Extra ROIs outside range: {sorted(extra)}"
    return True


if __name__ == "__main__":
    verify_coverage()
    sizes = get_network_sizes()
    total = sum(sizes.values())
    print(f"Yeo 17-network mapping: {len(sizes)} networks, {total} ROIs")
    for name, size in sizes.items():
        print(f"  {name:20s}: {size:3d} ROIs")
    pairs = get_fc_network_pairs()
    print(f"\nFC network pairs: {len(pairs)} (17 within + 136 between)")

    # Circuit configs
    for cfg_name in ["adhd_3", "adhd_2"]:
        cfg = get_circuit_config(cfg_name)
        circuits = get_circuit_roi_indices(cfg)
        total_rois = sum(len(idx) for _, idx in circuits)
        print(f"\n{cfg_name}: {len(circuits)} circuits, {total_rois} ROIs")
        for cname, idx in circuits:
            print(f"  {cname:20s}: {len(idx):3d} ROIs")
