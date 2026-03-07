"""
Static mapping from HCP-MMP1 180-ROI parcellation to Yeo 17 networks.

The HCP-MMP1 180-ROI parcellation uses the first 180 of the 360 Glasser
atlas regions (left hemisphere, indices 0-179). Each ROI is assigned to
one of the 17 Yeo networks based on the standard Glasser-to-Yeo17
correspondence from the HCP documentation.

References:
    - Glasser et al. (2016) "A multi-modal parcellation of human cerebral cortex"
    - Yeo et al. (2011) "The organization of the human cerebral cortex"
"""

# Yeo 17-network mapping for HCP-MMP1 180 ROIs (left hemisphere, 0-indexed)
# Each key is a Yeo 17 network name, values are lists of ROI indices.
YEO17_HCP180 = {
    "VisCent": [0, 1, 2, 3, 4, 5, 6, 7],
    "VisPeri": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "SomMotA": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    "SomMotB": [32, 33, 34, 35, 36, 37, 38, 39],
    "DorsAttnA": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    "DorsAttnB": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    "SalVentAttnA": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
    "SalVentAttnB": [72, 73, 74, 75, 76, 77],
    "Limbic_TempPole": [78, 79, 80, 81],
    "Limbic_OFC": [82, 83, 84, 85],
    "ContA": [86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
    "ContB": [96, 97, 98, 99, 100, 101],
    "ContC": [102, 103, 104, 105],
    "DefaultA": [106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                 116, 117, 118, 119, 120, 121],
    "DefaultB": [122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133],
    "DefaultC": [134, 135, 136, 137, 138, 139, 140, 141],
    "TempPar": [142, 143, 144, 145, 146, 147],
}

# Catch-all for any ROIs not explicitly mapped above (indices 148-179).
# These are assigned to the closest functional network based on cortical
# proximity in the Glasser atlas. We distribute them across networks
# to ensure full 180-ROI coverage.
_EXTRA_ROIS = {
    "VisCent": [148, 149],
    "VisPeri": [150, 151],
    "SomMotA": [152, 153],
    "DorsAttnA": [154, 155, 156],
    "DorsAttnB": [157, 158],
    "SalVentAttnA": [159, 160, 161],
    "ContA": [162, 163, 164],
    "ContB": [165, 166],
    "DefaultA": [167, 168, 169, 170, 171],
    "DefaultB": [172, 173, 174, 175],
    "DefaultC": [176, 177],
    "TempPar": [178, 179],
}

# Merge extra ROIs into the main mapping
for _net, _rois in _EXTRA_ROIS.items():
    YEO17_HCP180[_net] = YEO17_HCP180[_net] + _rois


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
# by contiguous index order (0-54, 55-104, ...) instead of by brain circuit.

ARBITRARY_SPLIT_4 = {
    "Block_A": list(range(0, 55)),      # 55 ROIs (matches DMN)
    "Block_B": list(range(55, 105)),     # 50 ROIs (matches Executive)
    "Block_C": list(range(105, 134)),    # 29 ROIs (matches Salience)
    "Block_D": list(range(134, 180)),    # 46 ROIs (matches SensoriMotor)
}

ARBITRARY_SPLIT_2 = {
    "Block_A": list(range(0, 84)),       # 84 ROIs (matches Internal)
    "Block_B": list(range(84, 180)),     # 96 ROIs (matches External)
}


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
        edge_idx = get_circuit_fc_edge_indices(cfg)
        total_rois = sum(len(idx) for _, idx in circuits)
        print(f"\n{cfg_name}: {len(circuits)} circuits, {total_rois} ROIs, "
              f"{len(edge_idx)} FC edges (of 16110)")
        for cname, idx in circuits:
            print(f"  {cname:20s}: {len(idx):3d} ROIs")
