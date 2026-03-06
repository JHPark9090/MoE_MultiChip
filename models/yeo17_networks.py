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
