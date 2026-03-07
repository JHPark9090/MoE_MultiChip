"""
HCP-MMP1 atlas ROI label lookup for the 180-ROI (bilateral) parcellation.

The ABCD HCP180 parcellation has 180 ROIs, each representing the **bilateral
average** of the corresponding left- and right-hemisphere Glasser atlas regions.
ROI index i (0-179) = mean signal of HCP-MMP1 label (i+1) [left] and label
(i+201) [right].  This was verified empirically: the HCP180 .npy files match
(LH + RH) / 2 from the 360-ROI version with MSE ~42x lower than LH-only.

The NIfTI parcellation used for extraction is stored at:
  dataloaders/HCPMMP1_for_ABCD.nii.gz
  Labels 1-180 = left hemisphere, 201-380 = right hemisphere.

Usage:
    from dataloaders.hcp_mmp1_labels import get_roi_name, get_roi_info

    get_roi_name(171)       # -> 'TGv' (bilateral)
    get_roi_info(171)       # -> {'name': 'TGv', 'long_name': 'Area_TG_Ventral',
                            #     'lobe': 'Temp', 'cortex': 'Lateral_Temporal',
                            #     'bilateral': True}
"""

import os
import pandas as pd
from functools import lru_cache

_CSV_PATH = os.path.join(os.path.dirname(__file__), "HCP-MMP1_UniqueRegionList.csv")


@lru_cache(maxsize=1)
def _load_atlas():
    """Load and cache the first 180 rows (left hemisphere template) of HCP-MMP1 atlas."""
    df = pd.read_csv(_CSV_PATH)
    return df.iloc[:180].reset_index(drop=True)


def get_roi_name(roi_index: int, with_suffix: bool = False) -> str:
    """Return the region name for a 0-indexed ROI.

    Args:
        roi_index: 0-179
        with_suffix: If True, return 'TGv_L' (left-hemisphere label).
                     If False (default), return 'TGv' (bilateral name).
    """
    df = _load_atlas()
    if not 0 <= roi_index < 180:
        raise ValueError(f"ROI index must be 0-179, got {roi_index}")
    name = df.iloc[roi_index]["regionName"]
    if not with_suffix and name.endswith("_L"):
        name = name[:-2]
    return name


def get_roi_info(roi_index: int) -> dict:
    """Return a dict with name, long_name, lobe, and cortex for a 0-indexed ROI.

    Names are bilateral (no _L/_R suffix) since HCP180 averages both hemispheres.
    """
    df = _load_atlas()
    if not 0 <= roi_index < 180:
        raise ValueError(f"ROI index must be 0-179, got {roi_index}")
    row = df.iloc[roi_index]
    name = row["regionName"]
    long_name = row["regionLongName"]
    # Strip hemisphere suffix for bilateral representation
    if name.endswith("_L"):
        name = name[:-2]
    if long_name.endswith("_L"):
        long_name = long_name[:-2]
    return {
        "name": name,
        "long_name": long_name,
        "lobe": row["Lobe"],
        "cortex": row["cortex"],
        "bilateral": True,
    }


def get_all_roi_names(with_suffix: bool = False) -> list:
    """Return list of 180 region names in order.

    Args:
        with_suffix: If True, keep '_L' suffix. If False (default), strip it.
    """
    df = _load_atlas()
    names = df["regionName"].tolist()
    if not with_suffix:
        names = [n[:-2] if n.endswith("_L") else n for n in names]
    return names
