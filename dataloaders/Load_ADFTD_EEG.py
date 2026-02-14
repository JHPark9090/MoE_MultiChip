import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import mne
from sklearn.model_selection import train_test_split


# Default data root
ADFTD_DATA_ROOT = "/pscratch/sd/j/junghoon/ADFTD_data"

N_CHANNELS = 19
NATIVE_SFREQ = 500  # Hz
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

# Group labels: A = Alzheimer's, C = Control, F = Frontotemporal Dementia
GROUP_LABEL_MAP = {"A": 0, "C": 1, "F": 2}
GROUP_NAMES = ["AD", "CN", "FTD"]


def load_adftd_eeg(
    seed,
    device,
    batch_size,
    data_source="derivatives",
    sampling_freq=500,
    segment_length=2500,
    label_type="group",
    normalize=True,
    clip_std=10.0,
    num_workers=0,
    data_root=ADFTD_DATA_ROOT,
):
    """
    Loads and preprocesses the ADFTD EEG dataset for dementia classification.

    The ADFTD dataset contains resting-state eyes-closed EEG recordings from
    88 subjects: 36 Alzheimer's Disease (AD), 23 Frontotemporal Dementia (FTD),
    and 29 healthy controls (CN). Recordings use 19 channels (10-20 system)
    at 500 Hz with variable duration (5-21 minutes per subject).

    Args:
        seed (int): Random seed for reproducibility.
        device (torch.device): The device to move the tensors to.
        batch_size (int): Number of samples per batch.
        data_source (str): Which data version to load.
            - "derivatives": Preprocessed (artifact-removed via ASR + ICA).
            - "raw": Original unprocessed recordings.
        sampling_freq (int): Target sampling frequency. If different from
            native 500 Hz, data will be resampled. Default 500 (native).
        segment_length (int): Number of samples per segment. Each subject's
            recording is split into non-overlapping fixed-length segments.
            Default 2500 (5 seconds at 500 Hz). Set to None to use full
            recordings (padded to shortest).
        label_type (str): Label scheme.
            - "group": 3-class (0=AD, 1=CN, 2=FTD).
            - "ad_vs_cn": Binary (0=AD, 1=CN). FTD subjects excluded.
            - "dementia_vs_cn": Binary (0=dementia [AD+FTD], 1=CN).
        normalize (bool): If True, apply per-channel z-score normalization
            fitted on the training set. Default True.
        clip_std (float or None): If set, clip values to ±clip_std after
            normalization. Default 10.0. Set to None to disable.
        num_workers (int): Number of DataLoader workers. Default 0.
        data_root (str): Root directory of the ADFTD dataset.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
            - train_loader: DataLoader for the training set.
            - val_loader: DataLoader for the validation set.
            - test_loader: DataLoader for the test set.
            - input_dim: Shape of a single input sample (channels, time).
    """

    # --- Step 1: Load participant metadata ---
    participants = pd.read_csv(
        os.path.join(data_root, "participants.tsv"), sep="\t"
    )

    # Filter by label_type
    if label_type == "group":
        # 3-class: AD=0, CN=1, FTD=2
        label_map = GROUP_LABEL_MAP
    elif label_type == "ad_vs_cn":
        # Binary: AD=0, CN=1 — exclude FTD
        participants = participants[participants["Group"].isin(["A", "C"])].reset_index(drop=True)
        label_map = {"A": 0, "C": 1}
    elif label_type == "dementia_vs_cn":
        # Binary: dementia=0, CN=1
        label_map = {"A": 0, "F": 0, "C": 1}
    else:
        raise ValueError(f"Unknown label_type: {label_type!r}")

    subject_ids = participants["participant_id"].values
    labels = np.array([label_map[g] for g in participants["Group"].values], dtype=np.int64)
    n_classes = len(set(label_map.values()))

    print(f"[ADFTD] Subjects: {len(subject_ids)}, Classes: {n_classes}, Label type: {label_type}")
    for val, name in enumerate(sorted(set(label_map.values()))):
        count = (labels == val).sum()
        print(f"[ADFTD]   Class {val}: {count} subjects")

    # --- Step 2: Stratified subject-level split ---
    indices = np.arange(len(subject_ids))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=labels[temp_idx]
    )

    print(f"[ADFTD] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # --- Step 3: Load EEG data ---
    all_eeg = []
    for sid in subject_ids:
        eeg = _load_subject_eeg(data_root, sid, data_source, sampling_freq)
        all_eeg.append(eeg)  # (19, T_variable)

    # --- Step 4: Split into train/val/test, then segment ---
    def _split_and_segment(idxs):
        segments = []
        seg_labels = []
        for i in idxs:
            eeg = all_eeg[i]  # (C, T)
            label = labels[i]
            if segment_length is not None:
                n_segs = eeg.shape[1] // segment_length
                for s in range(n_segs):
                    start = s * segment_length
                    segments.append(eeg[:, start:start + segment_length])
                    seg_labels.append(label)
            else:
                segments.append(eeg)
                seg_labels.append(label)
        return segments, np.array(seg_labels, dtype=np.int64)

    train_segs, y_train = _split_and_segment(train_idx)
    val_segs, y_val = _split_and_segment(val_idx)
    test_segs, y_test = _split_and_segment(test_idx)

    # Stack into arrays (pad to uniform length if segment_length is None)
    if segment_length is None:
        min_len = min(
            min(s.shape[1] for s in train_segs),
            min(s.shape[1] for s in val_segs),
            min(s.shape[1] for s in test_segs),
        )
        train_segs = [s[:, :min_len] for s in train_segs]
        val_segs = [s[:, :min_len] for s in val_segs]
        test_segs = [s[:, :min_len] for s in test_segs]

    X_train = np.stack(train_segs, axis=0).astype(np.float32)  # (N, C, T)
    X_val = np.stack(val_segs, axis=0).astype(np.float32)
    X_test = np.stack(test_segs, axis=0).astype(np.float32)

    # --- Step 5: Per-channel z-score normalization (train-only) ---
    if normalize:
        train_mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        train_std = X_train.std(axis=(0, 2), keepdims=True)
        train_std = np.where(train_std < 1e-8, 1.0, train_std)

        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        if clip_std is not None:
            X_train = np.clip(X_train, -clip_std, clip_std)
            X_val = np.clip(X_val, -clip_std, clip_std)
            X_test = np.clip(X_test, -clip_std, clip_std)

        print(f"\n[ADFTD] Normalization: per-channel z-score (train mean/std)")
        print(f"[ADFTD] Post-norm train stats: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

    print(f"\n[ADFTD] Training set shape: {X_train.shape}, "
          f"labels: {np.bincount(y_train, minlength=n_classes)}")
    print(f"[ADFTD] Validation set shape: {X_val.shape}, "
          f"labels: {np.bincount(y_val, minlength=n_classes)}")
    print(f"[ADFTD] Test set shape: {X_test.shape}, "
          f"labels: {np.bincount(y_test, minlength=n_classes)}")

    # --- Step 6: Create DataLoaders ---
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.long).to(device),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).to(device),
        torch.tensor(y_val, dtype=torch.long).to(device),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_test, dtype=torch.long).to(device),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    input_dim = X_train.shape[1:]  # (channels, time)

    return train_loader, val_loader, test_loader, input_dim


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_subject_eeg(data_root, subject_id, data_source, target_sfreq):
    """Load a single subject's EEG recording.

    Args:
        data_root: Root directory of ADFTD dataset.
        subject_id: e.g. "sub-001"
        data_source: "derivatives" or "raw"
        target_sfreq: Target sampling frequency.

    Returns:
        eeg: np.ndarray of shape (19, n_samples) float64
    """
    if data_source == "derivatives":
        fpath = os.path.join(
            data_root, "derivatives", subject_id, "eeg",
            f"{subject_id}_task-eyesclosed_eeg.set"
        )
    elif data_source == "raw":
        fpath = os.path.join(
            data_root, subject_id, "eeg",
            f"{subject_id}_task-eyesclosed_eeg.set"
        )
    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")

    raw = mne.io.read_raw_eeglab(fpath, preload=True, verbose="WARNING")

    # Resample if needed
    if target_sfreq != raw.info["sfreq"]:
        raw.resample(target_sfreq, npad="auto")

    # Convert to volts → microvolts for numerical stability
    eeg = raw.get_data() * 1e6  # (19, n_samples), in µV

    return eeg
