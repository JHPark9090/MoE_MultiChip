import numpy as np
import os
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from sklearn.model_selection import train_test_split


# SEED dataset labels: 15 trials per session
# 1 = positive, 0 = neutral, -1 = negative
SEED_TRIAL_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])

# Remap to 0-indexed: {-1: 0, 0: 1, 1: 2}  (negative, neutral, positive)
SEED_LABEL_MAP = {-1: 0, 0: 1, 1: 2}

# Default data root
SEED_DATA_ROOT = "/pscratch/sd/j/junghoon/quantum_hydra_mamba/SEED/SEED_EEG"


def load_seed_eeg(
    seed,
    device,
    batch_size,
    sampling_freq=200,
    sample_size=15,
    data_source="preprocessed",
    feature_type="de_LDS",
    window_size=1,
    segment_length=None,
    normalize=True,
    clip_std=10.0,
    num_workers=0,
    data_root=SEED_DATA_ROOT,
):
    """
    Loads and preprocesses the SEED EEG emotion recognition dataset.

    The SEED dataset contains EEG recordings from 15 subjects, each with
    3 sessions of 15 trials. Labels are: positive (1), neutral (0), negative (-1).

    Args:
        seed (int): Random seed for reproducibility.
        device (torch.device): The device to move the tensors to.
        batch_size (int): Number of samples per batch.
        sampling_freq (int): Target sampling frequency for resampling preprocessed
            data. Only used when data_source="preprocessed". Default 200 (native).
        sample_size (int): Number of subjects to load (1 to 15).
        data_source (str): Which data variant to load.
            - "preprocessed": Raw preprocessed EEG time series (62 ch x T).
            - "features": Extracted DE/PSD features (62 ch x T x 5 bands).
        feature_type (str): Feature key prefix when data_source="features".
            Options: "de_LDS", "de_movingAve", "psd_LDS", "psd_movingAve",
            "dasm_LDS", "rasm_LDS", "asm_LDS", "dcau_LDS", etc.
        window_size (int): Feature extraction window in seconds (1 or 4).
            Only used when data_source="features". Default 1.
        segment_length (int or None): If set, split each trial into fixed-length
            segments of this many samples. Only for data_source="preprocessed".
            If None, pads/truncates all trials to the length of the shortest trial.
        normalize (bool): If True, apply per-channel z-score normalization
            fitted on the training set. Default True.
        clip_std (float or None): If set, clip values to ±clip_std after
            normalization. Default 10.0. Set to None to disable.
        num_workers (int): Number of DataLoader workers. Default 0.
        data_root (str): Root directory of the SEED_EEG dataset.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
            - train_loader: DataLoader for the training set.
            - val_loader: DataLoader for the validation set.
            - test_loader: DataLoader for the test set.
            - input_dim: Shape of a single input sample (channels, time[, bands]).
    """

    # --- Step 1: Split Subject IDs ---
    subject_ids = np.arange(1, min(sample_size, 15) + 1)

    train_subjects, temp_subjects = train_test_split(
        subject_ids, test_size=0.3, random_state=seed
    )
    val_subjects, test_subjects = train_test_split(
        temp_subjects, test_size=0.5, random_state=seed
    )

    print(f"[SEED] Subjects in Training Set: {sorted(train_subjects)}")
    print(f"[SEED] Subjects in Validation Set: {sorted(val_subjects)}")
    print(f"[SEED] Subjects in Test Set: {sorted(test_subjects)}")

    # --- Step 2: Load data ---
    if data_source == "preprocessed":
        loader_fn = lambda subjs: _load_preprocessed(
            subjs, sampling_freq, segment_length, data_root
        )
    elif data_source == "features":
        loader_fn = lambda subjs: _load_features(
            subjs, feature_type, window_size, data_root
        )
    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")

    X_train, y_train = loader_fn(train_subjects)
    X_val, y_val = loader_fn(val_subjects)
    X_test, y_test = loader_fn(test_subjects)

    # --- Step 3: Per-channel z-score normalization (fit on train only) ---
    if normalize:
        if data_source == "preprocessed":
            # X shape: (N, 62, T) — normalize per channel across samples & time
            train_mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
            train_std = X_train.std(axis=(0, 2), keepdims=True)    # (1, C, 1)
        else:
            # X shape: (N, 62, T, 5) — normalize per channel across samples,
            # time windows, and frequency bands
            train_mean = X_train.mean(axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
            train_std = X_train.std(axis=(0, 2, 3), keepdims=True)    # (1, C, 1, 1)

        train_std = np.where(train_std < 1e-8, 1.0, train_std)

        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        if clip_std is not None:
            X_train = np.clip(X_train, -clip_std, clip_std)
            X_val = np.clip(X_val, -clip_std, clip_std)
            X_test = np.clip(X_test, -clip_std, clip_std)

        print(f"\n[SEED] Normalization: per-channel z-score (train mean/std)")
        print(f"[SEED] Post-norm train stats: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

    print(f"\n[SEED] Training set shape: {X_train.shape}, labels: {np.bincount(y_train)}")
    print(f"[SEED] Validation set shape: {X_val.shape}, labels: {np.bincount(y_val)}")
    print(f"[SEED] Test set shape: {X_test.shape}, labels: {np.bincount(y_test)}")

    # --- Step 4: Create DataLoaders ---
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

    input_dim = X_train.shape[1:]  # (channels, time) or (channels, time, bands)

    return train_loader, val_loader, test_loader, input_dim


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_session_files(subject_id, data_dir):
    """Return sorted list of .mat files for a given subject ID."""
    pattern = re.compile(rf"^{subject_id}_\d{{8}}\.mat$")
    files = [f for f in os.listdir(data_dir) if pattern.match(f)]
    return sorted(files)


def _load_preprocessed(subject_list, target_sfreq, segment_length, data_root):
    """Load preprocessed EEG time-series from .mat files.

    Each .mat file contains 15 keys like '<prefix>_eeg1' ... '<prefix>_eeg15',
    each of shape (62, n_samples). The prefix varies per subject.

    Returns:
        X: np.ndarray of shape (n_samples, 62, T) float32
        y: np.ndarray of shape (n_samples,) int64
    """
    native_sfreq = 200
    data_dir = os.path.join(data_root, "Preprocessed_EEG")

    all_X = []
    all_y = []

    for subj_id in subject_list:
        session_files = _get_session_files(subj_id, data_dir)
        for fname in session_files:
            mat = sio.loadmat(os.path.join(data_dir, fname))
            # Find keys that match the pattern <prefix>_eeg<N>
            trial_keys = {}
            for k in mat.keys():
                m = re.match(r"(.+?)(\d+)$", k)
                if m and not k.startswith("__"):
                    trial_keys[int(m.group(2))] = k

            for trial_idx in range(1, 16):
                key = trial_keys[trial_idx]
                eeg = mat[key]  # (62, n_samples)

                # Resample if needed
                if target_sfreq != native_sfreq:
                    from scipy.signal import resample
                    n_out = int(eeg.shape[1] * target_sfreq / native_sfreq)
                    eeg = resample(eeg, n_out, axis=1)

                label = SEED_LABEL_MAP[SEED_TRIAL_LABELS[trial_idx - 1]]

                if segment_length is not None:
                    # Split trial into fixed-length segments
                    n_segments = eeg.shape[1] // segment_length
                    for seg_i in range(n_segments):
                        start = seg_i * segment_length
                        seg = eeg[:, start : start + segment_length]
                        all_X.append(seg)
                        all_y.append(label)
                else:
                    all_X.append(eeg)
                    all_y.append(label)

    if segment_length is None:
        # Pad/truncate to shortest trial length for uniform shape
        min_len = min(x.shape[1] for x in all_X)
        all_X = [x[:, :min_len] for x in all_X]

    X = np.stack(all_X, axis=0).astype(np.float32)  # (N, 62, T)
    y = np.array(all_y, dtype=np.int64)

    return X, y


def _load_features(subject_list, feature_type, window_size, data_root):
    """Load extracted features (DE, PSD, etc.) from .mat files.

    Each .mat key like 'de_LDS1' has shape (62, T_windows, 5_bands).

    Returns:
        X: np.ndarray of shape (n_samples, 62, T, 5) float32
        y: np.ndarray of shape (n_samples,) int64
    """
    if window_size == 1:
        data_dir = os.path.join(data_root, "ExtractedFeatures_1s")
    elif window_size == 4:
        data_dir = os.path.join(data_root, "ExtractedFeatures_4s")
    else:
        raise ValueError(f"window_size must be 1 or 4, got {window_size}")

    all_X = []
    all_y = []

    for subj_id in subject_list:
        session_files = _get_session_files(subj_id, data_dir)
        for fname in session_files:
            mat = sio.loadmat(os.path.join(data_dir, fname))
            for trial_idx in range(1, 16):
                key = f"{feature_type}{trial_idx}"
                feat = mat[key]  # (62, T_windows, 5)
                label = SEED_LABEL_MAP[SEED_TRIAL_LABELS[trial_idx - 1]]
                all_X.append(feat)
                all_y.append(label)

    # Pad/truncate to shortest time dimension
    min_t = min(x.shape[1] for x in all_X)
    all_X = [x[:, :min_t, :] for x in all_X]

    X = np.stack(all_X, axis=0).astype(np.float32)  # (N, 62, T, 5)
    y = np.array(all_y, dtype=np.int64)

    return X, y
