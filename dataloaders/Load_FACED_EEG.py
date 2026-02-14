import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# FACED dataset: 9 emotions mapped to valence groups
# Emotion IDs: 0=anger, 1=disgust, 2=fear, 3=sadness, 4=neutral,
#              5=amusement, 6=inspiration, 7=joy, 8=tenderness
# Valence: -1 (negative: 0-3), 0 (neutral: 4), +1 (positive: 5-8)
FACED_EMOTION_NAMES = [
    "anger", "disgust", "fear", "sadness", "neutral",
    "amusement", "inspiration", "joy", "tenderness",
]

# Trial-to-emotion mapping: 28 trials, 3 per emotion (4 for neutral)
FACED_TRIAL_TO_EMOTION = np.array([
    0, 0, 0,   # anger
    1, 1, 1,   # disgust
    2, 2, 2,   # fear
    3, 3, 3,   # sadness
    4, 4, 4, 4, # neutral (4 trials)
    5, 5, 5,   # amusement
    6, 6, 6,   # inspiration
    7, 7, 7,   # joy
    8, 8, 8,   # tenderness
])

# Trial-to-valence mapping: 0=negative, 1=neutral, 2=positive
FACED_TRIAL_TO_VALENCE = np.array([
    0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,   # negative (12 trials)
    1, 1, 1, 1,                                 # neutral  (4 trials)
    2, 2, 2,  2, 2, 2,  2, 2, 2,  2, 2, 2,     # positive (12 trials)
])

# Default data root
FACED_DATA_ROOT = "/pscratch/sd/j/junghoon/quantum_hydra_mamba/Processed_data"

N_SUBJECTS = 123
N_CHANNELS = 32
N_TIMEPOINTS = 7500
SAMPLING_FREQ = 250  # Hz
TRIAL_DURATION = 30  # seconds


def load_faced_eeg(
    seed,
    device,
    batch_size,
    sampling_freq=250,
    sample_size=123,
    label_type="emotion",
    segment_length=None,
    normalize=True,
    clip_std=10.0,
    num_workers=0,
    data_root=FACED_DATA_ROOT,
):
    """
    Loads and preprocesses the FACED EEG emotion recognition dataset.

    The FACED dataset contains EEG recordings from 123 subjects watching
    emotional film clips. Each subject has 28 trials (32 channels, 250 Hz,
    30s each) labeled with 9 emotions or 3 valence classes.

    Args:
        seed (int): Random seed for reproducibility.
        device (torch.device): The device to move the tensors to.
        batch_size (int): Number of samples per batch.
        sampling_freq (int): Target sampling frequency. If different from native
            250 Hz, data will be resampled. Default 250 (native).
        sample_size (int): Number of subjects to load (1 to 123).
        label_type (str): Label granularity.
            - "emotion": 9-class (anger, disgust, fear, sadness, neutral,
              amusement, inspiration, joy, tenderness).
            - "valence": 3-class (negative, neutral, positive).
        segment_length (int or None): If set, split each 30s trial into
            fixed-length segments of this many samples. E.g., 250 for 1s
            segments, 1250 for 5s segments. If None, use full trials.
        normalize (bool): If True, apply per-channel z-score normalization
            fitted on the training set. Default True.
        clip_std (float or None): If set, clip values to ±clip_std after
            normalization. Default 10.0. Set to None to disable.
        num_workers (int): Number of DataLoader workers. Default 0.
        data_root (str): Directory containing sub000.pkl through sub122.pkl.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
            - train_loader: DataLoader for the training set.
            - val_loader: DataLoader for the validation set.
            - test_loader: DataLoader for the test set.
            - input_dim: Shape of a single input sample (channels, time).
    """

    n_subjects = min(sample_size, N_SUBJECTS)
    subject_ids = np.arange(n_subjects)

    # Build per-subject label for stratification
    # Use the dominant label type to stratify (first trial's label per subject)
    if label_type == "emotion":
        trial_labels = FACED_TRIAL_TO_EMOTION
    elif label_type == "valence":
        trial_labels = FACED_TRIAL_TO_VALENCE
    else:
        raise ValueError(f"Unknown label_type: {label_type!r}")

    # --- Step 1: Stratified subject-level split ---
    # For stratification, use the first trial's label per subject (all subjects
    # have identical trial-label structure, so this is constant — use valence
    # balance as the stratification target for valence mode)
    # Since every subject has the same label distribution, stratification is
    # equivalent to random split. We still pass stratify for correctness.
    # For emotion mode (9 classes, all balanced per subject), stratify is
    # not meaningful so we skip it.
    if label_type == "valence":
        # Can't stratify on per-subject label since all subjects have same
        # trial structure. Just use random split with seed.
        pass

    train_subjects, temp_subjects = train_test_split(
        subject_ids, test_size=0.3, random_state=seed
    )
    val_subjects, test_subjects = train_test_split(
        temp_subjects, test_size=0.5, random_state=seed
    )

    print(f"[FACED] Subjects in Training Set: {len(train_subjects)}")
    print(f"[FACED] Subjects in Validation Set: {len(val_subjects)}")
    print(f"[FACED] Subjects in Test Set: {len(test_subjects)}")

    # --- Step 2: Load data ---
    loader_fn = lambda subjs: _load_subjects(
        subjs, sampling_freq, label_type, segment_length, data_root
    )

    X_train, y_train = loader_fn(train_subjects)
    X_val, y_val = loader_fn(val_subjects)
    X_test, y_test = loader_fn(test_subjects)

    # --- Step 3: Per-channel z-score normalization (fit on train only) ---
    if normalize:
        # X shape: (N, C, T)
        train_mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        train_std = X_train.std(axis=(0, 2), keepdims=True)    # (1, C, 1)
        train_std = np.where(train_std < 1e-8, 1.0, train_std)

        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        if clip_std is not None:
            X_train = np.clip(X_train, -clip_std, clip_std)
            X_val = np.clip(X_val, -clip_std, clip_std)
            X_test = np.clip(X_test, -clip_std, clip_std)

        print(f"\n[FACED] Normalization: per-channel z-score (train mean/std)")
        print(f"[FACED] Post-norm train stats: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

    n_classes = 9 if label_type == "emotion" else 3
    print(f"\n[FACED] Training set shape: {X_train.shape}, labels: {np.bincount(y_train, minlength=n_classes)}")
    print(f"[FACED] Validation set shape: {X_val.shape}, labels: {np.bincount(y_val, minlength=n_classes)}")
    print(f"[FACED] Test set shape: {X_test.shape}, labels: {np.bincount(y_test, minlength=n_classes)}")

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

    input_dim = X_train.shape[1:]  # (channels, time)

    return train_loader, val_loader, test_loader, input_dim


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_subjects(subject_list, target_sfreq, label_type, segment_length, data_root):
    """Load pickle files for the given subject IDs.

    Each pickle is a numpy array of shape (28, 32, 7500) = (trials, channels, time).

    Returns:
        X: np.ndarray of shape (n_samples, 32, T) float32
        y: np.ndarray of shape (n_samples,) int64
    """
    native_sfreq = SAMPLING_FREQ

    if label_type == "emotion":
        trial_labels = FACED_TRIAL_TO_EMOTION
    elif label_type == "valence":
        trial_labels = FACED_TRIAL_TO_VALENCE
    else:
        raise ValueError(f"Unknown label_type: {label_type!r}")

    all_X = []
    all_y = []

    for subj_id in subject_list:
        fpath = os.path.join(data_root, f"sub{subj_id:03d}.pkl")
        data = np.load(fpath, allow_pickle=True)  # (28, 32, 7500)

        for trial_idx in range(data.shape[0]):
            eeg = data[trial_idx]  # (32, 7500)

            # Resample if needed
            if target_sfreq != native_sfreq:
                from scipy.signal import resample
                n_out = int(eeg.shape[1] * target_sfreq / native_sfreq)
                eeg = resample(eeg, n_out, axis=1)

            label = trial_labels[trial_idx]

            if segment_length is not None:
                n_segments = eeg.shape[1] // segment_length
                for seg_i in range(n_segments):
                    start = seg_i * segment_length
                    seg = eeg[:, start : start + segment_length]
                    all_X.append(seg)
                    all_y.append(label)
            else:
                all_X.append(eeg)
                all_y.append(label)

    X = np.stack(all_X, axis=0).astype(np.float32)  # (N, 32, T)
    y = np.array(all_y, dtype=np.int64)

    return X, y
