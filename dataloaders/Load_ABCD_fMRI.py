import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split


# Default data root
ABCD_DATA_ROOT = "/pscratch/sd/j/junghoon/ABCD"

N_TIMEPOINTS = 363
PARCEL_DIMS = {"HCP": 360, "HCP180": 180, "Schaefer": 400}


def load_abcd_fmri(
    seed,
    device,
    batch_size,
    parcel_type="HCP180",
    phenotypes_to_include=None,
    target_phenotype="isControl",
    categorical_features=None,
    task_type="classification",
    sample_size=None,
    min_timepoints=300,
    sequence_length=None,
    normalize=True,
    clip_std=10.0,
    num_workers=0,
    data_root=ABCD_DATA_ROOT,
):
    """
    Loads and preprocesses the ABCD resting-state fMRI dataset.

    The ABCD Study dataset contains ~5,472 subjects with HCP-MMP1-180
    parcellated fMRI (363 timepoints x 180 ROIs) and 124 phenotype features
    including demographics, cognitive scores, and psychiatric diagnoses.

    Args:
        seed (int): Random seed for reproducibility.
        device (torch.device): The device to move the tensors to.
        batch_size (int): Number of samples per batch.
        parcel_type (str): Parcellation type. Options: "HCP180" (180 ROIs,
            available on disk), "HCP" (360 ROIs), "Schaefer" (400 ROIs).
        phenotypes_to_include (list of str or None): Phenotype columns to
            append as extra features alongside fMRI data. E.g.,
            ["age", "sex", "BMI"]. If None, fMRI only.
        target_phenotype (str): Phenotype column to use as the label.
            Classification targets: "isControl", "ADHD_label", "ASD_label", etc.
            Regression targets: "nihtbx_fluidcomp_uncorrected", etc.
        categorical_features (list of str or None): Which columns in
            phenotypes_to_include are categorical (will not be z-scored).
        task_type (str): Determines label dtype and stratification.
            - "classification": integer labels (torch.long), stratified split.
            - "binary": float labels (torch.float32) for BCEWithLogitsLoss.
            - "regression": float labels (torch.float32), no stratification.
        sample_size (int or None): Maximum number of subjects to load. If None,
            load all subjects with valid fMRI and phenotype data.
        min_timepoints (int): Minimum number of fMRI timepoints required.
            Subjects with fewer timepoints are excluded. Remaining subjects
            are zero-padded to N_TIMEPOINTS (363). Default 300.
        sequence_length (int or None): If set, split each subject's time
            series into non-overlapping windows of this length. Each window
            inherits the subject's label. If None, use full time series.
        normalize (bool): If True, apply z-score normalization fitted on
            the training set. Default True.
        clip_std (float or None): If set, clip fMRI values to ±clip_std
            after normalization. Default 10.0. Set to None to disable.
        num_workers (int): Number of DataLoader workers. Default 0.
        data_root (str): Root directory containing the ABCD data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
            - train_loader: DataLoader for the training set.
            - val_loader: DataLoader for the validation set.
            - test_loader: DataLoader for the test set.
            - input_dim: Shape of the input data (n_samples, time, n_features).
    """
    if phenotypes_to_include is None:
        phenotypes_to_include = []
    if categorical_features is None:
        categorical_features = []

    # --- Step 1: Load phenotype data and find valid subjects ---
    csv_path = os.path.join(data_root, "ABCD_phenotype_total.csv")
    phenotypes = pd.read_csv(csv_path)

    # Keep only rows with complete data for selected columns + target
    required_cols = ["subjectkey", target_phenotype] + phenotypes_to_include
    phenotypes = phenotypes[required_cols].dropna().reset_index(drop=True)

    # Limit to sample_size before expensive fMRI validation
    if sample_size is not None and len(phenotypes) > sample_size:
        phenotypes = phenotypes.iloc[:sample_size].reset_index(drop=True)

    subject_ids = phenotypes["subjectkey"].values

    # Filter to subjects that have fMRI files on disk with enough timepoints
    valid_mask = []
    for sid in subject_ids:
        fpath = _get_fmri_path(data_root, sid, parcel_type)
        if os.path.exists(fpath):
            fmri = np.load(fpath)
            valid_mask.append(fmri.shape[0] >= min_timepoints)
        else:
            valid_mask.append(False)
    valid_mask = np.array(valid_mask)

    phenotypes = phenotypes[valid_mask].reset_index(drop=True)
    subject_ids = phenotypes["subjectkey"].values
    target_labels = phenotypes[target_phenotype].values

    print(f"[ABCD] Valid subjects with fMRI ({parcel_type}, T>={min_timepoints}): {len(subject_ids)}")
    print(f"[ABCD] Target: {target_phenotype}")

    # --- Step 2: Subject-level train/val/test split ---
    indices = np.arange(len(subject_ids))

    # Stratify for classification tasks
    stratify = None
    if task_type in ("classification", "binary"):
        stratify = target_labels.astype(int)

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=seed, stratify=stratify
    )
    stratify_temp = stratify[temp_idx] if stratify is not None else None
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=stratify_temp
    )

    print(f"[ABCD] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # --- Step 3: Load fMRI data ---
    n_rois = None
    all_fmri = []
    for sid in subject_ids:
        fmri = np.load(_get_fmri_path(data_root, sid, parcel_type))  # (T, R)
        n_rois = fmri.shape[1]
        # Center-crop to N_TIMEPOINTS if longer
        if fmri.shape[0] > N_TIMEPOINTS:
            start = (fmri.shape[0] - N_TIMEPOINTS) // 2
            fmri = fmri[start:start + N_TIMEPOINTS]
        # Zero-pad to N_TIMEPOINTS if shorter
        elif fmri.shape[0] < N_TIMEPOINTS:
            pad_len = N_TIMEPOINTS - fmri.shape[0]
            fmri = np.pad(fmri, ((0, pad_len), (0, 0)), mode='constant')
        all_fmri.append(fmri)

    fmri_all = np.stack(all_fmri, axis=0).astype(np.float32)  # (N, T, R)

    # --- Step 4: Extract phenotype features (raw, unnormalized) ---
    pheno_all = None
    continuous_col_idx = []
    if phenotypes_to_include:
        pheno_all = phenotypes[phenotypes_to_include].values.astype(np.float32)
        continuous_col_idx = [
            i for i, c in enumerate(phenotypes_to_include)
            if c not in categorical_features
        ]

    # --- Step 5: Split ---
    fmri_train = fmri_all[train_idx]
    fmri_val = fmri_all[val_idx]
    fmri_test = fmri_all[test_idx]
    y_train = target_labels[train_idx]
    y_val = target_labels[val_idx]
    y_test = target_labels[test_idx]

    pheno_train = pheno_val = pheno_test = None
    if pheno_all is not None:
        pheno_train = pheno_all[train_idx].copy()
        pheno_val = pheno_all[val_idx].copy()
        pheno_test = pheno_all[test_idx].copy()

    # --- Step 6: Normalization (train-only stats) ---
    if normalize:
        # fMRI: per-ROI z-score across train subjects & time
        fmri_mean = fmri_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, R)
        fmri_std = fmri_train.std(axis=(0, 1), keepdims=True)
        fmri_std = np.where(fmri_std < 1e-8, 1.0, fmri_std)

        fmri_train = (fmri_train - fmri_mean) / fmri_std
        fmri_val = (fmri_val - fmri_mean) / fmri_std
        fmri_test = (fmri_test - fmri_mean) / fmri_std

        if clip_std is not None:
            fmri_train = np.clip(fmri_train, -clip_std, clip_std)
            fmri_val = np.clip(fmri_val, -clip_std, clip_std)
            fmri_test = np.clip(fmri_test, -clip_std, clip_std)

        # Phenotype: per-feature z-score for continuous columns only
        if pheno_train is not None and continuous_col_idx:
            ci = continuous_col_idx
            p_mean = pheno_train[:, ci].mean(axis=0, keepdims=True)
            p_std = pheno_train[:, ci].std(axis=0, keepdims=True)
            p_std = np.where(p_std < 1e-8, 1.0, p_std)

            pheno_train[:, ci] = (pheno_train[:, ci] - p_mean) / p_std
            pheno_val[:, ci] = (pheno_val[:, ci] - p_mean) / p_std
            pheno_test[:, ci] = (pheno_test[:, ci] - p_mean) / p_std

        print(f"\n[ABCD] Normalization: per-ROI z-score (fMRI), per-feature z-score (phenotypes)")
        print(f"[ABCD] Post-norm fMRI train stats: mean={fmri_train.mean():.4f}, std={fmri_train.std():.4f}")

    # --- Step 7: Combine fMRI + phenotypes ---
    X_train = _combine_fmri_pheno(fmri_train, pheno_train)
    X_val = _combine_fmri_pheno(fmri_val, pheno_val)
    X_test = _combine_fmri_pheno(fmri_test, pheno_test)

    # --- Step 8: Create sequences (sliding window) ---
    if sequence_length is not None:
        X_train, y_train = _create_sequences(X_train, y_train, sequence_length)
        X_val, y_val = _create_sequences(X_val, y_val, sequence_length)
        X_test, y_test = _create_sequences(X_test, y_test, sequence_length)

    # Determine label dtype
    if task_type == "classification":
        label_dtype = torch.long
        y_train = y_train.astype(np.int64)
        y_val = y_val.astype(np.int64)
        y_test = y_test.astype(np.int64)
    else:
        label_dtype = torch.float32
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)

    if task_type in ("classification", "binary"):
        n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
        print(f"\n[ABCD] Training set shape: {X_train.shape}, "
              f"labels: {np.bincount(y_train.astype(int), minlength=n_classes)}")
        print(f"[ABCD] Validation set shape: {X_val.shape}, "
              f"labels: {np.bincount(y_val.astype(int), minlength=n_classes)}")
        print(f"[ABCD] Test set shape: {X_test.shape}, "
              f"labels: {np.bincount(y_test.astype(int), minlength=n_classes)}")
    else:
        print(f"\n[ABCD] Training set shape: {X_train.shape}, "
              f"target mean={y_train.mean():.4f}")
        print(f"[ABCD] Validation set shape: {X_val.shape}")
        print(f"[ABCD] Test set shape: {X_test.shape}")

    # --- Step 9: Create DataLoaders ---
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=label_dtype).to(device),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).to(device),
        torch.tensor(y_val, dtype=label_dtype).to(device),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_test, dtype=label_dtype).to(device),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    input_dim = X_train.shape

    return train_loader, val_loader, test_loader, input_dim


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_fmri_path(data_root, subject_id, parcel_type):
    """Return the fMRI file path for a given subject and parcellation."""
    if parcel_type == "HCP":
        return os.path.join(
            data_root, f"sub-{subject_id}", f"hcp_mmp1_sub-{subject_id}.npy"
        )
    elif parcel_type == "HCP180":
        return os.path.join(
            data_root, f"sub-{subject_id}", f"hcp_mmp1_180_sub-{subject_id}.npy"
        )
    elif parcel_type == "Schaefer":
        return os.path.join(
            data_root, f"sub-{subject_id}", f"schaefer_sub-{subject_id}.npy"
        )
    else:
        raise ValueError(f"Unknown parcel_type: {parcel_type!r}")


def _combine_fmri_pheno(fmri, pheno):
    """Concatenate phenotype features with fMRI along the feature dimension.

    Args:
        fmri: (N, T, R) array
        pheno: (N, P) array or None

    Returns:
        X: (N, T, R+P) array if pheno provided, else (N, T, R)
    """
    if pheno is None:
        return fmri
    n, t, _ = fmri.shape
    # Tile phenotype across all timepoints: (N, P) -> (N, T, P)
    pheno_tiled = np.tile(pheno[:, np.newaxis, :], (1, t, 1))
    return np.concatenate([fmri, pheno_tiled], axis=2).astype(np.float32)


def _create_sequences(X, y, sequence_length):
    """Create non-overlapping fixed-length sequences from each subject.

    Each window inherits the subject-level label.

    Args:
        X: (N, T, F) array
        y: (N,) array
        sequence_length: int, window size

    Returns:
        X_seq: (N_windows, sequence_length, F) array
        y_seq: (N_windows,) array
    """
    sequences = []
    labels = []
    for i in range(X.shape[0]):
        t = X.shape[1]
        for start in range(0, t - sequence_length + 1, sequence_length):
            sequences.append(X[i, start:start + sequence_length])
            labels.append(y[i])
    return np.stack(sequences, axis=0), np.array(labels)
