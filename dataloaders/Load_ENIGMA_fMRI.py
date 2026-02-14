import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


# Default data root
ENIGMA_DATA_ROOT = "/pscratch/sd/j/junghoon/ENIGMA"

N_SUBJECTS = 1706
N_ROIS = 318
N_FC_FEATURES = 50403  # 318 * 317 / 2
N_TIMEPOINTS = 115  # resampled time series length


def load_enigma_fmri(
    seed,
    device,
    batch_size,
    sample_size=None,
    data_source="fc",
    site_aware_split=False,
    covariates=None,
    normalize=True,
    clip_std=10.0,
    num_workers=0,
    data_root=ENIGMA_DATA_ROOT,
):
    """
    Loads the ENIGMA-OCD resting-state fMRI dataset for OCD vs. healthy control
    classification.

    The dataset contains 1706 subjects (869 OCD, 837 control) from 23 sites,
    with 318-ROI Schaefer parcellation. Two data formats are available:
    pre-computed functional connectivity features, or raw time series.

    Args:
        seed (int): Random seed for reproducibility.
        device (torch.device): The device to move the tensors to.
        batch_size (int): Number of samples per batch.
        sample_size (int or None): Number of subjects to load. If None, load all
            1706 subjects. Subjects are selected in order from the dataset.
        data_source (str): Which data format to load.
            - "fc": Pre-computed Fisher z-transformed FC vector (50403-dim).
            - "timeseries": Per-subject fMRI time series (115 timepoints x 318 ROIs).
            - "fc_matrix": Full 318x318 FC matrix computed from time series.
        site_aware_split (bool): If True, use site (Sample) as a grouping variable
            so subjects from the same site tend to stay in the same split. This
            reduces site-related data leakage. Default False (random split).
        covariates (list of str or None): Additional columns from the metadata CSV
            to return as a second feature tensor. E.g., ["Age", "Sex", "Mean_FD"].
            Returned as a float tensor concatenated column-wise. Missing values
            are filled with training-set column means (no data leakage).
        normalize (bool): If True, apply z-score normalization fitted on the
            training set. Default True.
        clip_std (float or None): If set, clip values to ±clip_std after
            normalization. Default 10.0. Set to None to disable.
        num_workers (int): Number of DataLoader workers. Default 0.
        data_root (str): Root directory containing the ENIGMA data files.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
            - train_loader: DataLoader for the training set.
            - val_loader: DataLoader for the validation set.
            - test_loader: DataLoader for the test set.
            - input_dim: Shape of the input data (n_samples, ...).
              "fc" -> (N, 50403), "timeseries" -> (N, 115, 318), "fc_matrix" -> (N, 318, 318)
    """

    # --- Step 1: Load metadata ---
    csv_path = os.path.join(data_root, "ENIGMA_QC_final_subject_list_UQ_FILTERED.csv")
    ids_path = os.path.join(data_root, "subject_ids.txt")

    with open(ids_path) as f:
        subject_ids = [line.strip() for line in f if line.strip()]

    df = pd.read_csv(csv_path)
    # Reindex CSV to match subject_ids.txt ordering (= FC matrix row order)
    df = df.set_index("Unique_ID").loc[subject_ids].reset_index()

    n_total = len(subject_ids)
    if sample_size is not None:
        n_total = min(sample_size, n_total)

    indices = np.arange(n_total)
    labels = df["OCD"].values[:n_total].astype(np.int64)
    sites = df["Sample"].values[:n_total]

    # --- Step 2: Train/Val/Test split ---
    if site_aware_split:
        train_idx, val_idx, test_idx = _site_aware_split(
            indices, labels, sites, seed
        )
    else:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=seed, stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=seed,
            stratify=labels[temp_idx],
        )

    print(f"[ENIGMA] Subjects in Training Set: {len(train_idx)}")
    print(f"[ENIGMA] Subjects in Validation Set: {len(val_idx)}")
    print(f"[ENIGMA] Subjects in Test Set: {len(test_idx)}")

    # --- Step 3: Load data ---
    if data_source == "fc":
        X_all = _load_fc(data_root, n_total)
    elif data_source == "timeseries":
        X_all = _load_timeseries(data_root, subject_ids[:n_total])
    elif data_source == "fc_matrix":
        X_all = _load_fc_matrix(data_root, subject_ids[:n_total])
    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")

    y_all = labels

    # Split into train/val/test
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # Optional covariates (impute missing using train-only means)
    cov_train = cov_val = cov_test = None
    if covariates:
        cov_train, cov_val, cov_test = _load_covariates_split(
            df, covariates, train_idx, val_idx, test_idx
        )

    # --- Step 4: Normalization (fit on train only) ---
    if normalize:
        if data_source == "fc":
            # X shape: (N, 50403) — normalize per feature
            train_mean = X_train.mean(axis=0, keepdims=True)  # (1, F)
            train_std = X_train.std(axis=0, keepdims=True)    # (1, F)
        elif data_source == "timeseries":
            # X shape: (N, 115, 318) — normalize per ROI across samples & time
            train_mean = X_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, R)
            train_std = X_train.std(axis=(0, 1), keepdims=True)    # (1, 1, R)
        elif data_source == "fc_matrix":
            # X shape: (N, 318, 318) — normalize globally
            train_mean = X_train.mean(axis=0, keepdims=True)  # (1, 318, 318)
            train_std = X_train.std(axis=0, keepdims=True)    # (1, 318, 318)

        train_std = np.where(train_std < 1e-8, 1.0, train_std)

        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        if clip_std is not None:
            X_train = np.clip(X_train, -clip_std, clip_std)
            X_val = np.clip(X_val, -clip_std, clip_std)
            X_test = np.clip(X_test, -clip_std, clip_std)

        print(f"\n[ENIGMA] Normalization: z-score (train mean/std)")
        print(f"[ENIGMA] Post-norm train stats: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

    print(f"\n[ENIGMA] Training set shape: {X_train.shape}, "
          f"labels: {np.bincount(y_train, minlength=2)}")
    print(f"[ENIGMA] Validation set shape: {X_val.shape}, "
          f"labels: {np.bincount(y_val, minlength=2)}")
    print(f"[ENIGMA] Test set shape: {X_test.shape}, "
          f"labels: {np.bincount(y_test, minlength=2)}")

    # --- Step 5: Create DataLoaders ---
    g = torch.Generator()
    g.manual_seed(seed)

    def _make_loader(X, y, cov, shuffle):
        tensors = [
            torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.long).to(device),
        ]
        if cov is not None:
            tensors.append(torch.tensor(cov, dtype=torch.float32).to(device))
        dataset = TensorDataset(*tensors)
        gen = g if shuffle else None
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=gen)

    train_loader = _make_loader(X_train, y_train, cov_train, shuffle=True)
    val_loader = _make_loader(X_val, y_val, cov_val, shuffle=False)
    test_loader = _make_loader(X_test, y_test, cov_test, shuffle=False)

    input_dim = X_train.shape

    return train_loader, val_loader, test_loader, input_dim


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_fc(data_root, n_subjects):
    """Load pre-computed FC feature vectors (Fisher z-transformed).

    Returns:
        X: np.ndarray of shape (n_subjects, 50403) float32
    """
    fc_path = os.path.join(data_root, "fc_feature_matrix_z.npy")
    fc = np.load(fc_path).astype(np.float32)
    return fc[:n_subjects]


def _load_timeseries(data_root, subject_ids):
    """Load per-subject fMRI time series.

    Returns:
        X: np.ndarray of shape (n_subjects, 115, 318) float32
    """
    ts_dir = os.path.join(data_root, "enigma_quantum_data")
    all_ts = []
    for sid in subject_ids:
        ts = np.load(os.path.join(ts_dir, sid, f"{sid}.npy"))
        all_ts.append(ts)
    X = np.stack(all_ts, axis=0).astype(np.float32)
    return X


def _load_fc_matrix(data_root, subject_ids):
    """Load per-subject time series and compute full FC matrices.

    Returns:
        X: np.ndarray of shape (n_subjects, 318, 318) float32
    """
    ts_dir = os.path.join(data_root, "enigma_quantum_data")
    all_fc = []
    for sid in subject_ids:
        ts = np.load(os.path.join(ts_dir, sid, f"{sid}.npy"))  # (115, 318)
        fc = np.corrcoef(ts.T)  # (318, 318)
        np.nan_to_num(fc, copy=False)
        # Fisher z-transform
        fc = np.clip(fc, -0.9999, 0.9999)
        fc = np.arctanh(fc)
        np.fill_diagonal(fc, 0.0)
        all_fc.append(fc)
    X = np.stack(all_fc, axis=0).astype(np.float32)
    return X


def _load_covariates_split(df, covariate_cols, train_idx, val_idx, test_idx):
    """Extract covariate columns, impute missing with TRAIN-ONLY column means.

    This avoids data leakage by fitting imputation statistics only on the
    training set, then applying them to val/test.

    Returns:
        cov_train, cov_val, cov_test: np.ndarray of shape (n, n_covariates) float32
    """
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    max_idx = all_idx.max() + 1
    sub_df = df.iloc[:max_idx]

    raw_cols = []
    for col in covariate_cols:
        vals = pd.to_numeric(sub_df[col], errors="coerce").values.astype(np.float64)
        raw_cols.append(vals)
    raw = np.stack(raw_cols, axis=1)  # (max_idx, n_cov)

    # Compute train-only means for imputation
    train_raw = raw[train_idx]
    train_col_means = np.nanmean(train_raw, axis=0)  # (n_cov,)

    # Impute missing with train means
    for j in range(raw.shape[1]):
        mask = np.isnan(raw[:, j])
        raw[mask, j] = train_col_means[j]

    cov_train = raw[train_idx].astype(np.float32)
    cov_val = raw[val_idx].astype(np.float32)
    cov_test = raw[test_idx].astype(np.float32)

    return cov_train, cov_val, cov_test


def _site_aware_split(indices, labels, sites, seed):
    """Split indices into train/val/test keeping same-site subjects together.

    Uses StratifiedGroupKFold with 5 folds: 3 train, 1 val, 1 test.
    Stratification is by OCD label; grouping is by site.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    # First split: separate test fold
    folds = list(sgkf.split(indices, labels, groups=sites))
    train_val_idx, test_idx = folds[0]

    # Second split on train_val to get val fold
    tv_indices = indices[train_val_idx]
    tv_labels = labels[train_val_idx]
    tv_sites = sites[train_val_idx]

    sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    folds2 = list(sgkf2.split(tv_indices, tv_labels, groups=tv_sites))
    train_sub_idx, val_sub_idx = folds2[0]

    train_idx = tv_indices[train_sub_idx]
    val_idx = tv_indices[val_sub_idx]
    test_idx_final = indices[test_idx]

    return train_idx, val_idx, test_idx_final
