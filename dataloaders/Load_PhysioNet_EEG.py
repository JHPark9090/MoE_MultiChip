import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import mne
from sklearn.model_selection import train_test_split

print('MNE Version :', mne.__version__)

def load_eeg_ts_revised(seed, device, batch_size, sampling_freq, sample_size,
                        normalize=True, clip_std=10.0, num_workers=0):
    """
    Loads and preprocesses the PhysioNet EEG Motor Imagery dataset for a specified number of subjects.

    Args:
        seed (int): Random seed for reproducibility.
        device (torch.device): The device to move the tensors to.
        batch_size (int): Number of samples per batch.
        sampling_freq (int): The target sampling frequency to resample the data to.
        sample_size (int): The number of subjects to load data from (1 to 109).
        normalize (bool): If True, apply per-channel z-score normalization fitted
            on the training set. Default True.
        clip_std (float or None): If set, clip values to ±clip_std after
            normalization. Default 10.0. Set to None to disable.
        num_workers (int): Number of DataLoader workers. Default 0.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader, input_dim).
               - train_loader: DataLoader for the training set.
               - val_loader: DataLoader for the validation set.
               - test_loader: DataLoader for the test set.
               - input_dim: The shape of the input data (n_trials, n_channels, n_timesteps).
    """

    # --- Step 1: Split Subject IDs into Train, Validation, and Test Sets ---
    N_SUBJECT = sample_size
    subject_ids = np.arange(1, N_SUBJECT + 1)

    # Split subjects: ~70% train, ~15% validation, ~15% test
    train_subjects, temp_subjects = train_test_split(
        subject_ids,
        test_size=0.3,
        random_state=seed
    )

    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=0.5,
        random_state=seed
    )

    print(f"Subjects in Training Set: {len(train_subjects)}")
    print(f"Subjects in Validation Set: {len(val_subjects)}")
    print(f"Subjects in Test Set: {len(test_subjects)}")

    # --- Step 2: Define a Helper Function to Load Data for a Given List of Subjects ---
    def _load_and_process_subjects(subject_list, sfreq):
        IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]

        DATA_PATH = "/pscratch/sd/j/junghoon/PhysioNet_EEG"

        physionet_paths = [
            mne.datasets.eegbci.load_data(
                subjects=subj_id,
                runs=IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
                path=DATA_PATH,
                update_path=False,
            ) for subj_id in subject_list
        ]
        physionet_paths = np.concatenate(physionet_paths)

        parts = []
        for path in physionet_paths:
            raw = mne.io.read_raw_edf(
                path, preload=True, stim_channel='auto', verbose='WARNING'
            )
            raw.resample(sfreq, npad="auto")
            parts.append(raw)

        raw = mne.concatenate_raws(parts)

        events, _ = mne.events_from_annotations(raw)
        eeg_channel_inds = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )
        epoched = mne.Epochs(
            raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,
            proj=False, picks=eeg_channel_inds, baseline=None, preload=True
        )

        X = (epoched.get_data() * 1e3).astype(np.float32)
        y = (epoched.events[:, 2] - 2).astype(np.int64)

        return X, y

    # --- Step 3: Load Data Separately for Each Subject Group ---
    X_train, y_train = _load_and_process_subjects(train_subjects, sampling_freq)
    X_val, y_val = _load_and_process_subjects(val_subjects, sampling_freq)
    X_test, y_test = _load_and_process_subjects(test_subjects, sampling_freq)

    # --- Step 4: Per-channel z-score normalization (fit on train only) ---
    if normalize:
        # X shape: (N, n_channels, n_timesteps)
        # Compute mean/std per channel across all training samples and time
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

        print(f"\nNormalization applied: per-channel z-score (train mean/std)")
        print(f"Post-norm train stats: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

    # --- Step 5: Create PyTorch Datasets and DataLoaders ---
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                  torch.tensor(y_train, dtype=torch.long).to(device))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device),
                                torch.tensor(y_val, dtype=torch.long).to(device))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.long).to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    input_dim = X_train.shape

    return train_loader, val_loader, test_loader, input_dim
