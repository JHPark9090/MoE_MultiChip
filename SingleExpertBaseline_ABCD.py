#!/usr/bin/env python3
"""
Single-Expert Baseline for ABCD Classification/Regression.

Provides matched baselines for Cluster-Informed MoE comparison.
Uses the exact same expert architectures (classical AllChannelExpert or
QuantumTSTransformer) but with NO MoE routing, gating, or cluster labels.

Usage:
    python SingleExpertBaseline_ABCD.py \
        --model-type=classical --task-type=binary \
        --target-phenotype=ADHD_label
"""

import os
import sys
import time
import random
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

# scipy.constants must be imported BEFORE pennylane on this system
import scipy.constants  # noqa: F401

# Data loader
from dataloaders.Load_ABCD_fMRI import load_abcd_fmri

# Logger from parent directory
sys.path.insert(0, "/pscratch/sd/j/junghoon")
from logger import TrainingLogger


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    elapsed = end_time - start_time
    return int(elapsed / 60), int(elapsed % 60)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AllChannelExpert(nn.Module):
    """Transformer expert processing ALL channels (no spatial split).

    Copied verbatim from ClusterInformedMoE_ABCD.py for exact comparison.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, nhead,
                 time_points, dropout):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, time_points, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True,
            dropout=dropout, dim_feedforward=hidden_dim * 4,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C) — full channel input
        Returns:
            (B, H) — temporal mean pooling
        """
        x = self.dropout(self.input_projection(x))  # (B, T, H)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.transformer(x)  # (B, T, H)
        return x.mean(dim=1)  # (B, H)


class SingleExpertBaseline(nn.Module):
    """Single expert baseline — no MoE routing, gating, or cluster labels.

    Uses the same expert architecture as ClusterInformedMoE but with only
    one expert and a direct classifier on top.
    """

    def __init__(self, total_channels, time_points,
                 expert_hidden_dim, model_type,
                 # Classical args:
                 expert_layers=2, nhead=4,
                 # Quantum args:
                 n_qubits=8, n_ansatz_layers=2, degree=3,
                 # Common:
                 num_classes=2, dropout=0.2,
                 device="cpu"):
        super().__init__()

        self.total_channels = total_channels
        self.time_points = time_points
        self.num_classes = num_classes
        self.model_type = model_type
        self.expert_hidden_dim = expert_hidden_dim

        # --- Single Expert ---
        if model_type == "classical":
            self.expert = AllChannelExpert(
                input_dim=total_channels,
                hidden_dim=expert_hidden_dim,
                num_layers=expert_layers,
                nhead=nhead,
                time_points=time_points,
                dropout=dropout,
            )
        elif model_type == "quantum":
            from models.QTSTransformer_v2_5 import QuantumTSTransformer
            self.expert = QuantumTSTransformer(
                n_qubits=n_qubits,
                n_timesteps=time_points,
                degree=degree,
                n_ansatz_layers=n_ansatz_layers,
                feature_dim=total_channels,
                output_dim=expert_hidden_dim,
                dropout=dropout,
                device=device,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        # --- Classifier ---
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(expert_hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C) for classical, permuted to (B, C, T) for quantum
        Returns:
            logits: (B,) for binary or (B, num_classes) for multiclass
        """
        if self.model_type == "quantum":
            h = self.expert(x.permute(0, 2, 1))  # QTS expects (B, C, T)
        else:
            h = self.expert(x)  # Classical expects (B, T, C)

        logits = self.classifier(h)
        if self.num_classes <= 2:
            logits = logits.squeeze(-1)

        return logits


# ---------------------------------------------------------------------------
# Transpose helper (no cluster labels)
# ---------------------------------------------------------------------------

def transpose_fmri_loaders(train_loader, val_loader, test_loader,
                           input_dim, batch_size):
    """Transpose fMRI data from (N, T, R) to (N, R, T) for 2-element datasets."""
    new_loaders = []
    for loader in [train_loader, val_loader, test_loader]:
        all_x, all_y = [], []
        for batch in loader:
            all_x.append(batch[0])
            all_y.append(batch[1])

        X = torch.cat(all_x, dim=0).permute(0, 2, 1)  # (N, T, R) -> (N, R, T)
        Y = torch.cat(all_y, dim=0)

        ds = TensorDataset(X, Y)
        shuffle = (loader is train_loader)
        new_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle))

    n_samples, n_time, n_rois = input_dim
    transposed_dim = (n_samples, n_rois, n_time)
    return new_loaders[0], new_loaders[1], new_loaders[2], transposed_dim


# ---------------------------------------------------------------------------
# Train / Validate / Test (no cluster labels, no balance loss)
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, criterion, device,
                num_classes, grad_clip, task_type="binary"):
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        data, target = batch[0].to(device), batch[1].to(device)
        data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        optimizer.zero_grad()
        logits = model(data)

        if task_type == "regression":
            target = target.float()
        elif num_classes > 2:
            target = target.long()
        else:
            target = target.float()
        loss = criterion(logits, target)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        epoch_loss += loss.item()

        with torch.no_grad():
            if task_type == "regression":
                preds = logits.cpu().numpy()
            elif num_classes == 2:
                preds = torch.sigmoid(logits).cpu().numpy()
            else:
                preds = F.softmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = epoch_loss / len(train_loader)

    if task_type == "regression":
        mse = float(np.mean((all_preds - all_labels) ** 2))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(all_labels, all_preds)) if len(all_labels) > 1 else 0.0
        return avg_loss, mse, rmse, r2
    else:
        all_labels = all_labels.astype(int)
        if num_classes == 2:
            binary_preds = (all_preds > 0.5).astype(int)
        else:
            binary_preds = all_preds.argmax(axis=-1)
        accuracy = accuracy_score(all_labels, binary_preds)
        try:
            if num_classes == 2:
                auc = roc_auc_score(all_labels, all_preds)
            else:
                auc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
        except Exception:
            auc = 0.0
        return avg_loss, accuracy, auc


def validate_epoch(model, val_loader, criterion, device, num_classes,
                   task_type="binary"):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            data, target = batch[0].to(device), batch[1].to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

            logits = model(data)

            if task_type == "regression":
                target = target.float()
            elif num_classes > 2:
                target = target.long()
            else:
                target = target.float()
            loss = criterion(logits, target)
            epoch_loss += loss.item()

            if task_type == "regression":
                preds = logits.cpu().numpy()
            elif num_classes == 2:
                preds = torch.sigmoid(logits).cpu().numpy()
            else:
                preds = F.softmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = epoch_loss / len(val_loader)

    if task_type == "regression":
        mse = float(np.mean((all_preds - all_labels) ** 2))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(all_labels, all_preds)) if len(all_labels) > 1 else 0.0
        return avg_loss, mse, rmse, r2
    else:
        all_labels = all_labels.astype(int)
        if num_classes == 2:
            binary_preds = (all_preds > 0.5).astype(int)
        else:
            binary_preds = all_preds.argmax(axis=-1)
        accuracy = accuracy_score(all_labels, binary_preds)
        try:
            if num_classes == 2:
                auc = roc_auc_score(all_labels, all_preds)
            else:
                auc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
        except Exception:
            auc = 0.0
        return avg_loss, accuracy, auc


def test_epoch(model, test_loader, criterion, device, num_classes,
               task_type="binary"):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            data, target = batch[0].to(device), batch[1].to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

            logits = model(data)

            if task_type == "regression":
                target = target.float()
            elif num_classes > 2:
                target = target.long()
            else:
                target = target.float()
            loss = criterion(logits, target)
            epoch_loss += loss.item()

            if task_type == "regression":
                preds = logits.cpu().numpy()
            elif num_classes == 2:
                preds = torch.sigmoid(logits).cpu().numpy()
            else:
                preds = F.softmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = epoch_loss / len(test_loader)

    if task_type == "regression":
        mse = float(np.mean((all_preds - all_labels) ** 2))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(all_labels, all_preds)) if len(all_labels) > 1 else 0.0
        return avg_loss, mse, rmse, r2
    else:
        all_labels = all_labels.astype(int)
        if num_classes == 2:
            binary_preds = (all_preds > 0.5).astype(int)
        else:
            binary_preds = all_preds.argmax(axis=-1)
        accuracy = accuracy_score(all_labels, binary_preds)
        try:
            if num_classes == 2:
                auc = roc_auc_score(all_labels, all_preds)
            else:
                auc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
        except Exception:
            auc = 0.0
        return avg_loss, accuracy, auc


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Single-Expert Baseline for ABCD Classification/Regression",
    )

    # Task type
    parser.add_argument("--task-type", type=str, default="binary",
                        choices=["binary", "regression"],
                        help="Task type: binary classification or regression")

    # Model
    parser.add_argument("--model-type", type=str, default="classical",
                        choices=["classical", "quantum"],
                        help="Expert type")

    # Architecture (classical)
    parser.add_argument("--expert-hidden-dim", type=int, default=64)
    parser.add_argument("--expert-layers", type=int, default=2,
                        help="Transformer layers (classical only)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Attention heads (classical only)")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Quantum circuit
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-ansatz-layers", type=int, default=2)
    parser.add_argument("--degree", type=int, default=3,
                        help="QSVT polynomial degree")

    # Training
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "step"])

    # Data
    parser.add_argument("--parcel-type", type=str, default="HCP180")
    parser.add_argument("--target-phenotype", type=str, default="ADHD_label")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Number of subjects (0=all)")

    # Experiment
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--job-id", type=str, default="SingleExpert")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--base-path", type=str, default="./checkpoints")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Task type -----------------------------------------------------------
    task_type = args.task_type
    is_regression = (task_type == "regression")
    num_classes = 1 if is_regression else args.num_classes

    # --- Data Loading -------------------------------------------------------
    print("\n" + "=" * 80)
    print("Loading ABCD fMRI Dataset (Single Expert Baseline)...")
    print("=" * 80)

    sample_sz = args.sample_size if args.sample_size > 0 else None
    train_loader, val_loader, test_loader, input_dim = load_abcd_fmri(
        seed=args.seed, device=device, batch_size=args.batch_size,
        parcel_type=args.parcel_type,
        target_phenotype=args.target_phenotype,
        task_type=task_type,
        sample_size=sample_sz,
    )

    # Transpose (N, T, R) -> (N, R, T) to match (B, C, T) convention
    train_loader, val_loader, test_loader, input_dim = \
        transpose_fmri_loaders(
            train_loader, val_loader, test_loader, input_dim,
            args.batch_size,
        )

    n_trials, n_channels, n_timesteps = input_dim
    print(f"\nInput: {n_trials} trials, {n_channels} channels, "
          f"{n_timesteps} timesteps")

    # --- Model --------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"Initializing Single Expert Baseline ({args.model_type.capitalize()})...")
    print("=" * 80)

    model = SingleExpertBaseline(
        total_channels=n_channels,
        time_points=n_timesteps,
        expert_hidden_dim=args.expert_hidden_dim,
        model_type=args.model_type,
        expert_layers=args.expert_layers,
        nhead=args.nhead,
        n_qubits=args.n_qubits,
        n_ansatz_layers=args.n_ansatz_layers,
        degree=args.degree,
        num_classes=num_classes,
        dropout=args.dropout,
        device=device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Model: {args.model_type}, Task: {task_type}")
    if args.model_type == "classical":
        print(f"Classical: {args.expert_layers} layers, {args.nhead} heads, "
              f"hidden={args.expert_hidden_dim}")
    else:
        print(f"Quantum: {args.n_qubits}Q, {args.n_ansatz_layers}L, "
              f"D={args.degree}, hidden={args.expert_hidden_dim}")

    # --- Optimizer, Loss, Scheduler -----------------------------------------
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if is_regression:
        criterion = nn.MSELoss()
    elif num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_epochs,
        )
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1,
        )

    # --- Checkpoint & Logger ------------------------------------------------
    os.makedirs(args.base_path, exist_ok=True)
    ckpt_name = f"SingleExpert_{args.model_type}_{args.job_id}.pt"
    checkpoint_path = os.path.join(args.base_path, ckpt_name)
    logger = TrainingLogger(save_dir=args.base_path, job_id=args.job_id)

    # --- Wandb --------------------------------------------------------------
    if args.wandb:
        import wandb
        wandb.init(project="single-expert-baseline", config=vars(args))

    # --- Resume -------------------------------------------------------------
    start_epoch = 0
    best_val_metric = float("inf") if is_regression else 0.0
    patience_counter = 0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_metric = ckpt.get("best_val_metric", best_val_metric)
        patience_counter = ckpt.get("patience_counter", 0)
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        metric_name = "Val Loss" if is_regression else "Val AUC"
        print(f"Resuming from epoch {start_epoch}, "
              f"Best {metric_name}: {best_val_metric:.4f}")

    # --- Training Loop ------------------------------------------------------
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)

    for epoch in range(start_epoch, args.n_epochs):
        start_time = time.time()

        if is_regression:
            train_loss, train_mse, train_rmse, train_r2 = train_epoch(
                model, train_loader, optimizer, criterion, device,
                num_classes, args.grad_clip, task_type=task_type,
            )
            val_loss, val_mse, val_rmse, val_r2 = validate_epoch(
                model, val_loader, criterion, device, num_classes,
                task_type=task_type,
            )
        else:
            train_loss, train_acc, train_auc = train_epoch(
                model, train_loader, optimizer, criterion, device,
                num_classes, args.grad_clip, task_type=task_type,
            )
            val_loss, val_acc, val_auc = validate_epoch(
                model, val_loader, criterion, device, num_classes,
                task_type=task_type,
            )

        if scheduler:
            scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"\nEpoch: {epoch + 1:03d}/{args.n_epochs} | "
              f"Time: {epoch_mins}m {epoch_secs}s")
        if is_regression:
            print(f"  Train Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | "
                  f"RMSE: {train_rmse:.4f} | R2: {train_r2:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | MSE: {val_mse:.4f} | "
                  f"RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")
        else:
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                  f"AUC: {train_auc:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
                  f"AUC: {val_auc:.4f}")

        if is_regression:
            logger.logging_per_epochs(
                epoch=epoch, train_rmse=train_rmse, train_loss=train_loss,
                val_rmse=val_rmse, val_loss=val_loss,
            )
        else:
            logger.logging_per_epochs(
                epoch=epoch, train_rmse=train_acc, train_loss=train_loss,
                val_rmse=val_acc, val_loss=val_loss,
            )

        if args.wandb:
            if is_regression:
                log_dict = {
                    "train/loss": train_loss, "train/mse": train_mse,
                    "train/rmse": train_rmse, "train/r2": train_r2,
                    "val/loss": val_loss, "val/mse": val_mse,
                    "val/rmse": val_rmse, "val/r2": val_r2,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            else:
                log_dict = {
                    "train/loss": train_loss, "train/acc": train_acc,
                    "train/auc": train_auc,
                    "val/loss": val_loss, "val/acc": val_acc, "val/auc": val_auc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            wandb.log(log_dict, step=epoch)

        # Checkpoint best model
        if is_regression:
            improved = val_loss < best_val_metric
        else:
            improved = val_auc > best_val_metric

        if improved:
            best_val_metric = val_loss if is_regression else val_auc
            patience_counter = 0
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_metric": best_val_metric,
                "patience_counter": patience_counter,
                "args": vars(args),
            }
            if scheduler:
                save_dict["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(save_dict, checkpoint_path)
            if is_regression:
                print(f"  *** New best model saved! Val Loss: {best_val_metric:.4f} ***")
            else:
                print(f"  *** New best model saved! Val AUC: {best_val_metric:.4f} ***")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch + 1} epochs "
                  f"(patience: {args.patience})")
            break

    # --- Final Test ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set...")
    print("=" * 80)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    if is_regression:
        test_loss, test_mse, test_rmse, test_r2 = test_epoch(
            model, test_loader, criterion, device, num_classes,
            task_type=task_type,
        )
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test R2: {test_r2:.4f}")
        print(f"\nBest Validation Loss: {best_val_metric:.4f}")
        print(f"Final Test RMSE: {test_rmse:.4f} | R2: {test_r2:.4f}")
    else:
        test_loss, test_acc, test_auc = test_epoch(
            model, test_loader, criterion, device, num_classes,
            task_type=task_type,
        )
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"\nBest Validation AUC: {best_val_metric:.4f}")
        print(f"Final Test AUC: {test_auc:.4f}")

    print(f"Trainable parameters: {n_params:,}")
    print(f"Config: {args.model_type} / single expert / {task_type}")
    print("=" * 80)

    if args.wandb:
        if is_regression:
            wandb.log({
                "test/loss": test_loss, "test/mse": test_mse,
                "test/rmse": test_rmse, "test/r2": test_r2,
            })
        else:
            wandb.log({
                "test/loss": test_loss, "test/acc": test_acc, "test/auc": test_auc,
            })
        wandb.finish()


if __name__ == "__main__":
    main()
