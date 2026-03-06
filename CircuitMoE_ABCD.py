#!/usr/bin/env python3
"""
Circuit-Specialized MoE for ABCD Classification/Regression.

Each expert processes a different brain circuit (DMN, Executive, Salience,
SensoriMotor) rather than all 180 ROIs. Gating uses the full input.

Usage:
    python CircuitMoE_ABCD.py \
        --model-type=classical --circuit-config=adhd_3 \
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


def load_balancing_loss(gate_weights: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch Transformer load-balancing auxiliary loss."""
    assignments = gate_weights.argmax(dim=-1)
    f = torch.zeros(num_experts, device=gate_weights.device)
    for i in range(num_experts):
        f[i] = (assignments == i).float().mean()
    P = gate_weights.mean(dim=0)
    return num_experts * (f * P).sum()


def transpose_fmri_loaders(train_loader, val_loader, test_loader,
                            input_dim, batch_size):
    """Transpose fMRI data from (N, T, R) to (N, R, T)."""
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
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, criterion, device,
                num_classes, balance_alpha, grad_clip, task_type="binary"):
    model.train()
    epoch_loss = 0.0
    epoch_bal_loss = 0.0
    all_preds = []
    all_labels = []
    num_experts = model.num_experts
    expert_counts = torch.zeros(num_experts)

    for batch in tqdm(train_loader, desc="Training", leave=False):
        data, target = batch[0].to(device), batch[1].to(device)
        data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        optimizer.zero_grad()
        logits, gate_weights = model(data)

        if task_type == "regression":
            target = target.float()
        elif num_classes > 2:
            target = target.long()
        else:
            target = target.float()
        task_loss = criterion(logits, target)

        bal_loss = load_balancing_loss(gate_weights, num_experts)
        loss = task_loss + balance_alpha * bal_loss

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        epoch_loss += task_loss.item()
        epoch_bal_loss += bal_loss.item()

        assignments = gate_weights.detach().argmax(dim=-1).cpu()
        for i in range(num_experts):
            expert_counts[i] += (assignments == i).sum().item()

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
    avg_bal_loss = epoch_bal_loss / len(train_loader)

    total_tokens = expert_counts.sum()
    utilization = (
        (expert_counts / total_tokens).tolist()
        if total_tokens > 0 else [0.0] * num_experts
    )

    if task_type == "regression":
        mse = float(np.mean((all_preds - all_labels) ** 2))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(all_labels, all_preds)) if len(all_labels) > 1 else 0.0
        return avg_loss, mse, rmse, r2, avg_bal_loss, utilization
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
        return avg_loss, accuracy, auc, avg_bal_loss, utilization


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

            logits, _ = model(data)

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

            logits, _ = model(data)

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
        description="Circuit-Specialized MoE for ABCD Classification/Regression",
    )

    # Task type
    parser.add_argument("--task-type", type=str, default="binary",
                        choices=["binary", "regression"])

    # Model / circuit
    parser.add_argument("--model-type", type=str, default="classical",
                        choices=["classical", "quantum"])
    parser.add_argument("--circuit-config", type=str, default="adhd_3",
                        choices=["adhd_3", "adhd_2"],
                        help="Circuit grouping: adhd_3 (4 experts) or adhd_2 (2 experts)")

    # MoE architecture
    parser.add_argument("--expert-hidden-dim", type=int, default=64)
    parser.add_argument("--expert-layers", type=int, default=2,
                        help="Transformer layers per expert (classical only)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Attention heads per expert (classical only)")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gating-noise-std", type=float, default=0.1)
    parser.add_argument("--balance-loss-alpha", type=float, default=0.1,
                        help="Weight for load-balancing loss")

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
    parser.add_argument("--job-id", type=str, default="CircuitMoE")
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

    # --- Task type ---
    task_type = args.task_type
    is_regression = (task_type == "regression")
    num_classes = 1 if is_regression else args.num_classes

    # --- Data Loading ---
    print("\n" + "=" * 80)
    print("Loading ABCD fMRI Dataset (Circuit MoE)...")
    print("=" * 80)

    sample_sz = args.sample_size if args.sample_size > 0 else None
    train_loader, val_loader, test_loader, input_dim = load_abcd_fmri(
        seed=args.seed, device=device, batch_size=args.batch_size,
        parcel_type=args.parcel_type,
        target_phenotype=args.target_phenotype,
        task_type=task_type,
        sample_size=sample_sz,
    )

    # Transpose (N, T, R) -> (N, R, T)
    train_loader, val_loader, test_loader, input_dim = \
        transpose_fmri_loaders(
            train_loader, val_loader, test_loader, input_dim,
            args.batch_size,
        )

    n_trials, n_channels, n_timesteps = input_dim
    print(f"\nInput: {n_trials} trials, {n_channels} channels, "
          f"{n_timesteps} timesteps")

    # --- Model ---
    print("\n" + "=" * 80)
    print(f"Initializing Circuit MoE ({args.model_type.capitalize()}, "
          f"{args.circuit_config})...")
    print("=" * 80)

    from models.CircuitMoE import CircuitMoE

    model = CircuitMoE(
        circuit_config_name=args.circuit_config,
        time_points=n_timesteps,
        expert_hidden_dim=args.expert_hidden_dim,
        model_type=args.model_type,
        expert_layers=args.expert_layers,
        nhead=args.nhead,
        n_qubits=args.n_qubits,
        n_ansatz_layers=args.n_ansatz_layers,
        degree=args.degree,
        gating_noise_std=args.gating_noise_std,
        total_channels=n_channels,
        num_classes=num_classes,
        dropout=args.dropout,
        device=device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Model: {args.model_type}, Circuit: {args.circuit_config}, "
          f"Experts: {model.num_experts}")
    if args.model_type == "classical":
        print(f"Classical: {args.expert_layers} layers, {args.nhead} heads, "
              f"hidden={args.expert_hidden_dim}")
    else:
        print(f"Quantum: {args.n_qubits}Q, {args.n_ansatz_layers}L, "
              f"D={args.degree}, hidden={args.expert_hidden_dim}")

    # --- Optimizer, Loss, Scheduler ---
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

    # --- Checkpoint & Logger ---
    os.makedirs(args.base_path, exist_ok=True)
    ckpt_name = (f"CircuitMoE_{args.model_type}_{args.circuit_config}_"
                 f"{args.job_id}.pt")
    checkpoint_path = os.path.join(args.base_path, ckpt_name)
    logger = TrainingLogger(save_dir=args.base_path, job_id=args.job_id)

    # --- Wandb ---
    if args.wandb:
        import wandb
        wandb.init(project="circuit-moe", config=vars(args))

    # --- Resume ---
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

    # --- Training Loop ---
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)

    for epoch in range(start_epoch, args.n_epochs):
        start_time = time.time()

        if is_regression:
            train_loss, train_mse, train_rmse, train_r2, bal_loss, utilization = \
                train_epoch(
                    model, train_loader, optimizer, criterion, device,
                    num_classes, args.balance_loss_alpha, args.grad_clip,
                    task_type=task_type,
                )
            val_loss, val_mse, val_rmse, val_r2 = validate_epoch(
                model, val_loader, criterion, device, num_classes,
                task_type=task_type,
            )
        else:
            train_loss, train_acc, train_auc, bal_loss, utilization = \
                train_epoch(
                    model, train_loader, optimizer, criterion, device,
                    num_classes, args.balance_loss_alpha, args.grad_clip,
                    task_type=task_type,
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
                  f"RMSE: {train_rmse:.4f} | R2: {train_r2:.4f} | Bal: {bal_loss:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | MSE: {val_mse:.4f} | "
                  f"RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")
        else:
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                  f"AUC: {train_auc:.4f} | Bal: {bal_loss:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
                  f"AUC: {val_auc:.4f}")

        util_str = " | ".join(
            f"{model.circuit_names[i]}:{u:.2f}"
            for i, u in enumerate(utilization)
        )
        print(f"  Expert util: [{util_str}]")

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
                    "train/balance_loss": bal_loss,
                    "val/loss": val_loss, "val/mse": val_mse,
                    "val/rmse": val_rmse, "val/r2": val_r2,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            else:
                log_dict = {
                    "train/loss": train_loss, "train/acc": train_acc,
                    "train/auc": train_auc, "train/balance_loss": bal_loss,
                    "val/loss": val_loss, "val/acc": val_acc, "val/auc": val_auc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            for i, u in enumerate(utilization):
                log_dict[f"expert_util/{model.circuit_names[i]}"] = u
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

    # --- Final Test ---
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
    print(f"Config: {args.model_type} / {args.circuit_config} / {task_type}")
    print(f"Experts: {model.num_experts} ({', '.join(model.circuit_names)})")
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
