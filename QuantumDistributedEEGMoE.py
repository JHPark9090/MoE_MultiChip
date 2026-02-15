#!/usr/bin/env python3
"""
Quantum Distributed EEG Mixture of Experts — Classical Distributor + QTS Quantum Experts.

Architecture:
- Classical input summary (temporal mean → Linear) drives gating network
- Region-aware channel splitting with halo overlap
- Multiple QuantumTSTransformer (sim14 + QSVT/LCU) instances as quantum experts
- Classical aggregator: weighted combination → classifier

Reuses GatingNetwork, load_balancing_loss, train/validate/test, and utilities
from DistributedEEGMoE.py.
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# scipy.constants must be imported BEFORE pennylane on this system
# (scipy 1.10.1 lazy-loading + PennyLane 0.42.3 compatibility)
import scipy.constants  # noqa: F401

# Reuse components from the classical MoE
from DistributedEEGMoE import (
    GatingNetwork,
    load_balancing_loss,
    set_all_seeds,
    epoch_time,
    transpose_fmri_loaders,
    train,
    validate,
    test,
)

# Quantum expert
from models.QTSTransformer_v2_5 import QuantumTSTransformer

# Data loaders
from dataloaders.Load_PhysioNet_EEG import load_eeg_ts_revised
from dataloaders.Load_ABCD_fMRI import load_abcd_fmri

# Logger from parent directory
sys.path.insert(0, "/pscratch/sd/j/junghoon")
from logger import TrainingLogger


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QuantumDistributedEEGMoE(nn.Module):
    """
    Mixture of Experts with a classical distributor/gating module
    and QuantumTSTransformer experts for feature extraction.

    Forward signature matches DistributedEEGMoE: forward(x) → (logits, gate_weights)
    so existing train/validate/test loops work unchanged.
    """

    def __init__(self, total_channels, time_points, num_experts,
                 expert_hidden_dim, n_qubits, n_ansatz_layers, degree,
                 halo_size, num_classes, dropout, gating_noise_std, device):
        super().__init__()

        self.num_experts = num_experts
        self.halo_size = halo_size
        self.total_channels = total_channels
        self.num_classes = num_classes

        # Compute per-expert channel ranges (handles uneven splits)
        self.expert_ranges = self._compute_expert_ranges()

        # Uniform input width: largest base allocation + 2 * halo
        base = total_channels // num_experts
        remainder = total_channels % num_experts
        max_base = base + (1 if remainder > 0 else 0)
        self.expert_input_dim = max_base + 2 * halo_size

        # --- QTS Quantum Experts ---
        self.experts = nn.ModuleList([
            QuantumTSTransformer(
                n_qubits=n_qubits,
                n_timesteps=time_points,
                degree=degree,
                n_ansatz_layers=n_ansatz_layers,
                feature_dim=self.expert_input_dim,
                output_dim=expert_hidden_dim,
                dropout=dropout,
                device=device,
            )
            for _ in range(num_experts)
        ])

        # --- Classical Gating ---
        self.input_summary = nn.Sequential(
            nn.Linear(total_channels, expert_hidden_dim),
            nn.ReLU(),
        )
        self.gate = GatingNetwork(
            hidden_dim=expert_hidden_dim, num_experts=num_experts,
            noise_std=gating_noise_std,
        )

        # --- Classifier ---
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(expert_hidden_dim, output_dim)

    # ----- helpers ----------------------------------------------------------

    def _compute_expert_ranges(self):
        """Distribute channels to experts; first r experts get +1 channel."""
        base = self.total_channels // self.num_experts
        remainder = self.total_channels % self.num_experts
        ranges = []
        offset = 0
        for k in range(self.num_experts):
            width = base + (1 if k < remainder else 0)
            ranges.append((offset, offset + width))
            offset += width
        return ranges

    def _split_with_halo(self, x):
        """
        Split (B, T, C) into expert chunks with halo overlap,
        padded to uniform width ``self.expert_input_dim``.
        """
        expert_inputs = []
        for start, end in self.expert_ranges:
            halo_start = max(0, start - self.halo_size)
            halo_end = min(self.total_channels, end + self.halo_size)

            chunk = x[:, :, halo_start:halo_end]

            pad_left = max(0, self.halo_size - start)
            current_width = halo_end - halo_start
            pad_right = self.expert_input_dim - current_width - pad_left

            if pad_left > 0 or pad_right > 0:
                chunk = F.pad(chunk, (pad_left, pad_right), "constant", 0)

            expert_inputs.append(chunk)
        return expert_inputs

    # ----- forward ----------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: (B, T, total_channels)
        Returns:
            logits:       (B,) for binary  or  (B, num_classes) for multiclass
            gate_weights: (B, num_experts) — expert routing weights
        """
        # 1. Classical input summary → gating weights
        summary = self.input_summary(x.mean(dim=1))     # (B, H)
        gate_weights = self.gate(summary)                # (B, K)

        # 2. Split channels with halo
        expert_inputs = self._split_with_halo(x)         # list of (B, T, C_local)

        # 3. Each QTS processes its channel subset
        expert_features = []
        for k, qts in enumerate(self.experts):
            chunk = expert_inputs[k].permute(0, 2, 1)    # (B, C_local, T)
            h = qts(chunk)                                # (B, H)
            expert_features.append(h)

        expert_stack = torch.stack(expert_features, dim=1)  # (B, K, H)

        # 4. Weighted combination + classification
        weighted = (gate_weights.unsqueeze(-1) * expert_stack).sum(dim=1)  # (B, H)
        logits = self.classifier(weighted)
        if self.num_classes == 2:
            logits = logits.squeeze(-1)

        return logits, gate_weights


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Quantum Distributed EEG Mixture of Experts",
    )

    # MoE architecture
    parser.add_argument("--num-experts", type=int, default=4,
                        help="Number of QTS experts")
    parser.add_argument("--expert-hidden-dim", type=int, default=64,
                        help="Hidden / output dimension per expert")
    parser.add_argument("--halo-size", type=int, default=2,
                        help="Channel overlap on each side")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="2 = binary (BCEWithLogitsLoss), >2 = multiclass")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gating-noise-std", type=float, default=0.1,
                        help="Std of noise added to gate logits during training")
    parser.add_argument("--balance-loss-alpha", type=float, default=0.01,
                        help="Weight for load-balancing auxiliary loss")

    # Quantum circuit
    parser.add_argument("--n-qubits", type=int, default=8,
                        help="Qubits per expert circuit")
    parser.add_argument("--n-ansatz-layers", type=int, default=2,
                        help="VQC depth (sim14 layers) per expert")
    parser.add_argument("--degree", type=int, default=3,
                        help="QSVT polynomial degree")

    # Training
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience (epochs)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm (0 = disabled)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "step"])

    # Data
    parser.add_argument("--dataset", type=str, default="PhysioNet_EEG",
                        choices=["PhysioNet_EEG", "ABCD_fMRI"],
                        help="Dataset to use")
    parser.add_argument("--parcel-type", type=str, default="HCP180",
                        help="fMRI parcellation type (ABCD only)")
    parser.add_argument("--target-phenotype", type=str, default="ADHD_label",
                        help="Target phenotype column (ABCD only)")
    parser.add_argument("--sampling-freq", type=int, default=16,
                        help="EEG resampling frequency (Hz)")
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Number of subjects (PhysioNet: 1-109, ABCD: 0=all)")

    # Experiment
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--job-id", type=str, default="QMoE_EEG")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--base-path", type=str, default="./checkpoints",
                        help="Directory for checkpoints and logs")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading -------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"Loading {args.dataset} Dataset...")
    print("=" * 80)

    if args.dataset == "PhysioNet_EEG":
        train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
            seed=args.seed, device=device, batch_size=args.batch_size,
            sampling_freq=args.sampling_freq, sample_size=args.sample_size,
        )
    elif args.dataset == "ABCD_fMRI":
        sample_sz = args.sample_size if args.sample_size > 0 else None
        train_loader, val_loader, test_loader, input_dim = load_abcd_fmri(
            seed=args.seed, device=device, batch_size=args.batch_size,
            parcel_type=args.parcel_type,
            target_phenotype=args.target_phenotype,
            task_type="binary",
            sample_size=sample_sz,
        )
        train_loader, val_loader, test_loader, input_dim = transpose_fmri_loaders(
            train_loader, val_loader, test_loader, input_dim,
            args.batch_size, device,
        )

    n_trials, n_channels, n_timesteps = input_dim
    print(f"\nInput: {n_trials} trials, {n_channels} channels, "
          f"{n_timesteps} timesteps")

    # --- Model --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Initializing Quantum Distributed EEG MoE...")
    print("=" * 80)
    model = QuantumDistributedEEGMoE(
        total_channels=n_channels,
        time_points=n_timesteps,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        n_qubits=args.n_qubits,
        n_ansatz_layers=args.n_ansatz_layers,
        degree=args.degree,
        halo_size=args.halo_size,
        num_classes=args.num_classes,
        dropout=args.dropout,
        gating_noise_std=args.gating_noise_std,
        device=device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Experts: {args.num_experts}, Hidden: {args.expert_hidden_dim}")
    print(f"Quantum: {args.n_qubits}Q, {args.n_ansatz_layers}L, D={args.degree}")
    print(f"Expert channel ranges: {model.expert_ranges}")
    print(f"Expert input dim (with halo): {model.expert_input_dim}")

    # --- Optimizer, Loss, Scheduler -----------------------------------------
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.num_classes == 2:
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
    checkpoint_path = os.path.join(
        args.base_path, f"QMoE_EEG_{args.job_id}.pt",
    )
    logger = TrainingLogger(save_dir=args.base_path, job_id=args.job_id)

    # --- Wandb --------------------------------------------------------------
    if args.wandb:
        import wandb
        wandb.init(project="quantum-distributed-eeg-moe", config=vars(args))

    # --- Resume -------------------------------------------------------------
    start_epoch = 0
    best_val_auc = 0.0
    patience_counter = 0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt.get("best_val_auc", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"Resuming from epoch {start_epoch}, "
              f"Best Val AUC: {best_val_auc:.4f}")

    # --- Training Loop ------------------------------------------------------
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)

    for epoch in range(start_epoch, args.n_epochs):
        start_time = time.time()

        train_loss, train_acc, train_auc, bal_loss, utilization = train(
            model, train_loader, optimizer, criterion, device,
            args.num_classes, args.balance_loss_alpha, args.grad_clip,
        )

        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device, args.num_classes,
        )

        if scheduler:
            scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"\nEpoch: {epoch + 1:03d}/{args.n_epochs} | "
              f"Time: {epoch_mins}m {epoch_secs}s")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"AUC: {train_auc:.4f} | Bal: {bal_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
              f"AUC: {val_auc:.4f}")
        util_str = " | ".join(
            f"E{i}:{u:.2f}" for i, u in enumerate(utilization)
        )
        print(f"  Expert util: [{util_str}]")

        logger.logging_per_epochs(
            epoch=epoch, train_rmse=train_acc, train_loss=train_loss,
            val_rmse=val_acc, val_loss=val_loss,
        )

        if args.wandb:
            log_dict = {
                "train/loss": train_loss, "train/acc": train_acc,
                "train/auc": train_auc, "train/balance_loss": bal_loss,
                "val/loss": val_loss, "val/acc": val_acc, "val/auc": val_auc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            for i, u in enumerate(utilization):
                log_dict[f"expert_util/expert_{i}"] = u
            wandb.log(log_dict, step=epoch)

        # Checkpoint best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc": best_val_auc,
                "patience_counter": patience_counter,
                "args": vars(args),
            }
            if scheduler:
                save_dict["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(save_dict, checkpoint_path)
            print(f"  *** New best model saved! Val AUC: {best_val_auc:.4f} ***")
        else:
            patience_counter += 1

        # Early stopping
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

    test_loss, test_acc, test_auc = test(
        model, test_loader, criterion, device, args.num_classes,
    )

    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"\nBest Validation AUC: {best_val_auc:.4f}")
    print(f"Final Test AUC: {test_auc:.4f}")
    print(f"Trainable parameters: {n_params:,}")
    print("=" * 80)

    if args.wandb:
        wandb.log({
            "test/loss": test_loss, "test/acc": test_acc,
            "test/auc": test_auc,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
