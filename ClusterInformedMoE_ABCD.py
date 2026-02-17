#!/usr/bin/env python3
"""
Cluster-Informed Mixture of Experts for ABCD ADHD Classification.

Key innovations over previous MoE approaches:
1. All experts process ALL 180 ROIs (no spatial decomposition)
2. Expert routing is informed by cluster labels from site-regressed
   coherence clustering (k=2 genuine neural subtypes)
3. Supports {Classical, Quantum} x {Soft gating, Hard routing}

Usage:
    python ClusterInformedMoE_ABCD.py \
        --model-type=classical --routing=soft \
        --cluster-file=results/coherence_clustering_site_regressed/cluster_assignments.csv
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
from sklearn.metrics import accuracy_score, roc_auc_score

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


def transpose_fmri_loaders_with_clusters(train_loader, val_loader, test_loader,
                                          input_dim, batch_size, device):
    """Transpose fMRI data from (N, T, R) to (N, R, T) for 3-element datasets."""
    new_loaders = []
    for loader in [train_loader, val_loader, test_loader]:
        all_x, all_y, all_c = [], [], []
        has_clusters = False
        for batch in loader:
            all_x.append(batch[0])
            all_y.append(batch[1])
            if len(batch) == 3:
                all_c.append(batch[2])
                has_clusters = True

        X = torch.cat(all_x, dim=0).permute(0, 2, 1)  # (N, T, R) -> (N, R, T)
        Y = torch.cat(all_y, dim=0)

        if has_clusters:
            C = torch.cat(all_c, dim=0)
            ds = TensorDataset(X, Y, C)
        else:
            ds = TensorDataset(X, Y)

        shuffle = (loader is train_loader)
        new_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle))

    n_samples, n_time, n_rois = input_dim
    transposed_dim = (n_samples, n_rois, n_time)
    return new_loaders[0], new_loaders[1], new_loaders[2], transposed_dim


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------

class AllChannelExpert(nn.Module):
    """Transformer expert processing ALL channels (no spatial split)."""

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


class GatingNetwork(nn.Module):
    """Gating network with optional exploration noise during training."""

    def __init__(self, hidden_dim, num_experts, noise_std=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_experts)
        self.noise_std = noise_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        return F.softmax(logits, dim=-1)


class ClusterInformedMoE(nn.Module):
    """
    Cluster-Informed Mixture of Experts.

    All experts process the full 180-ROI input. Routing is informed by
    cluster labels from site-regressed coherence clustering.

    Supports:
    - model_type: "classical" (TransformerEncoder) or "quantum" (QuantumTSTransformer)
    - routing: "soft" (gated, cluster-biased) or "hard" (deterministic by cluster)
    """

    def __init__(self, total_channels, time_points, num_experts,
                 expert_hidden_dim, model_type, routing,
                 # Classical args:
                 expert_layers=2, nhead=4,
                 # Quantum args:
                 n_qubits=8, n_ansatz_layers=2, degree=3,
                 # Common:
                 num_classes=2, dropout=0.2, gating_noise_std=0.1,
                 device="cpu"):
        super().__init__()

        self.num_experts = num_experts
        self.total_channels = total_channels
        self.time_points = time_points
        self.num_classes = num_classes
        self.model_type = model_type
        self.routing = routing
        self.expert_hidden_dim = expert_hidden_dim

        # --- Experts (all process full channels) ---
        if model_type == "classical":
            self.experts = nn.ModuleList([
                AllChannelExpert(
                    input_dim=total_channels,
                    hidden_dim=expert_hidden_dim,
                    num_layers=expert_layers,
                    nhead=nhead,
                    time_points=time_points,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ])
        elif model_type == "quantum":
            from models.QTSTransformer_v2_5 import QuantumTSTransformer
            self.experts = nn.ModuleList([
                QuantumTSTransformer(
                    n_qubits=n_qubits,
                    n_timesteps=time_points,
                    degree=degree,
                    n_ansatz_layers=n_ansatz_layers,
                    feature_dim=total_channels,
                    output_dim=expert_hidden_dim,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(num_experts)
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        # --- Routing ---
        if routing == "soft":
            # Gating input: temporal mean (C) + cluster one-hot (K)
            gate_input_dim = total_channels + num_experts
            self.input_summary = nn.Sequential(
                nn.Linear(gate_input_dim, expert_hidden_dim),
                nn.ReLU(),
            )
            self.gate = GatingNetwork(
                expert_hidden_dim, num_experts, gating_noise_std,
            )

        # --- Classifier ---
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(expert_hidden_dim, output_dim)

    def forward(self, x, cluster_labels):
        """
        Args:
            x: (B, T, C) — full fMRI input
            cluster_labels: (B,) — integer cluster assignments
        Returns:
            logits: (B,) for binary or (B, num_classes) for multiclass
            gate_weights: (B, K) — expert routing weights
        """
        B = x.size(0)

        if self.routing == "soft":
            # Cluster-informed soft gating
            cluster_onehot = F.one_hot(
                cluster_labels, self.num_experts
            ).float()  # (B, K)
            summary_input = torch.cat(
                [x.mean(dim=1), cluster_onehot], dim=-1
            )  # (B, C+K)
            summary = self.input_summary(summary_input)  # (B, H)
            gate_weights = self.gate(summary)  # (B, K)

            # Run all experts
            expert_outputs = []
            for expert in self.experts:
                if self.model_type == "quantum":
                    h = expert(x.permute(0, 2, 1))  # QTS expects (B, C, T)
                else:
                    h = expert(x)  # Classical expects (B, T, C)
                expert_outputs.append(h)

            expert_stack = torch.stack(expert_outputs, dim=1)  # (B, K, H)
            weighted = (
                gate_weights.unsqueeze(-1) * expert_stack
            ).sum(dim=1)  # (B, H)

        elif self.routing == "hard":
            # Deterministic routing by cluster label
            weighted = torch.zeros(
                B, self.expert_hidden_dim, device=x.device
            )
            gate_weights = F.one_hot(
                cluster_labels, self.num_experts
            ).float()  # (B, K)

            for k in range(self.num_experts):
                mask = (cluster_labels == k)
                if mask.any():
                    x_k = x[mask]
                    if self.model_type == "quantum":
                        h_k = self.experts[k](x_k.permute(0, 2, 1))
                    else:
                        h_k = self.experts[k](x_k)
                    weighted[mask] = h_k
        else:
            raise ValueError(f"Unknown routing: {self.routing!r}")

        logits = self.classifier(weighted)
        if self.num_classes == 2:
            logits = logits.squeeze(-1)

        return logits, gate_weights


# ---------------------------------------------------------------------------
# Train / Validate / Test (cluster-aware)
# ---------------------------------------------------------------------------

def train_cluster(model, train_loader, optimizer, criterion, device,
                  num_classes, balance_alpha, grad_clip, routing):
    model.train()
    epoch_loss = 0.0
    epoch_bal_loss = 0.0
    all_preds = []
    all_labels = []
    num_experts = model.num_experts
    expert_counts = torch.zeros(num_experts)

    for batch in tqdm(train_loader, desc="Training", leave=False):
        data, target, cluster = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        optimizer.zero_grad()
        logits, gate_weights = model(data, cluster)

        if num_classes > 2:
            target = target.long()
        else:
            target = target.float()
        task_loss = criterion(logits, target)

        # Load-balancing loss (soft routing only)
        if routing == "soft":
            bal_loss = load_balancing_loss(gate_weights, num_experts)
            loss = task_loss + balance_alpha * bal_loss
        else:
            bal_loss = torch.tensor(0.0)
            loss = task_loss

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        epoch_loss += task_loss.item()
        epoch_bal_loss += bal_loss.item()

        # Expert utilization tracking
        assignments = gate_weights.detach().argmax(dim=-1).cpu()
        for i in range(num_experts):
            expert_counts[i] += (assignments == i).sum().item()

        with torch.no_grad():
            if num_classes == 2:
                preds = torch.sigmoid(logits).cpu().numpy()
            else:
                preds = F.softmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)

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

    avg_loss = epoch_loss / len(train_loader)
    avg_bal_loss = epoch_bal_loss / len(train_loader)

    total_tokens = expert_counts.sum()
    utilization = (
        (expert_counts / total_tokens).tolist()
        if total_tokens > 0 else [0.0] * num_experts
    )

    return avg_loss, accuracy, auc, avg_bal_loss, utilization


def validate_cluster(model, val_loader, criterion, device, num_classes):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            data, target, cluster = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

            logits, _ = model(data, cluster)

            if num_classes > 2:
                target = target.long()
            else:
                target = target.float()
            loss = criterion(logits, target)
            epoch_loss += loss.item()

            if num_classes == 2:
                preds = torch.sigmoid(logits).cpu().numpy()
            else:
                preds = F.softmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)

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

    return epoch_loss / len(val_loader), accuracy, auc


def test_cluster(model, test_loader, criterion, device, num_classes):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            data, target, cluster = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

            logits, _ = model(data, cluster)

            if num_classes > 2:
                target = target.long()
            else:
                target = target.float()
            loss = criterion(logits, target)
            epoch_loss += loss.item()

            if num_classes == 2:
                preds = torch.sigmoid(logits).cpu().numpy()
            else:
                preds = F.softmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)

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

    return epoch_loss / len(test_loader), accuracy, auc


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Cluster-Informed MoE for ABCD ADHD Classification",
    )

    # Model / routing
    parser.add_argument("--model-type", type=str, default="classical",
                        choices=["classical", "quantum"],
                        help="Expert type")
    parser.add_argument("--routing", type=str, default="soft",
                        choices=["soft", "hard"],
                        help="Gating strategy")

    # Cluster
    parser.add_argument("--cluster-file", type=str, default=None,
                        help="Path to cluster assignments CSV")
    parser.add_argument("--cluster-column", type=str, default="km_cluster",
                        help="Column name for cluster labels")

    # MoE architecture
    parser.add_argument("--num-experts", type=int, default=2,
                        help="Number of experts (matches k clusters)")
    parser.add_argument("--expert-hidden-dim", type=int, default=64)
    parser.add_argument("--expert-layers", type=int, default=2,
                        help="Transformer layers per expert (classical only)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Attention heads per expert (classical only)")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gating-noise-std", type=float, default=0.1)
    parser.add_argument("--balance-loss-alpha", type=float, default=0.1,
                        help="Weight for load-balancing loss (soft only)")

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
    parser.add_argument("--job-id", type=str, default="ClusterMoE")
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

    # --- Data Loading -------------------------------------------------------
    print("\n" + "=" * 80)
    print("Loading ABCD fMRI Dataset with Cluster Labels...")
    print("=" * 80)

    sample_sz = args.sample_size if args.sample_size > 0 else None
    train_loader, val_loader, test_loader, input_dim = load_abcd_fmri(
        seed=args.seed, device=device, batch_size=args.batch_size,
        parcel_type=args.parcel_type,
        target_phenotype=args.target_phenotype,
        task_type="binary",
        sample_size=sample_sz,
        cluster_file=args.cluster_file,
        cluster_column=args.cluster_column,
    )

    # Transpose (N, T, R) -> (N, R, T) to match (B, C, T) convention
    train_loader, val_loader, test_loader, input_dim = \
        transpose_fmri_loaders_with_clusters(
            train_loader, val_loader, test_loader, input_dim,
            args.batch_size, device,
        )

    n_trials, n_channels, n_timesteps = input_dim
    print(f"\nInput: {n_trials} trials, {n_channels} channels, "
          f"{n_timesteps} timesteps")

    # --- Model --------------------------------------------------------------
    print("\n" + "=" * 80)
    config_name = f"{args.model_type.capitalize()} {args.routing.capitalize()}"
    print(f"Initializing Cluster-Informed MoE ({config_name})...")
    print("=" * 80)

    model = ClusterInformedMoE(
        total_channels=n_channels,
        time_points=n_timesteps,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        model_type=args.model_type,
        routing=args.routing,
        expert_layers=args.expert_layers,
        nhead=args.nhead,
        n_qubits=args.n_qubits,
        n_ansatz_layers=args.n_ansatz_layers,
        degree=args.degree,
        num_classes=args.num_classes,
        dropout=args.dropout,
        gating_noise_std=args.gating_noise_std,
        device=device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Model: {args.model_type}, Routing: {args.routing}, "
          f"Experts: {args.num_experts}, Hidden: {args.expert_hidden_dim}")
    if args.model_type == "classical":
        print(f"Classical: {args.expert_layers} layers, {args.nhead} heads")
    else:
        print(f"Quantum: {args.n_qubits}Q, {args.n_ansatz_layers}L, "
              f"D={args.degree}")

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
    ckpt_name = f"ClusterMoE_{args.model_type}_{args.routing}_{args.job_id}.pt"
    checkpoint_path = os.path.join(args.base_path, ckpt_name)
    logger = TrainingLogger(save_dir=args.base_path, job_id=args.job_id)

    # --- Wandb --------------------------------------------------------------
    if args.wandb:
        import wandb
        wandb.init(project="cluster-informed-moe", config=vars(args))

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

        train_loss, train_acc, train_auc, bal_loss, utilization = \
            train_cluster(
                model, train_loader, optimizer, criterion, device,
                args.num_classes, args.balance_loss_alpha, args.grad_clip,
                args.routing,
            )

        val_loss, val_acc, val_auc = validate_cluster(
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

    test_loss, test_acc, test_auc = test_cluster(
        model, test_loader, criterion, device, args.num_classes,
    )

    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"\nBest Validation AUC: {best_val_auc:.4f}")
    print(f"Final Test AUC: {test_auc:.4f}")
    print(f"Trainable parameters: {n_params:,}")
    print(f"Config: {args.model_type} / {args.routing}")
    print("=" * 80)

    if args.wandb:
        wandb.log({
            "test/loss": test_loss, "test/acc": test_acc, "test/auc": test_auc,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
