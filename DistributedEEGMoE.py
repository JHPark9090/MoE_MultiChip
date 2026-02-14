#!/usr/bin/env python3
"""
Distributed EEG Mixture of Experts (MoE) — Region-Aware Hierarchical Architecture
for PhysioNet Motor Imagery EEG Classification.

Implements:
- Region-aware channel splitting with halo overlap for spatial continuity
- Global memory token for inter-expert communication
- Gating network with exploration noise + load-balancing loss (Switch Transformer)
- Flexible binary / multiclass classification
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
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

# Data loader
from dataloaders.Load_PhysioNet_EEG import load_eeg_ts_revised

# Logger from parent directory
sys.path.insert(0, "/pscratch/sd/j/junghoon")
from logger import TrainingLogger


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Calculate elapsed time in minutes and seconds."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_balancing_loss(gate_weights: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Switch Transformer load-balancing auxiliary loss.

    L = N * sum_i(f_i * P_i)
      f_i = fraction of tokens routed to expert i  (hard argmax assignment)
      P_i = mean gate probability for expert i

    Args:
        gate_weights: (B, num_experts) softmax gate probabilities.
        num_experts:  number of experts.
    Returns:
        Scalar loss encouraging uniform expert utilization.
    """
    # f_i: fraction of batch where expert i has the highest weight
    assignments = gate_weights.argmax(dim=-1)  # (B,)
    f = torch.zeros(num_experts, device=gate_weights.device)
    for i in range(num_experts):
        f[i] = (assignments == i).float().mean()
    # P_i: mean probability for expert i across the batch
    P = gate_weights.mean(dim=0)  # (num_experts,)
    return num_experts * (f * P).sum()


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------

class EEGExpert(nn.Module):
    """Local Expert: Transformer Encoder over a spatial subset of EEG channels."""

    def __init__(self, input_dim, hidden_dim, num_layers, nhead,
                 num_time_points, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # +1 for the global memory token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_time_points + 1, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True,
            dropout=dropout, dim_feedforward=hidden_dim * 4,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

    def forward(self, x, global_token):
        """
        Args:
            x: (B, T, channels_local) — local channel subset with halo
            global_token: (B, 1, hidden_dim) — shared context token
        Returns:
            expert_summary:       (B, hidden_dim)
            updated_global_token: (B, 1, hidden_dim)
        """
        x = self.dropout(self.input_projection(x))     # (B, T, H)
        x = torch.cat([global_token, x], dim=1)        # (B, T+1, H)
        x = x + self.pos_embedding
        x_out = self.transformer(x)                     # (B, T+1, H)

        updated_global_token = x_out[:, 0:1, :]        # (B, 1, H)
        local_features = x_out[:, 1:, :]                # (B, T, H)
        expert_summary = local_features.mean(dim=1)     # (B, H)

        return expert_summary, updated_global_token


class GatingNetwork(nn.Module):
    """Gating network with optional exploration noise during training."""

    def __init__(self, hidden_dim, num_experts, noise_std=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_experts)
        self.noise_std = noise_std

    def forward(self, global_context):
        """
        Args:
            global_context: (B, hidden_dim)
        Returns:
            weights: (B, num_experts) — softmax expert weights
        """
        x = F.relu(self.fc1(global_context))
        logits = self.fc2(x)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        weights = F.softmax(logits, dim=-1)
        return weights


class DistributedEEGMoE(nn.Module):
    """
    Region-Aware Hierarchical Mixture of Experts for EEG.

    Splits channels into overlapping regions (with halo), processes each
    with a local Transformer expert, and combines via learned gating.
    A global memory token provides inter-expert communication.
    """

    def __init__(self, total_channels=64, time_points=300, num_experts=4,
                 expert_hidden_dim=64, expert_layers=2, nhead=4,
                 halo_size=2, num_classes=2, dropout=0.1,
                 gating_noise_std=0.1):
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

        # Experts
        self.experts = nn.ModuleList([
            EEGExpert(
                input_dim=self.expert_input_dim,
                hidden_dim=expert_hidden_dim,
                num_layers=expert_layers,
                nhead=nhead,
                num_time_points=time_points,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

        # Shared learnable global memory token
        self.global_token = nn.Parameter(
            torch.randn(1, 1, expert_hidden_dim)
        )

        # Gating network
        self.gate = GatingNetwork(
            hidden_dim=expert_hidden_dim, num_experts=num_experts,
            noise_std=gating_noise_std,
        )

        # Task head: binary → 1 output, multiclass → num_classes outputs
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(expert_hidden_dim, output_dim)

    # ----- helpers ----------------------------------------------------------

    def _compute_expert_ranges(self):
        """Distribute channels to experts; first *r* experts get +1 channel."""
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

            # Pad left when halo extends before channel 0
            pad_left = max(0, self.halo_size - start)
            # Pad right to reach uniform expert_input_dim
            current_width = halo_end - halo_start
            pad_right = self.expert_input_dim - current_width - pad_left

            if pad_left > 0 or pad_right > 0:
                # F.pad pads last dim: (left, right)
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
        batch_size = x.size(0)

        # 1. Split channels with halo overlap
        expert_inputs = self._split_with_halo(x)

        # 2. Broadcast global token
        global_token_batch = self.global_token.expand(batch_size, -1, -1)

        expert_outputs = []
        updated_global_tokens = []

        # 3. Run experts
        for k, expert in enumerate(self.experts):
            summary, updated_token = expert(
                expert_inputs[k], global_token_batch,
            )
            expert_outputs.append(summary)
            updated_global_tokens.append(updated_token)

        expert_outputs_stack = torch.stack(expert_outputs, dim=1)  # (B, K, H)

        # 4. Fuse global tokens (dependency bridge)
        global_context_fused = torch.mean(
            torch.stack(updated_global_tokens, dim=1), dim=1,
        ).squeeze(1)  # (B, H)

        # 5. Gating
        gate_weights = self.gate(global_context_fused)  # (B, K)

        # 6. Weighted combination
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (B, K, 1)
        final_representation = torch.sum(
            gate_weights_expanded * expert_outputs_stack, dim=1,
        )  # (B, H)

        # 7. Classification
        logits = self.classifier(final_representation)
        if self.num_classes == 2:
            logits = logits.squeeze(-1)  # (B,)

        return logits, gate_weights


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Distributed EEG Mixture of Experts",
    )

    # MoE architecture
    parser.add_argument("--num-experts", type=int, default=4,
                        help="Number of spatial experts")
    parser.add_argument("--expert-hidden-dim", type=int, default=64,
                        help="Hidden dimension inside each expert")
    parser.add_argument("--expert-layers", type=int, default=2,
                        help="Transformer layers per expert")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Attention heads per expert")
    parser.add_argument("--halo-size", type=int, default=2,
                        help="Channel overlap on each side")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="2 = binary (BCEWithLogitsLoss), >2 = multiclass")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gating-noise-std", type=float, default=0.1,
                        help="Std of noise added to gate logits during training")
    parser.add_argument("--balance-loss-alpha", type=float, default=0.01,
                        help="Weight for load-balancing auxiliary loss")

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
    parser.add_argument("--sampling-freq", type=int, default=16,
                        help="EEG resampling frequency (Hz)")
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Number of PhysioNet subjects (1-109)")

    # Experiment
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--job-id", type=str, default="MoE_EEG")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--base-path", type=str, default="./checkpoints",
                        help="Directory for checkpoints and logs")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train(model, train_loader, optimizer, criterion, device,
          num_classes, balance_alpha, grad_clip):
    model.train()
    epoch_loss = 0.0
    epoch_bal_loss = 0.0
    all_preds = []
    all_labels = []
    num_experts = model.num_experts
    expert_counts = torch.zeros(num_experts)

    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        optimizer.zero_grad()
        logits, gate_weights = model(data)

        # Task loss
        if num_classes > 2:
            target = target.long()
        task_loss = criterion(logits, target)

        # Load-balancing loss
        bal_loss = load_balancing_loss(gate_weights, num_experts)
        loss = task_loss + balance_alpha * bal_loss

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

        # Predictions
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


def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating", leave=False):
            data, target = data.to(device), target.to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

            logits, _ = model(data)

            if num_classes > 2:
                target = target.long()
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

    avg_loss = epoch_loss / len(val_loader)
    return avg_loss, accuracy, auc


def test(model, test_loader, criterion, device, num_classes):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

            logits, _ = model(data)

            if num_classes > 2:
                target = target.long()
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

    avg_loss = epoch_loss / len(test_loader)
    return avg_loss, accuracy, auc


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
    print("Loading PhysioNet EEG Dataset...")
    print("=" * 80)
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=args.seed, device=device, batch_size=args.batch_size,
        sampling_freq=args.sampling_freq, sample_size=args.sample_size,
    )
    n_trials, n_channels, n_timesteps = input_dim
    print(f"\nInput: {n_trials} trials, {n_channels} channels, "
          f"{n_timesteps} timesteps")

    # --- Model --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Initializing Distributed EEG MoE...")
    print("=" * 80)
    model = DistributedEEGMoE(
        total_channels=n_channels,
        time_points=n_timesteps,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        expert_layers=args.expert_layers,
        nhead=args.nhead,
        halo_size=args.halo_size,
        num_classes=args.num_classes,
        dropout=args.dropout,
        gating_noise_std=args.gating_noise_std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Experts: {args.num_experts}, Hidden: {args.expert_hidden_dim}, "
          f"Layers: {args.expert_layers}, Heads: {args.nhead}")
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
        args.base_path, f"MoE_EEG_{args.job_id}.pt",
    )
    logger = TrainingLogger(save_dir=args.base_path, job_id=args.job_id)

    # --- Wandb --------------------------------------------------------------
    if args.wandb:
        import wandb
        wandb.init(project="distributed-eeg-moe", config=vars(args))

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

        # CSV log (accuracy stored in rmse column for compatibility)
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
    print("=" * 80)

    if args.wandb:
        wandb.log({
            "test/loss": test_loss, "test/acc": test_acc,
            "test/auc": test_auc,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
