"""
Circuit-Specialized Mixture of Experts v2 — Quantum Fix Edition.

Addresses 4 identified bottlenecks in quantum Circuit MoE training:

1. poly_coeffs initialization: [0, 1, 0, 0] instead of random.
   Random init activates all QSVT iterations, causing gradient vanishing.
   [0, 1, 0, 0] starts with a single LCU round (input-dependent output).

2. Uniform mix_coeffs initialization: 1/sqrt(T) instead of random.
   Random complex coefficients create destructive interference at init.
   Uniform real-valued init gives equal weight to all timesteps.

3. Expert gradient scaling: compensates 1/K attenuation from soft gating.
   In MoE, dL/dh_i = dL/d(weighted) * gate_weight_i ≈ dL/d(weighted) * 1/K.
   Hook scales expert gradients by K during backward pass.

4. Temporal average pooling: 363 → ~36 timesteps before quantum circuit.
   Reduces LCU mix_coeffs from 363 to ~36 parameters.
   Reduces QNode batch from B*363 to B*36 (10x speedup).

All fixes apply only to quantum experts. Classical path is unchanged.

References:
    - Nigg et al. (2020) Biological Psychiatry: CNNI
    - Feng et al. (2024) EClinicalMedicine
    - Pan et al. (2026) JAMA Psychiatry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yeo17_networks import get_circuit_config, get_circuit_roi_indices


class CircuitExpert(nn.Module):
    """Transformer expert processing a specific ROI subset."""

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
            x: (B, T, n_rois_subset) -- circuit-specific ROI subset
        Returns:
            (B, H) -- temporal mean pooling
        """
        x = self.dropout(self.input_projection(x))  # (B, T, H)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.transformer(x)  # (B, T, H)
        return x.mean(dim=1)  # (B, H)


class CircuitGatingNetwork(nn.Module):
    """Gating network for circuit-specialized MoE."""

    def __init__(self, input_dim, num_experts, noise_std=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_experts)
        self.noise_std = noise_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        return F.softmax(logits, dim=-1)


class CircuitMoE(nn.Module):
    """
    Circuit-Specialized Mixture of Experts v2.

    Quantum-specific fixes over v1:
    - Temporal pooling before quantum experts (363 -> ~36 timesteps)
    - poly_coeffs = [0, 1, 0, 0] (single LCU round at init)
    - Uniform mix_coeffs = 1/sqrt(T_pooled)
    - Expert gradient scaling by num_experts
    """

    def __init__(self, circuit_config_name, time_points,
                 expert_hidden_dim, model_type,
                 # Classical args:
                 expert_layers=2, nhead=4,
                 # Quantum args:
                 n_qubits=8, n_ansatz_layers=2, degree=3,
                 # Gating:
                 use_cluster_gate=False, n_clusters=0,
                 gating_noise_std=0.1,
                 # Common:
                 total_channels=180, num_classes=2, dropout=0.2,
                 device="cpu",
                 # v2 quantum fixes:
                 pool_factor=10,
                 grad_scale=True):
        super().__init__()

        self.model_type = model_type
        self.expert_hidden_dim = expert_hidden_dim
        self.num_classes = num_classes
        self.total_channels = total_channels
        self.time_points = time_points
        self.use_cluster_gate = use_cluster_gate
        self.n_clusters = n_clusters
        self.pool_factor = pool_factor
        self.grad_scale = grad_scale

        # Get circuit ROI assignments
        circuit_config = get_circuit_config(circuit_config_name)
        circuit_roi_list = get_circuit_roi_indices(circuit_config)
        self.num_experts = len(circuit_roi_list)

        # Register ROI indices as buffers (not parameters)
        self.circuit_names = []
        for i, (name, roi_indices) in enumerate(circuit_roi_list):
            self.circuit_names.append(name)
            self.register_buffer(
                f"roi_idx_{i}",
                torch.tensor(roi_indices, dtype=torch.long),
            )

        # Print circuit info
        print(f"[CircuitMoE] Config: {circuit_config_name}, "
              f"{self.num_experts} experts")
        for i, (name, roi_indices) in enumerate(circuit_roi_list):
            print(f"  Expert {i} ({name}): {len(roi_indices)} ROIs")

        # Compute pooled timesteps for quantum experts
        if model_type == "quantum" and pool_factor > 1:
            self.quantum_timesteps = time_points // pool_factor
            print(f"  Temporal pooling: {time_points} -> {self.quantum_timesteps} "
                  f"(factor {pool_factor})")
        else:
            self.quantum_timesteps = time_points

        # --- Experts ---
        if model_type == "classical":
            self.experts = nn.ModuleList()
            for i, (name, roi_indices) in enumerate(circuit_roi_list):
                self.experts.append(CircuitExpert(
                    input_dim=len(roi_indices),
                    hidden_dim=expert_hidden_dim,
                    num_layers=expert_layers,
                    nhead=nhead,
                    time_points=time_points,
                    dropout=dropout,
                ))
        elif model_type == "quantum":
            from models.QTSTransformer_v2_5 import QuantumTSTransformer
            self.experts = nn.ModuleList()
            for i, (name, roi_indices) in enumerate(circuit_roi_list):
                self.experts.append(QuantumTSTransformer(
                    n_qubits=n_qubits,
                    n_timesteps=self.quantum_timesteps,  # FIX 4: pooled timesteps
                    degree=degree,
                    n_ansatz_layers=n_ansatz_layers,
                    feature_dim=len(roi_indices),
                    output_dim=expert_hidden_dim,
                    dropout=dropout,
                    device=device,
                ))

            # FIX 1: poly_coeffs = [0, 1, 0, 0] (single LCU round at init)
            # FIX 2: Uniform mix_coeffs = 1/sqrt(T_pooled)
            self._init_quantum_params()
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        # --- Gating Network ---
        gate_input_dim = total_channels
        if use_cluster_gate and n_clusters > 0:
            gate_input_dim += n_clusters
        self.gate = CircuitGatingNetwork(
            gate_input_dim, self.num_experts, gating_noise_std,
        )

        # --- Classifier ---
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(expert_hidden_dim, output_dim)

    def _init_quantum_params(self):
        """Apply quantum-specific parameter initialization fixes."""
        for i, expert in enumerate(self.experts):
            # FIX 1: poly_coeffs = [0, 1, 0, 0]
            # This starts with a single LCU round, so the output depends on
            # the input angles (feature_projection + mix_coeffs get gradients).
            # Random init activates all degree iterations, causing vanishing grads.
            with torch.no_grad():
                expert.poly_coeffs.zero_()
                expert.poly_coeffs[1] = 1.0
            print(f"  Expert {i} ({self.circuit_names[i]}): "
                  f"poly_coeffs init = {expert.poly_coeffs.data.tolist()}")

            # FIX 2: Uniform real-valued mix_coeffs = 1/sqrt(T)
            # Random complex coefficients create destructive interference.
            # Uniform init gives equal weight to all timesteps.
            T = self.quantum_timesteps
            with torch.no_grad():
                expert.mix_coeffs.data = torch.full(
                    (T,), 1.0 / T**0.5, dtype=torch.complex64,
                )
            print(f"  Expert {i} ({self.circuit_names[i]}): "
                  f"mix_coeffs init = uniform(1/sqrt({T}))")

    def forward(self, x, cluster_labels=None):
        """
        Args:
            x: (B, T, C) -- full 180-ROI fMRI input
            cluster_labels: (B,) -- optional cluster assignments for gating
        Returns:
            logits: (B,) for binary or (B, num_classes) for multiclass
            gate_weights: (B, num_experts) -- expert routing weights
        """
        B = x.size(0)

        # --- Gating ---
        gate_input = x.mean(dim=1)  # (B, C)
        if self.use_cluster_gate and cluster_labels is not None:
            cluster_onehot = F.one_hot(
                cluster_labels, self.n_clusters
            ).float()
            gate_input = torch.cat([gate_input, cluster_onehot], dim=-1)
        gate_weights = self.gate(gate_input)  # (B, num_experts)

        # --- Expert forward passes ---
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            roi_idx = getattr(self, f"roi_idx_{i}")
            x_subset = x[:, :, roi_idx]  # (B, T, n_rois_i)

            if self.model_type == "quantum":
                # FIX 4: Temporal average pooling before quantum circuit
                # (B, T, n_rois) -> permute -> (B, n_rois, T) -> pool -> permute back
                if self.pool_factor > 1:
                    x_q = x_subset.permute(0, 2, 1)  # (B, C, T)
                    # Truncate to exact multiple of pool_factor
                    T_trunc = (x_q.size(2) // self.pool_factor) * self.pool_factor
                    x_q = x_q[:, :, :T_trunc]
                    x_q = F.avg_pool1d(x_q, kernel_size=self.pool_factor)  # (B, C, T_pooled)
                else:
                    x_q = x_subset.permute(0, 2, 1)  # (B, C, T)

                h = expert(x_q)  # QTS expects (B, C, T)

                # FIX 3: Expert gradient scaling to compensate 1/K gating attenuation
                if self.training and self.grad_scale:
                    h = h * 1.0  # keep forward value unchanged
                    h.register_hook(
                        lambda grad, n=self.num_experts: grad * n
                    )
            else:
                h = expert(x_subset)  # Classical expects (B, T, C)

            expert_outputs.append(h)

        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, K, H)

        # --- Weighted combination ---
        weighted = (
            gate_weights.unsqueeze(-1) * expert_stack
        ).sum(dim=1)  # (B, H)

        # --- Classifier ---
        logits = self.classifier(weighted)
        if self.num_classes <= 2:
            logits = logits.squeeze(-1)

        return logits, gate_weights
