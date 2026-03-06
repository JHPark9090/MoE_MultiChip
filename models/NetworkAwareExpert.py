"""
Network-Aware Multi-Scale Expert (NAME) for ABCD fMRI.

Two expert variants:
  - NetworkAwareClassicalExpert: Transformer-based temporal mixing
  - NetworkAwareQuantumExpert: QSVT quantum circuit temporal mixing

Both share:
  1. Yeo 17-network grouped spatial projection
  2. Multi-scale dilated depthwise-separable convolutions
  3. Attentive temporal pooling
  4. Functional connectivity fingerprint branch
  5. Two-branch fusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yeo17_networks import get_network_indices, get_fc_network_pairs


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSepConv1d(nn.Module):
    """Depthwise separable 1D convolution with dilation."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, T)"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


def batched_corrcoef(x):
    """Compute batched Pearson correlation matrices.

    Args:
        x: (B, T, C) — time series for each ROI

    Returns:
        (B, C, C) — correlation matrix per sample
    """
    x = x - x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
    x = x / std
    return torch.bmm(x.transpose(1, 2), x) / (x.size(1) - 1)


# ---------------------------------------------------------------------------
# Classical Expert
# ---------------------------------------------------------------------------

class NetworkAwareClassicalExpert(nn.Module):
    """Network-aware classical expert with multi-scale temporal processing
    and functional connectivity branch."""

    def __init__(self, n_rois=180, d_net=8, d_model=128,
                 n_time=363, nhead=4, dropout=0.3):
        super().__init__()

        self.n_rois = n_rois
        self.d_net = d_net
        self.d_model = d_model

        # --- Branch 1: Network-Grouped Multi-Scale Temporal ---

        # Step 1: Per-network linear projections
        net_info = get_network_indices()
        self.network_names = [name for name, _ in net_info]
        self.network_indices = [idx for _, idx in net_info]
        self.n_networks = len(net_info)

        self.net_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(idx), d_net),
                nn.GELU(),
            )
            for _, idx in net_info
        ])
        concat_dim = self.n_networks * d_net  # 17 * 8 = 136

        # Step 2: Multi-scale dilated depthwise-separable convolutions
        self.conv_branch_a = DepthwiseSepConv1d(concat_dim, kernel_size=7, dilation=1, dropout=0.1)
        self.conv_branch_b = DepthwiseSepConv1d(concat_dim, kernel_size=7, dilation=4, dropout=0.1)
        self.conv_branch_c = DepthwiseSepConv1d(concat_dim, kernel_size=7, dilation=16, dropout=0.1)

        self.conv_merge = nn.Sequential(
            nn.Linear(concat_dim * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Step 3: 1-layer TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Step 4: Attentive temporal pooling
        self.attn_pool = nn.Linear(d_model, 1)

        # --- Branch 2: Functional Connectivity Fingerprint ---
        self.fc_pairs = get_fc_network_pairs()
        n_fc_features = len(self.fc_pairs)  # 153

        self.fc_mlp = nn.Sequential(
            nn.Linear(n_fc_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Fusion ---
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
        )

    def _network_projection(self, x):
        """Project ROIs into network-grouped features.

        Args:
            x: (B, T, C=180)
        Returns:
            (B, T, n_networks * d_net)
        """
        parts = []
        for i, idx in enumerate(self.network_indices):
            x_k = x[:, :, idx]  # (B, T, n_k)
            h_k = self.net_projections[i](x_k)  # (B, T, d_net)
            parts.append(h_k)
        return torch.cat(parts, dim=-1)  # (B, T, 17*d_net)

    def _multiscale_conv(self, h):
        """Apply multi-scale dilated convolutions.

        Args:
            h: (B, T, D) where D = n_networks * d_net
        Returns:
            (B, T, d_model)
        """
        # Conv1d expects (B, C, T)
        h_t = h.transpose(1, 2)  # (B, D, T)
        a = self.conv_branch_a(h_t)  # (B, D, T)
        b = self.conv_branch_b(h_t)
        c = self.conv_branch_c(h_t)
        cat = torch.cat([a, b, c], dim=1)  # (B, 3*D, T)
        cat = cat.transpose(1, 2)  # (B, T, 3*D)
        return self.conv_merge(cat)  # (B, T, d_model)

    def _attentive_pool(self, h):
        """Attentive temporal pooling.

        Args:
            h: (B, T, d_model)
        Returns:
            (B, d_model)
        """
        w = torch.softmax(self.attn_pool(h), dim=1)  # (B, T, 1)
        return (w * h).sum(dim=1)  # (B, d_model)

    def _fc_fingerprint(self, x):
        """Compute network-level FC fingerprint.

        Args:
            x: (B, T, C=180) — raw fMRI time series
        Returns:
            (B, 64)
        """
        corr = batched_corrcoef(x)  # (B, 180, 180)

        fc_features = []
        net_indices = self.network_indices
        for i, j in self.fc_pairs:
            idx_i = net_indices[i]
            idx_j = net_indices[j]
            # Extract submatrix and average
            block = corr[:, idx_i][:, :, idx_j]  # (B, n_i, n_j)
            fc_features.append(block.mean(dim=(1, 2)))  # (B,)

        fc_vec = torch.stack(fc_features, dim=1)  # (B, n_pairs)
        return self.fc_mlp(fc_vec)  # (B, 64)

    def forward(self, x):
        """
        Args:
            x: (B, T, C=180) — fMRI time series
        Returns:
            (B, 64) — expert features
        """
        # Branch 1: Temporal
        h = self._network_projection(x)      # (B, T, 136)
        h = self._multiscale_conv(h)          # (B, T, 128)
        h = self.transformer(h)               # (B, T, 128)
        temporal = self._attentive_pool(h)    # (B, 128)

        # Branch 2: FC
        fc = self._fc_fingerprint(x)          # (B, 64)

        # Fusion
        fused = torch.cat([temporal, fc], dim=1)  # (B, 192)
        return self.fusion(fused)                  # (B, 64)


# ---------------------------------------------------------------------------
# Quantum Expert
# ---------------------------------------------------------------------------

class NetworkAwareQuantumExpert(nn.Module):
    """Network-aware quantum expert. Same architecture as classical except
    the Transformer temporal mixing is replaced by the QSVT quantum circuit
    from QTSTransformer_v2_5."""

    def __init__(self, n_rois=180, d_net=8, d_model=128,
                 n_time=363, n_qubits=8, n_ansatz_layers=2,
                 degree=3, dropout=0.3, device="cpu"):
        super().__init__()

        self.n_rois = n_rois
        self.d_net = d_net
        self.d_model = d_model
        self.device = device

        # --- Branch 1: Network-Grouped Multi-Scale Temporal ---

        # Step 1: Per-network linear projections
        net_info = get_network_indices()
        self.network_names = [name for name, _ in net_info]
        self.network_indices = [idx for _, idx in net_info]
        self.n_networks = len(net_info)

        self.net_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(idx), d_net),
                nn.GELU(),
            )
            for _, idx in net_info
        ])
        concat_dim = self.n_networks * d_net

        # Step 2: Multi-scale dilated depthwise-separable convolutions
        self.conv_branch_a = DepthwiseSepConv1d(concat_dim, kernel_size=7, dilation=1, dropout=0.1)
        self.conv_branch_b = DepthwiseSepConv1d(concat_dim, kernel_size=7, dilation=4, dropout=0.1)
        self.conv_branch_c = DepthwiseSepConv1d(concat_dim, kernel_size=7, dilation=16, dropout=0.1)

        self.conv_merge = nn.Sequential(
            nn.Linear(concat_dim * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Step 3-Q: QSVT quantum temporal mixing (replaces Transformer)
        import pennylane as qml
        from models.QTSTransformer_v2_5 import (
            sim14_circuit, evaluate_polynomial_state_pl,
        )
        self._evaluate_polynomial_state_pl = evaluate_polynomial_state_pl

        self.n_qubits = n_qubits
        self.n_ansatz_layers = n_ansatz_layers
        self.degree = degree
        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1

        # Projection from d_model to quantum rotation angles
        self.quantum_projection = nn.Linear(d_model, self.n_rots)
        self.rot_sigm = nn.Sigmoid()
        self.quantum_dropout = nn.Dropout(dropout)

        # Trainable quantum parameters
        self.n_poly_coeffs = degree + 1
        self.poly_coeffs = nn.Parameter(torch.rand(self.n_poly_coeffs))
        self.qff_params = nn.Parameter(torch.rand(self.qff_n_rots))

        # PennyLane QNodes
        _n_qubits = n_qubits
        _n_ansatz_layers = n_ansatz_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _timestep_state_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(_n_qubits))
            sim14_circuit(params, wires=_n_qubits, layers=_n_ansatz_layers)
            return qml.state()

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _qff_qnode_expval(initial_state, params):
            qml.StatePrep(initial_state, wires=range(_n_qubits))
            sim14_circuit(params, wires=_n_qubits, layers=1)
            observables = [qml.PauliX(i) for i in range(_n_qubits)] + \
                          [qml.PauliY(i) for i in range(_n_qubits)] + \
                          [qml.PauliZ(i) for i in range(_n_qubits)]
            return [qml.expval(op) for op in observables]

        self.timestep_state_qnode = _timestep_state_qnode
        self.qff_qnode_expval = _qff_qnode_expval

        # Post-quantum projection back to d_model
        self.quantum_output = nn.Linear(3 * n_qubits, d_model)

        # Mix coefficients (n_time is used for LCU)
        self.n_time = n_time
        self.mix_coeffs = nn.Parameter(
            torch.rand(n_time, dtype=torch.complex64)
        )

        # Step 4: Attentive temporal pooling (on quantum output)
        self.attn_pool = nn.Linear(d_model, 1)

        # --- Branch 2: Functional Connectivity Fingerprint ---
        self.fc_pairs = get_fc_network_pairs()
        n_fc_features = len(self.fc_pairs)

        self.fc_mlp = nn.Sequential(
            nn.Linear(n_fc_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Fusion ---
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
        )

    def _network_projection(self, x):
        parts = []
        for i, idx in enumerate(self.network_indices):
            x_k = x[:, :, idx]
            h_k = self.net_projections[i](x_k)
            parts.append(h_k)
        return torch.cat(parts, dim=-1)

    def _multiscale_conv(self, h):
        h_t = h.transpose(1, 2)
        a = self.conv_branch_a(h_t)
        b = self.conv_branch_b(h_t)
        c = self.conv_branch_c(h_t)
        cat = torch.cat([a, b, c], dim=1)
        cat = cat.transpose(1, 2)
        return self.conv_merge(cat)

    def _quantum_temporal_mixing(self, h):
        """Replace the Transformer with QSVT quantum circuit.

        Args:
            h: (B, T, d_model) — enriched temporal features
        Returns:
            (B, d_model)
        """
        bsz = h.shape[0]

        # Project to rotation angles
        h_drop = self.quantum_dropout(h)
        timestep_params = self.rot_sigm(self.quantum_projection(h_drop)) * (2 * math.pi)

        # Initialize quantum state
        base_states = torch.zeros(
            bsz, 2 ** self.n_qubits,
            dtype=torch.complex64, device=h.device,
        )
        base_states[:, 0] = 1.0

        # LCU + QSVT polynomial
        mix_coeffs = self.mix_coeffs[:h.size(1)].repeat(bsz, 1)
        mixed_timestep = self._evaluate_polynomial_state_pl(
            base_states, timestep_params,
            self.timestep_state_qnode,
            self.n_qubits, mix_coeffs, self.poly_coeffs,
        )

        # Normalize
        norm = torch.linalg.vector_norm(mixed_timestep, dim=1, keepdim=True)
        normalized = mixed_timestep / (norm + 1e-9)

        # QFF expectation values
        exps = self.qff_qnode_expval(
            initial_state=normalized,
            params=self.qff_params,
        )
        exps = torch.stack(exps, dim=1).float()  # (B, 3*n_qubits)

        return self.quantum_output(exps)  # (B, d_model)

    def _attentive_pool(self, h):
        w = torch.softmax(self.attn_pool(h), dim=1)
        return (w * h).sum(dim=1)

    def _fc_fingerprint(self, x):
        corr = batched_corrcoef(x)
        fc_features = []
        net_indices = self.network_indices
        for i, j in self.fc_pairs:
            idx_i = net_indices[i]
            idx_j = net_indices[j]
            block = corr[:, idx_i][:, :, idx_j]
            fc_features.append(block.mean(dim=(1, 2)))
        fc_vec = torch.stack(fc_features, dim=1)
        return self.fc_mlp(fc_vec)

    def forward(self, x):
        """
        Args:
            x: (B, T, C=180) — fMRI time series
        Returns:
            (B, 64) — expert features
        """
        # Branch 1: Temporal with quantum mixing
        h = self._network_projection(x)        # (B, T, 136)
        h = self._multiscale_conv(h)            # (B, T, 128)
        temporal = self._quantum_temporal_mixing(h)  # (B, 128)

        # Branch 2: FC
        fc = self._fc_fingerprint(x)            # (B, 64)

        # Fusion
        fused = torch.cat([temporal, fc], dim=1)  # (B, 192)
        return self.fusion(fused)                  # (B, 64)
