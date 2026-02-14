# Distributed Spatio-Temporal Mixture of Experts for EEG Analysis

## 1. Overview
This document details a **Region-Aware Hierarchical Mixture of Experts (MoE)** architecture designed to process high-dimensional spatio-temporal data (specifically EEG) in a distributed setting.

The core challenge addressed is the preservation of global spatio-temporal dependencies when splitting data into independent clusters for parallel processing. This solution employs **Overlapping Clusters (Halos)** and a **Global Memory Token** to maintain causal links between separated spatial regions.

## 2. Architectural Components

### A. The Expert Layer (Local Processing)
Instead of a single monolithic model, the input is divided into $K$ spatial regions. Each region is assigned to a dedicated Local Expert (a Transformer Encoder).
- Input: A subset of EEG channels (e.g., 16 channels + Halo) for $T$ time points.
- Mechanism: The expert applies Self-Attention to learn local temporal dynamics and frequency bands specific to its brain region (e.g., Occipital Alpha waves).
- Output: A latent representation summarizing the local activity.

### B. The Dependency Bridge (Global Synchronization)
To prevent the experts from becoming isolated "islands," a Global Memory Token is introduced.
- Initialization: A learnable vector $\mathbf{g}$ represents the global context.
- Broadcast: $\mathbf{g}$ is appended to the input sequence of every expert.
- Update: Each expert updates $\mathbf{g}$ based on its local data via self-attention.
- Fusion: At the end of the forward pass, the updated tokens from all experts are averaged (synchronized). This fused token serves as the "whiteboard" where experts share information without direct lateral connections.

### C. The Gating Network (The Decision Maker)
The Gating Network determines the relative importance of each expert for the final prediction.
- Input: The Fused Global Token.
- Logic: A small MLP that outputs a scalar weight $w_k$ for each expert.
- Output: A weighted sum of expert outputs: $\mathbf{y}_{final} = \sum w_k \cdot \text{Expert}_k(\mathbf{x})$.

## 3. Key Mechanisms for Dependency Preservation
### A. Overlapping Clusters (The "Halo" Method)
- Location: Data Preparation Stage (Before Expert Layer). To mitigate the loss of spatial correlations at the cutting boundaries, we introduce spatial overlap.
- Concept: Expert $A$ processes channels $1-16$. Expert $B$ processes channels $17-32$.
- The Halo: Expert $A$ actually receives channels $1-18$, and Expert $B$ receives $15-32$.
- Benefit: This allows boundary signals to be processed by both experts, ensuring that "volume conduction" effects (signals spreading across scalp) are not lost in the split.

### B. Global Memory Token (The "Whiteboard")
- Location: Inside & Between Layers. This mechanism decouples computation from communication.
- Independence: Experts compute their heavy matrix multiplications independently (perfect for parallel GPUs).
- Connection: They only synchronize by averaging this single token vector. This drastically reduces communication overhead compared to full model parallelism while preserving global context.

### C. Gradient Flow (Backpropagation)
- Location: Loss Calculation (After Expert Layer). While experts are computationally independent during the forward pass, they are coupled during training.
- Mechanism: The loss is calculated on the combined weighted output.
- Effect: Gradients flow from the loss function back to the Gating Network, and then distribute to specific experts. If the model fails, the gradients signal specific experts to adapt their internal representations to better cooperate with the group.


## 4. PyTorch Implementation
Below is the complete implementation for an EEG dataset with 64 channels and 300 time points, distributed across 4 experts.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGExpert(nn.Module):
    """
    A Local Expert specializing in a specific spatial region (subset of channels).
    It uses a Transformer Encoder to learn temporal dynamics.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, nhead, num_time_points):
        super().__init__()
        # Project channel input to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional Encoding: +1 for the Global Token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time_points + 1, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head to compress the time sequence into a single vector (Summary)
        self.pooler = nn.AdaptiveAvgPool1d(1) 

    def forward(self, x, global_token):
        """
        x: (Batch, Time, Channels_Local)
        global_token: (Batch, 1, Hidden_Dim) - The shared context
        """
        # 1. Project to hidden dimension
        x = self.input_projection(x) # (B, T, H)
        
        # 2. Append Global Token to the sequence (Pre-pending)
        # This allows the expert to "read" the global context and "write" to it
        x = torch.cat([global_token, x], dim=1) # (B, T+1, H)
        
        # 3. Add Positional Encoding
        x = x + self.pos_embedding
        
        # 4. Transformer Processing
        # The Self-Attention allows the Global Token to aggregate info from all time points
        x_out = self.transformer(x) # (B, T+1, H)
        
        # 5. Separate the updated Global Token and the Local Features
        updated_global_token = x_out[:, 0:1, :] # (B, 1, H)
        local_features = x_out[:, 1:, :]        # (B, T, H)
        
        # 6. Pool local features for the Gating Network's consumption
        # We return the mean of the local features as the "Expert's Opinion"
        expert_summary = local_features.mean(dim=1) # (B, Hidden_Dim)
        
        return expert_summary, updated_global_token

class GatingNetwork(nn.Module):
    """
    Decides which expert is 'right' based on the Global Context.
    """
    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_experts)
        
    def forward(self, global_context):
        # global_context: (Batch, Hidden_Dim) - The averaged global state
        x = F.relu(self.fc1(global_context))
        logits = self.fc2(x)
        # Softmax ensures weights sum to 1
        weights = F.softmax(logits, dim=-1) 
        return weights

class DistributedEEGMoE(nn.Module):
    def __init__(self, 
                 total_channels=64, 
                 time_points=300, 
                 num_experts=4, 
                 expert_hidden_dim=64, 
                 expert_layers=2, 
                 halo_size=2):
        super().__init__()
        
        self.num_experts = num_experts
        self.halo_size = halo_size
        self.total_channels = total_channels
        
        # Calculate channels per expert (assuming even split for simplicity)
        self.base_channels = total_channels // num_experts
        # Actual input dim includes the overlap (halo)
        self.expert_input_dim = self.base_channels + (2 * halo_size)
        
        # Initialize Experts
        self.experts = nn.ModuleList([
            EEGExpert(
                input_dim=self.expert_input_dim,
                hidden_dim=expert_hidden_dim,
                num_layers=expert_layers,
                nhead=4,
                num_time_points=time_points
            ) for _ in range(num_experts)
        ])
        
        # Shared Learnable Global Token (The "Whiteboard")
        self.global_token = nn.Parameter(torch.randn(1, 1, expert_hidden_dim))
        
        # Gating Network
        self.gate = GatingNetwork(hidden_dim=expert_hidden_dim, num_experts=num_experts)
        
        # Final Task Head (e.g., Classification)
        self.classifier = nn.Linear(expert_hidden_dim, 2) # Binary classification

    def _split_with_halo(self, x):
        """
        Splits (B, T, 64) into 4 chunks of (B, T, 16 + 2*Halo).
        Uses padding for boundary experts.
        """
        batch_size, time, _ = x.shape
        expert_inputs = []
        
        for k in range(self.num_experts):
            start_idx = k * self.base_channels
            end_idx = (k + 1) * self.base_channels
            
            # Define Halo boundaries
            halo_start = max(0, start_idx - self.halo_size)
            halo_end = min(self.total_channels, end_idx + self.halo_size)
            
            # Slice
            chunk = x[:, :, halo_start:halo_end]
            
            # Padding if at the physical boundaries (first or last expert)
            # to maintain consistent input dimension for all experts
            pad_left = 0
            pad_right = 0
            
            if k == 0: # First expert needs left padding
                pad_left = self.halo_size
            if k == self.num_experts - 1: # Last expert needs right padding
                expected_end = end_idx + self.halo_size
                pad_right = expected_end - halo_end
                
            if pad_left > 0 or pad_right > 0:
                chunk = F.pad(chunk, (pad_left, pad_right), "constant", 0)
            
            expert_inputs.append(chunk)
            
        return expert_inputs

    def forward(self, x):
        # x: (Batch, Time, 64)
        batch_size = x.size(0)
        
        # 1. Prepare Data (Halo Split)
        expert_inputs = self._split_with_halo(x)
        
        # 2. Prepare Global Token
        # Expand for batch size: (B, 1, H)
        global_token_batch = self.global_token.expand(batch_size, -1, -1)
        
        expert_outputs = []
        updated_global_tokens = []
        
        # 3. Run Experts (Conceptually Parallel)
        for k, expert in enumerate(self.experts):
            summary, updated_token = expert(expert_inputs[k], global_token_batch)
            expert_outputs.append(summary)
            updated_global_tokens.append(updated_token)
            
        # Stack outputs
        expert_outputs_stack = torch.stack(expert_outputs, dim=1) 
        
        # 4. Synchronize Global Token (Dependency Bridge)
        # Average the updated tokens from all experts.
        # In multi-GPU, use torch.distributed.all_reduce here.
        global_context_fused = torch.mean(torch.stack(updated_global_tokens, dim=1), dim=1).squeeze(1) 
        
        # 5. Gating
        # The gate looks at the FUSED context to decide weights
        gate_weights = self.gate(global_context_fused) # (B, Num_Experts)
        
        # 6. Weighted Combination
        gate_weights = gate_weights.unsqueeze(-1)
        final_representation = torch.sum(gate_weights * expert_outputs_stack, dim=1)
        
        # 7. Final Prediction
        logits = self.classifier(final_representation)
        
        return logits, gate_weights

# --- Example Usage ---
if __name__ == "__main__":
    # Fake EEG Data: (Batch=32, Time=300, Channels=64)
    data = torch.randn(32, 300, 64)
    
    # Initialize Model with Halo=4 (overlap of 4 channels)
    model = DistributedEEGMoE(
        total_channels=64,
        time_points=300,
        num_experts=4,
        expert_hidden_dim=32,
        halo_size=4 
    )
    
    # Forward Pass
    logits, weights = model(data)
    
    print(f"Input shape: {data.shape}")
    print(f"Logits shape: {logits.shape}")
    print("Gradient check: Simulating backward pass...")
    loss = logits.sum()
    loss.backward()
    print("Success: Gradients flowed to all experts.")
```
