import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SVDSparsityEstimator(nn.Module):
    def __init__(self, weight: torch.Tensor, rank: int = 8):
        super().__init__()

        # Perform SVD decomposition: W = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)

        # Keep only top-k singular values/vectors
        U_k = U[:, :rank]  # [d_out, rank]
        S_k = S[:rank]     # [rank]
        Vh_k = Vh[:rank, :]  # [rank, d_in]

        # Split singular values evenly between the two matrices
        sqrt_S = torch.sqrt(S_k + 1e-8)

        # W ≈ (U @ sqrt(S)) @ (sqrt(S) @ Vh)
        self.register_buffer("W_A", U_k @ torch.diag(sqrt_S))  # [d_out, rank]
        self.register_buffer("W_B", torch.diag(sqrt_S) @ Vh_k)  # [rank, d_in]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = F.linear(x, self.W_B)
        out = F.linear(out, self.W_A)
        return out


class FFNEstimator(nn.Module):
    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        rank: int = 8
    ):
        super().__init__()
        self.gate_estimator = SVDSparsityEstimator(gate_weight, rank)
        self.up_estimator = SVDSparsityEstimator(up_weight, rank)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate FFN intermediate activations.
        For DistilBERT (gate_weight == up_weight): Returns single projection
        """
        gate_out = self.gate_estimator(x)
        up_out = self.up_estimator(x)
        # Apply SiLU activation: SiLU(gate) * up
        intermediate = F.silu(gate_out) * up_out
        return intermediate


def build_ffn_estimators(
    model,
    rank: int = 8,
    ffn_pattern: str = "ffn"
) -> Dict[str, FFNEstimator]:

    estimators = {}

    for name, module in model.named_modules():
        if ffn_pattern in name.lower() and "lin1" in name.lower():
            # For DistilBERT: ffn.lin1 is the intermediate projection
            # We use it for both gate and up in this simplified version
            try:
                estimators[name] = FFNEstimator(
                    gate_weight=module.weight.data,
                    up_weight=module.weight.data,  
                )
            except Exception as e:
                print(f"Warning: Failed to build FFN estimator for {name}: {e}")
                continue

    return estimators
