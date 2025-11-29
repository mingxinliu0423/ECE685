import torch
from typing import Optional


def select_ffn_channels(
    intermediate_approx: torch.Tensor,
    sparsity_ratio: float
) -> torch.Tensor:

    # Flatten batch and sequence dimensions
    batch_size, seq_len, intermediate_size = intermediate_approx.shape
    flat = intermediate_approx.reshape(-1, intermediate_size)  # [B*L, intermediate_size]

    # Compute L2 norm for each channel across all tokens
    importance = torch.norm(flat, p=2, dim=0)

    # Select top-k channels to keep
    keep_dim = int(intermediate_size * (1 - sparsity_ratio))
    keep_dim = max(1, keep_dim)

    top_k_indices = torch.topk(importance, keep_dim, dim=-1).indices

    return top_k_indices


def get_layer_sparsity_ratio(
    layer_name: str,
    default_sparsity: float,
    layer_sparsity_config: Optional[dict] = None
) -> float:
    if layer_sparsity_config is None:
        return default_sparsity

    # Check if any pattern matches the layer name
    for pattern, sparsity in layer_sparsity_config.items():
        if pattern in layer_name:
            return sparsity

    return default_sparsity
