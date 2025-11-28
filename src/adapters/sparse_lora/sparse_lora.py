import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from typing import Dict, Any, Optional


def compute_l1_loss(model, layer_weights: Optional[Dict[str, float]] = None):
    """
    Compute L1 loss for LoRA parameters with optional layer-wise weighting.

    Args:
        model: The model with LoRA parameters
        layer_weights: Optional dict mapping layer patterns to weight multipliers.
                      E.g., {"layer.0": 0.5, "layer.1": 0.5, "layer.2": 1.0}
                      Lower multiplier = less sparsity pressure
    """
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            # Determine layer-specific weight
            weight = 1.0
            if layer_weights:
                for pattern, w in layer_weights.items():
                    if pattern in name:
                        weight = w
                        break

            l1_loss += weight * torch.abs(param).sum()

    return l1_loss

def apply_magnitude_pruning(
    model,
    prune_ratio: float,
    device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:

    if prune_ratio <= 0.0:
        return {}

    if prune_ratio >= 1.0:
        raise ValueError(f"prune_ratio must be < 1.0, got {prune_ratio}")
    
    lora_params = []
    lora_names = []

    # Collect LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_params.append(param.data.view(-1))  # Flatten to 1D
            lora_names.append(name)

    if len(lora_params) == 0:
        print("Warning: No LoRA parameters found for pruning")
        return {}
    
    # Concatenate all LoRA parameters to compute global threshold
    all_weights = torch.cat(lora_params)
    threshold = torch.quantile(torch.abs(all_weights), prune_ratio)

    # Create and apply masks
    masks = {}
    total_pruned = 0

    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            # 1 if |weight| >= threshold, 0 otherwise
            mask = (torch.abs(param.data) >= threshold).float().to(device)
            masks[name] = mask

            # Apply mask to zero out pruned weights
            param.data *= mask

            pruned_count = (mask == 0).sum().item()
            total_pruned += pruned_count

    actual_sparsity = total_pruned / all_weights.numel()
    print(f"Actual sparsity achieved: {actual_sparsity:.2%}")

    return masks

# Compute sparsity statistics for LoRA parameters
def get_sparsity_stats(model) -> Dict[str, float]:
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            total_params += param.numel()
            zero_params += (torch.abs(param.data) < 1e-8).sum().item()

    if total_params == 0:
        return {
            "total_lora_params": 0,
            "zero_params": 0,
            "nonzero_params": 0,
            "sparsity_ratio": 0.0,
            "density_ratio": 0.0,
        }

    sparsity = zero_params / total_params

    return {
        "total_lora_params": total_params,
        "zero_params": zero_params,
        "nonzero_params": total_params - zero_params,
        "sparsity_ratio": sparsity,
        "density_ratio": 1.0 - sparsity,
    }

# using the masks to zero out pruned weights after each update
def apply_pruning_masks(model, masks: Dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name]


def build_sparse_lora_adapter(model, cfg: Dict[str, Any]):
    s = cfg["sparse_lora"]

    peft_config = LoraConfig(
        r=s["r"],
        lora_alpha=s["alpha"],
        lora_dropout=s["dropout"],
        target_modules=s["target_modules"],
        bias="none",
        task_type="SEQ_CLS",
    )

    peft_model = get_peft_model(model, peft_config)

    # Attach sparse configuration to model for use during training
    peft_model.sparse_config = {
        "method": s["method"],
        "l1_lambda": s.get("l1_lambda", 0.0),
        "prune_ratio": s.get("prune_ratio", 0.0),
    }

    print(f"Created Sparse LoRA adapter:")
    print(f"  Method: {s['method']}")
    if s["method"] == "l1":
        print(f"  L1 lambda: {s.get('l1_lambda', 0.0)}")
    else:
        print(f"  Prune ratio: {s.get('prune_ratio', 0.0):.1%}")
    print(f"  LoRA rank: {s['r']}")
    print(f"  Target modules: {s['target_modules']}")

    return peft_model