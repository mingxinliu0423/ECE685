import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from typing import Dict, Any, Optional

from .svd_estimator import build_ffn_estimators, FFNEstimator
from .sparsity_selector import select_ffn_channels, get_layer_sparsity_ratio
from .sparse_ffn_module import replace_ffn_with_sparse, get_sparsity_statistics


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


def apply_sparse_ffn_to_model(
    model,
    estimators: dict,
    cfg: Dict[str, Any],
    total_training_steps: int
) -> int:
    """
    Apply sparse FFN modules to the model.

    Args:
        model: The PEFT model
        estimators: SVD estimators dictionary
        cfg: Configuration dictionary
        total_training_steps: Total number of training steps

    Returns:
        Number of FFN layers replaced
    """
    s = cfg.get("sparse_lora", {})

    if s.get("method") != "svd" or estimators is None:
        return 0

    # Calculate warmup steps
    sparse_warmup_ratio = cfg.get("sparse_warmup_ratio", 0.0)
    sparse_warmup_steps = int(total_training_steps * sparse_warmup_ratio)

    # Get base model
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        if hasattr(base_model, 'model'):
            base_model = base_model.model

    print(f"\nApplying sparse FFN modules...")
    print(f"  Sparsity warmup: {sparse_warmup_steps} steps")
    print(f"  Default FFN sparsity: {s.get('ffn_sparsity', 0.5):.1%}")

    # Replace FFN modules
    replaced = replace_ffn_with_sparse(
        base_model,
        estimators,
        sparsity_ratio=s.get("ffn_sparsity", 0.5),
        sparse_warmup_steps=sparse_warmup_steps,
        layer_sparsity_config=s.get("layer_sparsity"),
    )

    print(f"  Replaced {replaced} FFN modules")

    return replaced


def build_svd_estimators_for_model(model, cfg: Dict[str, Any]) -> Optional[Dict]:
    """
    Build SVD estimators for the model (paper method).

    Args:
        model: The model to build estimators for (can be PEFT model)
        cfg: Configuration dictionary

    Returns:
        Dictionary containing SVD estimators, or None if not using SVD method
    """
    s = cfg.get("sparse_lora", {})

    if s.get("method") != "svd":
        return None

    rank = s.get("svd_rank", 8)
    print(f"\nBuilding SVD estimators (rank={rank})...")

    # If this is a PEFT model, get the base model
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        if hasattr(base_model, 'model'):  # PeftModelForSequenceClassification
            base_model = base_model.model

    # Build FFN estimators
    ffn_estimators = build_ffn_estimators(base_model, rank=rank)

    print(f"  Built {len(ffn_estimators)} FFN estimators")

    return {
        "ffn": ffn_estimators,
        "rank": rank,
    }


def build_sparse_lora_adapter(model, cfg: Dict[str, Any]):
    """
    Build Sparse LoRA adapter.

    Supports three methods:
    - "l1": L1 regularization on LoRA parameters (your original method)
    - "prune": Magnitude pruning on LoRA parameters (your original method)
    - "svd": SVD estimator with FFN channel sparsity (paper method)
    """
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
        "svd_rank": s.get("svd_rank", 8),
        "ffn_sparsity": s.get("ffn_sparsity", 0.5),
        "layer_sparsity": s.get("layer_sparsity", None),
    }

    print(f"Created Sparse LoRA adapter:")
    print(f"  Method: {s['method']}")
    if s["method"] == "l1":
        print(f"  L1 lambda: {s.get('l1_lambda', 0.0)}")
    elif s["method"] == "prune":
        print(f"  Prune ratio: {s.get('prune_ratio', 0.0):.1%}")
    elif s["method"] == "svd":
        print(f"  SVD rank: {s.get('svd_rank', 8)}")
        print(f"  FFN sparsity: {s.get('ffn_sparsity', 0.5):.1%}")
        if s.get("layer_sparsity"):
            print(f"  Layer-wise sparsity: {s.get('layer_sparsity')}")
    print(f"  LoRA rank: {s['r']}")
    print(f"  Target modules: {s['target_modules']}")

    return peft_model