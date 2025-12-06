import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from peft import PeftConfig, TaskType
from pathlib import Path

import json
import os



class HiraConfig(PeftConfig):
    """
    Configuration class for HIRA (Hadamard High-Rank Adaptation).
    
    HIRA applies element-wise Hadamard product between original weights and
    low-rank adapted weights: W_adapted = W_0 ⊙ (A @ B)
    """
    model_type = None
    
    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: Optional[list] = None,
        bias: str = "none",
        task_type: Optional[TaskType] = None,
        **kwargs,
    ):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules if target_modules is not None else []
        self.bias = bias
        super().__init__(task_type=task_type, **kwargs)

    def to_dict(self) -> dict:
        config = super().to_dict()
        config.update({
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        })
        return config


class HiraLayer(nn.Module):
    """
    HIRA layer that wraps a linear layer with Hadamard product adaptation.
    
    Formula: W_adapted = W_0 ⊙ (A @ B)
    where A: (out_features, rank), B: (rank, in_features)
    """
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.adapter_name = adapter_name
        
        # Store original weight dimensions
        if hasattr(base_layer, "weight"):
            out_features, in_features = base_layer.weight.shape
        else:
            raise ValueError("Base layer must have a weight attribute")
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Initialize low-rank matrices A and B
        # A: (out_features, r), B: (r, in_features)
        # Initialize such that (alpha/r) * (A @ B) approximates identity (matrix of ones)
        # This ensures W_adapted ≈ W_0 initially
        # If A and B have all elements = a, then A @ B has all elements = r * a^2
        # We want (alpha/r) * r * a^2 = 1, so a^2 = 1/alpha, hence a = 1/sqrt(alpha)
        init_scale = 1.0 / (alpha ** 0.5)
        self.hira_A = nn.Parameter(torch.ones(out_features, r) * init_scale)
        self.hira_B = nn.Parameter(torch.ones(r, in_features) * init_scale)
        
        # Add small noise to break symmetry and allow learning
        with torch.no_grad():
            self.hira_A.add_(torch.randn_like(self.hira_A) * 0.01)
            self.hira_B.add_(torch.randn_like(self.hira_B) * 0.01)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Mark base layer weight as non-trainable
        if hasattr(base_layer, "weight"):
            base_layer.weight.requires_grad = False
        if hasattr(base_layer, "bias") and base_layer.bias is not None:
            base_layer.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with HIRA adaptation.
        
        Computes: output = (W_0 ⊙ (scaling * A @ B)) @ x + b
        """
        # Compute low-rank adaptation: delta_W = A @ B
        delta_W = torch.matmul(self.hira_A, self.hira_B)  # (out_features, in_features)
        
        # Apply scaling
        delta_W = self.scaling * delta_W
        
        # Apply dropout to the adaptation
        delta_W = self.dropout(delta_W)
        
        # Get original weight
        original_weight = self.base_layer.weight
        
        # Compute Hadamard product: W_adapted = W_0 ⊙ delta_W
        adapted_weight = original_weight * delta_W
        
        # Apply adapted linear transformation
        if self.base_layer.bias is not None:
            output = torch.nn.functional.linear(x, adapted_weight, self.base_layer.bias)
        else:
            output = torch.nn.functional.linear(x, adapted_weight)
        
        return output


def _get_hira_adapter_candidates(model) -> Dict[str, nn.Module]:
    """Find all linear layers that can be adapted with HIRA."""
    adapter_candidates = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip if already adapted
            if any(isinstance(m, HiraLayer) for m in module.modules()):
                continue
            adapter_candidates[name] = module
    
    return adapter_candidates


def _replace_module_with_hira(
    parent: nn.Module,
    name: str,
    module: nn.Module,
    adapter_name: str,
    config: HiraConfig,
) -> None:
    """Replace a module with a HIRA-wrapped version."""
    hira_layer = HiraLayer(
        base_layer=module,
        adapter_name=adapter_name,
        r=config.r,
        alpha=config.alpha,
        dropout=config.dropout,
    )
    
    setattr(parent, name, hira_layer)


def _match_target_modules(name: str, target_modules: list) -> bool:
    """Check if a module name matches any target module pattern."""
    for target in target_modules:
        # Handle both exact matches and substring matches
        if target == name or target in name or name.endswith("." + target):
            return True
    return False


def inject_hira_adapters(model: nn.Module, config: HiraConfig, adapter_name: str = "default"):
    """
    Inject HIRA adapters into the model's target modules.
    
    Args:
        model: The base model to adapt
        config: HIRA configuration
        adapter_name: Name of the adapter
    """
    adapter_candidates = _get_hira_adapter_candidates(model)
    
    if not config.target_modules:
        raise ValueError("target_modules must be specified in HIRA config")
    
    # First, collect all modules that need to be replaced
    modules_to_replace = []
    for name, module in model.named_modules():
        # Skip if already a HIRA layer
        if isinstance(module, HiraLayer):
            continue
        
        # Check if this module matches target modules
        if isinstance(module, nn.Linear):
            if _match_target_modules(name, config.target_modules):
                modules_to_replace.append((name, module))
    
    if len(modules_to_replace) == 0:
        available = list(adapter_candidates.keys())[:10]
        raise ValueError(
            f"No modules matching {config.target_modules} were found. "
            f"Available modules: {available}..."
        )
    
    # Now replace all collected modules
    replaced_count = 0
    for name, module in modules_to_replace:
        # Find the parent module and child name
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        
        # Verify we found the right module
        if getattr(parent, child_name) is not module:
            continue
        
        # Replace the module
        _replace_module_with_hira(
            parent,
            child_name,
            module,
            adapter_name,
            config,
        )
        replaced_count += 1
    
    return replaced_count


def compute_hira_l1_loss(model, layer_weights: Optional[Dict[str, float]] = None):
    """
    Compute L1 loss for HIRA parameters with optional layer-wise weighting.

    Args:
        model: The model with HIRA parameters
        layer_weights: Optional dict mapping layer patterns to weight multipliers.
                      E.g., {"layer.0": 0.5, "layer.1": 0.5, "layer.2": 1.0}
                      Lower multiplier = less sparsity pressure
    """
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if "hira_A" in name or "hira_B" in name:
            if param.requires_grad:
                # Determine layer-specific weight
                weight = 1.0
                if layer_weights:
                    for pattern, w in layer_weights.items():
                        if pattern in name:
                            weight = w
                            break

                l1_loss += weight * torch.abs(param).sum()

    return l1_loss


def apply_hira_magnitude_pruning(
    model,
    prune_ratio: float,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Apply magnitude-based pruning to HIRA parameters.

    Args:
        model: Model with HIRA adapters
        prune_ratio: Fraction of parameters to prune (0.0 to 1.0)
        device: Device to place masks on

    Returns:
        Dictionary mapping parameter names to pruning masks
    """
    if prune_ratio <= 0.0:
        return {}

    if prune_ratio >= 1.0:
        raise ValueError(f"prune_ratio must be < 1.0, got {prune_ratio}")
    
    hira_params = []
    hira_names = []

    # Collect HIRA parameters
    for name, param in model.named_parameters():
        if ("hira_A" in name or "hira_B" in name) and param.requires_grad:
            hira_params.append(param.data.view(-1))  # Flatten to 1D
            hira_names.append(name)

    if len(hira_params) == 0:
        print("Warning: No HIRA parameters found for pruning")
        return {}
    
    # Concatenate all HIRA parameters to compute global threshold
    all_weights = torch.cat(hira_params)
    threshold = torch.quantile(torch.abs(all_weights), prune_ratio)

    # Create and apply masks
    masks = {}
    total_pruned = 0

    for name, param in model.named_parameters():
        if ("hira_A" in name or "hira_B" in name) and param.requires_grad:
            # 1 if |weight| >= threshold, 0 otherwise
            mask = (torch.abs(param.data) >= threshold).float().to(device)
            masks[name] = mask

            # Apply mask to zero out pruned weights
            param.data *= mask

            pruned_count = (mask == 0).sum().item()
            total_pruned += pruned_count

    actual_sparsity = total_pruned / all_weights.numel()
    print(f"Actual HIRA sparsity achieved: {actual_sparsity:.2%}")

    return masks


def get_hira_stats(model) -> Dict[str, float]:
    """
    Compute statistics for HIRA parameters.

    Args:
        model: Model with HIRA adapters

    Returns:
        Dictionary with HIRA parameter statistics
    """
    total_params = 0
    zero_params = 0
    hira_layers = 0

    for name, param in model.named_parameters():
        if "hira_A" in name or "hira_B" in name:
            if param.requires_grad:
                total_params += param.numel()
                zero_params += (torch.abs(param.data) < 1e-8).sum().item()
                if "hira_A" in name:
                    hira_layers += 1

    if total_params == 0:
        return {
            "total_hira_params": 0,
            "zero_params": 0,
            "nonzero_params": 0,
            "sparsity_ratio": 0.0,
            "density_ratio": 0.0,
            "num_hira_layers": 0,
        }

    sparsity = zero_params / total_params

    return {
        "total_hira_params": total_params,
        "zero_params": zero_params,
        "nonzero_params": total_params - zero_params,
        "sparsity_ratio": sparsity,
        "density_ratio": 1.0 - sparsity,
        "num_hira_layers": hira_layers,
    }


def apply_hira_pruning_masks(model, masks: Dict[str, torch.Tensor]) -> None:
    """
    Apply pruning masks to HIRA parameters after each update.

    Args:
        model: Model with HIRA adapters
        masks: Dictionary mapping parameter names to pruning masks
    """
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name]


def build_hira_adapter(model, cfg: Dict[str, Any]):
    """
    Build HIRA adapter for a model.
    
    Similar structure to build_sparse_lora_adapter, supporting various
    training configurations and utilities.
    
    Args:
        model: Base model to adapt
        cfg: Configuration dictionary containing "hira" key with HIRA settings
        
    Returns:
        Model with HIRA adapters injected
    """
    h = cfg.get("hira", {})
    
    # Create HIRA config
    peft_config = HiraConfig(
        r=h.get("r", 8),
        alpha=h.get("alpha", 16),
        dropout=h.get("dropout", 0.0),
        target_modules=h.get("target_modules", []),
        bias="none",
        task_type="SEQ_CLS",
    )
    
    # Inject HIRA adapters
    replaced = inject_hira_adapters(model, peft_config, adapter_name="default")
    
    # Attach HIRA configuration to model for use during training
    model.hira_config = {
        "r": peft_config.r,
        "alpha": peft_config.alpha,
        "dropout": peft_config.dropout,
        "target_modules": peft_config.target_modules,
        "l1_lambda": h.get("l1_lambda", 0.0),
        "prune_ratio": h.get("prune_ratio", 0.0),
    }
    
    print(f"Created HIRA adapter:")
    print(f"  Rank: {peft_config.r}")
    print(f"  Alpha: {peft_config.alpha}")
    print(f"  Dropout: {peft_config.dropout}")
    print(f"  Target modules: {peft_config.target_modules}")
    if h.get("l1_lambda", 0.0) > 0.0:
        print(f"  L1 lambda: {h.get('l1_lambda', 0.0)}")
    if h.get("prune_ratio", 0.0) > 0.0:
        print(f"  Prune ratio: {h.get('prune_ratio', 0.0):.1%}")
    print(f"  Replaced {replaced} modules")
    
    return model


def save_hira_adapter_config(model, output_dir: str, base_model_name: Optional[str] = None) -> None:
    """
    Save HIRA adapter configuration to adapter_config.json in the output directory.
    
    This function saves the HIRA configuration in a format compatible with
    how PEFT adapters save their config, making it easier to load later.
    
    Args:
        model: Model with HIRA adapters (must have hira_config attribute)
        output_dir: Directory path where adapter_config.json should be saved
        base_model_name: Optional base model name/path (if not provided, tries to infer from model)
    """
    if not hasattr(model, "hira_config"):
        raise ValueError("Model does not have hira_config attribute. Ensure build_hira_adapter was called.")
    
    config = model.hira_config.copy()
    
    # Try to get base model name from various sources
    base_model = base_model_name
    if base_model is None:
        # Try to get from model attributes
        base_model = getattr(model, "base_model_name_or_path", None)
        if base_model is None and hasattr(model, "config"):
            # Try to get from model config
            model_config = model.config
            if hasattr(model_config, "_name_or_path"):
                base_model = model_config._name_or_path
            elif hasattr(model_config, "name_or_path"):
                base_model = model_config.name_or_path
    
    # Add metadata to match PEFT adapter config format
    adapter_config = {
        "peft_type": "HIRA",
        "task_type": "SEQ_CLS",
        "base_model_name_or_path": base_model,
        "inference_mode": False,
        "r": config.get("r", 8),
        "alpha": config.get("alpha", 16),
        "dropout": config.get("dropout", 0.0),
        "target_modules": config.get("target_modules", []),
        "bias": "none",
        "l1_lambda": config.get("l1_lambda", 0.0),
        "prune_ratio": config.get("prune_ratio", 0.0),
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_file = output_path / "adapter_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"Saved HIRA adapter config to {config_file}")

