import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .svd_estimator import FFNEstimator
from .sparsity_selector import select_ffn_channels, get_layer_sparsity_ratio


class SparseFFN(nn.Module):
    def __init__(
        self,
        original_ffn,
        estimator: Optional[FFNEstimator] = None,
        sparsity_ratio: float = 0.5,
        layer_name: str = "",
        sparse_warmup_steps: int = 0,
        layer_sparsity_config: Optional[dict] = None,
    ):
        super().__init__()

        # Store original FFN components
        self.lin1 = original_ffn.lin1
        self.lin2 = original_ffn.lin2
        self.dropout = original_ffn.dropout
        self.activation = original_ffn.activation

        # Sparse configuration
        self.estimator = estimator
        self.default_sparsity = sparsity_ratio
        self.layer_name = layer_name
        self.sparse_warmup_steps = sparse_warmup_steps
        self.layer_sparsity_config = layer_sparsity_config

        # Training state
        self.current_step = 0
        self.training_mode = True

    def get_effective_sparsity(self) -> float:
        return get_layer_sparsity_ratio(
            self.layer_name,
            self.default_sparsity,
            self.layer_sparsity_config
        )

    def should_apply_sparsity(self) -> bool:
        # Only apply during training
        if not self.training_mode:
            return False

        # Check warmup
        if self.current_step < self.sparse_warmup_steps:
            return False

        # Check if estimator is available
        if self.estimator is None:
            return False

        return True

    def dense_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

    def sparse_forward(self, x: torch.Tensor) -> torch.Tensor:
        sparsity_ratio = self.get_effective_sparsity()
        # If sparsity is 0, use dense computation
        if sparsity_ratio == 0.0:
            return self.dense_forward(x)

        # Step 1: Estimate intermediate activations using SVD
        with torch.no_grad():
            intermediate_approx = self.estimator(x)

            # Step 2: Select important channels based on estimated activations
            selected_channels = select_ffn_channels(
                intermediate_approx,
                sparsity_ratio
            )

        # Step 3 & 4: Compute actual intermediate activations
        intermediate = self.lin1(x) 
        intermediate = self.activation(intermediate) # Silu

        # Step 5: Sparse computation on lin2
        # Only compute on selected channels
        batch_size, seq_len, intermediate_size = intermediate.shape
        hidden_size = self.lin2.out_features

        # Extract selected channels
        sparse_intermediate = intermediate[:, :, selected_channels]  # [batch, seq, num_selected]

        # Get corresponding weights from lin2
        sparse_weight = self.lin2.weight[:, selected_channels]  # [hidden_size, num_selected]

        # Sparse matrix multiplication
        output = F.linear(sparse_intermediate, sparse_weight, self.lin2.bias)

        # Apply dropout
        output = self.dropout(output)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Increment step counter during training
        if self.training_mode:
            self.current_step += 1

        # Choose sparse or dense forward
        if self.should_apply_sparsity():
            return self.sparse_forward(x)
        else:
            return self.dense_forward(x)

    def train(self, mode: bool = True):
        super().train(mode)
        self.training_mode = mode
        return self

    def eval(self):
        return self.train(False)


def replace_ffn_with_sparse(
    model,
    estimators: dict,
    sparsity_ratio: float = 0.5,
    sparse_warmup_steps: int = 0,
    layer_sparsity_config: Optional[dict] = None,
):
    replaced_count = 0
    ffn_estimators = estimators.get("ffn", {})

    # Find and replace FFN modules
    for name, module in model.named_modules():
        # Look for FFN modules
        if "ffn" in name.lower() and not any(x in name.lower() for x in ["lin1", "lin2", "dropout", "activation"]):
            estimator = None
            for est_name, est in ffn_estimators.items():
                if name in est_name or est_name in name:
                    estimator = est
                    break

            sparse_ffn = SparseFFN(
                original_ffn=module,
                estimator=estimator,
                sparsity_ratio=sparsity_ratio,
                layer_name=name,
                sparse_warmup_steps=sparse_warmup_steps,
                layer_sparsity_config=layer_sparsity_config,
            )


            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], sparse_ffn)

            replaced_count += 1
            print(f"  Replaced: {name}")

    return replaced_count


def get_sparsity_statistics(model) -> dict:
    stats = {
        "total_sparse_ffn": 0,
        "active_sparse_ffn": 0,
        "avg_sparsity": 0.0,
        "layer_sparsities": {},
    }

    total_sparsity = 0.0

    for name, module in model.named_modules():
        if isinstance(module, SparseFFN):
            stats["total_sparse_ffn"] += 1

            if module.should_apply_sparsity():
                stats["active_sparse_ffn"] += 1

            sparsity = module.get_effective_sparsity()
            stats["layer_sparsities"][name] = sparsity
            total_sparsity += sparsity

    if stats["total_sparse_ffn"] > 0:
        stats["avg_sparsity"] = total_sparsity / stats["total_sparse_ffn"]

    return stats
