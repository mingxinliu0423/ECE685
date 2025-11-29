## Evaluation Results Analysis

### SST-2 Dataset Results (WikiText-2 Training)

| Method              | Accuracy | F1 Score | Loss   | Sparsity | Params      | VRAM (MB) | Time (ms/step) |
|---------------------|----------|----------|--------|----------|-------------|-----------|----------------|
| LoRA (Baseline)     | 0.9060   | 0.9087   | 0.2674 | 0%       | 294,912     | 1100.6    | 78.2           |
| Sparse LoRA (L1)    | 0.8693   | 0.8707   | 0.2948 | 23.5%    | 225,473     | 833.8     | 87.0           |
| Sparse LoRA (Prune) | 0.9037   | 0.9065   | 0.2966 | 50%      | 147,456     | 835.0     | 79.3           |
| Sparse LoRA (SVD)   | 0.8945   | 0.8938   | 0.4903 | 69.2%*   | 294,912     | 845.4     | 90.1           |

*SVD applies 69.2% average FFN structured sparsity (layer-wise: 40-95%) but maintains full parameter count.

### IMDB Dataset Results

| Method              | Accuracy | F1 Score | Loss   | Sparsity | Params      | VRAM (MB) | Latency (ms) |
|---------------------|----------|----------|--------|----------|-------------|-----------|--------------|
| LoRA (Baseline)     | 0.8686   | 0.8709   | 0.3022 | 0%       | 294,912     | 415.95    | 10.290       |
| Sparse LoRA (L1)    | 0.8483   | 0.8495   | 0.3441 | 15.5%    | 249,126     | 415.95    | 9.518        |
| Sparse LoRA (Prune) | 0.8697   | 0.8716   | 0.3026 | 50%      | 147,456     | 415.95    | 10.176       |
| Sparse LoRA (SVD)   | 0.8458   | 0.8381   | 0.5404 | 69.2%*   | 294,912     | 415.95    | 9.624        |

*SVD applies 69.2% average FFN structured sparsity (layer-wise: 40-95%) but maintains full parameter count.
**Latency values represent median inference time per batch (50 batches measured).

### Method Descriptions

1. **LoRA (Baseline)** - Standard low-rank adaptation without sparsity. Uses rank-8 decomposition with alpha=16 and 5% dropout.

2. **Sparse LoRA (L1)** - Applies L1 regularization (λ=1e-3) during training to encourage weight sparsity, achieving 23.5% sparsity and reducing parameters to 225,473.

3. **Sparse LoRA (Prune)** - Post-training magnitude-based pruning with 50% pruning ratio, reducing parameters to 147,456.

4. **Sparse LoRA (SVD)** - Implements the SparseLoRA paper (ICML 2025) method using a training-free SVD sparsity estimator. Pre-computes low-rank approximations (W ≈ W_A @ W_B, rank=4) of pretrained weights via SVD decomposition. During training, dynamically selects important channels per batch using L2 norm criterion on SVD-estimated activations, applying layer-wise structured sparsity to FFN layers (40-95%, avg 69.2%). Unlike L1/Prune methods that sparsify LoRA parameters, this approach sparsifies the forward computation of backbone weights while keeping LoRA adapters fully dense. Uses 20% warmup steps for stability.

### Key Findings

#### SST-2 Dataset (WikiText-2 Training)

**Performance:**
- **Baseline LoRA** achieves best accuracy (90.6%) and F1 (90.87%), serving as the upper bound.
- **Sparse Prune** offers best accuracy-efficiency tradeoff: 90.37% accuracy (-0.23%) with 50% parameter reduction and minimal speed overhead.
- **Sparse SVD** shows moderate degradation: 89.45% accuracy (-1.15%) with 69.2% dynamic sparsity. Not suitable for small models/short sequences.
- **Sparse L1** shows significant degradation: 86.93% accuracy (-3.67%) with only 23.5% sparsity. L1 regularization (λ=1e-3) may be too aggressive.

**Efficiency:**
- **VRAM Reduction:** L1/Prune reduce VRAM by ~24% (1100.6 → 833-835 MB). SVD reduces by ~23% (1100.6 → 845.4 MB).
- **Speed:** Baseline LoRA fastest at 78.2 ms/step. Prune maintains similar speed (79.3 ms/step, +1.4%). SVD slower at 90.1 ms/step (+15.2%) due to SVD estimation overhead.
- **Parameter Reduction:** Pruning achieves 50% reduction (294,912 → 147,456) with minimal accuracy loss, enabling faster inference. SVD maintains full parameter count.

#### IMDB Dataset

**Performance:**
- **Baseline LoRA** achieves 86.86% accuracy and 87.09% F1, establishing the upper bound.
- **Sparse Prune** achieves best performance: 86.97% accuracy (+0.11% vs baseline!) and 87.16% F1 with 50% sparsity.
- **Sparse L1** shows moderate degradation: 84.83% accuracy (-2.03%) with 15.5% sparsity and 249,126 parameters.
- **Sparse SVD** shows significant degradation: 84.58% accuracy (-2.28%) with 69.2% dynamic FFN sparsity and higher loss (0.5404).

**Efficiency:**
- **VRAM Usage:** All methods use identical VRAM (415.95 MB) due to similar model architecture and batch processing.
- **Inference Speed:** L1 is fastest (9.518 ms median), followed by SVD (9.624 ms) and Prune (10.176 ms). Baseline is slowest (10.290 ms).
- **Parameter Reduction:** Prune achieves 50% reduction (294,912 → 147,456). L1 achieves 15.5% reduction (294,912 → 249,126). SVD maintains full parameter count but applies dynamic structured sparsity.

**Surprising Finding:**
On IMDB, **Sparse Prune actually outperforms the baseline** (+0.11% accuracy), suggesting that 50% magnitude pruning acts as effective regularization, removing noise and improving generalization. This contrasts with SST-2 where pruning showed slight degradation (-0.23%).

**Conclusion:**
For DistilBERT classification tasks, **Sparse Prune is the clear winner** across both datasets - achieving near-baseline or better accuracy with 50% fewer parameters and negligible speed overhead. The SVD method, designed for large-scale language models with long sequences, is not well-suited for small models like DistilBERT on short-sequence classification tasks. L1 regularization shows inconsistent sparsity levels (23.5% on SST-2 vs 15.5% on IMDB) and may require dataset-specific tuning.
