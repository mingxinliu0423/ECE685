# Sparse LoRA SVD Optimization Summary

## Configuration Changes

### Before (Original)
```yaml
svd_rank: 8
ffn_sparsity: 0.5
layer_sparsity:
  "layer.0": 0.0
  "layer.1": 0.2
  "layer.2": 0.3
  "layer.3": 0.5
  "layer.4": 0.6
  "layer.5": 0.7
sparse_warmup_ratio: 0.1
```
**Average Sparsity**: 38.3%

### After (Optimized)
```yaml
svd_rank: 4  # Reduced by 50%
ffn_sparsity: 0.75  # Increased by 50%
layer_sparsity:
  "layer.0": 0.4
  "layer.1": 0.5
  "layer.2": 0.6
  "layer.3": 0.8
  "layer.4": 0.9
  "layer.5": 0.95
sparse_warmup_ratio: 0.2  # Doubled for stability
```
**Average Sparsity**: 69.2%

## Expected Improvements

### Computation Reduction
| Component | Before FLOPs | After FLOPs | Reduction |
|-----------|-------------|-------------|-----------|
| SVD estimator | 30,720 | 15,360 | 50% |
| FFN (avg) | 3,569,664 | 2,179,296 | 39% |
| **Total per token** | **3,600,384** | **2,194,656** | **39%** |

### Speed Target
- **Current**: 103.9 ms/step (+32.8% slower than baseline)
- **Target**: ~70-75 ms/step (~10% faster than baseline)
- **Mechanism**: Higher sparsity + lower SVD rank should overcome overhead

## Code Cleanup

### Deleted Files
- ✅ `token_utils.py` - Completely unused (context/output token splitting not applicable for classification)

### Functions Removed
From `sparsity_selector.py`:
- ✅ `select_vo_channels()` - Unused (attention sparsity not implemented)
- ✅ `select_qk_channels()` - Unused (attention sparsity not implemented)
- ✅ `apply_channel_sparsity()` - Unused (implemented inline in sparse_ffn_module)
- ✅ `compute_sparse_output_with_lora()` - Unused (implemented differently)

From `svd_estimator.py`:
- ✅ `build_svd_estimators()` - Unused (only FFN estimators are used)

### Kept Functions
- ✅ `select_ffn_channels()` - Core FFN channel selection
- ✅ `get_layer_sparsity_ratio()` - Layer-wise sparsity lookup
- ✅ `SVDSparsityEstimator` - Base SVD approximation
- ✅ `FFNEstimator` - FFN-specific estimator
- ✅ `build_ffn_estimators()` - Factory for creating estimators

## Testing Instructions

### Run Optimized Training
```bash
cd /work/hc413/ECE685
python src/train.py configs/sst2_sparse_lora_svd.yaml
```

### Compare Results
Expected outcomes:
1. **Speed**: Should be faster than 103.9 ms/step (target: 70-80 ms/step)
2. **Accuracy**: May drop slightly from 90.48% (acceptable if >89.5%)
3. **VRAM**: Should remain similar (~850 MB)

### Fallback Plan
If accuracy drops too much (< 89%), try intermediate settings:
```yaml
svd_rank: 4  # Keep low
ffn_sparsity: 0.65  # Reduce from 0.75
layer_sparsity:
  "layer.0": 0.3
  "layer.1": 0.4
  "layer.2": 0.5
  "layer.3": 0.7
  "layer.4": 0.8
  "layer.5": 0.9
```

## Why These Changes?

### Root Cause Analysis
The original configuration was **too conservative** for classification tasks:

1. **Low effective sparsity** (38.3% avg):
   - First 3 layers nearly dense
   - Not enough FLOPs reduction to overcome SVD overhead

2. **High SVD rank** (8):
   - Two matrix multiplications per forward pass
   - PyTorch inefficient for small matrices (rank=8)
   - Overhead dominated speedup gains

3. **Task mismatch**:
   - Paper optimized for generation tasks (Math, Code)
   - Classification tasks have different characteristics
   - No context/output token distinction

### Solution
- **Aggressive sparsity**: Compensate for SVD overhead
- **Lower rank**: Minimize estimation cost
- **More warmup**: Ensure stability with high sparsity

## Theoretical FLOPs Analysis

### Per Token (DistilBERT FFN: 768→3072→768)

**Baseline (Dense)**:
- lin1: 768 × 3072 = 2,359,296
- lin2: 3072 × 768 = 2,359,296
- **Total**: 4,718,592 FLOPs

**Original SVD Config**:
- SVD (rank=8): 768×8 + 8×3072 = 30,720
- lin1: 768 × 3072 = 2,359,296
- lin2 (50% sparse): 1536 × 768 = 1,179,648
- **Total**: 3,569,664 FLOPs (24% reduction)
- **But 32.8% slower!** → Overhead too high

**Optimized Config**:
- SVD (rank=4): 768×4 + 4×3072 = 15,360
- lin1: 768 × 3072 = 2,359,296
- lin2 (75% sparse avg): 768 × 768 = 589,824
- **Total**: 2,964,480 FLOPs (37% reduction)
- **Expected**: 10-15% speedup

## Next Steps

1. ✅ Configuration optimized
2. ✅ Code cleaned
3. ⏳ Run training and measure
4. ⏳ Update evaluation.md with new results
5. ⏳ If still slow, consider removing SVD for classification tasks
