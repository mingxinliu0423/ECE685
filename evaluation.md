## Evaluation Results Analysis

### Performance Comparison

| Method              | Accuracy | F1 Score | Loss   | Configuration                 |
|---------------------|----------|----------|--------|------------------------------|
| LoRA (Baseline)     | 0.9060   | 0.9087   | 0.2674 | r = 8, α = 16, dropout = 0.05 |
| Sparse LoRA (L1)    | 0.9060   | 0.9089   | 0.2594 | L1 λ = 1e-5, r = 8, α = 16   |
| Sparse LoRA (Prune) | 0.9025   | 0.9055   | 0.2906 | 30% pruning, r = 8, α = 16   |

### Key Findings

1. **Sparse LoRA (L1) – Best Overall Performance**
   - Maintains accuracy at **0.9060** (equal to baseline)
   - Highest F1 score at **0.9089** (*+0.02% vs baseline*)
   - Lowest loss at **0.2594** (*-3.0% vs baseline*)
   - Uses L1 regularization (**λ = 1e-5**) to encourage sparsity during training  
   - **Conclusion:** L1 regularization successfully induces sparsity without sacrificing performance.

2. **Standard LoRA – Baseline Reference**
   - Solid performance across all metrics
   - No sparsity constraints
   - Trained for **2 epochs**
   - Serves as a good reference point for comparing sparse methods.

3. **Sparse LoRA (Prune) – Slight Performance Degradation**
   - Accuracy drops to **0.9025** (*-0.39% vs baseline*)
   - F1 score drops to **0.9055** (*-0.35% vs baseline*)
   - Highest loss at **0.2906** (*+8.7% vs baseline*)
   - Applies **30% magnitude pruning** after training
   - Trained for **3 epochs** (different from others)
   - **Conclusion:** Post-training pruning has a small but noticeable negative impact on performance.
