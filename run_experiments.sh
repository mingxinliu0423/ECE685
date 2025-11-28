#!/bin/bash
# Simple Sparse LoRA Experiments
# Runs: Baseline LoRA, Sparse LoRA (L1), Sparse LoRA (Prune)

set -e

echo "=========================================="
echo "Sparse LoRA Experiments"
echo "=========================================="

# 1. Baseline LoRA
echo ""
echo "[1/3] Training Baseline LoRA..."
python -m src.train --config configs/sst2_distilbert_lora.yaml

# 2. Sparse LoRA with L1 Regularization
echo ""
echo "[2/3] Training Sparse LoRA (L1 regularization)..."
python -m src.train --config configs/sst2_sparse_lora_l1.yaml

# 3. Sparse LoRA with Magnitude Pruning
echo ""
echo "[3/3] Training Sparse LoRA (Magnitude Pruning)..."
python -m src.train --config configs/sst2_sparse_lora_prune.yaml

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
python -c "
import json
from pathlib import Path

dirs = [
    'outputs/sst2_distilbert_lora',
    'outputs/sst2_sparse_lora_l1',
    'outputs/sst2_sparse_lora_prune'
]

print('\n| Method | Accuracy | Sparsity | Time(ms) |')
print('|--------|----------|----------|----------|')

for d in dirs:
    metrics_file = Path(d) / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
            name = d.split('/')[-1].replace('sst2_', '').replace('_', ' ').title()
            acc = data['val_acc']
            sparsity = data.get('final_sparsity', 0.0)
            time = data['avg_step_time_ms']
            print(f'| {name[:20]:20} | {acc:.4f} | {sparsity:.2%} | {time:.1f} |')
    else:
        print(f'| {d:20} | N/A | N/A | N/A |')
"
