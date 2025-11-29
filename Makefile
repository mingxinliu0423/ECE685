.PHONY: setup \
        train-sst2-lora eval-sst2-lora speed-sst2-lora \
        train-sst2-l1 eval-sst2-l1 speed-sst2-l1 \
        train-sst2-prune eval-sst2-prune speed-sst2-prune \
        train-sst2-svd eval-sst2-svd speed-sst2-svd \
        train-imdb-lora eval-imdb-lora speed-imdb-lora \
        train-imdb-l1 eval-imdb-l1 speed-imdb-l1 \
        train-imdb-prune eval-imdb-prune speed-imdb-prune \
        train-imdb-svd eval-imdb-svd speed-imdb-svd \
        train-wikitext2-lora eval-wikitext2-lora speed-wikitext2-lora \
        train-wikitext2-l1 eval-wikitext2-l1 speed-wikitext2-l1 \
        train-wikitext2-prune eval-wikitext2-prune speed-wikitext2-prune \
        train-wikitext2-svd eval-wikitext2-svd speed-wikitext2-svd \
        train-sst2-all eval-sst2-all speed-sst2-all \
        train-imdb-all eval-imdb-all speed-imdb-all \
        train-wikitext2-all eval-wikitext2-all speed-wikitext2-all \
        train-all eval-all speed-all

setup:
	python -m pip install -r requirements.txt

# ========================================
# SST-2 Dataset (4 methods)
# ========================================

# SST-2 - LoRA
train-sst2-lora:
	python -m src.train --config configs/sst2_distilbert_lora.yaml

eval-sst2-lora:
	python -m src.eval --config configs/sst2_distilbert_lora.yaml

speed-sst2-lora:
	python -m src.infer_speed --config configs/sst2_distilbert_lora.yaml

# SST-2 - Sparse LoRA (L1)
train-sst2-l1:
	python -m src.train --config configs/sst2_sparse_lora_l1.yaml

eval-sst2-l1:
	python -m src.eval --config configs/sst2_sparse_lora_l1.yaml

speed-sst2-l1:
	python -m src.infer_speed --config configs/sst2_sparse_lora_l1.yaml

# SST-2 - Sparse LoRA (Prune)
train-sst2-prune:
	python -m src.train --config configs/sst2_sparse_lora_prune.yaml

eval-sst2-prune:
	python -m src.eval --config configs/sst2_sparse_lora_prune.yaml

speed-sst2-prune:
	python -m src.infer_speed --config configs/sst2_sparse_lora_prune.yaml

# SST-2 - Sparse LoRA (SVD)
train-sst2-svd:
	python -m src.train --config configs/sst2_sparse_lora_svd.yaml

eval-sst2-svd:
	python -m src.eval --config configs/sst2_sparse_lora_svd.yaml

speed-sst2-svd:
	python -m src.infer_speed --config configs/sst2_sparse_lora_svd.yaml

# ========================================
# IMDB Dataset (4 methods)
# ========================================

# IMDB - LoRA
train-imdb-lora:
	python -m src.train --config configs/imdb_distilbert_lora.yaml

eval-imdb-lora:
	python -m src.eval --config configs/imdb_distilbert_lora.yaml

speed-imdb-lora:
	python -m src.infer_speed --config configs/imdb_distilbert_lora.yaml

# IMDB - Sparse LoRA (L1)
train-imdb-l1:
	python -m src.train --config configs/imdb_sparse_lora_l1.yaml

eval-imdb-l1:
	python -m src.eval --config configs/imdb_sparse_lora_l1.yaml

speed-imdb-l1:
	python -m src.infer_speed --config configs/imdb_sparse_lora_l1.yaml

# IMDB - Sparse LoRA (Prune)
train-imdb-prune:
	python -m src.train --config configs/imdb_sparse_lora_prune.yaml

eval-imdb-prune:
	python -m src.eval --config configs/imdb_sparse_lora_prune.yaml

speed-imdb-prune:
	python -m src.infer_speed --config configs/imdb_sparse_lora_prune.yaml

# IMDB - Sparse LoRA (SVD)
train-imdb-svd:
	python -m src.train --config configs/imdb_sparse_lora_svd.yaml

eval-imdb-svd:
	python -m src.eval --config configs/imdb_sparse_lora_svd.yaml

speed-imdb-svd:
	python -m src.infer_speed --config configs/imdb_sparse_lora_svd.yaml

# ========================================
# WikiText2 Dataset (4 methods)
# ========================================

# WikiText2 - LoRA
train-wikitext2-lora:
	python -m src.train --config configs/wikitext2_distilbert_lora.yaml

eval-wikitext2-lora:
	python -m src.eval --config configs/wikitext2_distilbert_lora.yaml

speed-wikitext2-lora:
	python -m src.infer_speed --config configs/wikitext2_distilbert_lora.yaml

# WikiText2 - Sparse LoRA (L1)
train-wikitext2-l1:
	python -m src.train --config configs/wikitext2_sparse_lora_l1.yaml

eval-wikitext2-l1:
	python -m src.eval --config configs/wikitext2_sparse_lora_l1.yaml

speed-wikitext2-l1:
	python -m src.infer_speed --config configs/wikitext2_sparse_lora_l1.yaml

# WikiText2 - Sparse LoRA (Prune)
train-wikitext2-prune:
	python -m src.train --config configs/wikitext2_sparse_lora_prune.yaml

eval-wikitext2-prune:
	python -m src.eval --config configs/wikitext2_sparse_lora_prune.yaml

speed-wikitext2-prune:
	python -m src.infer_speed --config configs/wikitext2_sparse_lora_prune.yaml

# WikiText2 - Sparse LoRA (SVD)
train-wikitext2-svd:
	python -m src.train --config configs/wikitext2_sparse_lora_svd.yaml

eval-wikitext2-svd:
	python -m src.eval --config configs/wikitext2_sparse_lora_svd.yaml

speed-wikitext2-svd:
	python -m src.infer_speed --config configs/wikitext2_sparse_lora_svd.yaml

# ========================================
# Batch Operations - SST-2
# ========================================

train-sst2-all:
	@echo "=== Training all SST-2 models (LoRA, L1, Prune, SVD) ==="
	@echo "[1/4] SST-2 LoRA..."
	$(MAKE) train-sst2-lora
	@echo "[2/4] SST-2 Sparse L1..."
	$(MAKE) train-sst2-l1
	@echo "[3/4] SST-2 Sparse Prune..."
	$(MAKE) train-sst2-prune
	@echo "[4/4] SST-2 Sparse SVD..."
	$(MAKE) train-sst2-svd
	@echo "=== SST-2 training complete! ==="

eval-sst2-all:
	@echo "=== Evaluating all SST-2 models (LoRA, L1, Prune, SVD) ==="
	@echo "[1/4] SST-2 LoRA..."
	$(MAKE) eval-sst2-lora
	@echo "[2/4] SST-2 Sparse L1..."
	$(MAKE) eval-sst2-l1
	@echo "[3/4] SST-2 Sparse Prune..."
	$(MAKE) eval-sst2-prune
	@echo "[4/4] SST-2 Sparse SVD..."
	$(MAKE) eval-sst2-svd
	@echo "=== SST-2 evaluation complete! ==="

speed-sst2-all:
	@echo "=== Measuring inference speed for all SST-2 models ==="
	@echo "[1/4] SST-2 LoRA..."
	$(MAKE) speed-sst2-lora
	@echo "[2/4] SST-2 Sparse L1..."
	$(MAKE) speed-sst2-l1
	@echo "[3/4] SST-2 Sparse Prune..."
	$(MAKE) speed-sst2-prune
	@echo "[4/4] SST-2 Sparse SVD..."
	$(MAKE) speed-sst2-svd
	@echo "=== SST-2 speed measurement complete! ==="

# ========================================
# Batch Operations - IMDB
# ========================================

train-imdb-all:
	@echo "=== Training all IMDB models (LoRA, L1, Prune, SVD) ==="
	@echo "[1/4] IMDB LoRA..."
	$(MAKE) train-imdb-lora
	@echo "[2/4] IMDB Sparse L1..."
	$(MAKE) train-imdb-l1
	@echo "[3/4] IMDB Sparse Prune..."
	$(MAKE) train-imdb-prune
	@echo "[4/4] IMDB Sparse SVD..."
	$(MAKE) train-imdb-svd
	@echo "=== IMDB training complete! ==="

eval-imdb-all:
	@echo "=== Evaluating all IMDB models (LoRA, L1, Prune, SVD) ==="
	@echo "[1/4] IMDB LoRA..."
	$(MAKE) eval-imdb-lora
	@echo "[2/4] IMDB Sparse L1..."
	$(MAKE) eval-imdb-l1
	@echo "[3/4] IMDB Sparse Prune..."
	$(MAKE) eval-imdb-prune
	@echo "[4/4] IMDB Sparse SVD..."
	$(MAKE) eval-imdb-svd
	@echo "=== IMDB evaluation complete! ==="

speed-imdb-all:
	@echo "=== Measuring inference speed for all IMDB models ==="
	@echo "[1/4] IMDB LoRA..."
	$(MAKE) speed-imdb-lora
	@echo "[2/4] IMDB Sparse L1..."
	$(MAKE) speed-imdb-l1
	@echo "[3/4] IMDB Sparse Prune..."
	$(MAKE) speed-imdb-prune
	@echo "[4/4] IMDB Sparse SVD..."
	$(MAKE) speed-imdb-svd
	@echo "=== IMDB speed measurement complete! ==="

# ========================================
# Batch Operations - WikiText2
# ========================================

train-wikitext2-all:
	@echo "=== Training all WikiText2 models (LoRA, L1, Prune, SVD) ==="
	@echo "[1/4] WikiText2 LoRA..."
	$(MAKE) train-wikitext2-lora
	@echo "[2/4] WikiText2 Sparse L1..."
	$(MAKE) train-wikitext2-l1
	@echo "[3/4] WikiText2 Sparse Prune..."
	$(MAKE) train-wikitext2-prune
	@echo "[4/4] WikiText2 Sparse SVD..."
	$(MAKE) train-wikitext2-svd
	@echo "=== WikiText2 training complete! ==="

eval-wikitext2-all:
	@echo "=== Evaluating all WikiText2 models (LoRA, L1, Prune, SVD) ==="
	@echo "[1/4] WikiText2 LoRA..."
	$(MAKE) eval-wikitext2-lora
	@echo "[2/4] WikiText2 Sparse L1..."
	$(MAKE) eval-wikitext2-l1
	@echo "[3/4] WikiText2 Sparse Prune..."
	$(MAKE) eval-wikitext2-prune
	@echo "[4/4] WikiText2 Sparse SVD..."
	$(MAKE) eval-wikitext2-svd
	@echo "=== WikiText2 evaluation complete! ==="

speed-wikitext2-all:
	@echo "=== Measuring inference speed for all WikiText2 models ==="
	@echo "[1/4] WikiText2 LoRA..."
	$(MAKE) speed-wikitext2-lora
	@echo "[2/4] WikiText2 Sparse L1..."
	$(MAKE) speed-wikitext2-l1
	@echo "[3/4] WikiText2 Sparse Prune..."
	$(MAKE) speed-wikitext2-prune
	@echo "[4/4] WikiText2 Sparse SVD..."
	$(MAKE) speed-wikitext2-svd
	@echo "=== WikiText2 speed measurement complete! ==="

# ========================================
# Global Batch Operations (All Datasets)
# ========================================

train-all:
	@echo "========================================"
	@echo "Training ALL models across ALL datasets"
	@echo "========================================"
	$(MAKE) train-sst2-all
	@echo ""
	$(MAKE) train-imdb-all
	@echo ""
	$(MAKE) train-wikitext2-all
	@echo "========================================"
	@echo "ALL TRAINING COMPLETE!"
	@echo "========================================"

eval-all:
	@echo "========================================="
	@echo "Evaluating ALL models across ALL datasets"
	@echo "========================================="
	$(MAKE) eval-sst2-all
	@echo ""
	$(MAKE) eval-imdb-all
	@echo ""
	$(MAKE) eval-wikitext2-all
	@echo "========================================"
	@echo "ALL EVALUATION COMPLETE!"
	@echo "========================================"

speed-all:
	@echo "============================================"
	@echo "Measuring speed for ALL models/datasets"
	@echo "============================================"
	$(MAKE) speed-sst2-all
	@echo ""
	$(MAKE) speed-imdb-all
	@echo ""
	$(MAKE) speed-wikitext2-all
	@echo "========================================"
	@echo "ALL SPEED MEASUREMENT COMPLETE!"
	@echo "========================================"
