.PHONY: setup train-sst2 eval-sst2 speed-sst2 \
        train-sparse-l1 eval-sparse-l1 speed-sparse-l1 \
        train-sparse-prune eval-sparse-prune speed-sparse-prune

setup:
	python -m pip install -r requirements.txt

# ========== LoRA ==========
train-sst2:
	python -m src.train --config configs/sst2_distilbert_lora.yaml

eval-sst2:
	python -m src.eval --config configs/sst2_distilbert_lora.yaml

speed-sst2:
	python -m src.infer_speed --config configs/sst2_distilbert_lora.yaml

# ========== Sparse LoRA (L1) ==========
train-sparse-l1:
	python -m src.train --config configs/sst2_sparse_lora_l1_test.yaml

eval-sparse-l1:
	python -m src.eval --config configs/sst2_sparse_lora_l1_test.yaml

speed-sparse-l1:
	python -m src.infer_speed --config configs/sst2_sparse_lora_l1_test.yaml

# ========== Sparse LoRA (prune) ==========
train-sparse-prune:
	python -m src.train --config configs/sst2_sparse_lora_prune_test.yaml

eval-sparse-prune:
	python -m src.eval --config configs/sst2_sparse_lora_prune_test.yaml

speed-sparse-prune:
	python -m src.infer_speed --config configs/sst2_sparse_lora_prune_test.yaml

# ========== Train All ==========
train-all:
	@echo "=== training lora,Sparse LoRA (L1),and Sparse LoRA (Prune) ==="
	@echo "1/3  LoRA..."
	$(MAKE) train-sst2
	@echo "2/3  Sparse LoRA (L1)..."
	$(MAKE) train-sparse-l1
	@echo "3/3  Sparse LoRA (Prune)..."
	$(MAKE) train-sparse-prune
	@echo " === All training done! ==="

# ========== Eval All ==========
eval-all:
	@echo "=== evaluating lora,Sparse LoRA (L1),and Sparse LoRA (Prune) ==="
	@echo "1/3  LoRA..."
	$(MAKE) eval-sst2
	@echo "2/3  Sparse LoRA (L1)..."
	$(MAKE) eval-sparse-l1
	@echo "3/3  Sparse LoRA (Prune)..."
	$(MAKE) eval-sparse-prune
	@echo " === All evaluation done! ==="


# ========== Speed All ==========
speed-all:	
	@echo "=== measuring inference speed of lora,Sparse LoRA (L1),and Sparse LoRA (Prune) ==="
	@echo "1/3  LoRA..."
	$(MAKE) speed-sst2
	@echo "2/3  Sparse LoRA (L1)..."
	$(MAKE) speed-sparse-l1
	@echo "3/3  Sparse LoRA (Prune)..."
	$(MAKE) speed-sparse-prune
	@echo " === All speed measurement done! ==="