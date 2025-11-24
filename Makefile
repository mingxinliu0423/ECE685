.PHONY: setup train-sst2 eval-sst2 speed-sst2

setup:
	python -m pip install -r requirements.txt

train-sst2:
	python -m src.train --config configs/sst2_distilbert_lora.yaml

eval-sst2:
	python -m src.eval --config configs/sst2_distilbert_lora.yaml

speed-sst2:
	python -m src.infer_speed --config configs/sst2_distilbert_lora.yaml