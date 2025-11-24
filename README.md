# ECE685 Project — DistilBERT LoRA Baseline

Minimal, reproducible LoRA fine-tuning baseline on DistilBERT for GLUE SST-2. Provides a clean starting point for teammates extending with SparseLoRA (Person B) or HiRA (Person C).

## Resources
- SST-2 (GLUE): https://huggingface.co/datasets/glue/viewer/sst2 - programmatic id `("glue", "sst2")`
- IMDB (optional smoke test): https://huggingface.co/datasets/stanfordnlp/imdb - programmatic id `"stanfordnlp/imdb"`
- WikiText-2 subset (optional smoke test): https://huggingface.co/datasets/mindchain/wikitext2 - programmatic id `"mindchain/wikitext2"`
- DistilBERT base uncased: https://huggingface.co/distilbert-base-uncased - programmatic id `"distilbert-base-uncased"`

## Setup & Usage
```bash
python -m pip install -r requirements.txt
make train-sst2
make eval-sst2
make speed-sst2
```
`make train-sst2` trains a LoRA adapter once (2 epochs) to validate the loop, storing outputs under `outputs/sst2_distilbert_lora/`. Evaluation loads the saved adapter from `outputs/.../ckpt/adapter`. Speed testing benchmarks validation batches and reports latency/VRAM.

## Offline / Manual Data or Model Drop-in
If automated downloads fail, follow these steps:
1. Create `data/glue/sst2/` and place `train.tsv` and `dev.tsv` files with `sentence<TAB>label` columns (0/1 labels). A `data/README.md` file will be generated with these instructions the first time a failure is detected.
2. (If needed) Download `distilbert-base-uncased` from Hugging Face, place it under `models/distilbert-base-uncased/`, and set `TRANSFORMERS_CACHE=models/` before running any scripts.

## Extending for B and C
- Person B (SparseLoRA): implement `build_sparse_lora_adapter` in `src/adapters/sparse_lora/sparse_lora.py` and swap the import used in `src/model.py`.
- Person C (HiRA): implement `build_hira_adapter` in `src/adapters/hira/hira_adapter.py` and swap the import in `src/model.py`.
Keep the training/eval loops untouched to ensure changes stay comparable to this baseline.