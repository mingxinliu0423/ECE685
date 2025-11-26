# ECE685 Project — DistilBERT LoRA Experiment

Minimal, reproducible LoRA fine-tuning on DistilBERT for GLUE SST-2. Intended as a clean starting point for adapter experiments (e.g., SparseLoRA, HiRA) while keeping the training/eval loop stable.

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
`make train-sst2` runs a 2-epoch LoRA fine-tune and saves outputs under `outputs/sst2_distilbert_lora/`. Evaluation loads the saved adapter from `outputs/.../ckpt/adapter`. Speed testing benchmarks validation batches and reports latency/VRAM.

## Offline / Manual Data or Model Drop-in
If automated downloads fail:
1. Create `data/glue/sst2/` and place `train.tsv` and `dev.tsv` with `sentence<TAB>label` columns (0/1 labels). A `data/README.md` will be generated with these instructions on failure.
2. (If needed) Download `distilbert-base-uncased`, place it under `models/distilbert-base-uncased/`, and set `TRANSFORMERS_CACHE=models/` before running scripts.

## Adapter Experiments
- Add new adapter builders in `src/adapters/` (e.g., `src/adapters/sparse_lora/` or `src/adapters/hira/`) and switch `src/model.py` to use them.
- Keep the config/output structure unchanged so results stay comparable.
