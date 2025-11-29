import argparse
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from . import data, utils


def parse_args():
    parser = argparse.ArgumentParser(description="Measure inference speed")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--batches", type=int, default=50, help="Number of validation batches")
    return parser.parse_args()


def load_model(cfg):
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2
    )
    adapter_dir = Path(cfg["output_dir"]) / "ckpt" / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter checkpoint not found at {adapter_dir}. Run training first."
        )
    return PeftModel.from_pretrained(base, adapter_dir)


def main():
    args = parse_args()
    cfg = utils.load_config(args.config)
    utils.set_seed(cfg.get("seed", 42))

    task = cfg.get("task_name", "sst2").lower()
    if task == "sst2":
        _, val_loader, _ = data.get_sst2_splits(
            cfg["model_name"], cfg["max_seq_len"], cfg["train_batch_size"], cfg["eval_batch_size"]
        )
    elif task == "imdb":
        _, val_loader, _ = data.get_imdb_splits(
            cfg["model_name"], cfg["max_seq_len"], cfg["train_batch_size"], cfg["eval_batch_size"]
        )
    elif task == "wikitext2":
        _, val_loader, _ = data.get_wikitext2_splits(
            cfg["model_name"], cfg["max_seq_len"], cfg["train_batch_size"], cfg["eval_batch_size"]
        )
    else:
        raise NotImplementedError(f"Task {task} is not implemented in this baseline.")

    model = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    total_batches = min(args.batches, len(val_loader))
    if total_batches == 0:
        raise RuntimeError("Validation loader is empty.")

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= total_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            start = time.perf_counter()
            _ = model(**batch)
            latencies.append((time.perf_counter() - start) * 1000.0)

    mean_latency = float(sum(latencies) / len(latencies))
    median_latency = float(statistics.median(latencies))
    peak_vram = utils.get_max_vram_mb()

    print(f"Batches measured: {len(latencies)}")
    print(f"Mean latency (ms): {mean_latency:.3f}")
    print(f"Median latency (ms): {median_latency:.3f}")
    print(f"Peak VRAM (MB): {peak_vram:.2f}")


if __name__ == "__main__":
    main()