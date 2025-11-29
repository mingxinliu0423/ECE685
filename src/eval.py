import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from . import data, utils


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter")
    parser.add_argument("--config", required=True, help="Config file used for training")
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
    model = PeftModel.from_pretrained(base, adapter_dir)
    return model


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

    preds, labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.detach().float()
            pred = torch.argmax(outputs.logits, dim=-1)
            preds.extend(pred.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    avg_loss = float(total_loss / max(1, len(val_loader)))
    acc = utils.compute_accuracy(preds, labels)
    f1 = utils.compute_f1(preds, labels)

    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation F1: {f1:.4f}")
    print(f"Validation loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()