import argparse
import copy
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from . import data, model, utils
from .adapters.sparse_lora.sparse_lora import (
    compute_l1_loss,
    get_sparsity_stats,
    apply_magnitude_pruning,
    apply_pruning_masks,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA baseline trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def evaluate(accelerator, model, dataloader):
    model.eval()
    preds, labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.detach().float()
            pred = torch.argmax(logits, dim=-1)
            preds.extend(accelerator.gather(pred).cpu().tolist())
            labels.extend(accelerator.gather(batch["labels"]).cpu().tolist())
    avg_loss = float(total_loss / max(1, len(dataloader)))
    return {
        "accuracy": utils.compute_accuracy(preds, labels),
        "f1": utils.compute_f1(preds, labels),
        "loss": avg_loss,
    }


def get_dataloaders(cfg):
    task = cfg.get("task_name", "sst2").lower()
    if task != "sst2":
        raise NotImplementedError(f"Task {task} is not implemented in this baseline.")
    return data.get_sst2_splits(
        cfg["model_name"],
        cfg["max_seq_len"],
        cfg["train_batch_size"],
        cfg["eval_batch_size"],
    )


def save_checkpoint(unwrapped_model, output_dir: Path):
    ckpt_dir = output_dir / "ckpt"
    adapter_dir = ckpt_dir / "adapter"
    full_dir = ckpt_dir / "full_model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model.save_pretrained(adapter_dir, safe_serialization=True)
    merged_model = copy.deepcopy(unwrapped_model).merge_and_unload()
    merged_model.save_pretrained(full_dir, safe_serialization=True)


def main():
    args = parse_args()
    cfg = utils.load_config(args.config)
    utils.set_seed(cfg.get("seed", 42))
    accelerator = Accelerator(mixed_precision="fp16" if cfg.get("fp16", False) else "no")
    accelerator.print(f"Loaded config from {args.config}")
    device = accelerator.device
    accelerator.print(f"Accelerator processes: {accelerator.state.num_processes}")
    accelerator.print(f"Using device: {device}")
    if device.type == "cuda":
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        accelerator.print(f"CUDA available: True | GPU index: {gpu_index} | GPU name: {gpu_name}")
    else:
        accelerator.print("CUDA available: False; running on CPU")

    output_dir = Path(cfg["output_dir"])
    utils.ensure_dir(str(output_dir))
    utils.save_config(cfg, str(output_dir / "config_used.yaml"))

    train_loader, val_loader, _ = get_dataloaders(cfg)
    base_model = model.create_model(cfg)

    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    optimizer = AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)

    num_update_steps_per_epoch = len(train_loader)
    total_training_steps = num_update_steps_per_epoch * cfg["epochs"]
    warmup_ratio = float(cfg.get("warmup_ratio", 0.0))
    warmup_steps = int(total_training_steps * warmup_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    base_model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        base_model, optimizer, train_loader, val_loader, lr_scheduler
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step_times = []
    global_step = 0
    best_metrics = {"accuracy": 0.0, "f1": 0.0, "loss": float("inf"), "epoch": -1}

    save_freq = max(1, cfg.get("save_every_epochs", cfg["epochs"]))

    # Check if using Sparse LoRA
    unwrapped_model = accelerator.unwrap_model(base_model)
    is_sparse_lora = hasattr(unwrapped_model, "sparse_config")
    pruning_masks = None

    if is_sparse_lora:
        sparse_cfg = unwrapped_model.sparse_config
        accelerator.print(f"\n=== Sparse LoRA Configuration ===")
        accelerator.print(f"  Method: {sparse_cfg['method']}")
        if sparse_cfg["method"] == "l1":
            accelerator.print(f"  L1 lambda: {sparse_cfg['l1_lambda']}")
        else:
            accelerator.print(f"  Prune ratio: {sparse_cfg['prune_ratio']:.1%}")
        accelerator.print("=================================\n")

    for epoch in range(cfg["epochs"]):
        accelerator.print(f"Epoch {epoch + 1}/{cfg['epochs']}")
        base_model.train()
        running_loss = 0.0
        running_l1_loss = 0.0
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}", disable=not accelerator.is_local_main_process)
        for batch in train_loader:
            global_step += 1
            step_start = time.time()
            outputs = base_model(**batch)
            loss = outputs.loss

            # Add L1 regularization for Sparse LoRA
            if is_sparse_lora and sparse_cfg["method"] == "l1":
                l1_loss = compute_l1_loss(base_model)
                l1_weighted = sparse_cfg["l1_lambda"] * l1_loss
                loss = loss + l1_weighted
                running_l1_loss += l1_weighted.detach().float()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Apply pruning masks if using prune method
            if is_sparse_lora and sparse_cfg["method"] == "prune" and pruning_masks is not None:
                apply_pruning_masks(base_model, pruning_masks)

            running_loss += loss.detach().float()
            step_times.append((time.time() - step_start) * 1000.0)
            progress_bar.update(1)
        progress_bar.close()

        # Print training loss
        avg_train_loss = float(running_loss / max(1, len(train_loader)))
        if is_sparse_lora and sparse_cfg["method"] == "l1":
            avg_l1_loss = float(running_l1_loss / max(1, len(train_loader)))
            accelerator.print(f"  Train loss: {avg_train_loss:.4f} (L1 loss: {avg_l1_loss:.6f})")
        else:
            accelerator.print(f"  Train loss: {avg_train_loss:.4f}")

        # Log sparsity statistics
        if is_sparse_lora:
            stats = get_sparsity_stats(base_model)
            accelerator.print(f"  Sparsity: {stats['sparsity_ratio']:.2%} | Nonzero params: {stats['nonzero_params']:,}")

        metrics = evaluate(accelerator, base_model, val_loader)
        accelerator.print(
            f"  Val acc: {metrics['accuracy']:.4f} | Val f1: {metrics['f1']:.4f} | Loss: {metrics['loss']:.4f}"
        )
        if metrics["accuracy"] >= best_metrics["accuracy"]:
            best_metrics = {**metrics, "epoch": epoch + 1}

        should_save = ((epoch + 1) % save_freq == 0) or (epoch + 1 == cfg["epochs"])
        if should_save:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(base_model)
            save_checkpoint(unwrapped, output_dir)

    # Apply final pruning if using Sparse LoRA with prune method
    if is_sparse_lora and sparse_cfg["method"] == "prune":
        accelerator.print("\n=== Applying Final Magnitude Pruning ===")
        unwrapped_final = accelerator.unwrap_model(base_model)
        pruning_masks = apply_magnitude_pruning(
            unwrapped_final,
            prune_ratio=sparse_cfg["prune_ratio"],
            device=device
        )

        # Save final sparsity stats
        final_stats = get_sparsity_stats(unwrapped_final)
        accelerator.print(f"Final Sparsity: {final_stats['sparsity_ratio']:.2%}")
        accelerator.print(f"Nonzero params: {final_stats['nonzero_params']:,} / {final_stats['total_lora_params']:,}")

        # Save pruned model
        accelerator.wait_for_everyone()
        save_checkpoint(unwrapped_final, output_dir)
        accelerator.print("Pruned model saved.\n")

    avg_step_time = float(sum(step_times) / max(1, len(step_times)))
    metrics_payload = {
        "val_acc": best_metrics["accuracy"],
        "val_f1": best_metrics["f1"],
        "loss": best_metrics["loss"],
        "best_epoch": best_metrics["epoch"],
        "max_allocated_vram_mb": utils.get_max_vram_mb(),
        "avg_step_time_ms": avg_step_time,
    }

    # Add sparsity metrics if using Sparse LoRA
    if is_sparse_lora:
        final_stats = get_sparsity_stats(accelerator.unwrap_model(base_model))
        metrics_payload.update({
            "sparse_method": sparse_cfg["method"],
            "final_sparsity": final_stats["sparsity_ratio"],
            "nonzero_params": final_stats["nonzero_params"],
            "total_lora_params": final_stats["total_lora_params"],
        })
        if sparse_cfg["method"] == "l1":
            metrics_payload["l1_lambda"] = sparse_cfg["l1_lambda"]
        else:
            metrics_payload["prune_ratio"] = sparse_cfg["prune_ratio"]

    utils.write_json(metrics_payload, str(output_dir / "metrics.json"))
    accelerator.print(f"Metrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
