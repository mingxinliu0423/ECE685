import argparse
import copy
import time
from pathlib import Path
import math

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup,AutoModelForSequenceClassification,AutoModelForMaskedLM


from . import data, model, utils
from .adapters.sparse_lora.sparse_lora import (
    compute_l1_loss,
    get_sparsity_stats,
    apply_magnitude_pruning,
    apply_pruning_masks,
    build_svd_estimators_for_model,
    apply_sparse_ffn_to_model,
    get_sparsity_statistics,
)
from .adapters.hira.hira_adapter import (
    compute_hira_l1_loss,
    get_hira_stats,
    apply_hira_magnitude_pruning,
    apply_hira_pruning_masks,
    save_hira_adapter_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA baseline trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def evaluate(accelerator, model, dataloader, tokenizer):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    # For MLM metrics
    total_mlm_correct = 0
    total_mlm_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            logits = outputs.logits          # could be (B, C) or (B, T, V)

            # ---- classification vs MLM detection ----
            if logits.dim() == 2:
                # ====== SEQUENCE CLASSIFICATION (SST-2 / IMDB) =======
                # logits: (B, num_labels)
                loss = outputs.loss          # HF computes CE if batch has "labels"
                total_loss += loss.detach().item()

                # gather preds + labels for accuracy/F1
                preds = torch.argmax(logits, dim=-1)   # (B,)
                preds = accelerator.gather(preds).cpu().tolist()
                labels = accelerator.gather(batch["labels"]).cpu().tolist()

                all_preds.extend(preds)
                all_labels.extend(labels)

            elif logits.dim() == 3:
                # ====== MASKED LM (WikiText-2) =======
                # logits: (B, T, V)
                # use input_ids as labels unless you already prepared batch["labels"]
                labels = batch.get("labels", batch["input_ids"]).clone()  # (B, T)

                # ignore pads
                if tokenizer.pad_token_id is not None:
                    labels[labels == tokenizer.pad_token_id] = -100

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),  # (B*T, V)
                    labels.view(-1),                   # (B*T,)
                )
                total_loss += loss.detach().item()

                # masked-token accuracy (optional but useful)
                preds = logits.argmax(dim=-1)         # (B, T)
                mask = labels != -100                 # (B, T)
                correct = ((preds == labels) & mask).sum()
                tokens = mask.sum()

                # gather across processes
                correct = accelerator.gather(correct).sum()
                tokens = accelerator.gather(tokens).sum()

                total_mlm_correct += correct.item()
                total_mlm_tokens += tokens.item()

            else:
                raise RuntimeError(f"Unexpected logits dim: {logits.dim()}")

    avg_loss = total_loss / max(1, len(dataloader))

    # ---- Final metric summary ----
    if total_mlm_tokens > 0:
        # We ran MLM at least once → return MLM metrics
        mlm_accuracy = total_mlm_correct / max(1, total_mlm_tokens)
        perplexity = math.exp(avg_loss)
        return {
            "accuracy": mlm_accuracy,
            "perplexity": perplexity,
            "loss": avg_loss,
        }
    else:
        # Only classification → return accuracy / F1
        return {
            "accuracy": utils.compute_accuracy(all_preds, all_labels),
            "f1": utils.compute_f1(all_preds, all_labels),
            "loss": avg_loss,
        }


def get_dataloaders(cfg):
    task = cfg.get("task_name", "sst2").lower()
    if task == "sst2":
        return data.get_sst2_splits(
            cfg["model_name"],
            cfg["max_seq_len"],
            cfg["train_batch_size"],
            cfg["eval_batch_size"],
        )
    elif task == "imdb":
        return data.get_imdb_splits(
            cfg["model_name"],
            cfg["max_seq_len"],
            cfg["train_batch_size"],
            cfg["eval_batch_size"],
        )
    elif task == "wikitext2":
        return data.get_wikitext2_splits(
            cfg["model_name"],
            cfg["max_seq_len"],
            cfg["train_batch_size"],
            cfg["eval_batch_size"],
        )
    else:
        raise NotImplementedError(f"Task {task} is not implemented in this baseline.")


def save_checkpoint(unwrapped_model, output_dir: Path):
    ckpt_dir = output_dir / "ckpt"
    adapter_dir = ckpt_dir / "adapter"
    full_dir = ckpt_dir / "full_model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model.save_pretrained(adapter_dir, safe_serialization=True)
    
    # Only merge if it's a PEFT model (LoRA/Sparse LoRA), not for custom adapters like HIRA
    if hasattr(unwrapped_model, 'merge_and_unload'):
        try:
            merged_model = copy.deepcopy(unwrapped_model).merge_and_unload()
            merged_model.save_pretrained(full_dir, safe_serialization=True)
        except Exception as e:
            print(f"Error merging model: {e}")
            print(f"Saving full model without merging")
            unwrapped_model.save_pretrained(full_dir, safe_serialization=True)
    else:
        # For custom adapters like HIRA, save the full model directly
        unwrapped_model.save_pretrained(full_dir, safe_serialization=True)
        if hasattr(unwrapped_model, 'hira_config'):
            try:
                save_hira_adapter_config(unwrapped_model, str(adapter_dir))
            except Exception as e:
                print(f"Warning: Failed to save HIRA adapter config: {e}")


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

    train_loader, val_loader, tokenizer = get_dataloaders(cfg)
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
    svd_estimators = None

    # Training schedule for sparse methods
    sparse_warmup_ratio = cfg.get("sparse_warmup_ratio", 0.0)
    sparse_start_step = int(total_training_steps * sparse_warmup_ratio)
    prune_at_epoch = -1

    # Layer-wise L1 weights (for non-uniform sparsity)
    layer_weights = cfg.get("layer_l1_weights", None)
    
    # Check if using HIRA
    is_hira = hasattr(unwrapped_model, "hira_config")
    hira_pruning_masks = None
    hira_cfg = None

    if is_sparse_lora:
        sparse_cfg = unwrapped_model.sparse_config
        accelerator.print(f"\n=== Sparse LoRA Configuration ===")
        accelerator.print(f"  Method: {sparse_cfg['method']}")
        if sparse_cfg["method"] == "l1":
            accelerator.print(f"  L1 lambda: {sparse_cfg['l1_lambda']}")
            if sparse_warmup_ratio > 0:
                accelerator.print(f"  Sparse warmup: first {sparse_warmup_ratio:.1%} steps are dense")
                accelerator.print(f"  Sparse training starts at step {sparse_start_step}/{total_training_steps}")
            if layer_weights:
                accelerator.print(f"  Layer-wise L1 weights: {layer_weights}")
        elif sparse_cfg["method"] == "prune":
            accelerator.print(f"  Prune ratio: {sparse_cfg['prune_ratio']:.1%}")
            # Prune at the second-to-last epoch, then fine-tune in the last epoch
            prune_at_epoch = max(1, cfg["epochs"] - 1)
            accelerator.print(f"  Will prune at epoch {prune_at_epoch}, then fine-tune")
        elif sparse_cfg["method"] == "svd":
            accelerator.print(f"  SVD rank: {sparse_cfg['svd_rank']}")
            accelerator.print(f"  FFN sparsity: {sparse_cfg['ffn_sparsity']:.1%}")
            if sparse_warmup_ratio > 0:
                accelerator.print(f"  Sparse warmup: first {sparse_warmup_ratio:.1%} steps are dense")
                accelerator.print(f"  Sparse training starts at step {sparse_start_step}/{total_training_steps}")
            if sparse_cfg.get('layer_sparsity'):
                accelerator.print(f"  Layer-wise sparsity configured")

            # Build SVD estimators (only on main process to avoid duplication)
            if accelerator.is_main_process:
                svd_estimators = build_svd_estimators_for_model(unwrapped_model, cfg)

                # Apply sparse FFN modules to the model
                if svd_estimators:
                    apply_sparse_ffn_to_model(
                        unwrapped_model,
                        svd_estimators,
                        cfg,
                        total_training_steps
                    )
            accelerator.wait_for_everyone()
    elif is_hira:
        hira_cfg = unwrapped_model.hira_config
        accelerator.print(f"\n=== HIRA Configuration ===")
        accelerator.print(f"  Rank: {hira_cfg['r']}")
        accelerator.print(f"  Alpha: {hira_cfg['alpha']}")
        accelerator.print(f"  Dropout: {hira_cfg['dropout']}")
        accelerator.print(f"  Target modules: {hira_cfg['target_modules']}")
        if hira_cfg.get("l1_lambda", 0.0) > 0.0:
            accelerator.print(f"  L1 lambda: {hira_cfg['l1_lambda']}")
            if layer_weights:
                accelerator.print(f"  Layer-wise L1 weights: {layer_weights}")
        if hira_cfg.get("prune_ratio", 0.0) > 0.0:
            prune_at_epoch = max(1, cfg["epochs"] - 1)
            accelerator.print(f"  Prune ratio: {hira_cfg['prune_ratio']:.1%}")
            accelerator.print(f"  Will prune at epoch {prune_at_epoch}, then fine-tune")

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
            if cfg["task_name"] == "wikitext2":
              logits = outputs["logits"]              # (B, T, V)
              labels = batch["input_ids"]          # (B, T) for now

              # (optional but recommended) ignore pad tokens in the loss
              if tokenizer.pad_token_id is not None:
                  labels = labels.clone()
                  labels[labels == tokenizer.pad_token_id] = -100

              loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

              # reshape:
              #   (B, T, V) → (B*T, V)
              #   (B, T)    → (B*T)
              loss = loss_fct(
                  logits.view(-1, logits.size(-1)),  # (B*T, V)
                  labels.view(-1)                    # (B*T,)
              )
            else:
              loss = outputs.loss

            # Add L1 regularization for Sparse LoRA after warmup
            if is_sparse_lora and sparse_cfg["method"] == "l1":
                if global_step >= sparse_start_step:
                    l1_lambda = float(sparse_cfg.get("l1_lambda", 0.0))
                    l1_loss = compute_l1_loss(base_model, layer_weights=layer_weights)
                    l1_weighted = l1_lambda * l1_loss
                    loss = loss + l1_weighted
                    running_l1_loss += l1_weighted.detach().float()
            
            # Add L1 regularization for HIRA
            if is_hira and hira_cfg is not None:
                l1_lambda = float(hira_cfg.get("l1_lambda", 0.0))
                if l1_lambda > 0.0:
                    hira_l1_loss = compute_hira_l1_loss(base_model, layer_weights=layer_weights)
                    l1_weighted = l1_lambda * hira_l1_loss
                    loss = loss + l1_weighted
                    running_l1_loss += l1_weighted.detach().float()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Apply pruning masks if using prune method
            if is_sparse_lora and sparse_cfg["method"] == "prune" and pruning_masks is not None:
                apply_pruning_masks(base_model, pruning_masks)
            
            # Apply pruning masks for HIRA
            if is_hira and hira_pruning_masks is not None:
                apply_hira_pruning_masks(base_model, hira_pruning_masks)

            running_loss += loss.detach().float()
            step_times.append((time.time() - step_start) * 1000.0)
            progress_bar.update(1)
        progress_bar.close()

        # Print training loss
        avg_train_loss = float(running_loss / max(1, len(train_loader)))
        if is_sparse_lora and sparse_cfg["method"] == "l1":
            avg_l1_loss = float(running_l1_loss / max(1, len(train_loader)))
            accelerator.print(f"  Train loss: {avg_train_loss:.4f} (L1 loss: {avg_l1_loss:.6f})")
        elif is_hira and hira_cfg is not None and hira_cfg.get("l1_lambda", 0.0) > 0.0:
            avg_l1_loss = float(running_l1_loss / max(1, len(train_loader)))
            accelerator.print(f"  Train loss: {avg_train_loss:.4f} (L1 loss: {avg_l1_loss:.6f})")
        else:
            accelerator.print(f"  Train loss: {avg_train_loss:.4f}")

        # Apply pruning at the designated epoch
        if is_sparse_lora and sparse_cfg["method"] == "prune":
            if epoch + 1 == prune_at_epoch and pruning_masks is None:
                accelerator.print(f"\n=== Applying Magnitude Pruning at Epoch {epoch + 1} ===")
                unwrapped = accelerator.unwrap_model(base_model)
                pruning_masks = apply_magnitude_pruning(
                    unwrapped,
                    prune_ratio=sparse_cfg["prune_ratio"],
                    device=device
                )
                apply_pruning_masks(base_model, pruning_masks)
                accelerator.print(f"Pruning applied. Will fine-tune for {cfg['epochs'] - prune_at_epoch} more epoch(s)\n")
        
        # Apply pruning at the designated epoch for HIRA
        if is_hira and hira_cfg is not None and hira_cfg.get("prune_ratio", 0.0) > 0.0:
            if epoch + 1 == prune_at_epoch and hira_pruning_masks is None:
                accelerator.print(f"\n=== Applying HIRA Magnitude Pruning at Epoch {epoch + 1} ===")
                unwrapped = accelerator.unwrap_model(base_model)
                hira_pruning_masks = apply_hira_magnitude_pruning(
                    unwrapped,
                    prune_ratio=hira_cfg["prune_ratio"],
                    device=device
                )
                apply_hira_pruning_masks(base_model, hira_pruning_masks)
                accelerator.print(f"HIRA pruning applied. Will fine-tune for {cfg['epochs'] - prune_at_epoch} more epoch(s)\n")

        # Log sparsity statistics
        if is_sparse_lora:
            if sparse_cfg["method"] == "svd":
                # For SVD, log sparse FFN statistics
                sparse_stats = get_sparsity_statistics(base_model)
                if sparse_stats["total_sparse_ffn"] > 0:
                    accelerator.print(
                        f"  Sparse FFN: {sparse_stats['active_sparse_ffn']}/{sparse_stats['total_sparse_ffn']} active | "
                        f"Avg sparsity: {sparse_stats['avg_sparsity']:.1%}"
                    )
            else:
                # For L1/Prune, log parameter sparsity
                stats = get_sparsity_stats(base_model)
                accelerator.print(f"  Sparsity: {stats['sparsity_ratio']:.2%} | Nonzero params: {stats['nonzero_params']:,}")
        
        # Log HIRA statistics
        if is_hira:
            hira_stats = get_hira_stats(base_model)
            if hira_stats["total_hira_params"] > 0:
                accelerator.print(
                    f"  HIRA layers: {hira_stats['num_hira_layers']} | "
                    f"Sparsity: {hira_stats['sparsity_ratio']:.2%} | "
                    f"Nonzero params: {hira_stats['nonzero_params']:,}"
                )

        metrics = evaluate(accelerator, base_model, val_loader,tokenizer)
        try:
          accelerator.print(
              f"  Val acc: {metrics['accuracy']:.4f} | Val f1: {metrics['f1']:.4f} | Loss: {metrics['loss']:.4f}"
          )
        except:
          accelerator.print(
            f"  Val acc: {metrics['accuracy']:.4f} | Val perplexity: {metrics['perplexity']:.4f} | Loss: {metrics['loss']:.4f}"
          )
        if metrics["accuracy"] >= best_metrics["accuracy"]:
            best_metrics = {**metrics, "epoch": epoch + 1}

        should_save = ((epoch + 1) % save_freq == 0) or (epoch + 1 == cfg["epochs"])
        if should_save:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(base_model)
            save_checkpoint(unwrapped, output_dir)


    avg_step_time = float(sum(step_times) / max(1, len(step_times)))
    try:
      metrics_payload = {
          "val_acc": best_metrics["accuracy"],
          "val_f1": best_metrics["f1"],
          "loss": best_metrics["loss"],
          "best_epoch": best_metrics["epoch"],
          "max_allocated_vram_mb": utils.get_max_vram_mb(),
          "avg_step_time_ms": avg_step_time,
      }
    except:
      metrics_payload = {
          "val_acc": best_metrics["accuracy"],
          "val_perplexity": best_metrics["perplexity"],
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
        elif sparse_cfg["method"] == "prune":
            metrics_payload["prune_ratio"] = sparse_cfg["prune_ratio"]
        elif sparse_cfg["method"] == "svd":
            metrics_payload["svd_rank"] = sparse_cfg["svd_rank"]
            metrics_payload["ffn_sparsity"] = sparse_cfg["ffn_sparsity"]
            if svd_estimators:
                metrics_payload["num_ffn_estimators"] = len(svd_estimators.get("ffn", {}))
    
    # Add HIRA metrics
    if is_hira and hira_cfg is not None:
        final_hira_stats = get_hira_stats(accelerator.unwrap_model(base_model))
        metrics_payload.update({
            "adapter_type": "hira",
            "hira_rank": hira_cfg["r"],
            "hira_alpha": hira_cfg["alpha"],
            "final_sparsity": final_hira_stats["sparsity_ratio"],
            "nonzero_params": final_hira_stats["nonzero_params"],
            "total_hira_params": final_hira_stats["total_hira_params"],
            "num_hira_layers": final_hira_stats["num_hira_layers"],
        })
        if hira_cfg.get("l1_lambda", 0.0) > 0.0:
            metrics_payload["l1_lambda"] = hira_cfg["l1_lambda"]
        if hira_cfg.get("prune_ratio", 0.0) > 0.0:
            metrics_payload["prune_ratio"] = hira_cfg["prune_ratio"]

    utils.write_json(metrics_payload, str(output_dir / "metrics.json"))
    accelerator.print(f"Metrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
