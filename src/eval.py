import argparse
from pathlib import Path
import math
from safetensors.torch import load_file as load_safetensors 

import torch
from peft import PeftModel,PeftConfig
# Note: You might need to import AutoModelForCausalLM if your base model is a LM,
# but keeping AutoModelForSequenceClassification for consistency with your code.
from transformers import AutoModelForSequenceClassification,AutoModelForMaskedLM

from . import data, utils

from .adapters.hira.hira_adapter import inject_hira_adapters, HiraConfig


try:
    from peft.utils import infer_device
except ImportError:
    def infer_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter")
    parser.add_argument("--config", required=True, help="Config file used for training")
    return parser.parse_args()


def load_model(cfg):
    model_args = {}
    
    # Check if the output directory (where wikitext2 task results would be saved) 
    # indicates a language modeling task.
    output_dir_str = str(cfg["output_dir"]).lower()
    
    # --- LOGIC TO OMIT num_labels=2 for wikitext2 ---
    # This logic assumes 'wikitext2' implies a language modeling task
    if cfg["task_name"]=='wikitext2':
        BaseModelClass = AutoModelForMaskedLM # <--- MUST be this!
    else:
        BaseModelClass = AutoModelForSequenceClassification
        model_args["num_labels"] = 2
        
    base = BaseModelClass.from_pretrained(
        cfg["model_name"], 
        **model_args
    )
    # --- PeftModel Loading Fix (as previously discussed) ---
    adapter_dir = Path(cfg["output_dir"]) / "ckpt" / "adapter"
    full_model_dir = Path(cfg["output_dir"]) / "ckpt" / "full_model"
    adapter_path_str = str(adapter_dir.resolve())
    # ... (Load Base Model and HIRA Config) ...

    # --- 3. Determine Weight Path and Load ---
    try:
        # We must load the HIRA config to get parameters like r, alpha, target_modules
        peft_config_obj = HiraConfig.from_pretrained(adapter_path_str)
    except Exception as e:
        # Fallback for standard PEFT types if HIRA config fails
        try:
             peft_config_obj = PeftConfig.from_pretrained(adapter_path_str)
        except Exception:
             raise RuntimeError(f"Failed to load any configuration from {adapter_path_str}.")

    adapter_type = getattr(peft_config_obj, 'peft_type', None)
    if adapter_type == 'HIRA':
        
        # Check the full_model_dir for the single valid weight file
        weights_path = full_model_dir / "model.safetensors" 
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing weights. The required file '{weights_path.name}' was not found in {full_model_dir}."
            )
        
        print(f"Loading HIRA snapshot from: {weights_path}")

        # A. Inject Structure (Mandatory step for HIRA)
        inject_hira_adapters(base, peft_config_obj, adapter_name="default")
        
        # B. Load Weights: Use the dedicated safetensors loader
        # map_location='cpu' is often necessary when loading large files
        adapter_state_dict = load_safetensors(weights_path, device='cpu') 
        
        # C. Load the state dictionary into the adapted model structure
        
        # Apply prefix cleanup (critical because the saved snapshot is a wrapped model)
        cleaned_weights = {}
        for k, v in adapter_state_dict.items():
            # Remove common PEFT saving prefixes (if present)
            k = k.replace("base_model.model.", "")
            k = k.replace("base_model.", "")
            cleaned_weights[k] = v

        # strict=False is often required when loading a wrapped model snapshot
        base.load_state_dict(cleaned_weights, strict=False)
        model = base
        
    else:
        # Standard PEFT Loading (for LoRA, etc.)
        adapter_path_str = str(adapter_dir.resolve())
        model = PeftModel.from_pretrained(base, adapter_path_str, local_files_only=True)
        
    return model


def main():
    args = parse_args()
    cfg = utils.load_config(args.config)
    utils.set_seed(cfg.get("seed", 42))

    task = cfg.get("task_name", "sst2").lower()
    if task == "sst2":
        _, val_loader, tokenizer = data.get_sst2_splits(
            cfg["model_name"], cfg["max_seq_len"], cfg["train_batch_size"], cfg["eval_batch_size"]
        )
    elif task == "imdb":
        _, val_loader, tokenizer = data.get_imdb_splits(
            cfg["model_name"], cfg["max_seq_len"], cfg["train_batch_size"], cfg["eval_batch_size"]
        )
    elif task == "wikitext2":
        _, val_loader, tokenizer = data.get_wikitext2_splits(
            cfg["model_name"], cfg["max_seq_len"], cfg["train_batch_size"], cfg["eval_batch_size"]
        )
    else:
        raise NotImplementedError(f"Task {task} is not implemented in this baseline.")

    model = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0
    
    # For MLM metrics
    total_mlm_correct = 0
    total_mlm_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits  # could be (B, C) or (B, T, V)

            # Classification vs MLM detection (matching train.py evaluate function)
            # Check dimensions to determine task type, matching train.py lines 53-98
            if cfg['task_name']=='wikitext2':
                # ====== MASKED LM (WikiText-2) =======
                
                # 1. Prepare token-level labels (labels_mlm)
                labels_mlm = batch.get('labels',batch["input_ids"]).clone() 
                if tokenizer.pad_token_id is not None:
                    labels_mlm[labels_mlm == tokenizer.pad_token_id] = -100

                # 2. Loss calculation (uses flattened tensors)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                num_tokens = labels_mlm.numel() 
                loss = loss_fct(
                    logits.reshape(num_tokens, -1), 
                    labels_mlm.reshape(-1) 
                )
                total_loss += loss.detach().item()

                # 3. Prediction Calculation
                preds_mlm = logits.argmax(dim=-1)  # (B, T) -> (32, 128)

                # 4. Accuracy Calculation (Comparison must be BxT vs BxT)
                mask = labels_mlm != -100  # (B, T)
                
                # Compare the token predictions (BxT) against the token labels (BxT)
                correct = ((preds_mlm == labels_mlm) & mask).sum() 
                tokens = mask.sum()

                total_mlm_correct += correct.item()
                total_mlm_tokens += tokens.item()
            else:
                # ====== SEQUENCE CLASSIFICATION (SST-2 / IMDB) =======
                # This logic is UNCHANGED - only affects classification tasks
                # logits: (B, num_labels) - already set above
                if outputs.loss is not None:
                    loss = outputs.loss  # HF computes CE if batch has "labels"
                else:
                    # Manual loss computation if outputs.loss is None
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits, batch["labels"])
                total_loss += loss.detach().item()

                # gather preds + labels for accuracy/F1
                preds = torch.argmax(logits, dim=-1)  # (B,)
                preds = preds.cpu().tolist()
                labels = batch["labels"].cpu().tolist()

                all_preds.extend(preds)
                all_labels.extend(labels)

    avg_loss = total_loss / max(1, len(val_loader))

    # Final metric summary (matching train.py)
    if total_mlm_tokens > 0:
        # We ran MLM → return MLM metrics
        mlm_accuracy = total_mlm_correct / max(1, total_mlm_tokens)
        perplexity = math.exp(avg_loss)
        print(f"Validation accuracy: {mlm_accuracy:.4f}")
        print(f"Validation perplexity: {perplexity:.4f}")
        print(f"Validation loss: {avg_loss:.4f}")
    else:
        # Only classification → return accuracy / F1
        acc = utils.compute_accuracy(all_preds, all_labels)
        f1 = utils.compute_f1(all_preds, all_labels)
        print(f"Validation accuracy: {acc:.4f}")
        print(f"Validation F1: {f1:.4f}")
        print(f"Validation loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()