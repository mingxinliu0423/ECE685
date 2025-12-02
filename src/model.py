from transformers import AutoModelForSequenceClassification,AutoModelForMaskedLM
from pathlib import Path
import os

from . import utils
from .adapters import build_lora_adapter
from .adapters.sparse_lora.sparse_lora import build_sparse_lora_adapter

def create_model(config):
    model_name = config["model_name"]
    dataset = config["task_name"]
    if dataset == 'wikitext2':
        ModelCls = AutoModelForMaskedLM
        pretrained_kwargs = {}  # no num_labels for MLM
    else:
        ModelCls = AutoModelForSequenceClassification
        pretrained_kwargs = {"num_labels": 2}
    
    # Set TRANSFORMERS_CACHE to models/ if not already set and model exists locally
    models_dir = Path("models").absolute()
    local_model_path = models_dir / model_name
    
    # If model exists locally, prefer loading from there
    if local_model_path.exists() and (local_model_path / "config.json").exists():
        # Set cache to models directory
        if "TRANSFORMERS_CACHE" not in os.environ:
            os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
        
        # Try loading directly from local path (as a path, not by name)
        try:
            model = ModelCls.from_pretrained(
                str(local_model_path), local_files_only=False, **pretrained_kwargs,
            )
        except Exception as local_err:
            # If that fails, try loading by name with cache_dir pointing to models/
            try:
                model = ModelCls.from_pretrained(
                    model_name, cache_dir=str(models_dir), local_files_only=False, **pretrained_kwargs,
                )
            except Exception as err:
                utils.ensure_models_readme()
                raise RuntimeError(
                    f"Model loading failed. Local model found at {local_model_path} but failed to load. "
                    f"Original error: {local_err}. Secondary error: {err}. "
                    f"Please check model files per models/README.md."
                ) from err
    else:
        # Model not found locally, try downloading
        # Set cache directory
        if "TRANSFORMERS_CACHE" not in os.environ:
            os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
        
        try:
            model = ModelCls.from_pretrained(
                model_name, cache_dir=str(models_dir), **pretrained_kwargs,
            )
        except Exception as err:
            utils.ensure_models_readme()
            raise RuntimeError(
                f"Model download failed for {model_name}. "
                f"Place model files in models/{model_name}/ per models/README.md. "
                f"Error: {err}"
            ) from err
    
    if "sparse_lora" in config:
        model = build_sparse_lora_adapter(model, config)
    elif "hira" in config:
        from .adapters.hira.hira_adapter import build_hira_adapter
        model = build_hira_adapter(model, config)
    elif "lora" in config:
        model = build_lora_adapter(model, config)


    return model
