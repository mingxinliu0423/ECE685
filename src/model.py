from transformers import AutoModelForSequenceClassification

from . import utils
from .adapters import build_lora_adapter
from .adapters.sparse_lora.sparse_lora import build_sparse_lora_adapter

def create_model(config):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=2
        )
    except Exception as err:
        utils.ensure_models_readme()
        raise RuntimeError(
            "Model download failed. Place model files per models/README.md."
        ) from err
    
    if "sparse_lora" in config:
        model = build_sparse_lora_adapter(model, config)
    elif "lora" in config:
        model = build_lora_adapter(model, config)


    return model
