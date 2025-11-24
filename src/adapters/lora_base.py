from peft import LoraConfig, get_peft_model


def build_lora_adapter(model, cfg):
    l = cfg["lora"]
    peft_cfg = LoraConfig(
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        target_modules=l["target_modules"],
        bias="none",
        task_type="SEQ_CLS",
    )
    return get_peft_model(model, peft_cfg)