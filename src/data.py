import csv
from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from . import utils


LOCAL_DATA_DIR = Path("data")
LOCAL_INSTRUCTIONS = (
    "For SST-2, place a GLUE-style directory: data/glue/sst2/train.tsv, dev.tsv with sentence\\tlabel.\n"
    "Or run an included utility function to convert HF format to TSV for caching."
)


def _ensure_data_readme() -> None:
    readme_path = LOCAL_DATA_DIR / "README.md"
    if readme_path.exists():
        return
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(LOCAL_INSTRUCTIONS + "\n", encoding="utf-8")


def _load_local_sst2() -> Optional[DatasetDict]:
    sst2_dir = LOCAL_DATA_DIR / "glue" / "sst2"
    train_path = sst2_dir / "train.tsv"
    dev_path = sst2_dir / "dev.tsv"
    if not train_path.exists() or not dev_path.exists():
        return None

    def _read_split(path: Path) -> Dataset:
        sentences, labels = [], []
        with path.open(encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                if row[0].strip().lower() == "sentence" and row[-1].strip().lower() == "label":
                    # Header row
                    continue
                sentences.append(row[0])
                labels.append(int(row[-1]))
        return Dataset.from_dict({"sentence": sentences, "label": labels})

    return DatasetDict({
        "train": _read_split(train_path),
        "validation": _read_split(dev_path),
    })


def _load_sst2_dataset() -> DatasetDict:
    try:
        return load_dataset("glue", "sst2")
    except Exception as err:
        local_ds = _load_local_sst2()
        if local_ds is not None:
            return local_ds
        _ensure_data_readme()
        raise RuntimeError(
            "SST-2 download failed. Place files under data/glue/sst2/ per data/README.md"
        ) from err


def _tokenize_dataset(ds: DatasetDict, model_name: str, max_len: int):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as err:
        utils.ensure_models_readme()
        raise RuntimeError(
            "Tokenizer download failed. Place model files under models/ per models/README.md"
        ) from err

    def _encode(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    ds = ds.map(_encode, batched=True, desc="Tokenizing")
    if "label" in ds["train"].column_names:
        ds = ds.rename_column("label", "labels")
    columns = ["input_ids", "attention_mask", "labels"]
    ds.set_format(type="torch", columns=columns)
    return ds, tokenizer


def get_sst2_splits(model_name: str, max_len: int, batch_train: int, batch_eval: int) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    ds = _load_sst2_dataset()
    ds, tokenizer = _tokenize_dataset(ds, model_name, max_len)
    dl_train = DataLoader(ds["train"], batch_size=batch_train, shuffle=True)
    dl_val = DataLoader(ds["validation"], batch_size=batch_eval)
    return dl_train, dl_val, tokenizer
