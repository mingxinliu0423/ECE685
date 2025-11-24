from .lora_base import build_lora_adapter

try:
    from .sparse_lora.sparse_lora import build_sparse_lora_adapter  # noqa: F401
except Exception:
    build_sparse_lora_adapter = None

try:
    from .hira.hira_adapter import build_hira_adapter  # noqa: F401
except Exception:
    build_hira_adapter = None