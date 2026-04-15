from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


class Settings(BaseModel):
    app_name: str
    database_url: str
    uploads_dir: Path
    qwen_model_id: str
    qwen_adapter_path: Optional[Path]
    qwen_max_tokens: int
    qwen_temperature: float
    qwen_top_p: float
    yolo_weights_path: Path
    yolo_confidence: float


settings = Settings(
    app_name=_get_env("APP_NAME", "insurance-claim-api"),
    database_url=_get_env("DATABASE_URL", "sqlite:///./api.db"),
    uploads_dir=Path(_get_env("UPLOADS_DIR", "data/uploads")),
    qwen_model_id=_get_env("QWEN_MODEL_ID", "mlx-community/Qwen3-8B-4bit"),
    qwen_adapter_path=Path(
        _get_env("QWEN_ADAPTER_PATH", "qwen3/adapters/qwen3_claim_sft")
    ),
    qwen_max_tokens=_get_env_int("QWEN_MAX_TOKENS", 3000),
    qwen_temperature=_get_env_float("QWEN_TEMPERATURE", 0.1),
    qwen_top_p=_get_env_float("QWEN_TOP_P", 1.0),
    yolo_weights_path=Path(
        _get_env(
            "YOLO_WEIGHTS_PATH",
            "runs/detect/runs/detect/models/modelo_large_v3/weights/best.pt",
        )
    ),
    yolo_confidence=_get_env_float("YOLO_CONFIDENCE", 0.25),
)
