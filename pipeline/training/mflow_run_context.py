"""Mô-đun quản lý MLflow run context cho training."""
from __future__ import annotations

from datetime import datetime

import mlflow

from pipeline.training.config import TrainConfig
from pipeline.mlflow_manager import MLflowManager


def _default_run_name() -> str:
    """Tạo tên run mặc định dựa trên thời gian hiện tại.

    Returns:
        Chuỗi tên run theo định dạng train_YYYYMMdd_HHMMSS.
    """
    return f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def start_train_run(cfg: TrainConfig, manager: MLflowManager):
    """
    Bước 3: Start run (context manager).
    """
    run_name: str = cfg.run_name or _default_run_name()
    tags = manager.get_default_tags() if manager else {}
    return mlflow.start_run(run_name=run_name, tags=tags)
