"""Mô-đun khởi tạo cấu hình MLflow cho pipeline."""
from __future__ import annotations

from pipeline.training.config import TrainConfig
from pipeline.mlflow_manager import MLflowManager


def setup_mlflow(cfg: TrainConfig) -> MLflowManager:
    """
    Bước 1-2: Init MLflow + set experiment.
    Dùng MLflowManager của project để đảm bảo thống nhất config/tags.
    """
    manager = MLflowManager.configure(
        tracking_uri=cfg.tracking_uri,
        experiment_name=cfg.experiment_name,
        default_tags=cfg.default_tags,
    )
    return manager
