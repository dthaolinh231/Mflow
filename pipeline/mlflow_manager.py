"""
MLflow Manager
Singleton pattern cho cấu hình MLflow dùng chung toàn pipeline

Bước 1: Init MLflow
- set_tracking_uri
- (optional) set_experiment
- set default tags
"""

from __future__ import annotations

import os
from typing import Optional, Dict

import mlflow


class MLflowManager:
    """Singleton class để quản lý cấu hình MLflow"""

    _instance: Optional["MLflowManager"] = None
    _configured: bool = False

    # --------------------------------------------------
    # Singleton
    # --------------------------------------------------
    def __new__(cls) -> "MLflowManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # --------------------------------------------------
    # Configure (RUN ONCE)
    # --------------------------------------------------
    @classmethod
    def configure(
        cls,
        tracking_uri: str,
        experiment_name: Optional[str] = None,
        default_tags: Optional[Dict[str, str]] = None,
    ) -> "MLflowManager":
        """
        Khởi tạo cấu hình MLflow (idempotent – chỉ chạy 1 lần)

        Args:
            tracking_uri: MLflow tracking URI (file path hoặc http)
            experiment_name: Experiment mặc định (vd: churn_prediction)
            default_tags: Tags mặc định cho mọi run
        """
        instance = cls()

        if cls._configured:
            # Đã cấu hình rồi → không làm gì thêm
            return instance

        # TODO: validate
