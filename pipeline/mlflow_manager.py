"""MLflow Manager - Singleton pattern cho cấu hình MLflow dùng chung."""

from __future__ import annotations

from typing import Optional, Dict

import mlflow


class MLflowManager:
    """Singleton class để quản lý cấu hình MLflow"""

    _instance: Optional["MLflowManager"] = None
    _configured: bool = False
    _default_tags: Dict[str, str] = {}

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
            return instance

        mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        if default_tags:
            for key, value in default_tags.items():
                mlflow.set_tag(key, value)

        cls._configured = True
        return instance

    def get_default_tags(self) -> Dict[str, str]:
        """Trả về default tags cho MLflow runs (placeholder)."""
        return {}

    def apply_default_tags(self) -> None:
        """Apply default_tags vào active run hiện tại.

        Quy ước sử dụng:
        - Gọi method này SAU khi đã mlflow.start_run()
        - Nếu chưa có active run -> raise để fail fast (đúng kỳ vọng).

        Raises:
            RuntimeError: nếu chưa có active run.
        """
        if mlflow.active_run() is None:
            raise RuntimeError(
                "No active MLflow run. Call mlflow.start_run() before apply_default_tags()."
            )

        for key, value in self._default_tags.items():
            mlflow.set_tag(key, value)
