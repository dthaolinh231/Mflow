"""
Model Registry
Đăng ký model vào MLflow Model Registry và gán alias/stage

Bước 10: Register model
- tạo registered model name nếu chưa có
- tạo model version từ run artifact: runs:/<run_id>/model

Bước 11: Promote stage (tuỳ team còn dùng stage hay không)
- Production / Staging / Archived

Bước 12: Gán alias deploy (khuyến nghị)
- alias "production" trỏ tới 1 version tại 1 thời điểm
- service load: models:/Name@production
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RegistryConfig:
    """Cấu hình chung cho Model Registry"""
    tracking_uri: str
    # thường = tracking_uri (nếu server chung)
    registry_uri: Optional[str] = None


class ModelRegistry:
    """Wrapper cho các thao tác Model Registry (Bước 10–12)"""

    def __init__(self, config: RegistryConfig) -> None:
        """Khởi tạo Model Registry wrapper.

        Args:
            config: Cấu hình registry.
        """
        self.config = config
        # Import mlflow.tracking.MlflowClient khi cần sử dụng
        self.client = None

    # --------------------------------------------------
    # Bước 10: Register model version từ 1 run
    # --------------------------------------------------
    def register_from_run(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
    ) -> int:
        """Register model từ artifact của run.

        Args:
            run_id: MLflow run id.
            model_name: Tên registered model (vd: churn_model).
            artifact_path: Thường là "model".

        Returns:
            Version (int) của model vừa tạo.

        Raises:
            NotImplementedError: Placeholder chưa implement.
        """
        raise NotImplementedError

    # --------------------------------------------------
    # Bước 11: Promote stage (nếu team còn dùng stage)
    # --------------------------------------------------
    def set_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing_versions: bool = False,
    ) -> None:
        """Set stage cho 1 version (Staging/Production/Archived).

        Args:
            model_name: Tên registered model.
            version: Version number.
            stage: Mục tiêu stage.
            archive_existing_versions: Có archive version cũ không.

        Raises:
            NotImplementedError: Placeholder chưa implement.
        """
        raise NotImplementedError

    # --------------------------------------------------
    # Bước 12: Set alias (khuyến nghị)
    # --------------------------------------------------
    def set_alias(self, model_name: str, alias: str, version: int) -> None:
        """Gán alias trỏ vào 1 version.

        Args:
            model_name: Tên registered model.
            alias: Tên alias (vd: production).
            version: Version number.

        Raises:
            NotImplementedError: Placeholder chưa implement.
        """
        raise NotImplementedError
