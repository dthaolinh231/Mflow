"""
Evaluate Pipeline - Bước 9
Load model từ MLflow và đánh giá độc lập (CI-friendly)

Mục tiêu:
- KHÔNG train lại
- Load model bằng model_uri (runs:/... hoặc models:/...)
- Tính metrics + log artifacts (report/cm)
- Có thể log vào MLflow (run riêng cho evaluate)

Input:
- model_uri: "runs:/<run_id>/model"
- experiment_name (optional): để group các run evaluate

Output:
- dict metrics (+ optional artifacts)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import os


@dataclass(frozen=True)
class EvaluateConfig:
    """Cấu hình cho Evaluate Pipeline."""
    tracking_uri: str
    experiment_name: Optional[str] = None
    model_uri: str = ""
    run_name: Optional[str] = None
    out_dir: str = "outputs_eval"
    default_tags: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config thành dict.

        Returns:
            Dict chứa toàn bộ cấu hình.
        """
        return {
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "model_uri": self.model_uri,
            "run_name": self.run_name,
            "out_dir": self.out_dir,
            "default_tags": self.default_tags,
        }


def ensure_out_dir(out_dir: str) -> Path:
    """Đảm bảo thư mục output tồn tại"""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


class EvaluatePipeline:
    """Đánh giá model bằng cách load từ MLflow (Bước 9)"""

    def __init__(self, config: EvaluateConfig) -> None:
        self.config = config
        self.out_dir = ensure_out_dir(config.out_dir)

    def validate(self) -> bool:
        """Validate cấu hình.

        Returns:
            True nếu cấu hình hợp lệ.
        """
        return bool(self.config.tracking_uri and self.config.model_uri)

    # -----------------------------
    # Data
    # -----------------------------
    def _prepare_eval_data(self) -> Tuple[Any, Any]:
        """
        Chuẩn bị dữ liệu evaluate

        TODO:
        - Load dataset eval/test thật (từ file/DB/feature store)
        - Chuẩn hóa tiền xử lý giống lúc train
        - Trả về: X_eval, y_eval
        """
        raise NotImplementedError("Placeholder: implement _prepare_eval_data")

    # -----------------------------
    # Model load
    # -----------------------------
    def _load_model(self, model_uri: str) -> Any:
        """
        Load model từ MLflow

        TODO:
        - Dùng mlflow.pyfunc.load_model(model_uri)
        - Hoặc mlflow.sklearn.load_model nếu là sklearn
        - Validations: model_uri hợp lệ, có quyền truy cập
        """
        raise NotImplementedError

    # -----------------------------
    # Evaluate
    # -----------------------------
    def _evaluate(
        self,
        model: Any,
        x_eval: Any,
        y_eval: Any,
    ) -> Dict[str, float]:
        """Tính metrics cho model.

        Args:
            model: Model cần đánh giá.
            x_eval: Feature evaluation data.
            y_eval: Target evaluation data.

        Returns:
            Dict metrics accuracy, f1, precision, recall.
        """
        raise NotImplementedError("Placeholder: implement _evaluate")

    # -----------------------------
    # Artifacts (optional)
    # -----------------------------
    def _build_artifacts(
        self,
    ) -> Dict[str, Path]:
        """Tạo artifacts đánh giá.

        Returns:
            Dict đường dẫn artifacts (ảnh, báo cáo).
        """
        return {}

    def run(self) -> Dict[str, float]:
        """
        Chạy evaluate pipeline (placeholder).
        Gợi ý các bước thực hiện thật:
        1) Validate `model_uri`
        2) Cấu hình MLflow (tracking_uri, experiment_name, tags)
        3) Bắt đầu run evaluate (mlflow.start_run)
        4) Load model theo `model_uri`
        5) Chuẩn bị dữ liệu `X_eval, y_eval`
        6) Tính metrics và log vào MLflow
        7) Sinh và log artifacts (cm, report, ...)
        8) Trả về metrics
        """
        raise NotImplementedError("Placeholder: implement EvaluatePipeline.run")


def main() -> None:
    """Entry point chạy thử local"""
    cfg = EvaluateConfig(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
        experiment_name=os.getenv(
            "MLFLOW_EXPERIMENT",
            "churn_prediction",
        ),
        model_uri=os.getenv("MODEL_URI", ""),  # bắt buộc truyền
        out_dir=os.getenv("OUT_DIR", "outputs_eval"),
        default_tags=None,
    )
    try:
        metrics = EvaluatePipeline(cfg).run()
        print("✓ Evaluate done:", metrics)
    except NotImplementedError as e:
        print("Evaluate placeholder:", e)


if __name__ == "__main__":
    main()
