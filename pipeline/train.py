"""
Train Pipeline - Bước 1–8
Thực hiện huấn luyện model và log đầy đủ thông tin vào MLflow

Các bước:
1) Khởi tạo MLflow (Init MLflow)
2) Chọn bài toán (Set experiment)
3) Tạo run (Start run)
4) Log cấu hình / hyperparameters
5) Log metrics đánh giá
6) Log artifacts (file)
7) Gán metadata tags
8) Log model
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn

from mlpipeline.tracking.mlflow_manager import MLflowManager


# =========================================================
# Cấu hình pipeline (skeleton)
# =========================================================
@dataclass(frozen=True)
class TrainConfig:
    """
    Cấu hình cho Train Pipeline
    """
    tracking_uri: str              # MLflow tracking URI
    experiment_name: str           # Tên experiment (bài toán)
    run_name: Optional[str] = None # Tên run (nếu không truyền sẽ tự sinh)
    out_dir: str = "outputs"       # Thư mục lưu artifacts
    default_tags: Optional[Dict[str, str]] = None  # Tags mặc định


# =========================================================
# Helper functions cho artifacts (SKELETON)
# =========================================================
def ensure_out_dir(out_dir: str) -> Path:
    """
    Đảm bảo thư mục output tồn tại
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_feature_config(config: Dict[str, Any], out_path: Path) -> Path:
    """
    Lưu cấu hình (feature / hyperparameter) ra file YAML
    TODO:
    - Implement ghi YAML
    """
    # TODO: ghi file YAML
    return out_path


def save_classification_report(report_text: str, out_path: Path) -> Path:
    """
    Lưu classification report ra file text
    TODO:
    - Implement ghi file .txt
    """
    # TODO: ghi file text
    return out_path


def plot_and_save_confusion_matrix(y_true, y_pred, out_path: Path) -> Path:
    """
    Vẽ và lưu confusion matrix ra file ảnh

    TODO:
    - Implement plotting (matplotlib / seaborn)
    """
    # TODO: vẽ confusion matrix và lưu .png
    return out_path


# =========================================================
# Train Pipeline chính – MLflow
# =========================================================
class TrainPipeline:
    """
    Điều phối toàn bộ quá trình train và log MLflow (Bước 1–8)
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.out_dir = ensure_out_dir(config.out_dir)

    def _prepare_data(self) -> Tuple[Any, Any, Any, Any]:
        """
        Chuẩn bị dữ liệu train / test

        TODO:
        - Load dataset
        - Chia train / test
        - Trả về: X_train, X_test, y_train, y_test
        """
        raise NotImplementedError

    def _build_model(self, params: Dict[str, Any]) -> Any:
        """
        Khởi tạo model từ hyperparameters

        TODO:
        - Tạo model (sklearn / custom)
        - Trả về instance model
        """
        raise NotImplementedError

    def _train(self, model: Any, X_train: Any, y_train: Any) -> Any:
        """
        Huấn luyện model

        TODO:
        - Fit model
        - Trả về model đã train
        """
        raise NotImplementedError

    def _evaluate(self, model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
        """
        Đánh giá model và trả về metrics

        TODO:
        - Predict
        - Tính accuracy / f1 / precision / recall / ...
        - Trả về dict metrics
        """
        raise NotImplementedError

    def run(self) -> str:
        """
        Chạy toàn bộ pipeline train và trả về MLflow run_id
        """

        # --------------------------------------------------
        # Bước 1–2: Init MLflow + set experiment
        # --------------------------------------------------
        manager = MLflowManager.configure(
            tracking_uri=self.config.tracking_uri,
            experiment_name=self.config.experiment_name,
            default_tags=self.config.default_tags,
        )

        # --------------------------------------------------
        # Chuẩn bị hyperparameters / config để log
        # --------------------------------------------------
        # TODO: định nghĩa hyperparameters thực tế
        params: Dict[str, Any] = {
            # "model_type": "...",
            # "seed": 42,
            # "feature_version": "v1",
        }

        # Quy ước đặt tên run (model + thời gian)
        run_name = self.config.run_name or (
            f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # --------------------------------------------------
        # Bước 3: Start MLflow run
        # --------------------------------------------------
        with mlflow.start_run(
            run_name=run_name,
            tags=manager.get_default_tags()
        ) as run:

            run_id = run.info.run_id

            # --------------------------------------------------
            # Bước 4: Log hyperparameters
            # --------------------------------------------------
            # TODO: flatten config nếu cần
            for k, v in params.items():
                mlflow.log_param(k, v)

            # --------------------------------------------------
            # Train & Evaluate
            # --------------------------------------------------
            X_train, X_test, y_train, y_test = self._prepare_data()
            model = self._build_model(params=params)
            model = self._train(model=model, X_train=X_train, y_train=y_train)

            # --------------------------------------------------
            # Bước 5: Log metrics
            # --------------------------------------------------
            metrics = self._evaluate(
                model=model,
                X_test=X_test,
                y_test=y_test,
            )
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # --------------------------------------------------
            # Bước 6: Log artifacts (file)
            # --------------------------------------------------
            # TODO:
            # - confusion_matrix.png
            # - feature_config.yaml
            # - classification_report.txt
            cm_path = plot_and_save_confusion_matrix(
                y_true=y_test,
                y_pred=None,  # TODO: truyền y_pred thực tế
                out_path=self.out_dir / "confusion_matrix.png",
            )

            cfg_path = save_feature_config(
                config=params,
                out_path=self.out_dir / "feature_config.yaml",
            )

            rpt_path = save_classification_report(
                report_text="",  # TODO: sinh classification report
                out_path=self.out_dir / "classification_report.txt",
            )

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(cfg_path))
            mlflow.log_artifact(str(rpt_path))

            # Có thể log toàn bộ thư mục outputs
            mlflow.log_artifacts(str(self.out_dir), artifact_path="outputs")

            # --------------------------------------------------
            # Bước 7: Gán metadata tags
            # --------------------------------------------------
            # TODO: gắn tag nghiệp vụ
            extra_tags = {
                # "author": "...",
                # "purpose": "...",
                # "status": "completed",
            }
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)

            # --------------------------------------------------
            # Bước 8: Log model
            # --------------------------------------------------
            # TODO: bổ sung signature / input_example
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=None,  # Register ở bước 10+
                signature=None,
                input_example=None,
            )

            return run_id


def main() -> None:
    """
    Entry point cho CLI (skeleton)
    """
    cfg = TrainConfig(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT", "churn_prediction"),
        run_name=None,
        out_dir=os.getenv("OUT_DIR", "outputs"),
        default_tags=None,
    )

    pipeline = TrainPipeline(config=cfg)
    run_id = pipeline.run()
    print(f"✓ Train pipeline hoàn thành. run_id={run_id}")


if __name__ == "__main__":
    import os
    main()
