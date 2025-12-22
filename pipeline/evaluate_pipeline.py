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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from typing_extensions import Protocol, runtime_checkable

# sklearn imports
from sklearn.utils.multiclass import type_of_target
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

# MLflow
import mlflow

# Artifacts helpers
from pipeline.training.mflow_artifacts import (
    save_classification_report, plot_and_save_confusion_matrix
)


# pylint: disable=too-few-public-methods
@runtime_checkable
class SupportsPredict(Protocol):
    """Model interface tối thiểu cho EvaluatePipeline.
    - Ưu tiên predict(DataFrame) theo pyfunc convention.
    - Nếu model không support DataFrame, ta sẽ bọc adapter.
    ... là cú pháp của Python dùng trong Protocol / interface
    """
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return predictions for input features."""
        # pylint: disable=unnecessary-ellipsis
        ...


# pylint: disable=too-many-instance-attributes,too-few-public-methods
@dataclass(frozen=True)
class EvaluateConfig:
    """Cấu hình cho Evaluate Pipeline..
    """
    tracking_uri: str
    experiment_name: Optional[str] = None
    model_uri: str = ""
    run_name: Optional[str] = None
    out_dir: str = "outputs_eval"
    default_tags: Optional[Dict[str, str]] = None
    load_flavor: str = "pyfunc"  # "pyfunc" | "sklearn"
    task_type: str = "classification"  # classification | regression
    average: Optional[str] = None      # binary | macro | weighted
    pos_label: int = 1    # chỉ dùng khi average="binary"

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
            "load_flavor": self.load_flavor,
            "task_type": self.task_type,
            "average": self.average,
            "pos_label": self.pos_label,
        }


def ensure_out_dir(out_dir: str) -> Path:
    """Đảm bảo thư mục output tồn tại."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_dataframe(x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Chuẩn hoá input X thành DataFrame.

    - Nếu đã là DataFrame: giữ nguyên.
    - Nếu là ndarray: bọc thành DataFrame với cột f_00..f_nn.
    """
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, np.ndarray):
        cols = [f"f_{i:02d}" for i in range(x.shape[1])]
        return pd.DataFrame(x, columns=cols)
    raise TypeError(f"Unsupported X type: {type(x)}")


def _ensure_series(y: pd.Series | np.ndarray) -> pd.Series:
    """Chuẩn hoá y thành Series."""
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, np.ndarray):
        return pd.Series(y, name="target")
    raise TypeError(f"Unsupported y type: {type(y)}")


class EvaluatePipeline:
    """Đánh giá model bằng cách load từ MLflow (Bước 9)
    Implements model evaluation pipeline for MLflow models.
    """

    def __init__(self, config: EvaluateConfig) -> None:
        """Initialize EvaluatePipeline with config."""
        self.config = config
        self.out_dir = ensure_out_dir(config.out_dir)

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid.
        """
        return bool(self.config.tracking_uri and self.config.model_uri)

    def _prepare_eval_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare evaluation data (placeholder).

        Returns:
            Tuple of X_eval (DataFrame), y_eval (Series)
        """
        x, y = make_classification(
            n_samples=300,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42,
        )
        cols = [f"f_{i:02d}" for i in range(x.shape[1])]
        x_df = pd.DataFrame(x, columns=cols)
        y_s = pd.Series(y, name="target")
        return x_df, y_s

    def _load_model(self, model_uri: str) -> SupportsPredict:
        """Load model from MLflow."""
        if self.config.load_flavor == "sklearn":
            model = mlflow.sklearn.load_model(model_uri)
        else:
            model = mlflow.pyfunc.load_model(model_uri)

        if not hasattr(model, "predict"):
            raise TypeError("Loaded model does not have .predict()")
        return cast(SupportsPredict, model)

    def _resolve_average(self, y_true: pd.Series) -> str:
        """Chọn average cho precision/recall/f1 theo config (Cách 2), có fallback.

        Rules:
        - Nếu config.average set -> dùng đúng cái đó (fail fast nếu không phù hợp).
        - Nếu average None -> fallback:
            - binary -> "binary"
            - multiclass -> "macro"
        """
        tgt = type_of_target(y_true)
        avg = self.config.average

        # Fallback nếu user không set
        if avg is None:
            return "binary" if tgt == "binary" else "macro"

        # Validate cơ bản để fail fast
        if avg == "binary" and tgt != "binary":
            raise ValueError(
                f"average='binary' is invalid for target_type='{tgt}'. "
                "Set average to 'macro'/'weighted' for multiclass."
            )
        return avg

    def _normalize_pred(self, y_pred: Any) -> np.ndarray:
        """Normalize y_pred -> 1D numpy array (labels)."""
        if isinstance(y_pred, (pd.Series, pd.DataFrame)):
            y_pred = np.asarray(y_pred)

        if isinstance(y_pred, list):
            y_pred = np.asarray(y_pred)

        if not isinstance(y_pred, np.ndarray):
            y_pred = np.asarray(y_pred)

        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        return y_pred

    def _evaluate(
        self,
        model: SupportsPredict,
        x_eval: pd.DataFrame,
        y_eval: pd.Series,
    ) -> Tuple[Dict[str, float], np.ndarray, str]:
        """Tính metrics cho model.

        Args:
            model: Model cần đánh giá.
            x_eval: Feature evaluation data.
            y_eval: Target evaluation data.

        Returns:
            Dict metrics accuracy, f1, precision, recall.
        """
        if self.config.task_type != "classification":
            raise NotImplementedError(
                f"task_type='{self.config.task_type}' not implemented. "
                "Current pipeline implements classification only."
            )

        x_eval = _ensure_dataframe(x_eval)
        y_eval = _ensure_series(y_eval)

        y_pred = self._normalize_pred(model.predict(x_eval))
        avg = self._resolve_average(y_eval)

        metric_kwargs: Dict[str, Any] = {"average": avg}
        if avg == "binary":
            metric_kwargs["pos_label"] = self.config.pos_label

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_eval, y_pred)),
            "f1": float(f1_score(y_eval, y_pred, **metric_kwargs)),
            "precision": float(precision_score(y_eval, y_pred, **metric_kwargs)),
            "recall": float(recall_score(y_eval, y_pred, **metric_kwargs)),
        }

        report_text = classification_report(y_eval, y_pred)
        return metrics, y_pred, report_text

    def _build_artifacts(
        self,
        y_true: Any,
        y_pred: Any,
        report_text: str,
    ) -> Dict[str, Path]:
        """Tạo artifacts đánh giá.

        Returns:
            Dict đường dẫn artifacts (ảnh, báo cáo).
        """
        report_path = save_classification_report(
            report_text=report_text,
            out_path=self.out_dir / "classification_report.txt",
        )
        cm_path = plot_and_save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            out_path=self.out_dir / "confusion_matrix.png",
        )
        return {"report": report_path, "confusion_matrix": cm_path}

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
        if not self.validate():
            raise ValueError("EvaluateConfig invalid: tracking_uri and model_uri are required")

        mlflow.set_tracking_uri(self.config.tracking_uri)
        if self.config.experiment_name:
            mlflow.set_experiment(self.config.experiment_name)

        run_name = self.config.run_name or "evaluate"

        with mlflow.start_run(run_name=run_name):
            # tags (optional)
            if self.config.default_tags:
                for k, v in self.config.default_tags.items():
                    mlflow.set_tag(str(k), str(v))

            model = self._load_model(self.config.model_uri)
            x_eval, y_eval = self._prepare_eval_data()
            metrics, y_pred, report_text = self._evaluate(model, x_eval, y_eval)
            mlflow.log_metrics(metrics)
            artifacts = self._build_artifacts(y_eval, y_pred, report_text)
            for _, p in artifacts.items():
                mlflow.log_artifact(str(p))
            return metrics


def main() -> None:
    """Entry point chạy thử local"""

    cfg = EvaluateConfig(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT", "churn_prediction"),
        model_uri=os.getenv("MODEL_URI", ""),
        out_dir=os.getenv("OUT_DIR", "outputs_eval"),
        load_flavor=os.getenv("LOAD_FLAVOR", "pyfunc"),
        task_type=os.getenv("TASK_TYPE", "classification"),
        average=os.getenv("AVERAGE") or None,   # ví dụ: "macro"
        pos_label=int(os.getenv("POS_LABEL", "1")),
    )
    metrics = EvaluatePipeline(cfg).run()
    print("✓ Evaluate done:", metrics)


if __name__ == "__main__":
    main()
