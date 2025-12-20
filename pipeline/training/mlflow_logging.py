"""Mô-đun logging dữ liệu training vào MLflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mlflow
import mlflow.sklearn


def log_params(params: Dict[str, Any]) -> None:
    """Log hyperparameters vào MLflow run hiện tại.

    Chuyển đổi giá trị sang kiểu hợp lệ (str/int/float/bool).

    Args:
        params: Dict chứa tên parameter và giá trị.
    """
    # mlflow.log_params chỉ nhận str/int/float/bool; convert nhẹ cho an toàn
    safe_params: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            safe_params[str(k)] = v
        else:
            safe_params[str(k)] = str(v)
    mlflow.log_params(safe_params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics đánh giá vào MLflow run hiện tại.

    Args:
        metrics: Dict chứa tên metric và giá trị.
        step: Bước training (nếu logging theo step).
    """
    safe_metrics: Dict[str, float] = {
        str(k): float(v) for k, v in metrics.items()
    }
    if step is None:
        mlflow.log_metrics(safe_metrics)
    else:
        for k, v in safe_metrics.items():
            mlflow.log_metric(k, v, step=step)


def set_tags(tags: Dict[str, str]) -> None:
    """Gán tags metadata cho MLflow run hiện tại.

    Tags dùng để phân loại, tìm kiếm runs.

    Args:
        tags: Dict chứa tên tag và giá trị.
    """
    for k, v in tags.items():
        if v is None:
            continue
        mlflow.set_tag(str(k), str(v))


def log_artifacts(
    artifact_paths: Iterable[Path],
    also_log_dir: Optional[Path] = None,
    artifact_dir_name: str = "outputs",
) -> None:
    """Log file artifacts (ảnh, báo cáo, config) vào MLflow.

    Args:
        artifact_paths: Danh sách đường dẫn file cần log.
        also_log_dir: Thư mục bổ sung cần log (nếu có).
        artifact_dir_name: Tên folder trong MLflow artifacts.
    """
    for p in artifact_paths:
        mlflow.log_artifact(str(p))

    # Log cả folder outputs cho tiện trace
    if also_log_dir is not None and also_log_dir.exists():
        mlflow.log_artifacts(
            str(also_log_dir),
            artifact_path=artifact_dir_name,
        )


def log_sklearn_model(
    model: Any,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
) -> None:
    """Log scikit-learn model vào MLflow artifacts.

    Args:
        model: Fitted scikit-learn model.
        artifact_path: Đường dẫn lưu trong MLflow.
        registered_model_name: Tên khi đăng ký vào Model Registry.
    """
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
        signature=None,
        input_example=None,
    )
