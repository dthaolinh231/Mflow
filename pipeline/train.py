"""
Train Pipeline - Bước 1–8
Thực hiện huấn luyện model và log đầy đủ thông tin vào MLflow
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path

from pipeline.training.config import TrainConfig
from pipeline.training.mlflow_setup import setup_mlflow
from pipeline.training.mflow_run_context import start_train_run
from pipeline.training.mlflow_logging import (
    log_params,
    log_metrics,
    log_artifacts,
    set_tags,
    log_sklearn_model,
)
from pipeline.training.mflow_trainer import (
    prepare_data,
    build_model,
    train_model,
    evaluate_model,
)
from pipeline.training.mflow_artifacts import (
    ensure_out_dir,
    save_feature_config,
    save_classification_report,
    plot_and_save_confusion_matrix,
)


@dataclass
class TrainState:
    """State container cho training process.

    Attributes:
        params: Hyperparameters dict.
        out_dir: Output directory.
        extra_tags: Extra tags để gán.
    """
    params: dict
    out_dir: Path
    extra_tags: dict


def main(cfg: TrainConfig) -> str:
    """Chạy training pipeline từ chuẩn bị dữ liệu đến lưu model.

    Thực hiện các bước 1-8 của MLflow training:
    - Khởi tạo MLflow + set experiment
    - Log hyperparameters
    - Train model
    - Log metrics và artifacts
    - Gán tags
    - Lưu model

    Args:
        cfg: Cấu hình training (TrainConfig).

    Returns:
        run_id: ID của MLflow run vừa tạo.
    """
    # Bước 1-2: Init MLflow + set experiment
    manager = setup_mlflow(cfg)
    params = cfg.params or {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
        "feature_version": "v1",
        "threshold": 0.5,
    }
    out_dir = ensure_out_dir(cfg.out_dir)
    state = TrainState(params=params, out_dir=out_dir,
                       extra_tags=cfg.extra_tags or {})

    # Bước 3: Start MLflow run
    with start_train_run(cfg, manager) as run:
        run_id = run.info.run_id
        _log_and_train(state)

    return run_id


@dataclass
class ArtifactState:
    """State container cho artifact saving.

    Attributes:
        out_dir: Output directory.
        y_test: Test labels.
        y_pred: Predictions.
        params: Hyperparameters.
        report_text: Classification report.
        extra_tags: Extra tags.
    """
    out_dir: Path
    y_test: object
    y_pred: object
    params: dict
    report_text: str
    extra_tags: dict


def _log_and_train(state: TrainState) -> None:
    """Train và log model sử dụng state object.

    Args:
        state: TrainState object chứa tất cả parameters.
    """
    # Bước 4: Log hyperparameters
    log_params(state.params)

    # Train & Evaluate
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(build_model(state.params), x_train, y_train)
    metrics, y_pred, report_text = evaluate_model(model, x_test, y_test)

    # Bước 5: Log metrics
    log_metrics(metrics)

    # Bước 6-7: Save artifacts & Tags
    artifact_state = ArtifactState(
        out_dir=state.out_dir, y_test=y_test, y_pred=y_pred,
        params=state.params, report_text=report_text,
        extra_tags=state.extra_tags,
    )
    _save_and_log_artifacts(artifact_state)

    # Bước 8: Log model
    log_sklearn_model(model, artifact_path="model")


def _save_and_log_artifacts(state: ArtifactState) -> None:
    """Save artifacts và log vào MLflow.

    Args:
        state: ArtifactState object chứa tất cả artifacts data.
    """
    cm_path = plot_and_save_confusion_matrix(
        y_true=state.y_test,
        y_pred=state.y_pred,
        out_path=state.out_dir / "confusion_matrix.png",
    )

    cfg_path = save_feature_config(
        state.params, state.out_dir / "feature_config.yaml",
    )

    rpt_path = save_classification_report(
        report_text=state.report_text,
        out_path=state.out_dir / "classification_report.txt",
    )

    log_artifacts([cfg_path, rpt_path, cm_path],
                  also_log_dir=state.out_dir)

    # Tags
    set_tags(state.extra_tags or {})

    if __name__ == "__main__":
        cfg = TrainConfig(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT", "churn_prediction"),
            run_name=None,
            out_dir=os.getenv("OUT_DIR", "outputs"),
            default_tags=None,
            extra_tags=None,
            params=None,
        )

        run_id = main(cfg)
        print(f"✓ Train pipeline hoàn thành. run_id={run_id}")
