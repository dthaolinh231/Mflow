from __future__ import annotations

from pathlib import Path

from pipeline.train import main
from pipeline.training.config import TrainConfig


class DummyRun:
    """Giả lập mlflow.start_run context manager"""

    class Info:
        run_id = "dummy-run-id"

    info = Info()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyModel:
    """Model giả, không train thật"""
    pass


def test_train_main_happy_path(monkeypatch, tmp_path):
    # Mock MLflow setup & run (PATCH ĐÚNG MODULE: pipeline.train)
    monkeypatch.setattr("pipeline.train.setup_mlflow", lambda cfg: object())
    monkeypatch.setattr("pipeline.train.start_train_run", lambda cfg, manager: DummyRun())

    # Mock trainer functions
    monkeypatch.setattr("pipeline.train.prepare_data", lambda: ("xtr", "xte", "ytr", "yte"))
    monkeypatch.setattr("pipeline.train.build_model", lambda params: DummyModel())
    monkeypatch.setattr("pipeline.train.train_model", lambda model, x, y: model)
    monkeypatch.setattr(
        "pipeline.train.evaluate_model",
        lambda model, x, y: ({"accuracy": 1.0}, ["y_pred"], "classification report"),
    )

    # Mock MLflow logging
    monkeypatch.setattr("pipeline.train.log_params", lambda params: None)
    monkeypatch.setattr("pipeline.train.log_metrics", lambda metrics: None)
    monkeypatch.setattr("pipeline.train.log_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr("pipeline.train.log_sklearn_model", lambda *args, **kwargs: None)
    monkeypatch.setattr("pipeline.train.log_pyfunc_model", lambda *args, **kwargs: None)
    monkeypatch.setattr("pipeline.train.set_tags", lambda tags: None)

    # Mock artifact helpers
    monkeypatch.setattr("pipeline.train.ensure_out_dir", lambda _: Path(tmp_path))
    monkeypatch.setattr("pipeline.train.save_feature_config", lambda *args, **kwargs: tmp_path / "cfg.yaml")
    monkeypatch.setattr("pipeline.train.save_classification_report", lambda *args, **kwargs: tmp_path / "rpt.txt")
    monkeypatch.setattr("pipeline.train.plot_and_save_confusion_matrix", lambda *args, **kwargs: tmp_path / "cm.png")

    cfg = TrainConfig(
        tracking_uri="./mlruns",
        experiment_name="test_exp",
        out_dir=str(tmp_path),
        params={"n_estimators": 10},
    )

    run_id = main(cfg)
    assert run_id == "dummy-run-id"
