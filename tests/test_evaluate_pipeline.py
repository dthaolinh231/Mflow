"""
Unit tests cho EvaluatePipeline.

EvaluatePipeline làm 2 nhóm việc:
1) Logic thuần (không side-effect):
   - _resolve_average()
   - _normalize_pred()
   - _evaluate() (tính metric)
2) Orchestration (side-effect):
   - run(): set_tracking_uri, set_experiment, start_run, log_metrics, log_artifact, ...

Unit test cần đạt:
- Test logic thuần: dùng dữ liệu nhỏ, deterministic, assert giá trị rõ ràng.
- Test orchestration: monkeypatch MLflow + artifact helpers để:
    + không tạo mlruns thật
    + không tạo file ảnh thật
    + không phụ thuộc environment
    + vẫn đảm bảo "đúng luồng gọi" và "đúng tham số"

Lưu ý:
------
- Không cần integration test ở đây (integration đã dành cho MLflowManager).
- Khi test run(), phải đảm bảo không bị lỗi MLflow "active run" do global state:
  -> ta mock start_run bằng context manager giả để tránh dính state MLflow thật.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import mlflow

from pipeline.evaluate_pipeline import EvaluateConfig, EvaluatePipeline


# Helpers: Dummy model + dummy run context
class DummyRun:
    """Giả lập context manager của mlflow.start_run.

    Tại sao cần?
    - mlflow.start_run thật dùng global state.
    - Nếu test fail giữa chừng, run có thể bị "active" → test sau crash.
    - Do đó unit test nên mock start_run bằng context manager giả.
    """

    class Info:
        run_id = "dummy-run-id"

    info = Info()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # False = không nuốt exception
        return False


class DummyPredictModel:
    """Model giả có predict() đúng contract, trả về nhãn dự đoán."""

    def __init__(self, y_pred: Any):
        self._y_pred = y_pred

    def predict(self, x: pd.DataFrame) -> Any:
        # cố tình không dùng x để test pure behavior
        return self._y_pred


# Unit tests: _resolve_average()
def test_resolve_average_default_binary():
    """Nếu average=None và y_true là binary → fallback 'binary'."""
    cfg = EvaluateConfig(tracking_uri=".", model_uri="runs:/x/model", average=None)
    pipe = EvaluatePipeline(cfg)

    y_true = pd.Series([0, 1, 0, 1], name="target")
    assert pipe._resolve_average(y_true) == "binary"


def test_resolve_average_default_multiclass():
    """Nếu average=None và y_true là multiclass → fallback 'macro'."""
    cfg = EvaluateConfig(tracking_uri=".", model_uri="runs:/x/model", average=None)
    pipe = EvaluatePipeline(cfg)

    y_true = pd.Series([0, 1, 2, 1], name="target")
    assert pipe._resolve_average(y_true) == "macro"


def test_resolve_average_fail_fast_when_binary_on_multiclass():
    """Nếu user set average='binary' nhưng y_true là multiclass → raise ValueError."""
    cfg = EvaluateConfig(tracking_uri=".", model_uri="runs:/x/model", average="binary")
    pipe = EvaluatePipeline(cfg)

    y_true = pd.Series([0, 1, 2, 1], name="target")
    with pytest.raises(ValueError):
        pipe._resolve_average(y_true)


# Unit tests: _normalize_pred()
def test_normalize_pred_from_list_to_1d_numpy():
    """_normalize_pred() phải chuẩn hoá list → np.ndarray 1D."""
    cfg = EvaluateConfig(tracking_uri=".", model_uri="runs:/x/model")
    pipe = EvaluatePipeline(cfg)

    out = pipe._normalize_pred([0, 1, 0])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


def test_normalize_pred_ravel_2d_to_1d():
    """_normalize_pred() phải ravel array 2D thành 1D."""
    cfg = EvaluateConfig(tracking_uri=".", model_uri="runs:/x/model")
    pipe = EvaluatePipeline(cfg)

    out = pipe._normalize_pred(np.array([[0], [1], [0]]))
    assert out.shape == (3,)


# Unit tests: _evaluate() – tính metrics
def test_evaluate_binary_metrics_are_correct():
    """Test _evaluate() trả metrics đúng với bài toán binary.

    Setup:
    - y_true = [0,1,0,1]
    - y_pred = [0,1,1,1]
    => accuracy = 3/4 = 0.75

    F1/precision/recall phụ thuộc pos_label=1 và average='binary'
    """
    cfg = EvaluateConfig(
        tracking_uri=".",
        model_uri="runs:/x/model",
        task_type="classification",
        average="binary",
        pos_label=1,
    )
    pipe = EvaluatePipeline(cfg)

    x_eval = pd.DataFrame({"f_00": [1, 2, 3, 4]})
    y_eval = pd.Series([0, 1, 0, 1], name="target")
    model = DummyPredictModel([0, 1, 1, 1])

    metrics, y_pred, report = pipe._evaluate(model, x_eval, y_eval)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert isinstance(y_pred, np.ndarray)
    assert "precision" in report  # classification_report text


def test_evaluate_raises_when_task_not_classification():
    """Nếu config.task_type != classification → _evaluate raise NotImplementedError."""
    cfg = EvaluateConfig(
        tracking_uri=".",
        model_uri="runs:/x/model",
        task_type="regression",
    )
    pipe = EvaluatePipeline(cfg)

    x_eval = pd.DataFrame({"f_00": [1, 2, 3]})
    y_eval = pd.Series([0, 1, 0], name="target")
    model = DummyPredictModel([0, 1, 0])

    with pytest.raises(NotImplementedError):
        pipe._evaluate(model, x_eval, y_eval)


# Unit tests: _load_model() – chọn flavor + validate predict
def test_load_model_uses_pyfunc_when_default(monkeypatch):
    """Nếu load_flavor='pyfunc' (default) → gọi mlflow.pyfunc.load_model."""
    # load_flavor="pyfunc" là điều kiện kích hoạt nhánh pyfunc trong _load_model()    
    cfg = EvaluateConfig(tracking_uri=".", model_uri="runs:/x/model", load_flavor="pyfunc")
    pipe = EvaluatePipeline(cfg)

    # Tạo model giả để giả lập kết quả load từ MLflow
    fake_model = DummyPredictModel([0, 1, 0])
    calls: Dict[str, str] = {}
    def fake_load_model(uri):
        """
        Hàm giả thay thế mlflow.pyfunc.load_model.

        Vai trò:
        - Ghi lại uri được truyền vào (để assert sau)
        - Trả về fake_model (để pipeline chạy tiếp)
        """

        calls["uri"] = uri
        return fake_model
    # Patch pyfunc → fake_load_model (CHO PHÉP gọi)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", fake_load_model)

    # Patch sklearn → FAIL nếu bị gọi
    monkeypatch.setattr(
        mlflow.sklearn,
        "load_model",
        lambda _: (_ for _ in ()).throw(
            AssertionError("Should not call sklearn")
        ),
    )
    # Gọi hàm cần test
    model = pipe._load_model("runs:/abc/model")

    # Assert kết quả trả về
    assert model is fake_model
    assert calls["uri"] == "runs:/abc/model"


def test_load_model_uses_sklearn_when_configured(monkeypatch):
    """Nếu load_flavor='sklearn' → gọi mlflow.sklearn.load_model."""
    cfg = EvaluateConfig(tracking_uri=".", model_uri="runs:/x/model", load_flavor="sklearn")
    pipe = EvaluatePipeline(cfg)

    fake_model = DummyPredictModel([0, 1, 0])
    calls: Dict[str, str] = {}

    cfg = EvaluateConfig(tracking_uri=".", model_uri="runs:/x/model", load_flavor="sklearn")
    pipe = EvaluatePipeline(cfg)

    fake_model = DummyPredictModel([0, 1, 0])
    calls: Dict[str, str] = {}

    def fake_sklearn_load_model(uri):
        calls["uri"] = uri
        return fake_model

    monkeypatch.setattr(mlflow.sklearn, "load_model", fake_sklearn_load_model)
    monkeypatch.setattr(
        mlflow.pyfunc,
        "load_model",
        lambda _: (_ for _ in ()).throw(
            AssertionError("Should not call pyfunc")
        ),
    )

    model = pipe._load_model("runs:/abc/model")

    assert model is fake_model
    assert calls["uri"] == "runs:/abc/model"


def test_load_model_raises_if_no_predict(monkeypatch):
    """Nếu model load ra không có .predict() → raise TypeError."""
    cfg = EvaluateConfig(tracking_uri=".", model_uri="runs:/x/model", load_flavor="pyfunc")
    pipe = EvaluatePipeline(cfg)

    class NoPredict:  # pylint: disable=too-few-public-methods
        pass

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda _: NoPredict())

    with pytest.raises(TypeError):
        pipe._load_model("runs:/abc/model")


# ============================================================
# Unit tests: run() – orchestration + logging đúng
# ============================================================

def test_run_happy_path_logs_metrics_and_artifacts(monkeypatch, tmp_path: Path):
    """Test run() happy path.

    Mục tiêu:
    - Không dùng MLflow thật (tránh active run state)
    - Ensure pipeline gọi đúng các bước:
        + set_tracking_uri
        + set_experiment (nếu có)
        + start_run
        + log_metrics
        + log_artifact cho từng artifact
    - Return metrics dict

    Đây là test quan trọng nhất cho orchestration.
    """
    cfg = EvaluateConfig(
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="test_exp",
        model_uri="runs:/abc/model",
        out_dir=str(tmp_path / "outputs_eval"),
        load_flavor="pyfunc",
        default_tags={"env": "test"},
        average="binary",
        pos_label=1,
    )
    pipe = EvaluatePipeline(cfg)

    # ----- Mock MLflow core calls
    calls: Dict[str, Any] = {"tags": {}, "artifacts": [], "metrics": None}

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: calls.setdefault("tracking_uri", uri))
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: calls.setdefault("experiment", name))
    monkeypatch.setattr(mlflow, "set_tag", lambda k, v: calls["tags"].__setitem__(k, v))
    monkeypatch.setattr(mlflow, "start_run", lambda run_name=None: DummyRun())
    monkeypatch.setattr(mlflow, "log_metrics", lambda m: calls.__setitem__("metrics", m))
    monkeypatch.setattr(mlflow, "log_artifact", lambda p: calls["artifacts"].append(p))

    # ----- Mock load_model → trả Dummy model
    monkeypatch.setattr(pipe, "_load_model", lambda _: DummyPredictModel([0, 1, 0, 1]))

    # ----- Mock _prepare_eval_data → data nhỏ deterministic
    x_eval = pd.DataFrame({"f_00": [1, 2, 3, 4]})
    y_eval = pd.Series([0, 1, 0, 1], name="target")
    monkeypatch.setattr(pipe, "_prepare_eval_data", lambda: (x_eval, y_eval))

    # ----- Mock artifact helper functions → trả path giả tồn tại
    # (run() chỉ cần string path để log_artifact)
    report_path = tmp_path / "outputs_eval" / "classification_report.txt"
    cm_path = tmp_path / "outputs_eval" / "confusion_matrix.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("dummy report", encoding="utf-8")
    cm_path.write_bytes(b"fakepng")

    monkeypatch.setattr(
        "pipeline.evaluate_pipeline.save_classification_report",
        lambda report_text, out_path: report_path,
    )
    monkeypatch.setattr(
        "pipeline.evaluate_pipeline.plot_and_save_confusion_matrix",
        lambda y_true, y_pred, out_path: cm_path,
    )

    metrics = pipe.run()

    # ----- Assert return + logging behavior
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics

    assert calls["tracking_uri"] == str(tmp_path / "mlruns")
    assert calls["experiment"] == "test_exp"
    assert calls["tags"]["env"] == "test"

    assert calls["metrics"] is not None
    assert any(str(p).endswith("classification_report.txt") for p in calls["artifacts"])
    assert any(str(p).endswith("confusion_matrix.png") for p in calls["artifacts"])


def test_run_raises_when_config_invalid(tmp_path: Path):
    """Nếu config thiếu tracking_uri hoặc model_uri → run() phải raise ValueError."""
    cfg = EvaluateConfig(
        tracking_uri="",           # invalid
        model_uri="",              # invalid
        out_dir=str(tmp_path / "outputs"),
    )
    pipe = EvaluatePipeline(cfg)

    with pytest.raises(ValueError):
        pipe.run()
