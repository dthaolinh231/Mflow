from __future__ import annotations

import mlflow
import pytest

from pipeline.mlflow_manager import MLflowManager


# ==================================================
# Helper: reset singleton giữa các test
# ==================================================
def reset_mlflow_manager():
    """
    Reset trạng thái singleton của MLflowManager.

    Vì MLflowManager dùng pattern Singleton + biến class (_instance, _configured),
    nên nếu không reset, test sau sẽ bị ảnh hưởng bởi test trước
    (configure chỉ chạy 1 lần).

    → BẮT BUỘC reset trước mỗi test để test độc lập, CI không bị flaky.
    """
    MLflowManager._instance = None
    MLflowManager._configured = False


# ==================================================
# Tests
# ==================================================

def test_singleton_identity():
    """
    Test: MLflowManager đúng là Singleton.

    Mục tiêu:
    - Gọi MLflowManager() nhiều lần
    - Phải trả về CÙNG 1 instance (cùng địa chỉ bộ nhớ)

    Nếu fail:
    - Singleton bị implement sai
    - Có nguy cơ config MLflow nhiều lần → bug nguy hiểm
    """
    reset_mlflow_manager()

    m1 = MLflowManager()
    m2 = MLflowManager()

    # is → so sánh identity (cùng object), KHÔNG phải == (so sánh giá trị)
    assert m1 is m2


def test_configure_sets_tracking_experiment_and_tags(monkeypatch):
    """
    Test: configure() gọi đúng các hàm MLflow:
    - set_tracking_uri
    - set_experiment
    - set_tag cho từng default tag

    Lưu ý:
    - KHÔNG test MLflow thật
    - Chỉ test xem code có gọi đúng API MLflow không
    """
    reset_mlflow_manager()

    # Dict để ghi nhận xem các hàm MLflow có được gọi không
    calls = {
        "tracking_uri": None,
        "experiment": None,
        "tags": {},
    }

    # Mock mlflow.set_tracking_uri
    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        lambda uri: calls.update(tracking_uri=uri),
    )

    # Mock mlflow.set_experiment
    monkeypatch.setattr(
        mlflow,
        "set_experiment",
        lambda name: calls.update(experiment=name),
    )

    # Mock mlflow.set_tag
    monkeypatch.setattr(
        mlflow,
        "set_tag",
        lambda k, v: calls["tags"].update({k: v}),
    )

    # Gọi configure với đầy đủ tham số
    mgr = MLflowManager.configure(
        tracking_uri="./mlruns",
        experiment_name="churn_prediction",
        default_tags={"env": "test", "owner": "ml-team"},
    )

    # Kiểm tra trả về đúng kiểu object
    assert isinstance(mgr, MLflowManager)

    # Kiểm tra MLflow API được gọi đúng dữ liệu
    assert calls["tracking_uri"] == "./mlruns"
    assert calls["experiment"] == "churn_prediction"
    assert calls["tags"] == {
        "env": "test",
        "owner": "ml-team",
    }


def test_configure_is_idempotent(monkeypatch):
    """
    Test: configure() chỉ chạy SIDE-EFFECT đúng 1 lần.

    Ý nghĩa:
    - configure() có thể bị gọi nhiều lần trong pipeline
    - Nhưng MLflow chỉ được set_tracking_uri MỘT LẦN
    - Đây là điểm quan trọng nhất của MLflowManager
    """
    reset_mlflow_manager()

    # Đếm số lần mlflow.set_tracking_uri được gọi
    call_count = {"tracking": 0}

    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        lambda _: call_count.__setitem__(
            "tracking", call_count["tracking"] + 1
        ),
    )

    # Gọi configure 2 lần
    MLflowManager.configure(tracking_uri="./mlruns")
    MLflowManager.configure(tracking_uri="./mlruns-again")

    # Kỳ vọng: chỉ gọi MLflow đúng 1 lần
    assert call_count["tracking"] == 1


def test_get_default_tags_returns_dict():
    """
    Test: get_default_tags() trả về dict.

    Hiện tại method này là placeholder (return {}),
    nên test chỉ cần đảm bảo:
    - Trả về dict
    - Không bị None / type sai

    Sau này nếu mở rộng logic,
    test này vẫn dùng được (không phải sửa).
    """
    reset_mlflow_manager()

    mgr = MLflowManager.configure(tracking_uri="./mlruns")
    tags = mgr.get_default_tags()

    assert isinstance(tags, dict)
    assert tags == {}
