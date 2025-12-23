"""
Unit tests cho ModelRegistry.

- ModelRegistry là wrapper mỏng (thin wrapper) quanh MLflow API.
- Không chứa logic nghiệp vụ phức tạp.
- Vì vậy:
    + Unit test là ĐỦ và ĐÚNG.
    + Monkeypatch là CHỦ ĐÍCH và HỢP LÝ.
    + KHÔNG test MLflow thật (tránh flaky, chậm, phụ thuộc môi trường).

Mục tiêu của test:
1) Đảm bảo ModelRegistry:
   - Build đúng tham số (model_uri, version, stage, alias)
   - Gọi đúng MLflow API / MlflowClient API
   - Không silent bug khi refactor

2) Nếu test FAIL:
   - Code wrapper có bug
   - Hoặc interface MLflow bị dùng sai
   → cần sửa code, KHÔNG sửa test để pass
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlflow

from pipeline.model_registry import ModelRegistry, RegistryConfig


# Test: _get_client() – lazy init MlflowClient
def test_get_client_lazy_init(monkeypatch):
    """
    Test: ModelRegistry._get_client() chỉ khởi tạo MlflowClient khi cần.

    - Tránh tạo MlflowClient sớm (side-effect không cần thiết).
    - Client chỉ được tạo ở lần gọi đầu tiên.
    - Các lần gọi sau phải reuse cùng instance.
    """
    registry = ModelRegistry(
        RegistryConfig(tracking_uri="./mlruns")
    )

    fake_client = MagicMock()

    # Monkeypatch class MlflowClient trong module đang test
    monkeypatch.setattr(
        "pipeline.model_registry.MlflowClient",
        lambda: fake_client,
    )

    client_1 = registry._get_client()
    client_2 = registry._get_client()

    assert client_1 is fake_client
    assert client_2 is fake_client


# Test: register_from_run()
def test_register_from_run(monkeypatch):
    """
    Test: register_from_run() gọi đúng mlflow.register_model
    và trả về version dưới dạng int.
    - model_uri phải đúng format: runs:/<run_id>/<artifact_path>
    - name truyền đúng
    - version trả về phải là int (không phải str)

    Nếu fail:
    - Model URI build sai → registry load model sẽ lỗi
    - Ép kiểu version sai → downstream code crash
    """
    registry = ModelRegistry(
        RegistryConfig(tracking_uri="./mlruns")
    )

    # Fake ModelVersion object trả về từ mlflow.register_model
    fake_model_version = MagicMock()
    fake_model_version.version = "3"  # MLflow trả string

    calls = {}

    def fake_register_model(model_uri: str, name: str):
        calls["model_uri"] = model_uri
        calls["name"] = name
        return fake_model_version

    monkeypatch.setattr(
        mlflow,
        "register_model",
        fake_register_model,
    )

    version = registry.register_from_run(
        run_id="run_123",
        model_name="churn_model",
        artifact_path="model",
    )

    assert version == 3
    assert calls["model_uri"] == "runs:/run_123/model"
    assert calls["name"] == "churn_model"


# Test: set_stage()
def test_set_stage_calls_client_transition(monkeypatch):
    """
    Test: set_stage() gọi đúng
    MlflowClient.transition_model_version_stage().
    Ý nghĩa:
    - Đảm bảo wrapper không truyền sai tham số
    - Stage, version, archive flag phải chính xác

    Nếu fail:
    - Promote model sai stage
    - Có thể archive nhầm version production
    """
    registry = ModelRegistry(
        RegistryConfig(tracking_uri="./mlruns")
    )

    fake_client = MagicMock()

    # Bypass lazy init, inject client giả
    monkeypatch.setattr(
        registry,
        "_get_client",
        lambda: fake_client,
    )

    registry.set_stage(
        model_name="churn_model",
        version=5,
        stage="Production",
        archive_existing_versions=True,
    )

    fake_client.transition_model_version_stage.assert_called_once_with(
        name="churn_model",
        version="5",  # MLflow API yêu cầu string
        stage="Production",
        archive_existing_versions=True,
    )


# Test: set_alias()
def test_set_alias_calls_client_set_alias(monkeypatch):
    """
    Test: set_alias() gọi đúng
    MlflowClient.set_registered_model_alias().
    Ý nghĩa:
    - Alias (vd: production) là best practice thay cho stage
    - Nếu gọi sai → service load sai model version

    Nếu fail:
    - Alias không trỏ đúng version
    - Deploy production bị sai model
    """
    registry = ModelRegistry(
        RegistryConfig(tracking_uri="./mlruns")
    )

    fake_client = MagicMock()

    monkeypatch.setattr(
        registry,
        "_get_client",
        lambda: fake_client,
    )

    registry.set_alias(
        model_name="churn_model",
        alias="production",
        version=7,
    )

    fake_client.set_registered_model_alias.assert_called_once_with(
        name="churn_model",
        alias="production",
        version="7",
    )
