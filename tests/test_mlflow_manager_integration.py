"""
Integration tests cho MLflowManager.

Mục tiêu của integration test:
- KHÔNG monkeypatch các hàm mlflow
- Dùng tracking_uri trỏ vào thư mục tạm (tmp_path) để:
  + không ghi bẩn vào project,
  + không phụ thuộc môi trường máy,
  + chạy được trong CI.

Vì sao cần integration test?
- Unit test (dùng monkeypatch) chỉ chứng minh code của ta "gọi đúng API".
- Nhưng vẫn có thể xảy ra tình huống:
  + mlflow version thay đổi,
  + tracking uri format sai,
  + quyền ghi file bị lỗi,
  + backend store gặp vấn đề
  → lúc chạy thật mới fail.
- Integration test sẽ bắt các lỗi này sớm.

Lưu ý quan trọng:
- MLflowManager là Singleton + idempotent (configure chỉ chạy 1 lần).
- Do đó mỗi test phải reset trạng thái singleton để đảm bảo độc lập.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from pipeline.mlflow_manager import MLflowManager


# Đánh dấu toàn bộ file này là integration tests.
# Khi chạy pytest, ta có thể tách:
#   - Unit tests: pytest -m "not integration"
#   - Integration tests: pytest -m integration
pytestmark = pytest.mark.integration


def reset_mlflow_manager() -> None:
    """Reset trạng thái singleton của MLflowManager giữa các test.

    Vì MLflowManager dùng:
      - MLflowManager._instance
      - MLflowManager._configured

    Nếu không reset:
    - Test trước gọi configure() sẽ set _configured=True
    - Test sau gọi configure() sẽ return sớm, không chạy setup MLflow thật
    -> Làm integration test sai và flaky.

    Do đó integration test phải reset để:
    - mỗi test chạy độc lập
    - không phụ thuộc thứ tự chạy
    """
    MLflowManager._instance = None
    MLflowManager._configured = False


@pytest.fixture(autouse=True)
def _cleanup_mlflow_active_run() -> None:
    """Luôn đảm bảo không có MLflow run nào đang active trước/sau mỗi test.

    Vì MLflow dùng global state (active run) theo process,
    nên nếu test trước không end_run đúng cách hoặc bị crash,
    test sau sẽ gặp lỗi:
      - Run already active
      - Run not found
    """
    # Trước test: end hết run đang active (nếu có)
    while mlflow.active_run() is not None:
        mlflow.end_run()

    yield

    # Sau test: end hết run đang active (phòng test fail giữa chừng)
    while mlflow.active_run() is not None:
        mlflow.end_run()


def test_configure_then_can_create_run_and_log_metric(tmp_path: Path) -> None:
    """Integration: configure() xong thì MLflow có thể start_run và log_metric thật.

    Ý nghĩa test:
    1) Gọi MLflowManager.configure() với tracking_uri là thư mục tạm.
    2) Gọi mlflow.start_run() thật.
    3) Log metric thật.
    4) Kiểm tra thư mục tracking (mlruns) được tạo và có dữ liệu bên trong.

    Nếu test fail thường do:
    - tracking_uri không hợp lệ / không ghi được
    - MLflow backend store có lỗi
    - Permission/FS lỗi trong môi trường chạy (đặc biệt WSL/CI)
    """
    reset_mlflow_manager()

    tracking_dir = tmp_path / "mlruns"

    # Configure MLflow "thật" (không monkeypatch)
    MLflowManager.configure(
        tracking_uri=str(tracking_dir),
        experiment_name="it_exp",
        default_tags={"env": "integration"},
    )

    # BẮT BUỘC – gọi lại lần nữa cho chắc
    mlflow.end_run()

    with mlflow.start_run(run_name="it_run"):
        mlflow.log_metric("it_metric", 1.0)

    assert tracking_dir.exists()
    #tracking_dir phải có dữ liệu vì đã tạo run/metric.
    assert any(tracking_dir.iterdir())

def test_configure_is_idempotent_in_real_environment(tmp_path: Path) -> None:
    """Integration: configure() idempotent - gọi 2 lần không làm thay đổi cấu hình MLflow.

    Ý nghĩa:
    - MLflowManager.configure() chỉ chạy "side-effect" 1 lần.
    - Gọi configure lần 2 phải return sớm (vì _configured=True).
    - Điều ta cần kiểm chứng trong môi trường thật:
        + configure lần 2 không làm hỏng khả năng start_run/log_metric.

    Cách kiểm:
    1) configure lần 1: tracking_uri=dir_a
    2) configure lần 2: tracking_uri=dir_b (cố tình khác)
    3) start_run + log_metric
    4) Kiểm tra dữ liệu được ghi vào dir_a (vì lần 2 bị bỏ qua)

    Nếu fail:
    - Có thể idempotent logic sai
    - Hoặc singleton reset không đúng
    """
    reset_mlflow_manager()

    dir_a = tmp_path / "mlruns_a"
    dir_b = tmp_path / "mlruns_b"

    MLflowManager.configure(
        tracking_uri=str(dir_a),
        experiment_name="it_exp_a",
        default_tags={"env": "integration"},
    )

    # Gọi lần 2 với tracking_uri khác.
    # Theo logic idempotent, lần này sẽ return ngay và không set_tracking_uri lại.
    MLflowManager.configure(
        tracking_uri=str(dir_b),
        experiment_name="it_exp_b",
        default_tags={"env": "integration-2"},
    )

    # end run ngay trước start_run để đảm bảo k bị lỗi "Run already active"
    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name="it_run_idempotent") as run:
        mlflow.log_metric("it_metric", 2.0)

    # Kỳ vọng: dữ liệu được ghi vào dir_a, và dir_b không nhất thiết tồn tại
    assert dir_a.exists(), "dir_a phải tồn tại vì configure lần 1 đã set tracking_uri vào đó."
    assert any(dir_a.iterdir()), "dir_a phải có dữ liệu vì đã tạo run/metric."

    # dir_b có thể không tồn tại vì configure lần 2 bị bỏ qua hoàn toàn
    assert not dir_b.exists() or not any(dir_b.iterdir()), (
        "dir_b không nên có dữ liệu đáng kể vì configure lần 2 phải bị bỏ qua (idempotent)."
    )

    assert run.info.run_id, "run_id rỗng - run không hợp lệ."
