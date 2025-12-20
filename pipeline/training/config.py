"""Mô-đun cấu hình cho training pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainConfig:
    """
    Cấu hình cho Train Pipeline
    """
    tracking_uri: str              # MLflow tracking URI
    experiment_name: str           # Tên experiment (bài toán)
    run_name: Optional[str] = None  # Tên run (nếu không truyền sẽ tự sinh)
    out_dir: str = "outputs"        # Thư mục lưu artifacts
    # Tags mặc định (trộn vào default tags của MLflowManager nếu có)
    default_tags: Optional[Dict[str, str]] = None
    # Tags phát sinh theo run, hỗ trợ filter run dễ dàng hơn
    extra_tags: Optional[Dict[str, str]] = None
    # Params/hyperparameters
    # MLflow bắt buộc phải log params → để so sánh run
    params: Optional[Dict[str, Any]] = None
