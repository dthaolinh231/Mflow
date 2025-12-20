"""Mô-đun xử lý artifacts (file, hình ảnh) cho training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def ensure_out_dir(out_dir: str) -> Path:
    """
    Đảm bảo thư mục output tồn tại
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_feature_config(config: Dict[str, Any], out_path: Path) -> Path:
    """Lưu cấu hình (feature / hyperparameter) ra file YAML.

    Args:
        config: Dict cấu hình cần lưu.
        out_path: Đường dẫn file YAML đầu ra.

    Returns:
        Đường dẫn file đã lưu.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return out_path


def save_classification_report(report_text: str, out_path: Path) -> Path:
    """Lưu classification report ra file text.

    Args:
        report_text: Nội dung report từ sklearn.
        out_path: Đường dẫn file text đầu ra.

    Returns:
        Đường dẫn file đã lưu.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    return out_path


def plot_and_save_confusion_matrix(y_true, y_pred, out_path: Path) -> Path:
    """Vẽ và lưu confusion matrix ra file ảnh.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        out_path: Đường dẫn file PNG đầu ra.

    Returns:
        Đường dẫn file ảnh đã lưu.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path
