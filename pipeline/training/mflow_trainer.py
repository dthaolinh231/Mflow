"""Mô-đun huấn luyện và đánh giá model machine learning."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def prepare_data() -> Tuple[Any, Any, Any, Any]:
    """Chuẩn bị dữ liệu train/test.

    Demo bằng make_classification, sau có thể thay bằng loader thật.

    Returns:
        Tuple (x_train, x_test, y_train, y_test).
    """
    x, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    return x_train, x_test, y_train, y_test


def build_model(params: Dict[str, Any]) -> Any:
    """
    Khởi tạo model từ hyperparameters

    TODO:
    - Tạo model (sklearn RandomForestClassifier)
    - Trả về instance model
    """
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", 10)),
        min_samples_split=int(params.get("min_samples_split", 2)),
        random_state=int(params.get("random_state", 42)),
    )


def train_model(model: Any, x_train: Any, y_train: Any) -> Any:
    """Huấn luyện model.

    Args:
        model: Model sklearn cần train.
        x_train: Feature training data.
        y_train: Target training data.

    Returns:
        Trained model.
    """
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: Any, x_test: Any, y_test: Any):
    """Đánh giá model và tính metrics.

    Args:
        model: Fitted model.
        x_test: Feature test data.
        y_test: Target test data.

    Returns:
        Tuple (metrics dict, predictions, classification report text).
    """
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    report_text = classification_report(y_test, y_pred)
    return metrics, y_pred, report_text
