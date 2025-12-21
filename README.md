# MLflow Pipeline - Complete Implementation

Pipeline đầy đủ 13 bước MLflow từ training đến deployment.

## 🎯 Mục tiêu

Implement đầy đủ MLflow pipeline theo best practices:
- ✅ MLflow Tracking
- ✅ Model Registry
- ✅ Model Versioning
- ✅ Stage Management (Development → Production)
- ✅ Alias Management
- ✅ Reproducibility

## 📋 13 Bước Pipeline

| Bước | Tên | File | Mô tả |
|------|-----|------|-------|
| 1 | Init MLflow | `mlflow_manager.py` | Singleton configuration |
| 2 | Chọn experiment | `train.py` | Set experiment name |
| 3 | Tạo run | `train.py` | Start run với tags |
| 4 | Log params | `train.py` | Log hyperparameters |
| 5 | Log metrics | `train.py` | Log accuracy, f1, etc. |
| 6 | Log artifacts | `train.py` | Log plots, configs |
| 7 | Log metadata | `train.py` | Log tags |
| 8 | Log model | `train.py` | Save model to artifacts |
| 9 | Load model | `evaluate.py` | Load từ runs:/ URI |
| 10 | Register model | `register_model.py` | Đưa vào Registry |
| 11 | Set stage | `promote_model.py` | Production stage |
| 12 | Set alias | `promote_model.py` | @production alias |
| 13 | Deploy | `deploy.py` | Load models:/ URI |

## 🚀 Quick Start

### 1. Setup môi trường

**WSL/Linux:**
```bash
make setup
```

**Windows PowerShell:**
```powershell
pip install -r requirements.txt
mkdir data, outputs, mlruns, models -Force
```

### 2. Start MLflow UI (Terminal riêng)

```bash
make init-mlflow
# hoặc
mlflow ui --host 0.0.0.0 --port 5000
```

Mở browser: http://localhost:5000

### 3. Chạy pipeline

**Chạy từng bước:**
```bash
# Bước 1-8: Train
make train

# Bước 9: Evaluate
make evaluate

# Bước 10: Register
make register

# Bước 11-12: Promote to Production
make promote

# Bước 13: Deploy/Load
make deploy
```

**Hoặc chạy toàn bộ:**
```bash
make pipeline
```

## 📁 Cấu trúc dự án

```
.
├── Makefile                    # Build commands
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── src/
│   ├── mlflow_manager.py      # Bước 1: MLflow config 
│   ├── train.py               # Bước 2-8: Training pipeline
│   ├── evaluate.py            # Bước 9: Evaluation pipeline
│   ├── register_model.py      # Bước 10: Model registration
│   ├── promote_model.py       # Bước 11-12: Promotion
│   └── deploy.py              # Bước 13: Deployment
├── data/                       # Data folder (gitignored)
├── outputs/                    # Artifacts (gitignored)
├── mlruns/                     # MLflow tracking data
└── models/                     # Registered models cache
```

## 🔍 Chi tiết từng bước

### Bước 1-8: Training (`make train`)

```python
# Init MLflow
manager = MLflowManager.configure(
    tracking_uri="./mlruns",
    experiment_name="churn_prediction"
)

# Start run
with mlflow.start_run(run_name="rf_model", tags={...}):
    # Log params
    mlflow.log_param("n_estimators", 100)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Bước 9: Evaluation (`make evaluate`)

```python
# Load model từ run
model_uri = "runs:/abc123/model"
model = mlflow.pyfunc.load_model(model_uri)

# Evaluate
predictions = model.predict(X_test)
```

### Bước 10: Registration (`make register`)

```python
# Register model to Registry
mlflow.register_model(
    model_uri="runs:/abc123/model",
    name="churn_prediction_model"
)
```

### Bước 11-12: Promotion (`make promote`)

```python
# Set stage
client.transition_model_version_stage(
    name="churn_prediction_model",
    version="1",
    stage="Production"
)

# Set alias (khuyến nghị)
client.set_registered_model_alias(
    name="churn_prediction_model",
    alias="production",
    version="1"
)
```

### Bước 13: Deployment (`make deploy`)

```python
# Load from Registry
model = mlflow.pyfunc.load_model(
    "models:/churn_prediction_model@production"
)

# Predict
predictions = model.predict(new_data)
```

## 🎨 MLflow UI Features

Sau khi chạy `make init-mlflow`, mở http://localhost:5000:

1. **Experiments**: Xem tất cả runs theo experiment
2. **Compare**: So sánh metrics/params giữa các runs
3. **Models**: Xem Model Registry, versions, stages
4. **Artifacts**: Download confusion matrix, configs, models

## 🧪 Testing

```bash
# Test toàn bộ pipeline
make pipeline

# Xem kết quả trong MLflow UI
# http://localhost:5000
```

## 📊 Outputs

Sau khi train, bạn sẽ có:

```
outputs/
├── confusion_matrix.png        # Confusion matrix plot
├── feature_config.yaml         # Hyperparameters config
└── classification_report.txt   # Sklearn classification report

mlruns/
└── 0/                          # Experiment ID
    └── abc123/                 # Run ID
        ├── artifacts/
        │   ├── model/          # Logged model
        │   └── outputs/        # Logged artifacts
        ├── metrics/            # Logged metrics
        ├── params/             # Logged params
        └── tags/               # Logged tags
```

## 🔧 Troubleshooting

### MLflow UI không mở được
```bash
# Check port 5000
netstat -ano | findstr :5000

# Dùng port khác
mlflow ui --port 5001
```

### Model không load được
```bash
# Check run_id
python -c "from src.evaluate import get_latest_run_id; print(get_latest_run_id())"

# Load manually
python -c "import mlflow; print(mlflow.pyfunc.load_model('runs:/YOUR_RUN_ID/model'))"
```

### Registry không có model
```bash
# Check registered models
python -c "from mlflow.tracking import MlflowClient; print(MlflowClient().search_registered_models())"
```

## 📚 References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)

## 🎓 Next Steps

1. ✅ Hoàn thành 13 bước cơ bản
2. 🔄 Thêm CI/CD pipeline
3. 🐳 Dockerize deployment
4. ☁️ Deploy to cloud (AWS/Azure/GCP)
5. 📈 Thêm model monitoring
6. 🔁 Thêm auto-retraining

---

Made with ❤️ for MLOps learning
