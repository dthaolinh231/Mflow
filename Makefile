.PHONY: help setup init-mlflow train evaluate register promote deploy clean test

help:
	@echo "MLflow Pipeline Commands:"
	@echo "  make setup          - Cài đặt môi trường (Python packages)"
	@echo "  make init-mlflow    - Khởi động MLflow UI"
	@echo "  make train          - Train model và log vào MLflow"
	@echo "  make evaluate       - Evaluate model từ MLflow"
	@echo "  make register       - Register model vào Model Registry"
	@echo "  make promote        - Promote model to Production"
	@echo "  make deploy         - Deploy/Load model từ Registry"
	@echo "  make test           - Test Python syntax và imports"
	@echo "  make pipeline       - Chạy toàn bộ pipeline (train -> eval -> register)"
	@echo "  make clean          - Xóa outputs và cache"

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Creating directories..."
	mkdir -p data outputs mlruns models
	@echo "Setup completed!"

init-mlflow:
	@echo "Starting MLflow UI at http://localhost:5000"
	mlflow ui --host 0.0.0.0 --port 5000

train:
	@echo "Step 1-8: Training model with MLflow tracking..."
	python -m mlpipeline.cli.train

evaluate:
	@echo "Step 9: Evaluating model from MLflow..."
	python src/evaluate.py

register:
	@echo "Step 10: Registering model to Model Registry..."
	python src/register_model.py

promote:
	@echo "Step 11-12: Promoting model to Production..."
	python src/promote_model.py

deploy:
	@echo "Step 13: Loading model for deployment..."
	python src/deploy.py

test:
	@echo "Running tests..."
	@echo "Checking Python syntax..."
	python3 -m py_compile mlpipeline/tracking/mlflow_manager.py
	python3 -m py_compile mlpipeline/pipelines/train_pipeline.py
	python3 -m py_compile mlpipeline/cli/train.py
	@echo "✓ All syntax checks passed!"

pipeline: train evaluate register
	@echo "Pipeline completed! Check MLflow UI at http://localhost:5000"

clean:
	@echo "Cleaning up..."
	rm -rf outputs/*.png outputs/*.txt __pycache__ src/__pycache__
	@echo "Clean completed!"
