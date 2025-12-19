"""
CLI Entrypoint: Train Pipeline

Usage:
    python -m mlpipeline.cli.train
    
Environment Variables:
    MLFLOW_TRACKING_URI - MLflow tracking URI (default: ./mlruns)
    MLFLOW_EXPERIMENT   - Experiment name (default: churn_prediction)
    OUT_DIR             - Output directory (default: outputs)
    RUN_NAME            - Custom run name (optional)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mlpipeline.pipelines.train_pipeline import TrainConfig, TrainPipeline


def main() -> int:
    """
    Main entrypoint cho training pipeline
    
    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print("=" * 60)
    print("MLflow Training Pipeline - CLI")
    print("=" * 60)
    
    # Parse config từ environment variables
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "churn_prediction")
    out_dir = os.getenv("OUT_DIR", "outputs")
    run_name = os.getenv("RUN_NAME", None)
    
    # Default tags
    default_tags = {
        "env": os.getenv("ENV", "development"),
        "project": "mlflow-pipeline",
        "dataset_version": "v1.0",
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
    }
    
    # Tạo config
    config = TrainConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        out_dir=out_dir,
        default_tags=default_tags,
    )
    
    print(f"\nConfiguration:")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Output Dir: {out_dir}")
    print(f"  Run Name: {run_name or '(auto-generated)'}")
    print()
    
    try:
        # Chạy pipeline
        pipeline = TrainPipeline(config=config)
        run_id = pipeline.run()
        
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print(f"  Run ID: {run_id}")
        print(f"  View results: mlflow ui --host 0.0.0.0 --port 5000")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Training failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
