"""
MLflow CLI entrypoint module

Allows running: python -m mlpipeline.cli.train
"""

import sys
from mlpipeline.cli.train import main

if __name__ == "__main__":
    sys.exit(main())
