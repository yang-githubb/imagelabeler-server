"""
======================================================
train.py
------------------------------------------------------
Responsibility:
- Execute YOLO training using Ultralytics
- Accept configuration via CLI arguments
- Save results to specified project/run directory

Design rules:
- No hardcoded paths
- No dependency on server structure
- Fully controlled by external caller (API/service layer)
======================================================
"""

import argparse
from ultralytics import YOLO


# --------------------------------------------------
# CLI argument parsing
# --------------------------------------------------
def parse_args():
    """
    Parse CLI arguments passed from the service layer.

    Arguments:
    - data: path to dataset.yaml
    - model: model name or .pt path
    - epochs: number of training epochs
    - imgsz: image size
    - batch: batch size
    - name: run name
    - project: output directory

    Returns:
    - argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="YOLO Training")

    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", required=True, help="Model name or .pt path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", required=True, help="Run name")
    parser.add_argument("--project", required=True, help="Training output root")

    return parser.parse_args()


# --------------------------------------------------
# Main training execution
# --------------------------------------------------
def main():
    """
    Entry point for training execution.

    Behavior:
    - Loads YOLO model from given input
    - Runs training with provided configuration
    - Outputs results to project/run directory

    Notes:
    - Model path can be local or pretrained name
    - Ultralytics resolves model loading automatically
    """
    args = parse_args()

    # Initialize model
    model = YOLO(args.model)

    # Execute training
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        exist_ok=True,
        verbose=True
    )


# --------------------------------------------------
# Script entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()