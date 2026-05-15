from pathlib import Path

"""
======================================================
core/config.py
------------------------------------------------------
Responsibility:
- Define global project paths and shared constants
- Ensure a single source of truth for filesystem layout

Design rules:
- No business logic here
- No side effects (no file creation except where explicit)
- Only constants and path definitions
======================================================
"""

# --------------------------------------------------
# Project root
# --------------------------------------------------

# Resolves to the root directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# Core directories
# --------------------------------------------------

# Root directory for all datasets (training images, labels, etc.)
DATASET_ROOT = BASE_DIR / "datasets"

# Root directory for all training runs (outputs, logs, weights)
RUNS_DIR = BASE_DIR / "runs"


# --------------------------------------------------
# Standard file names (used across services)
# --------------------------------------------------

# Training output files
RESULTS_FILE = "results.csv"        # YOLO training metrics (per epoch)
CONFIG_FILE = "run_config.json"    # Stored training configuration

# Weights structure
WEIGHTS_DIR = "weights"            # Folder storing model weights
BEST_WEIGHTS = "best.pt"           # Best checkpoint file
