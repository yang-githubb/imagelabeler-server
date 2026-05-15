from pathlib import Path

# Root of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Shared paths
DATASET_ROOT = BASE_DIR / "datasets"
RUNS_DIR = BASE_DIR / "runs"

RESULTS_FILE = "results.csv"
CONFIG_FILE = "run_config.json"
WEIGHTS_DIR = "weights"
BEST_WEIGHTS = "best.pt"
