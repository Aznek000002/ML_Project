"""
Configuration file for the machine learning project.
Contains all paths, hyperparameters and model configurations.
"""

import os
from pathlib import Path

# Base directory of the project
PROJECT_ROOT = Path(__file__).parent.resolve()

# Relative path for the "model" folder (compatible cloud/local)
MODEL_DIR = PROJECT_ROOT / "model"

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# === Model files (portable paths) ===
MODEL_FREQ_PATH = MODEL_DIR / "model_freq.pkl"
MODEL_CM_PATH = MODEL_DIR / "model_cm.pkl"
MODEL_MONTANT_PATH = MODEL_DIR / "model_montant.pkl"
COLS_TO_DROP_PATH = MODEL_DIR / "cols_to_drop.pkl"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"

# === Optional: use relative data files for training if needed ===
DATA_DIR = PROJECT_ROOT / "data"

X_TRAIN_PATH = DATA_DIR / "x_train.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
X_TEST_PATH = DATA_DIR / "x_test.csv"


# === Training parameters ===
TEST_SIZE = 0.2
RANDOM_STATE = 42

# === API config ===
API_HOST = "0.0.0.0"
API_PORT = 8000

# === Other parameters ===
NAN_THRESHOLD = 0.9
N_ESTIMATORS = 100
N_JOBS = -1
CHUNK_SIZE = 100000
DTYPE_OPTIMIZATION = True
