"""
Configuration and constants for the Digit Classification ML project.

This module centralizes all configuration parameters used throughout the project,
including data paths, model hyperparameters, and preprocessing settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_DATASET_PATH = DATA_DIR / "kaggle_dataset"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create necessary directories
for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Kaggle dataset configuration
KAGGLE_DATASET_NAME = "pintowar/numerical-images"

# Image preprocessing parameters
IMAGE_SIZE = (28, 28)
IMAGE_MODE = "grayscale"  # cv2.IMREAD_GRAYSCALE = 0

# Data preprocessing parameters
DATASET_GROUP = "Hnd"  # Handwritten images
DATASET_ORIGIN = "mnist"  # MNIST PNG directory
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_LABELS = True

# PCA configuration
USE_PCA = False
PCA_N_COMPONENTS = 500
PCA_EXPLAINED_VARIANCE_THRESHOLD = 0.95

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 0
}

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "random_state": RANDOM_STATE
}

XGBOOST_PARAMS = {
    "num_class": 10,
    "max_depth": 6,
    "eta": 0.1,
    "random_state": RANDOM_STATE,
    "objective": "multi:softmax",
    "eval_metric": "mlogloss"
}

# Training parameters
CROSS_VALIDATION_SPLITS = 2
RANDOM_SEARCH_ITERATIONS = 10
N_JOBS = -1  # Use all available cores

# Hyperparameter tuning ranges (for RandomizedSearchCV)
RF_PARAM_DIST = {
    "max_depth": [5, 10, 15, 20, None],
    "max_features": ["sqrt", "log2", None],
    "min_samples_leaf": [1, 2, 4],
    "n_estimators": [100, 200, 300],
    "min_samples_split": [2, 5, 10]
}

SVC_PARAM_DIST = {
    "svc__C": [0.001, 0.01, 0.05, 0.1, 0.4, 1],
    "svc__gamma": [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, "scale", "auto"],
    "svc__kernel": ["linear", "rbf", "poly"]
}

XGBOOST_PARAM_DIST = {
    "max_depth": [3, 4, 5, 8, 10],
    "eta": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [3, 5],
    "gamma": [0, 0.1, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1],
    "reg_lambda": [0, 0.01, 0.1, 0.5, 1],
    "n_estimators": [50, 100, 150, 300]
}

# Model names for file naming
MODEL_NAMES = {
    "random_forest": "random_forest_model.pkl",
    "svm": "svm_model.pkl",
    "xgboost": "xgboost_model.json"
}

# Logging
LOG_FILE = PROJECT_ROOT / "training.log"
VERBOSE = True
