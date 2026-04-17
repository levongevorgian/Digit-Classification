"""
Data loading and preprocessing module for digit classification.

This module handles:
- Downloading datasets from Kaggle
- Loading and filtering datasets
- Preprocessing images
- Creating train/test splits
- Caching preprocessed data
"""

import os
import shutil
import cv2
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict

import config
from preprocessing import preprocess_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset() -> str:
    """
    Download the Kaggle numerical images dataset.
    
    Uses kagglehub to download the dataset if not already present.
    Requires Kaggle API credentials to be configured.
    
    Returns:
        str: Path to the downloaded dataset
    
    Raises:
        ImportError: If kagglehub is not installed
        FileNotFoundError: If dataset download fails
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub not installed. Install with: pip install kagglehub"
        )
    
    if config.KAGGLE_DATASET_PATH.exists():
        logger.info(f"Dataset already exists at {config.KAGGLE_DATASET_PATH}")
        return str(config.KAGGLE_DATASET_PATH)
    
    logger.info(f"Downloading dataset: {config.KAGGLE_DATASET_NAME}")
    dataset_path = kagglehub.dataset_download(config.KAGGLE_DATASET_NAME)
    
    config.KAGGLE_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset_path, config.KAGGLE_DATASET_PATH, dirs_exist_ok=True)
    
    logger.info(f"Dataset downloaded to {config.KAGGLE_DATASET_PATH}")
    return str(config.KAGGLE_DATASET_PATH)


def load_and_filter_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and filter the dataset CSV file.
    
    Applies filters for:
    - Dataset group (e.g., 'Hnd' for handwritten)
    - Dataset origin (e.g., 'mnist')
    
    Args:
        csv_path (str): Path to the CSV file containing image metadata
    
    Returns:
        pd.DataFrame: Filtered dataframe with image metadata
    """
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    initial_count = len(df)
    
    # Apply filters
    df = df[df["group"] == config.DATASET_GROUP]
    df = df[df["origin"] == config.DATASET_ORIGIN]
    
    final_count = len(df)
    logger.info(
        f"Filtered dataset from {initial_count} to {final_count} images "
        f"(group='{config.DATASET_GROUP}', origin='{config.DATASET_ORIGIN}')"
    )
    
    return df.reset_index(drop=True)


def preprocess_dataset(
    df: pd.DataFrame,
    images_dir: str,
    force_reprocess: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Load and preprocess all images from the dataset.
    
    Reads each image from disk, applies preprocessing, and creates
    feature matrix X and label vector y. Supports caching to avoid
    reprocessing on repeated runs.
    
    Args:
        df (pd.DataFrame): Dataframe with image metadata (must have 'file' and 'label' columns)
        images_dir (str): Directory containing the image files
        force_reprocess (bool): If True, ignore cache and reprocess all images
    
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict]: 
            - X: Feature matrix (n_samples, n_features)
            - y: Label vector (n_samples,)
            - label_map: Mapping from label values to indices
    """
    cache_files = [
        config.PROCESSED_DATA_DIR / "X.npy",
        config.PROCESSED_DATA_DIR / "y.npy",
        config.PROCESSED_DATA_DIR / "label_map.npy"
    ]
    
    # Check cache
    if all(f.exists() for f in cache_files) and not force_reprocess:
        logger.info("Loading preprocessed data from cache")
        X = np.load(cache_files[0], allow_pickle=True)
        y = np.load(cache_files[1], allow_pickle=True)
        label_map = np.load(cache_files[2], allow_pickle=True).item()
        return X, y, label_map
    
    logger.info(f"Preprocessing {len(df)} images from {images_dir}")
    
    X = []
    y = []
    
    # Create label mapping
    unique_labels = sorted(df["label"].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    logger.info(f"Label mapping: {label_map}")
    
    # Process each image
    errors = 0
    for idx, row in df.iterrows():
        try:
            img_path = os.path.join(images_dir, "numbers", row["file"])
            
            # Read image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image at {img_path}")
            
            # Preprocess image
            processed_img = preprocess_image(img, config.IMAGE_SIZE)
            
            X.append(processed_img)
            y.append(label_map[row["label"]])
            
            # Log progress every 5000 images
            if (idx + 1) % 5000 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} images")
        
        except Exception as e:
            errors += 1
            if errors <= 5:  # Log first 5 errors
                logger.warning(f"Error processing image {row['file']}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Preprocessing complete. Skipped {errors} images due to errors.")
    logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
    
    # Save to cache
    np.save(cache_files[0], X)
    np.save(cache_files[1], y)
    np.save(cache_files[2], label_map)
    logger.info(f"Cached preprocessed data to {config.PROCESSED_DATA_DIR}")
    
    return X, y, label_map


def prepare_data(
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    force_reprocess: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete data preparation pipeline.
    
    Orchestrates the entire data loading process:
    1. Download dataset (if needed)
    2. Load and filter dataset
    3. Preprocess images
    4. Split into train/test sets
    
    Args:
        test_size (float): Proportion of data to use for testing (0.0-1.0)
        random_state (int): Random seed for reproducibility
        force_reprocess (bool): Force reprocessing even if cache exists
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
    """
    from sklearn.model_selection import train_test_split
    
    # Download dataset
    dataset_path = download_kaggle_dataset()
    
    # Load and filter dataset
    csv_path = os.path.join(dataset_path, "numbers.csv")
    df = load_and_filter_dataset(csv_path)
    
    # Preprocess images
    images_dir = dataset_path
    X, y, label_map = preprocess_dataset(df, images_dir, force_reprocess=force_reprocess)
    
    # Split into train/test
    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if config.STRATIFY_LABELS else None
    )
    
    logger.info(
        f"Train/test split complete:\n"
        f"  Train: X={X_train.shape}, y={y_train.shape}\n"
        f"  Test:  X={X_test.shape}, y={y_test.shape}"
    )
    
    return X_train, X_test, y_train, y_test
