"""
Model definitions and utilities for digit classification.

Provides factory functions to create and configure different classifiers:
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import logging
from typing import Tuple

import config

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(use_pca: bool = False) -> Pipeline:
    """
    Create a preprocessing pipeline with StandardScaler and optional PCA.
    
    Args:
        use_pca (bool): If True, include PCA for dimensionality reduction
    
    Returns:
        Pipeline: Scikit-learn Pipeline object
    """
    steps = [("scaler", StandardScaler())]
    
    if use_pca:
        steps.append(("pca", PCA(n_components=config.PCA_N_COMPONENTS)))
        logger.info(f"Creating pipeline with PCA (n_components={config.PCA_N_COMPONENTS})")
    else:
        logger.info("Creating pipeline without PCA")
    
    return Pipeline(steps)


def create_random_forest_model(
    n_estimators: int = None,
    max_depth: int = None,
    **kwargs
) -> RandomForestClassifier:
    """
    Create a Random Forest classifier.
    
    Random Forest is an ensemble learning method that builds multiple decision trees
    and aggregates their predictions for improved accuracy and robustness.
    
    Args:
        n_estimators (int): Number of trees in forest (default: config value)
        max_depth (int): Maximum depth of trees (default: config value)
        **kwargs: Additional arguments passed to RandomForestClassifier
    
    Returns:
        RandomForestClassifier: Configured Random Forest model
    """
    params = config.RANDOM_FOREST_PARAMS.copy()
    
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if max_depth is not None:
        params["max_depth"] = max_depth
    
    params.update(kwargs)
    
    logger.info(f"Creating Random Forest with params: {params}")
    return RandomForestClassifier(**params)


def create_svm_model(
    kernel: str = None,
    C: float = None,
    gamma: str = None,
    **kwargs
) -> SVC:
    """
    Create a Support Vector Machine (SVM) classifier.
    
    SVM finds the optimal hyperplane to separate classes, particularly effective
    in high-dimensional spaces for binary and multi-class classification.
    
    Args:
        kernel (str): Kernel type ('linear', 'rbf', 'poly') (default: config value)
        C (float): Regularization parameter (default: config value)
        gamma (str): Kernel coefficient (default: config value)
        **kwargs: Additional arguments passed to SVC
    
    Returns:
        SVC: Configured SVM classifier
    """
    params = config.SVM_PARAMS.copy()
    
    if kernel is not None:
        params["kernel"] = kernel
    if C is not None:
        params["C"] = C
    if gamma is not None:
        params["gamma"] = gamma
    
    params.update(kwargs)
    
    logger.info(f"Creating SVM with params: {params}")
    return SVC(**params)


def create_xgboost_model(
    max_depth: int = None,
    eta: float = None,
    n_estimators: int = None,
    **kwargs
) -> xgb.XGBClassifier:
    """
    Create an XGBoost classifier.
    
    XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting framework
    designed for speed and performance, using ensemble methods with trees.
    
    Args:
        max_depth (int): Maximum tree depth (default: config value)
        eta (float): Learning rate (default: config value)
        n_estimators (int): Number of boosting rounds (default: config value)
        **kwargs: Additional arguments passed to XGBClassifier
    
    Returns:
        xgb.XGBClassifier: Configured XGBoost classifier
    """
    params = config.XGBOOST_PARAMS.copy()
    
    if max_depth is not None:
        params["max_depth"] = max_depth
    if eta is not None:
        params["eta"] = eta
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    
    params.update(kwargs)
    
    logger.info(f"Creating XGBoost with params: {params}")
    return xgb.XGBClassifier(**params)


def preprocess_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    use_pca: bool = False
) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
    """
    Apply preprocessing pipeline to training and test data.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        use_pca (bool): If True, apply PCA
    
    Returns:
        Tuple containing:
            - X_train_processed: Preprocessed training features
            - X_test_processed: Preprocessed test features
            - pipeline: Fitted preprocessing pipeline
    """
    pipeline = create_preprocessing_pipeline(use_pca=use_pca)
    
    logger.info(f"Preprocessing data (use_pca={use_pca})")
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    logger.info(
        f"Preprocessing complete:\n"
        f"  Original shape:     {X_train.shape}\n"
        f"  Processed shape:    {X_train_processed.shape}"
    )
    
    return X_train_processed, X_test_processed, pipeline
