"""
Model training module for digit classification.

Provides functions to train models with various configurations,
perform hyperparameter tuning, and save trained models.
"""

import time
import pickle
import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

import config
import models as model_factory
from evaluation import evaluate_model

logger = logging.getLogger(__name__)


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "Model",
    verbose: bool = True
) -> Tuple[Any, float]:
    """
    Train a model on training data.
    
    Args:
        model: Model instance (sklearn or xgboost)
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        model_name (str): Name for logging
        verbose (bool): Print progress
    
    Returns:
        Tuple containing:
            - Trained model
            - Training time in seconds
    """
    if verbose:
        logger.info(f"Training {model_name}...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    if verbose:
        logger.info(f"{model_name} trained in {elapsed_time:.2f} seconds")
    
    return model, elapsed_time


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    plot_confusion: bool = True
) -> Dict[str, Any]:
    """
    Complete training and evaluation pipeline.
    
    Args:
        model: Model instance
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
        model_name (str): Name for logging
        plot_confusion (bool): Whether to plot confusion matrix
    
    Returns:
        Dict with keys:
            - model: Trained model
            - metrics: Evaluation metrics dictionary
            - train_time: Training time in seconds
    """
    # Train
    trained_model, train_time = train_model(
        model, X_train, y_train, model_name=model_name
    )
    
    # Predict
    y_pred = trained_model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_model(
        y_test, y_pred,
        model_name=model_name,
        plot_confusion_matrix=plot_confusion
    )
    
    return {
        "model": trained_model,
        "metrics": metrics,
        "train_time": train_time
    }


def hyperparameter_tuning(
    base_model,
    param_dist: Dict[str, list],
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "Model",
    n_iter: int = None,
    cv_splits: int = None,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Args:
        base_model: Base model instance
        param_dist (Dict): Parameter distribution for search
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        model_name (str): Name for logging
        n_iter (int): Number of iterations (default: config value)
        cv_splits (int): Number of CV splits (default: config value)
        verbose (bool): Print progress
    
    Returns:
        Tuple containing:
            - Best trained model
            - Dictionary with:
                - best_params: Best hyperparameters found
                - best_score: Best cross-validation score
                - search_results: Full search results
    """
    if n_iter is None:
        n_iter = config.RANDOM_SEARCH_ITERATIONS
    if cv_splits is None:
        cv_splits = config.CROSS_VALIDATION_SPLITS
    
    if verbose:
        logger.info(
            f"Starting hyperparameter tuning for {model_name}\n"
            f"  Iterations: {n_iter}\n"
            f"  CV Splits:  {cv_splits}\n"
            f"  Scoring:    accuracy"
        )
    
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=config.RANDOM_STATE)
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv_strategy,
        refit=True,
        n_jobs=config.N_JOBS,
        verbose=2 if verbose else 0,
        random_state=config.RANDOM_STATE
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    best_model = search.best_estimator_
    
    if verbose:
        logger.info(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best CV accuracy: {search.best_score_:.4f}")
        logger.info(f"Best hyperparameters: {search.best_params_}")
    
    return best_model, {
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "search_results": search.cv_results_
    }


def save_model(model, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Supports both scikit-learn models (pkl) and XGBoost models (json).
    
    Args:
        model: Trained model
        filepath (str): Path to save model to
    """
    try:
        if isinstance(model, xgb.XGBClassifier):
            model.save_model(filepath)
            logger.info(f"Saved XGBoost model to {filepath}")
        else:
            with open(filepath, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.
    
    Args:
        filepath (str): Path to model file
    
    Returns:
        Loaded model
    """
    try:
        if filepath.endswith(".json"):
            model = xgb.XGBClassifier()
            model.load_model(filepath)
            logger.info(f"Loaded XGBoost model from {filepath}")
        else:
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
