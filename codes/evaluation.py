"""
Model evaluation utilities for digit classification.

Provides functions to evaluate model performance using standard
classification metrics and visualization tools.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    verbose: bool = True,
    plot_confusion_matrix: bool = True
) -> dict:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Computes and displays:
    - Accuracy, Precision, Recall, F1 Score (macro and weighted)
    - Classification Report (per-class metrics)
    - Confusion Matrix (optional visualization)
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Name of model for logging purposes
        verbose (bool): If True, print results to console
        plot_confusion_matrix (bool): If True, display confusion matrix
    
    Returns:
        dict: Dictionary containing all computed metrics:
            - accuracy: Overall accuracy
            - precision_macro: Macro-averaged precision
            - precision_weighted: Weighted-averaged precision
            - recall_macro: Macro-averaged recall
            - recall_weighted: Weighted-averaged recall
            - f1_macro: Macro-averaged F1 score
            - f1_weighted: Weighted-averaged F1 score
    """
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:                  {accuracy:.4f}")
        print(f"Precision (Macro/Weighted): {precision_macro:.4f} / {precision_weighted:.4f}")
        print(f"Recall    (Macro/Weighted): {recall_macro:.4f} / {recall_weighted:.4f}")
        print(f"F1 Score  (Macro/Weighted): {f1_macro:.4f} / {f1_weighted:.4f}")
        print(f"\n{'-'*60}")
        print("Classification Report:")
        print(f"{'-'*60}")
        print(classification_report(y_true, y_pred, zero_division=0))
    
    # Plot confusion matrix if requested
    if plot_confusion_matrix:
        try:
            plt.figure(figsize=(10, 8))
            ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                cmap="Blues",
                xticks_rotation="vertical"
            )
            plt.title(f"Confusion Matrix: {model_name}")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"Could not plot confusion matrix: {e}")
    
    return metrics


def compare_models(
    model_results: dict,
    metric: str = "f1_macro"
) -> str:
    """
    Compare performance of multiple models.
    
    Args:
        model_results (dict): Dictionary with model names as keys and metric dicts as values
        metric (str): Metric to compare on (default: f1_macro)
    
    Returns:
        str: Name of best performing model
    """
    print(f"\n{'='*60}")
    print(f"Model Comparison (sorted by {metric})")
    print(f"{'='*60}")
    
    sorted_models = sorted(
        model_results.items(),
        key=lambda x: x[1].get(metric, 0),
        reverse=True
    )
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        value = metrics.get(metric, "N/A")
        print(f"{rank}. {model_name:20s} - {metric}: {value:.4f}")
    
    best_model = sorted_models[0][0]
    print(f"\nBest Model: {best_model}")
    print(f"{'='*60}\n")
    
    return best_model


def plot_model_learning_curves(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    param_values: np.ndarray,
    param_name: str,
    model_name: str = "Model"
):
    """
    Plot learning curves for hyperparameter tuning.
    
    Args:
        train_scores (np.ndarray): Training scores for each parameter value
        val_scores (np.ndarray): Validation scores for each parameter value
        param_values (np.ndarray): Parameter values tested
        param_name (str): Name of the hyperparameter
        model_name (str): Name of the model
    """
    plt.figure(figsize=(10, 6))
    
    # Compute mean and std
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Plot
    plt.plot(param_values, train_mean, label="Training score", marker="o")
    plt.fill_between(param_values, train_mean - train_std, train_mean + train_std, alpha=0.2)
    
    plt.plot(param_values, val_mean, label="Validation score", marker="s")
    plt.fill_between(param_values, val_mean - val_std, val_mean + val_std, alpha=0.2)
    
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve: {model_name} vs {param_name}")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
