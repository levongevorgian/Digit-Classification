"""
Main training script for digit classification project.

This script orchestrates the complete machine learning pipeline:
1. Data preparation (download, preprocess, split)
2. Model training with three different classifiers
3. Hyperparameter tuning (optional)
4. Model evaluation and comparison
5. Model persistence
"""

import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline

import config
import data_loader
import models as model_factory
import training as training_module
from evaluation import evaluate_model, compare_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main(
    tune_hyperparameters: bool = False,
    save_models: bool = True,
    plot_confusion_matrices: bool = True
):
    """
    Main training pipeline.
    
    Args:
        tune_hyperparameters (bool): If True, perform hyperparameter tuning
        save_models (bool): If True, save trained models to disk
        plot_confusion_matrices (bool): If True, display confusion matrices
    """
    logger.info("="*70)
    logger.info("Starting Digit Classification ML Pipeline")
    logger.info("="*70)
    
    # Step 1: Data Preparation
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Data Preparation")
    logger.info("="*70)
    
    try:
        X_train, X_test, y_train, y_test = data_loader.prepare_data()
        logger.info(f"Data prepared successfully")
        logger.info(f"  Train set: {X_train.shape}, Test set: {X_test.shape}")
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        return
    
    # Step 2: Train models
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Model Training")
    logger.info("="*70)
    
    results = {}
    
    # Random Forest
    logger.info("\n[1/3] Random Forest Classifier")
    logger.info("-" * 70)
    try:
        rf_model = model_factory.create_random_forest_model()
        rf_results = training_module.train_and_evaluate(
            rf_model, X_train, X_test, y_train, y_test,
            model_name="Random Forest",
            plot_confusion=plot_confusion_matrices
        )
        results["Random Forest"] = rf_results["metrics"]
        
        if save_models:
            training_module.save_model(
                rf_results["model"],
                str(config.MODELS_DIR / config.MODEL_NAMES["random_forest"])
            )
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
    
    # SVM Classifier
    logger.info("\n[2/3] Support Vector Machine (SVM)")
    logger.info("-" * 70)
    try:
        svm_model = model_factory.create_svm_model()
        svm_results = training_module.train_and_evaluate(
            svm_model, X_train, X_test, y_train, y_test,
            model_name="SVM (RBF)",
            plot_confusion=plot_confusion_matrices
        )
        results["SVM"] = svm_results["metrics"]
        
        if save_models:
            training_module.save_model(
                svm_results["model"],
                str(config.MODELS_DIR / config.MODEL_NAMES["svm"])
            )
    except Exception as e:
        logger.error(f"SVM training failed: {e}")
    
    # XGBoost Classifier
    logger.info("\n[3/3] XGBoost Classifier")
    logger.info("-" * 70)
    try:
        xgb_model = model_factory.create_xgboost_model()
        xgb_results = training_module.train_and_evaluate(
            xgb_model, X_train, X_test, y_train, y_test,
            model_name="XGBoost",
            plot_confusion=plot_confusion_matrices
        )
        results["XGBoost"] = xgb_results["metrics"]
        
        if save_models:
            training_module.save_model(
                xgb_results["model"],
                str(config.MODELS_DIR / config.MODEL_NAMES["xgboost"])
            )
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
    
    # Step 3: Model Comparison
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Model Comparison")
    logger.info("="*70)
    
    best_model = compare_models(results, metric="f1_macro")
    
    # Step 4: Hyperparameter Tuning (Optional)
    if tune_hyperparameters:
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Hyperparameter Tuning")
        logger.info("="*70)
        
        logger.info("\nNote: Hyperparameter tuning can take a long time.")
        logger.info("Consider using a subset of data or fewer iterations for testing.\n")
        
        # Random Forest Tuning
        logger.info("[1/3] Random Forest Hyperparameter Tuning")
        logger.info("-" * 70)
        try:
            rf_base = model_factory.create_random_forest_model()
            rf_tuned, rf_tune_info = training_module.hyperparameter_tuning(
                rf_base,
                config.RF_PARAM_DIST,
                X_train, y_train,
                model_name="Random Forest",
                n_iter=5,  # Reduced for quicker testing
                cv_splits=2
            )
            
            # Evaluate tuned model
            y_pred = rf_tuned.predict(X_test)
            tuned_metrics = evaluate_model(
                y_test, y_pred,
                model_name="Random Forest (Tuned)",
                plot_confusion_matrix=False
            )
            logger.info(f"Tuned RF F1 (macro): {tuned_metrics['f1_macro']:.4f}")
        
        except Exception as e:
            logger.error(f"Random Forest tuning failed: {e}")
    
    logger.info("\n" + "="*70)
    logger.info("Pipeline Complete")
    logger.info("="*70)
    logger.info(f"\nBest Model: {best_model}")
    logger.info(f"Models saved to: {config.MODELS_DIR}")
    logger.info(f"Logs saved to: {config.LOG_FILE}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train digit classification models"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning (slow)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save trained models"
    )
    parser.add_argument(
        "--no-confusion",
        action="store_true",
        help="Don't display confusion matrices"
    )
    
    args = parser.parse_args()
    
    main(
        tune_hyperparameters=args.tune,
        save_models=not args.no_save,
        plot_confusion_matrices=not args.no_confusion
    )
