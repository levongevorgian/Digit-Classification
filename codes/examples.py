"""
Example usage script demonstrating how to use the digit classification modules.

This script shows:
1. Data loading and preprocessing
2. Model training
3. Hyperparameter tuning
4. Making predictions
5. Model comparison
"""

import numpy as np
from pathlib import Path

# Import project modules
import config
import data_loader
import models as model_factory
import training as training_module
import predict
from evaluation import evaluate_model, compare_models


def example_1_load_data():
    """Example 1: Load and prepare data."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Data Loading and Preparation")
    print("="*70)
    
    # Prepare data (downloads if needed, caches processed data)
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    
    print(f"\nData loaded successfully!")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set:     {X_test.shape}")
    print(f"  Classes:      {np.unique(y_train)}")


def example_2_train_single_model():
    """Example 2: Train a single model."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Training a Single Model")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    
    # Create and train Random Forest
    print("\nTraining Random Forest Classifier...")
    rf_model = model_factory.create_random_forest_model(n_estimators=50)  # Fewer trees for speed
    
    results = training_module.train_and_evaluate(
        rf_model, X_train, X_test, y_train, y_test,
        model_name="Random Forest (50 trees)",
        plot_confusion=False  # Set to True to display confusion matrix
    )
    
    print(f"\nModel trained in {results['train_time']:.2f} seconds")
    print(f"F1 Score (Macro): {results['metrics']['f1_macro']:.4f}")


def example_3_compare_models():
    """Example 3: Train and compare multiple models."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Training and Comparing Models")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    
    results = {}
    
    # Random Forest
    print("\n[1/3] Random Forest...")
    rf_model = model_factory.create_random_forest_model(n_estimators=50)
    rf_results = training_module.train_and_evaluate(
        rf_model, X_train, X_test, y_train, y_test,
        model_name="Random Forest",
        plot_confusion=False
    )
    results["Random Forest"] = rf_results["metrics"]
    
    # SVM (Linear kernel for speed)
    print("\n[2/3] SVM (Linear kernel)...")
    svm_model = model_factory.create_svm_model(kernel="linear")
    svm_results = training_module.train_and_evaluate(
        svm_model, X_train, X_test, y_train, y_test,
        model_name="SVM (Linear)",
        plot_confusion=False
    )
    results["SVM"] = svm_results["metrics"]
    
    # XGBoost
    print("\n[3/3] XGBoost...")
    xgb_model = model_factory.create_xgboost_model(n_estimators=50)
    xgb_results = training_module.train_and_evaluate(
        xgb_model, X_train, X_test, y_train, y_test,
        model_name="XGBoost",
        plot_confusion=False
    )
    results["XGBoost"] = xgb_results["metrics"]
    
    # Compare
    best_model = compare_models(results, metric="f1_macro")


def example_4_hyperparameter_tuning():
    """Example 4: Hyperparameter tuning with RandomizedSearchCV."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Hyperparameter Tuning")
    print("="*70)
    
    # Load data (using smaller subset for faster tuning)
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    
    # Use subset for faster testing
    X_train_subset = X_train[:1000]  # Use first 1000 samples
    y_train_subset = y_train[:1000]
    
    print(f"\nTuning on subset: {X_train_subset.shape}")
    
    # Create base model
    base_model = model_factory.create_random_forest_model()
    
    # Simplified parameter grid for quick demo
    param_dist = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20],
    }
    
    # Tune
    best_model, tune_info = training_module.hyperparameter_tuning(
        base_model,
        param_dist,
        X_train_subset, y_train_subset,
        model_name="Random Forest",
        n_iter=4,  # Very few iterations for demo
        cv_splits=2
    )
    
    # Evaluate on full test set
    y_pred = best_model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, model_name="Tuned Random Forest", plot_confusion_matrix=False)
    
    print(f"\nBest parameters found: {tune_info['best_params']}")
    print(f"Best CV score: {tune_info['best_score']:.4f}")
    print(f"Test F1 (macro): {metrics['f1_macro']:.4f}")


def example_5_save_and_load_models():
    """Example 5: Save and load trained models."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Saving and Loading Models")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    
    # Train model
    print("\nTraining model...")
    model = model_factory.create_random_forest_model(n_estimators=50)
    model.fit(X_train, y_train)
    
    # Save
    model_path = config.MODELS_DIR / "example_model.pkl"
    training_module.save_model(model, str(model_path))
    
    # Load
    print(f"Loading model from {model_path}...")
    loaded_model = training_module.load_model(str(model_path))
    
    # Test
    y_pred = loaded_model.predict(X_test[:10])
    print(f"Predictions from loaded model: {y_pred}")


def example_6_prediction_on_images():
    """Example 6: Make predictions on images."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Making Predictions")
    print("="*70)
    
    # Load data and train model
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    
    model = model_factory.create_random_forest_model(n_estimators=50)
    model.fit(X_train, y_train)
    
    # Predict on array of features
    print("\n[1] Predicting on feature array...")
    predictions = predict.predict_array(X_test[:5], return_confidence=False)
    print(f"Predictions: {predictions}")
    
    # You would use predict_single_image() with actual image files:
    # prediction, confidence = predict.predict_single_image(
    #     model,
    #     "path/to/digit_image.png",
    #     return_confidence=True
    # )
    # print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")


def example_7_preprocessing_pipeline():
    """Example 7: Using preprocessing with PCA."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Preprocessing with PCA")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    
    # Preprocess with PCA
    print("\nApplying preprocessing with PCA...")
    X_train_processed, X_test_processed, pipeline = model_factory.preprocess_data(
        X_train, X_test, use_pca=True
    )
    
    print(f"Original shape:    {X_train.shape}")
    print(f"After PCA shape:   {X_train_processed.shape}")
    print(f"Dimensionality reduction: {X_train.shape[1]} -> {X_train_processed.shape[1]} features")
    
    # Train on reduced features
    print("\nTraining on PCA-reduced features...")
    model = model_factory.create_random_forest_model(n_estimators=50)
    model.fit(X_train_processed, y_train)
    
    y_pred = model.predict(X_test_processed)
    metrics = evaluate_model(y_test, y_pred, model_name="RF with PCA", plot_confusion_matrix=False)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("Digit Classification ML - Usage Examples")
    print("="*70)
    print("\nChoose an example to run:")
    print("  1. Load data")
    print("  2. Train single model")
    print("  3. Train and compare models")
    print("  4. Hyperparameter tuning")
    print("  5. Save and load models")
    print("  6. Make predictions")
    print("  7. Preprocessing with PCA")
    print("  0. Run all examples")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (0-7): ").strip()
    
    examples = {
        "1": example_1_load_data,
        "2": example_2_train_single_model,
        "3": example_3_compare_models,
        "4": example_4_hyperparameter_tuning,
        "5": example_5_save_and_load_models,
        "6": example_6_prediction_on_images,
        "7": example_7_preprocessing_pipeline,
    }
    
    if choice == "0":
        for example in examples.values():
            try:
                example()
            except Exception as e:
                print(f"Error: {e}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice")
    
    print("\n" + "="*70)
    print("Examples completed")
    print("="*70 + "\n")
