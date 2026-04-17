"""
QUICK START GUIDE
=================

This file provides quick setup and execution instructions.
For detailed documentation, see README.md
"""

# ============================================================================
# STEP 1: SETUP (Run once)
# ============================================================================

"""
1. Install Python 3.8+ from https://www.python.org/

2. Open terminal/command prompt and navigate to project folder:
   cd path/to/DigitClassificationML

3. Create virtual environment:
   python -m venv venv
   
4. Activate virtual environment:
   - macOS/Linux:  source venv/bin/activate
   - Windows:      venv\Scripts\activate

5. Install dependencies:
   pip install -r requirements.txt

6. Setup Kaggle API (required for data download):
   a. Go to https://www.kaggle.com/settings/account
   b. Click "Create New API Token" (downloads kaggle.json)
   c. Place kaggle.json in ~/.kaggle/kaggle.json
   d. Set permissions (macOS/Linux): chmod 600 ~/.kaggle/kaggle.json
"""

# ============================================================================
# STEP 2: RUN TRAINING (Main workflow)
# ============================================================================

"""
Basic training:
    python main.py

Train with hyperparameter tuning (slow, takes hours):
    python main.py --tune

Skip saving models:
    python main.py --no-save

Skip confusion matrix plots:
    python main.py --no-confusion
"""

# ============================================================================
# STEP 3: USE THE MODELS
# ============================================================================

# Example 1: Load a trained model and make predictions
# ====================================================

from training import load_model
from predict import predict_single_image

# Load the best model (Random Forest)
model = load_model("models/random_forest_model.pkl")

# Predict on an image
prediction, confidence = predict_single_image(
    model,
    "path/to/your/digit_image.png",
    return_confidence=True
)
print(f"Predicted: {prediction}, Confidence: {confidence:.4f}")


# Example 2: Train a custom model
# ================================

from data_loader import prepare_data
from models import create_random_forest_model
from training import train_and_evaluate

# Get data
X_train, X_test, y_train, y_test = prepare_data()

# Create model with custom parameters
model = create_random_forest_model(n_estimators=200, max_depth=15)

# Train and evaluate
results = train_and_evaluate(
    model, X_train, X_test, y_train, y_test,
    model_name="Custom RF",
    plot_confusion=True
)

print(f"F1 Score: {results['metrics']['f1_macro']:.4f}")


# Example 3: Hyperparameter tuning
# =================================

from models import create_random_forest_model
from training import hyperparameter_tuning
from data_loader import prepare_data

# Get data
X_train, X_test, y_train, y_test = prepare_data()

# Define hyperparameters to search
param_dist = {
    "max_depth": [5, 10, 15, 20],
    "n_estimators": [50, 100, 200],
}

# Create base model
base_model = create_random_forest_model()

# Perform tuning
best_model, info = hyperparameter_tuning(
    base_model,
    param_dist,
    X_train, y_train,
    model_name="Tuned RF",
    n_iter=10,
    cv_splits=3
)

print(f"Best params: {info['best_params']}")


# Example 4: Batch predictions
# =============================

from predict import predict_batch, predict_array
import numpy as np

# Predict on multiple image files
image_paths = ["img1.png", "img2.png", "img3.png"]
predictions, confidences = predict_batch(
    model, image_paths, return_confidence=True
)

# Or predict on preprocessed feature arrays
predictions = predict_array(model, X_test[:100])
print(f"Predictions: {predictions}")


# Example 5: Model comparison
# ============================

from models import create_random_forest_model, create_svm_model, create_xgboost_model
from training import train_and_evaluate
from evaluation import compare_models
from data_loader import prepare_data

X_train, X_test, y_train, y_test = prepare_data()

results = {}

# Train RF
rf_model = create_random_forest_model()
rf_results = train_and_evaluate(
    rf_model, X_train, X_test, y_train, y_test,
    model_name="Random Forest",
    plot_confusion=False
)
results["Random Forest"] = rf_results["metrics"]

# Train SVM
svm_model = create_svm_model()
svm_results = train_and_evaluate(
    svm_model, X_train, X_test, y_train, y_test,
    model_name="SVM",
    plot_confusion=False
)
results["SVM"] = svm_results["metrics"]

# Train XGBoost
xgb_model = create_xgboost_model()
xgb_results = train_and_evaluate(
    xgb_model, X_train, X_test, y_train, y_test,
    model_name="XGBoost",
    plot_confusion=False
)
results["XGBoost"] = xgb_results["metrics"]

# Compare
best_model_name = compare_models(results, metric="f1_macro")


# Example 6: Using preprocessing pipeline
# ========================================

from models import preprocess_data, create_random_forest_model
from data_loader import prepare_data

X_train, X_test, y_train, y_test = prepare_data()

# Preprocess with PCA
X_train_proc, X_test_proc, pipeline = preprocess_data(
    X_train, X_test, use_pca=True
)

print(f"Original features: {X_train.shape[1]}")
print(f"After PCA: {X_train_proc.shape[1]}")

# Train on reduced features
model = create_random_forest_model()
model.fit(X_train_proc, y_train)
score = model.score(X_test_proc, y_test)
print(f"Accuracy with PCA: {score:.4f}")


# ============================================================================
# USEFUL CONFIGURATION
# ============================================================================

"""
Edit config.py to customize:

1. DATA SETTINGS
   - TEST_SIZE: 0.2 (20% for testing)
   - RANDOM_STATE: 42 (for reproducibility)
   - IMAGE_SIZE: (28, 28)

2. MODEL HYPERPARAMETERS
   - RANDOM_FOREST_PARAMS
   - SVM_PARAMS
   - XGBOOST_PARAMS

3. TRAINING SETTINGS
   - CROSS_VALIDATION_SPLITS: 2
   - N_JOBS: -1 (use all CPU cores)
   - PCA_N_COMPONENTS: 500

4. PATHS
   - Change DATA_DIR, MODELS_DIR, etc. as needed
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Q: Kaggle download fails?
A: Ensure ~/.kaggle/kaggle.json exists with correct API credentials

Q: "ModuleNotFoundError" when running code?
A: Activate virtual environment and check requirements are installed:
   pip install -r requirements.txt

Q: Out of memory errors?
A: Set USE_PCA = True in config.py to reduce dimensionality

Q: Training is slow?
A: - Use PCA
   - Reduce RANDOM_SEARCH_ITERATIONS
   - Use fewer CROSS_VALIDATION_SPLITS

Q: Why are results slightly different each time?
A: Check RANDOM_STATE in config.py (should be 42 for reproducibility)
"""

# ============================================================================
# NEXT STEPS
# ============================================================================

"""
1. Run: python main.py
   (Trains all three models, takes ~1-2 minutes)

2. Check results in:
   - Console output
   - models/ directory (saved models)
   - training.log (detailed logs)

3. Try: python examples.py
   (Interactive examples of different functionalities)

4. See README.md for full documentation
"""
