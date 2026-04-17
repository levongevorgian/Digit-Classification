# Digit Classification ML Project

A modular, production-ready machine learning project for classifying handwritten digits (0-9) using multiple algorithms. This project demonstrates best practices in ML development with clean code architecture, comprehensive documentation, and reproducibility.

## Overview

This project trains and evaluates three different classification algorithms on the MNIST handwritten digit dataset:
- **Random Forest**: Ensemble learning with decision trees
- **Support Vector Machine (SVM)**: High-dimensional hyperplane separation
- **XGBoost**: Optimized gradient boosting framework

The expected accuracy is ~96-97% on the test set.

## Project Structure

```
DigitClassificationML/
├── config.py              # Configuration and constants
├── preprocessing.py       # Image preprocessing utilities
├── data_loader.py         # Data loading and preparation
├── models.py              # Model factory functions
├── training.py            # Training pipelines and utilities
├── evaluation.py          # Model evaluation metrics
├── predict.py             # Prediction utilities
├── main.py                # Main training script
├── requirements.txt       # Project dependencies
├── README.md              # This file
│
├── data/                  # Data directory
│   ├── kaggle_dataset/    # Downloaded MNIST dataset
│   └── processed/         # Cached preprocessed data
│
├── models/                # Trained models
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── xgboost_model.json
│
├── cache/                 # joblib cache for expensive operations
└── training.log           # Training logs
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. Clone or download this project:
```bash
cd DigitClassificationML
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Kaggle API (one-time setup):
   - Download your API key from https://www.kaggle.com/settings/account
   - Place it at `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Quick Start

### Training Models

Run the basic training pipeline:
```bash
python main.py
```

Options:
```bash
# Perform hyperparameter tuning (slow, takes ~hours)
python main.py --tune

# Skip saving models
python main.py --no-save

# Skip displaying confusion matrices
python main.py --no-confusion

# Combine options
python main.py --tune --no-confusion
```

### Making Predictions

```python
from predict import predict_single_image
from training import load_model

# Load a trained model
model = load_model("models/random_forest_model.pkl")

# Predict on a single image
prediction, confidence = predict_single_image(
    model,
    "path/to/digit_image.png",
    return_confidence=True
)
print(f"Predicted digit: {prediction}, Confidence: {confidence:.4f}")
```

See [predict.py](predict.py) for more prediction utilities.

## Module Documentation

### config.py
Centralized configuration for the entire project:
- **Data paths**: Dataset, cache, and model directories
- **Preprocessing parameters**: Image size, normalization
- **Model hyperparameters**: Default settings for RF, SVM, XGBoost
- **Training parameters**: Cross-validation, random seed

**Key Constants:**
- `IMAGE_SIZE = (28, 28)` - Standard MNIST image size
- `TEST_SIZE = 0.2` - 80/20 train/test split
- `RANDOM_STATE = 42` - For reproducibility
- `PCA_N_COMPONENTS = 500` - Dimensionality reduction (optional)

### preprocessing.py
Image preprocessing functions with detailed documentation:

**Main Functions:**
- `preprocess_image()` - Preprocess single image (resize, binarize, normalize, flatten)
- `batch_preprocess_images()` - Process multiple images

**Key Steps:**
1. Resize to 28×28 pixels
2. Apply Otsu's thresholding for binarization
3. Normalize pixel values to [0, 1]
4. Flatten to 1D vector (784 dimensions)

### data_loader.py
Complete data pipeline management:

**Functions:**
- `download_kaggle_dataset()` - Download MNIST from Kaggle
- `load_and_filter_dataset()` - Load CSV and apply filters
- `preprocess_dataset()` - Process all images with caching
- `prepare_data()` - Complete pipeline orchestrator

**Features:**
- Automatic dataset download
- Filtering by group and origin
- Preprocessed data caching to avoid reprocessing
- Stratified train/test split
- Detailed logging

### models.py
Factory functions for creating configured models:

**Functions:**
- `create_random_forest_model()` - Random Forest classifier
- `create_svm_model()` - Support Vector Machine
- `create_xgboost_model()` - XGBoost classifier
- `create_preprocessing_pipeline()` - ScalerPCA pipeline
- `preprocess_data()` - Apply preprocessing to X_train/X_test

**Features:**
- Easily customizable hyperparameters
- Optional PCA dimensionality reduction
- Comprehensive logging

### training.py
Model training and tuning utilities:

**Functions:**
- `train_model()` - Basic training with timing
- `train_and_evaluate()` - Complete train + eval pipeline
- `hyperparameter_tuning()` - RandomizedSearchCV wrapper
- `save_model()` / `load_model()` - Model persistence

**Features:**
- Unified interface for different model types
- Automatic model saving (supports both sklearn and XGBoost)
- Hyperparameter tuning with cross-validation
- Timing and logging

### evaluation.py
Comprehensive model evaluation:

**Functions:**
- `evaluate_model()` - Complete evaluation with metrics and confusion matrix
- `compare_models()` - Compare multiple models
- `plot_model_learning_curves()` - Hyperparameter tuning visualization

**Metrics Computed:**
- Accuracy, Precision, Recall, F1 Score
- Macro and weighted averages (handles class imbalance)
- Per-class classification report
- Confusion matrix visualization

### predict.py
Inference utilities for single images or batches:

**Functions:**
- `predict_single_image()` - Predict from image file
- `predict_batch()` - Predict from multiple files
- `predict_array()` - Predict from feature array
- `print_prediction_result()` - Formatted output

**Features:**
- Optional confidence scores
- Error handling for invalid images
- Batch prediction with progress tracking

### main.py
Main training orchestration script:

**Features:**
- Complete ML pipeline (data → train → eval → save)
- Three classifiers trained and evaluated
- Optional hyperparameter tuning
- Model comparison and ranking
- Command-line interface with arguments
- Comprehensive logging to file and console

## Data Format

### Input Images
- **Format**: Any format supported by OpenCV (PNG, JPG, etc.)
- **Recommended size**: 28×28 pixels (will be resized if different)
- **Color**: Grayscale or color (converted to grayscale)
- **Content**: Black background with white digit foreground
- **Dataset**: MNIST handwritten digits (0-9)

### Feature Format
After preprocessing, images are represented as:
- **Shape**: (784,) for single image, (n_samples, 784) for batch
- **Range**: [0.0, 1.0] (normalized pixel values)
- **Type**: np.float32

### Labels
- **Type**: Integer (0-9)
- **Mapping**: Digit value → index (0→0, 1→1, ..., 9→9)

## Reproducibility

This project ensures reproducible results through:

1. **Fixed Random Seeds**
   - `RANDOM_STATE = 42` throughout
   - Stratified splits preserve class distribution

2. **Versioned Dependencies**
   - All packages pinned in `requirements.txt`

3. **Centralized Configuration**
   - All parameters in `config.py`
   - Easy to modify and track changes

4. **Detailed Logging**
   - All operations logged to file and console
   - Saved to `training.log`

To reproduce results:
```bash
python main.py
# Results will be identical across runs
```

## Extending the Project

### Adding a New Model

1. Add factory function in `models.py`:
```python
def create_my_model():
    """Create My Model classifier."""
    return MyModel(**config.MY_MODEL_PARAMS)
```

2. Add hyperparameters to `config.py`:
```python
MY_MODEL_PARAMS = {...}
MY_MODEL_PARAM_DIST = {...}
```

3. Add training code in `main.py`:
```python
my_model = model_factory.create_my_model()
my_results = training_module.train_and_evaluate(
    my_model, X_train, X_test, y_train, y_test,
    model_name="My Model"
)
results["My Model"] = my_results["metrics"]
```

### Using Custom Data

Replace the `prepare_data()` call in `main.py`:

```python
from data_loader import preprocess_dataset
from sklearn.model_selection import train_test_split

# Load your own dataset
X, y = load_my_data()

# Preprocess if needed
X_processed = [preprocess_image(img) for img in X]
X = np.array(X_processed)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Continue with training...
```

## Performance Expectations

Based on the original notebook analysis:

| Model | Accuracy | F1 (Macro) | Training Time |
|-------|----------|-----------|---------------|
| Random Forest | ~96% | ~0.96 | ~2-5 sec |
| SVM (RBF) | ~95.7% | ~0.957 | ~10-30 sec |
| XGBoost | ~97% | ~0.97 | ~5-10 sec |

*Times vary by hardware; these are approximate.*

## Troubleshooting

### Kaggle Dataset Download Fails
- Ensure Kaggle API credentials are configured correctly
- Check internet connection
- Verify `~/.kaggle/kaggle.json` exists and has correct permissions

### Out of Memory Errors
- Use PCA: Set `USE_PCA = True` in `config.py` to reduce dimensionality
- Reduce `random_search_iterations` for faster (less thorough) tuning
- Process data in smaller batches

### Slow Training
- Use fewer hyperparameter tuning iterations
- Reduce `cv_splits` (cross-validation splits)
- Set `USE_PCA = True` for faster training

### Predictions Return -1 Confidence
- Model doesn't support `predict_proba()` (e.g., SVM without probability calibration)
- This is expected; still valid prediction

## Files Overview

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| config.py | Configuration | All configuration constants |
| preprocessing.py | Image processing | `preprocess_image()`, `batch_preprocess_images()` |
| data_loader.py | Data pipeline | `prepare_data()`, `download_kaggle_dataset()` |
| models.py | Model creation | `create_random_forest_model()`, `create_svm_model()` |
| training.py | Training utilities | `train_and_evaluate()`, `hyperparameter_tuning()` |
| evaluation.py | Model evaluation | `evaluate_model()`, `compare_models()` |
| predict.py | Inference | `predict_single_image()`, `predict_batch()` |
| main.py | Main pipeline | `main()` - orchestrates everything |

## License

This project is provided as-is for educational and research purposes.

## References

- **MNIST Dataset**: [Kaggle MNIST](https://www.kaggle.com/datasets/pintowar/numerical-images)
- **scikit-learn**: [Documentation](https://scikit-learn.org/)
- **XGBoost**: [Documentation](https://xgboost.readthedocs.io/)
- **OpenCV**: [Documentation](https://opencv-python-tutroals.readthedocs.io/)

## Support

For issues, questions, or suggestions:
1. Check the troubleshooting section above
2. Review the inline code documentation
3. Check `training.log` for detailed error messages
4. Inspect `config.py` to understand current settings

## Future Enhancements

Potential improvements:
- [ ] Deep learning models (CNN, ResNet)
- [ ] Transfer learning with pretrained models
- [ ] Ensemble methods combining all three models
- [ ] Real-time webcam digit recognition
- [ ] Model explainability (SHAP, LIME)
- [ ] Automated ML (AutoML) frameworks
- [ ] REST API for model serving
- [ ] Web UI for interactive predictions
