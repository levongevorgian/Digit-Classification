"""
ARCHITECTURE OVERVIEW
=====================

This document describes the architecture and data flow of the Digit Classification ML project.
"""

PROJECT_ARCHITECTURE = r"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DIGIT CLASSIFICATION ML PROJECT                         │
│                              ARCHITECTURE DIAGRAM                           │
└─────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════╗
║                            INPUT LAYER                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

                         KAGGLE DATASET (MNIST)
                              │
                              ▼
                    [data_loader.py]
                    ├─ download_kaggle_dataset()
                    ├─ load_and_filter_dataset()
                    └─ preprocess_dataset()
                              │
                              ▼
                    RAW IMAGE DATA (PNG FILES)
                              │
                    CSV METADATA (numbers.csv)


╔═══════════════════════════════════════════════════════════════════════════╗
║                       PREPROCESSING LAYER                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

    Image → [preprocessing.py] → Feature Vector
    │       preprocess_image()   │
    │       • Resize (28×28)     │
    │       • Binarize (Otsu)    │
    │       • Normalize [0,1]    │
    │       • Flatten (784-dim)  │
    └──────────────────────────────┘
                                     │
                                     ▼
                            CACHED FEATURES
                        (X.npy, y.npy, labels)


╔═══════════════════════════════════════════════════════════════════════════╗
║                      CONFIGURATION LAYER                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

                              [config.py]
                              (Centralized)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              DATA CONFIG    MODEL PARAMS    TRAINING PARAMS
              • Paths        • Hyperparams   • CV Splits
              • Image size   • RF, SVM       • Random seed
              • Split ratio  • XGBoost       • N_jobs


╔═══════════════════════════════════════════════════════════════════════════╗
║                       FEATURE ENGINEERING LAYER                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

        Raw Features (784-dim)
               │
        [models.py]
        create_preprocessing_pipeline()
               │
               ├─ StandardScaler()
               │  └─ Normalize features
               │
               └─ PCA() [Optional]
                  └─ Reduce to 500 dims
               │
               ▼
        Processed Features


╔═══════════════════════════════════════════════════════════════════════════╗
║                        TRAINING LAYER                                     ║
╚═══════════════════════════════════════════════════════════════════════════╝

    Processed Data (X_train, y_train)
              │
              ├──────────┬──────────┬──────────┐
              │          │          │          │
              ▼          ▼          ▼          ▼
        [models.py]   [models.py]  [models.py]
        Random Forest  SVM         XGBoost
              │          │          │
              ▼          ▼          ▼
        [training.py] : train_and_evaluate()
              │
              ├─ Train on X_train
              ├─ Predict on X_test
              └─ Evaluate metrics
              │
              ▼
        Trained Models


╔═══════════════════════════════════════════════════════════════════════════╗
║                       EVALUATION LAYER                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

    Predictions (y_pred)
              │
        [evaluation.py]
        evaluate_model()
              │
              ├─ Compute metrics
              │  ├─ Accuracy
              │  ├─ Precision
              │  ├─ Recall
              │  └─ F1 Score
              │
              ├─ Classification report
              └─ Confusion matrix
              │
              ▼
        EVALUATION METRICS & VISUALIZATIONS


╔═══════════════════════════════════════════════════════════════════════════╗
║                       PERSISTENCE LAYER                                   ║
╚═══════════════════════════════════════════════════════════════════════════╝

    Trained Models
              │
        [training.py]
        save_model()
              │
              ├─ RF Model      → random_forest_model.pkl
              ├─ SVM Model     → svm_model.pkl
              └─ XGBoost Model → xgboost_model.json
              │
              ▼
        DISK STORAGE (models/)


╔═══════════════════════════════════════════════════════════════════════════╗
║                        INFERENCE LAYER                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

    Image File
        │
        ▼
    [preprocessing.py]
    preprocess_image()
        │
        ▼
    Feature Vector
        │
        ├─ Load Model (training.load_model())
        │
        ▼
    [predict.py]
    predict_single_image()
        │
        ├─ Predict class (0-9)
        └─ Return confidence (optional)
        │
        ▼
    PREDICTION & CONFIDENCE


╔═══════════════════════════════════════════════════════════════════════════╗
║                       ORCHESTRATION LAYER                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

    main.py (Complete Pipeline)
         │
         ├─ Prepare Data
         │  └─ data_loader.prepare_data()
         │
         ├─ Train 3 Models
         │  ├─ Random Forest
         │  ├─ SVM
         │  └─ XGBoost
         │
         ├─ Evaluate & Compare
         │  └─ evaluation.compare_models()
         │
         ├─ Hyperparameter Tuning (Optional)
         │  └─ training.hyperparameter_tuning()
         │
         └─ Save Models
            └─ training.save_model()


╔═══════════════════════════════════════════════════════════════════════════╗
║                        MODULE DEPENDENCIES                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

                                config.py
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
            preprocessing.py  data_loader.py  models.py
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
                training.py   evaluation.py  predict.py
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                                   ▼
                                main.py
                           (Orchestration)


╔═══════════════════════════════════════════════════════════════════════════╗
║                         DATA FLOW EXAMPLE                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

1. DATA PREPARATION
   ─────────────────
   Kaggle Dataset
        │
        ▼ download_kaggle_dataset()
   PNG Files (60,000 images)
        │
        ▼ load_and_filter_dataset()
   Filtered Dataset (60,000 handwritten MNIST)
        │
        ▼ preprocess_dataset()
   Feature Matrix X (60000, 784)
   Label Vector y (60000,)
        │
        ▼ train_test_split()
   X_train (48000, 784), X_test (12000, 784)
   y_train (48000,), y_test (12000,)

2. MODEL TRAINING
   ───────────────
   X_train, y_train
        │
        ├─ Random Forest
        │       ▼
        │  train() → 96% accuracy
        │
        ├─ SVM
        │       ▼
        │  train() → 95.7% accuracy
        │
        └─ XGBoost
                ▼
            train() → 97% accuracy

3. EVALUATION
   ──────────
   y_test vs y_pred
        │
        ▼ evaluate_model()
   Metrics:
   ├─ Accuracy: 0.97
   ├─ Precision: 0.97
   ├─ Recall: 0.97
   ├─ F1: 0.97
   └─ Confusion Matrix (visualization)

4. INFERENCE
   ──────────
   New Image (digit.png)
        │
        ▼ preprocess_image()
   Feature Vector (784,)
        │
        ├─ Load trained model
        │
        ▼ predict()
   Prediction: 7
   Confidence: 0.98


╔═══════════════════════════════════════════════════════════════════════════╗
║                        DIRECTORY STRUCTURE                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

DigitClassificationML/
│
├── Core Modules (Python files)
│   ├── config.py              # Configuration hub
│   ├── preprocessing.py       # Image processing
│   ├── data_loader.py         # Data pipeline
│   ├── models.py              # Model factory
│   ├── training.py            # Training utilities
│   ├── evaluation.py          # Metrics & eval
│   ├── predict.py             # Inference
│   ├── main.py                # Orchestrator
│   └── examples.py            # Example usage
│
├── Documentation
│   ├── README.md              # Full documentation
│   ├── QUICKSTART.md          # Quick reference
│   ├── PROJECT_SUMMARY.txt    # This summary
│   └── requirements.txt       # Dependencies
│
├── Data Directory (auto-created)
│   ├── kaggle_dataset/        # Downloaded MNIST
│   └── processed/             # Cached features
│       ├── X.npy
│       ├── y.npy
│       └── label_map.npy
│
├── Models Directory (auto-created)
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── xgboost_model.json
│
├── Cache Directory
│   └── (joblib cache)
│
└── Logs
    └── training.log


╔═══════════════════════════════════════════════════════════════════════════╗
║                        EXECUTION FLOW                                     ║
╚═══════════════════════════════════════════════════════════════════════════╝

STANDARD USAGE (main.py)
────────────────────────
$ python main.py

    main()
        │
        ├─ prepare_data()
        │       └─ Download + Preprocess + Split
        │
        ├─ Train RF
        │   └─ evaluate_model()
        │
        ├─ Train SVM
        │   └─ evaluate_model()
        │
        ├─ Train XGBoost
        │   └─ evaluate_model()
        │
        ├─ compare_models()
        │   └─ Print ranking
        │
        └─ save_model() (all 3 models)


CUSTOM WORKFLOW (examples.py)
─────────────────────────────
$ python examples.py

    Choose example (1-7):
    
    1. Load data
    2. Train single model
    3. Compare models
    4. Hyperparameter tuning
    5. Save/load models
    6. Make predictions
    7. Preprocessing with PCA


INTERACTIVE USAGE (Python)
──────────────────────────
from data_loader import prepare_data
from models import create_random_forest_model
from training import train_and_evaluate
from predict import predict_single_image

# Get data
X_train, X_test, y_train, y_test = prepare_data()

# Create model
model = create_random_forest_model()

# Train & evaluate
results = train_and_evaluate(...)

# Predict
pred = predict_single_image(model, "image.png")
"""

print(PROJECT_ARCHITECTURE)
