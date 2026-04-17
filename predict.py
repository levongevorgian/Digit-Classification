"""
Prediction utilities for digit classification.

Provides functions to make predictions on single images or batches,
with optional confidence scores.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Union

from preprocessing import preprocess_image
import config

logger = logging.getLogger(__name__)


def predict_single_image(
    model,
    image_path: str,
    return_confidence: bool = False
) -> Union[int, Tuple[int, float]]:
    """
    Predict the digit in a single image.
    
    The image should preferably be:
    - 28x28 pixels (will be resized if not)
    - Grayscale
    - Black background with white foreground
    
    Args:
        model: Trained classifier model
        image_path (str): Path to image file
        return_confidence (bool): If True, return prediction confidence/probability
    
    Returns:
        Union[int, Tuple[int, float]]:
            - If return_confidence is False: Predicted digit (0-9)
            - If return_confidence is True: Tuple of (predicted_digit, confidence)
    
    Raises:
        FileNotFoundError: If image file not found
        ValueError: If image cannot be read or is invalid
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")
    
    try:
        # Preprocess
        preprocessed = preprocess_image(img, config.IMAGE_SIZE)
        X = preprocessed.reshape(1, -1)  # Reshape to (1, n_features)
        
        # Predict
        prediction = model.predict(X)[0]
        
        if return_confidence:
            # Try to get confidence scores
            try:
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            except AttributeError:
                # Model doesn't have predict_proba (e.g., SVM without probability=True)
                confidence = None
            
            return prediction, confidence
        else:
            return prediction
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        raise


def predict_batch(
    model,
    image_paths: list,
    return_confidence: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict digits for a batch of images.
    
    Args:
        model: Trained classifier model
        image_paths (list): List of image file paths
        return_confidence (bool): If True, return confidences as well
    
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - If return_confidence is False: Array of predictions
            - If return_confidence is True: Tuple of (predictions, confidences)
    """
    predictions = []
    confidences = []
    failed_images = []
    
    logger.info(f"Predicting {len(image_paths)} images...")
    
    for idx, image_path in enumerate(image_paths):
        try:
            if return_confidence:
                pred, conf = predict_single_image(
                    model, image_path, return_confidence=True
                )
                predictions.append(pred)
                confidences.append(conf if conf is not None else -1.0)
            else:
                pred = predict_single_image(model, image_path, return_confidence=False)
                predictions.append(pred)
        
        except Exception as e:
            logger.warning(f"Failed to predict {image_path}: {e}")
            failed_images.append(image_path)
            predictions.append(-1)  # Error indicator
            if return_confidence:
                confidences.append(-1.0)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(image_paths)} images")
    
    if failed_images:
        logger.warning(f"Failed to process {len(failed_images)} images")
    
    predictions = np.array(predictions)
    
    if return_confidence:
        confidences = np.array(confidences)
        return predictions, confidences
    else:
        return predictions


def predict_array(
    model,
    X: np.ndarray,
    return_confidence: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict digits for preprocessed feature arrays.
    
    Use this when you already have preprocessed features.
    
    Args:
        model: Trained classifier model
        X (np.ndarray): Preprocessed feature matrix (n_samples, n_features)
        return_confidence (bool): If True, return confidence scores
    
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - If return_confidence is False: Prediction array
            - If return_confidence is True: Tuple of (predictions, confidences)
    """
    predictions = model.predict(X)
    
    if return_confidence:
        try:
            confidences = model.predict_proba(X).max(axis=1)
        except AttributeError:
            logger.warning("Model does not support predict_proba, returning confidence=-1")
            confidences = np.full(len(predictions), -1.0)
        
        return predictions, confidences
    else:
        return predictions


def print_prediction_result(
    prediction: int,
    confidence: float = None,
    image_path: str = None
) -> None:
    """
    Print a formatted prediction result.
    
    Args:
        prediction (int): Predicted digit
        confidence (float): Prediction confidence (0-1)
        image_path (str): Optional image path for reference
    """
    msg = f"Predicted: {prediction}"
    
    if image_path:
        msg = f"{image_path} -> {msg}"
    
    if confidence is not None and confidence >= 0:
        msg += f" (confidence: {confidence:.4f})"
    
    print(msg)
