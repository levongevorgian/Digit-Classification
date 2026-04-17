"""
Image preprocessing utilities for digit classification.

This module provides functions to preprocess images for the digit classification
model, including resizing, binarization, normalization, and flattening.
"""

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from config import IMAGE_SIZE


def preprocess_image(img: np.ndarray, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Preprocess a single image for model input.
    
    Performs the following operations:
    1. Resizes image to target size (default 28x28)
    2. Applies Otsu's thresholding for binarization (if grayscale)
    3. Normalizes pixel values to 0-1 range
    4. Flattens to 1D array (784 dimensions for 28x28 images)
    
    Args:
        img (np.ndarray): Input image as numpy array
        target_size (tuple): Target size as (height, width). Default is (28, 28)
    
    Returns:
        np.ndarray: Preprocessed, flattened image vector
    
    Raises:
        ValueError: If image is None or invalid
    
    Example:
        >>> img = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)
        >>> preprocessed = preprocess_image(img)
        >>> print(preprocessed.shape)  # (784,)
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Image must be a valid numpy array")
    
    # Resize image to target size if needed
    if img.shape != target_size[::-1]:  # OpenCV uses (height, width)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Binarize using Otsu's method if not already binary
    if len(np.unique(img)) > 2:
        thresh = threshold_otsu(img)
        img = (img > thresh).astype(np.uint8) * 255
    
    # Normalize pixel values to 0-1 range
    img = img.astype(np.float32) / 255.0
    
    # Flatten to 1D vector
    return img.flatten()


def batch_preprocess_images(images: list, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Preprocess a batch of images.
    
    Args:
        images (list): List of numpy arrays representing images
        target_size (tuple): Target size as (height, width)
    
    Returns:
        np.ndarray: 2D array of preprocessed images (n_samples, n_features)
    """
    preprocessed = []
    for img in images:
        try:
            preprocessed.append(preprocess_image(img, target_size))
        except ValueError as e:
            print(f"Warning: Failed to preprocess image - {e}")
            continue
    
    return np.array(preprocessed)
