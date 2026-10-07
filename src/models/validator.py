import numpy as np
import torch
import cv2

from src.config import IMAGE_SIZE


def validate_wafer_input(input_img):
    """
    Validates whether an input numpy array or image is a valid wafer map representation.
    
    Returns:
        is_valid (bool): True if valid wafer map, False otherwise.
        status_type (str): 'valid_normal', 'valid_defective', 'unknown_invalid', or 'outlier_pattern'.
        message (str): Explanatory message.
        processed_matrix (np.ndarray or None): Clean 2D wafer matrix if valid.
    """
    if input_img is None:
        return False, "unknown_invalid", "Input image is empty or invalid file format.", None

    if not isinstance(input_img, np.ndarray):
        return False, "unknown_invalid", "Input does not appear to be a valid wafer map.", None

    # Handle RGB/BGR or 2D image
    if input_img.ndim == 3:
        if input_img.shape[2] == 4: # RGBA
            input_img = cv2.cvtColor(input_img, cv2.COLOR_RGBA2GRAY)
        elif input_img.shape[2] == 3:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    if input_img.ndim != 2:
        return False, "unknown_invalid", "Input does not appear to be a valid wafer map.", None

    h, w = input_img.shape
    if h < 5 or w < 5:
        return False, "unknown_invalid", "Input resolution is too small to be a wafer map.", None

    aspect_ratio = float(w) / float(h)
    if aspect_ratio < 0.3 or aspect_ratio > 3.3:
        return False, "unknown_invalid", "Input aspect ratio is inconsistent with wafer maps.", None

    # Inspect discrete value distribution
    unique_vals = np.unique(input_img)

    # Standard WM-811K 2D die matrix contains values {0, 1, 2}
    if set(unique_vals).issubset({0, 1, 2}) or set(unique_vals).issubset({0, 1, 2, 255}):
        # Direct discrete wafer map
        matrix = input_img.copy()
        matrix[matrix == 255] = 2
        non_bg_ratio = np.mean(matrix > 0)
        if non_bg_ratio < 0.05 or non_bg_ratio > 0.98:
            return False, "unknown_invalid", "Input does not appear to be a valid wafer map (invalid boundary).", None
        return True, "valid", "Valid wafer map representation detected.", matrix

    # Grayscale image uploaded (e.g. exported PNG of wafer map)
    # Check for wafer circular/oval boundary using Otsu thresholding
    _, thresh = cv2.threshold(input_img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    non_zero_ratio = np.mean(thresh > 0)

    if non_zero_ratio < 0.05 or non_zero_ratio > 0.98:
        return False, "unknown_invalid", "Input does not appear to be a valid wafer map.", None

    # Convert arbitrary grayscale uploaded image into standard 0,1,2 wafer die matrix
    # Normalize grayscale to 0, 1, 2 die states
    matrix = np.zeros_like(thresh, dtype=np.int32)
    matrix[thresh > 0] = 1 # Normal die region

    # Identify defective dies as high intensity or distinct sub-regions
    high_intensity = (input_img > np.percentile(input_img[thresh > 0], 85)) if np.any(thresh > 0) else np.zeros_like(thresh, dtype=bool)
    matrix[high_intensity & (thresh > 0)] = 2

    return True, "valid", "Valid wafer map image loaded.", matrix


def detect_anomaly_confidence(probabilities, confidence_threshold=0.45, max_entropy_threshold=1.8):
    """
    Detects low confidence / out-of-distribution wafer patterns using prediction entropy.
    """
    probs = np.clip(probabilities, 1e-8, 1.0)
    top_conf = float(np.max(probs))
    entropy = float(-np.sum(probs * np.log(probs)))

    is_anomaly = (top_conf < confidence_threshold) or (entropy > max_entropy_threshold)

    return is_anomaly, top_conf, entropy
