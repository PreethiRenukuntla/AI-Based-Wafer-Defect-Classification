import sys
import os
from pathlib import Path
import torch
from PIL import Image

# Add app directory to sys.path if needed
BASE_DIR = Path(r"c:\Users\preet\OneDrive\Desktop\AI-Wafer-Defect-Classification")
sys.path.insert(0, str(BASE_DIR))

# Import logic from app.app
from app.app import (
    load_model,
    predict_image,
    validate_wafer_image,
    create_gradcam,
    CLASS_NAMES,
    PROJECT_NAME
)

print(f"=== PROJECT NAME CHECK ===")
print(f"Project Name: {PROJECT_NAME}")
assert PROJECT_NAME == "AI-Based Wafer Defect Classification Using Deep Learning", "Project name mismatch!"

print("\n=== MODEL LOADING CHECK ===")
model = load_model()
print("Model loaded successfully!")

print("\n=== TEST 1: HEALTHY WAFER (Normal Class) ===")
normal_img_path = BASE_DIR / "data" / "processed" / "images" / "Normal"
normal_files = list(normal_img_path.glob("*.png"))
if normal_files:
    test_normal = Image.open(normal_files[0])
    is_valid, err = validate_wafer_image(test_normal)
    print(f"Validation: valid={is_valid}, err='{err}'")
    pred_class, conf, probs, pred_idx = predict_image(test_normal)
    print(f"Predicted Class: {pred_class}, Confidence: {conf*100:.2f}%")
else:
    print("No normal image found to test.")

print("\n=== TEST 2: DEFECTIVE WAFER (Center Class) ===")
center_img_path = BASE_DIR / "data" / "processed" / "images" / "Center" / "wafer_000004.png"
if center_img_path.exists():
    test_center = Image.open(center_img_path)
    is_valid, err = validate_wafer_image(test_center)
    print(f"Validation: valid={is_valid}, err='{err}'")
    pred_class, conf, probs, pred_idx = predict_image(test_center)
    print(f"Predicted Class: {pred_class}, Confidence: {conf*100:.2f}%")
    cam_img = create_gradcam(test_center, pred_idx)
    print(f"Grad-CAM Generated: {cam_img is not None}")
else:
    print("Center image not found.")

print("\n=== TEST 3: INVALID IMAGE DETECTION ===")
# Create a dummy RGB photograph-like image with high color variance
dummy_photo = Image.new("RGB", (100, 100))
import numpy as np
arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
# Add strong color separation
arr[:, :, 0] = 255
arr[:, :, 1] = 0
arr[:, :, 2] = 128
dummy_photo = Image.fromarray(arr)
is_valid, err = validate_wafer_image(dummy_photo)
print(f"Photo Validation: valid={is_valid}, err='{err}'")
assert not is_valid, "Invalid image was incorrectly marked as valid!"

print("\n=== ALL AUTOMATED CHECKS PASSED SUCCESSFULLY! ===")
