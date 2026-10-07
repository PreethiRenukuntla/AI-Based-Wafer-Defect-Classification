import numpy as np
import torch
import torch.nn.functional as F

from src.config import BEST_MODEL_PATH, DEVICE, DEFECT_CLASSES, NUM_CLASSES, IMAGE_SIZE
from src.data.dataset import preprocess_wafer_matrix
from src.models.architecture import WaferDefectResNet
from src.models.validator import validate_wafer_input, detect_anomaly_confidence
from src.analysis.yield_severity import calculate_wafer_yield_stats, analyze_wafer_severity
from src.explainability.gradcam import generate_gradcam_overlay
from src.search.embedding_search import search_similar_wafers
from src.intelligence.insights_recommendations import generate_ai_insights, generate_engineering_recommendations

_LOADED_MODEL = None


def get_prediction_model():
    """Loads and caches single instance of WaferDefectResNet."""
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    model = WaferDefectResNet(num_classes=NUM_CLASSES).to(DEVICE)
    if BEST_MODEL_PATH.exists():
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {BEST_MODEL_PATH}")
    else:
        print("Warning: Trained checkpoint not found. Using initialized model weights.")

    model.eval()
    _LOADED_MODEL = model
    return _LOADED_MODEL


def predict_single_wafer(input_img, index_data=None, lot_info=None):
    """
    Full 11-Step Uploaded Image Prediction Pipeline.
    """
    model = get_prediction_model()

    # STEP 1 & STEP 2: Input Validation & Mode Determination
    is_valid, status_type, val_message, matrix = validate_wafer_input(input_img)

    if not is_valid or matrix is None:
        return {
            'is_valid': False,
            'status_type': 'unknown_invalid',
            'message': val_message,
            'predicted_class': 'Unknown / Invalid Input',
            'confidence_pct': 0.0,
            'probabilities': {cls: 0.0 for cls in DEFECT_CLASSES},
            'yield_stats': calculate_wafer_yield_stats(None),
            'severity_info': analyze_wafer_severity(None, 'none', 0.0),
            'gradcam': None,
            'similar_wafers': [],
            'insights': ["Input image failed validation check. Input does not appear to be a valid wafer map."],
            'recommendations': {
                'suggested_investigation': 'Invalid Input Uploaded',
                'action_type': 'None',
                'target_investigation_areas': ['Upload a valid 2D semiconductor wafer map image.'],
                'disclaimer': 'N/A',
                'recommendation_effectiveness_status': 'N/A'
            }
        }

    # STEP 3: Generate Prediction Probabilities
    tensor = preprocess_wafer_matrix(matrix, target_size=IMAGE_SIZE).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(np.argmax(probs))
    predicted_class = DEFECT_CLASSES[top_idx]
    confidence_pct = round(float(probs[top_idx]) * 100.0, 2)

    # Anomaly / Unknown pattern check
    is_anomaly, top_conf, entropy = detect_anomaly_confidence(probs)

    prob_dict = {DEFECT_CLASSES[i]: round(float(probs[i]) * 100.0, 2) for i in range(len(DEFECT_CLASSES))}

    # STEP 6 & STEP 7: Wafer-level yield statistics & Severity Analysis
    yield_stats = calculate_wafer_yield_stats(matrix)
    severity_info = analyze_wafer_severity(matrix, predicted_class, confidence_pct)

    # STEP 8: Grad-CAM Explainability
    orig_rgb, heatmap_rgb, overlay_rgb, _, _, _ = generate_gradcam_overlay(matrix, model, target_class_idx=top_idx)

    # STEP 9 & STEP 10: Similar Wafers, Insights, and Engineering Recommendations
    similar_wafers = []
    if index_data is not None:
        similar_wafers = search_similar_wafers(matrix, model, index_data, top_k=5)

    insights = generate_ai_insights(matrix, predicted_class, confidence_pct, yield_stats, severity_info, lot_info)
    recommendations = generate_engineering_recommendations(predicted_class, severity_info)

    if is_anomaly and confidence_pct < 65.0:
        insights.insert(0, "Low-confidence prediction alert: Human review recommended.")

    return {
        'is_valid': True,
        'status_type': 'valid_defective' if predicted_class != 'none' else 'valid_normal',
        'message': val_message,
        'predicted_class': predicted_class,
        'confidence_pct': confidence_pct,
        'probabilities': prob_dict,
        'yield_stats': yield_stats,
        'severity_info': severity_info,
        'wafer_matrix': matrix,
        'gradcam': {
            'orig_rgb': orig_rgb,
            'heatmap_rgb': heatmap_rgb,
            'overlay_rgb': overlay_rgb
        },
        'similar_wafers': similar_wafers,
        'insights': insights,
        'recommendations': recommendations
    }
