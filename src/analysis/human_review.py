import json
from datetime import datetime
from pathlib import Path

from src.config import HUMAN_REVIEWS_PATH, PREDICTIONS_DIR


def load_human_reviews():
    """Loads recorded human-in-the-loop review history from local JSON file."""
    if HUMAN_REVIEWS_PATH.exists():
        try:
            with open(HUMAN_REVIEWS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_human_review(wafer_id, predicted_class, confidence, reviewer_decision, corrected_class=None, notes=""):
    """
    Saves a human review entry to outputs/predictions/human_reviews.json.
    """
    reviews = load_human_reviews()

    entry = {
        'review_id': f"REV-{len(reviews) + 1:04d}",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'wafer_id': str(wafer_id),
        'predicted_class': str(predicted_class),
        'confidence_pct': round(float(confidence), 2),
        'reviewer_decision': str(reviewer_decision),
        'corrected_class': str(corrected_class) if corrected_class else str(predicted_class),
        'notes': str(notes)
    }

    reviews.append(entry)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(HUMAN_REVIEWS_PATH, "w") as f:
        json.dump(reviews, f, indent=4)

    return entry
