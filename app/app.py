
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================================
# Project setup
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DEFECT_CLASSES,
    BASELINE_ACCURACY,
    OUTPUTS_DIR,
    FIGURES_DIR,
    HUMAN_REVIEWS_PATH,
)
from src.data.wm811k_loader import parse_wm811k_metadata
from src.data.dataset import preprocess_wafer_matrix
from src.inference.predictor import WaferPredictor
from src.search.embedding_search import build_embedding_index, search_similar_wafers
from src.analysis.human_review import save_human_review, load_human_reviews


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Wafer Defect AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .project-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .project-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }

    .module-note {
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.45;
        min-height: 42px;
    }

    .workflow-strip {
        border: 1px solid rgba(96,165,250,0.20);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: rgba(30,64,175,0.12);
        margin: 0.8rem 0 1.2rem 0;
    }

    .status-strip {
        border: 1px solid rgba(34,197,94,0.22);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        background: rgba(22,101,52,0.12);
        margin-bottom: 1rem;
    }

    .nav-module-number {
        color: #60a5fa;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        margin: 0.1rem 0 0.35rem 0;
    }

    .nav-module-number + div {
        margin-bottom: 0.15rem;
    }

    .module-title {
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.88rem;
    }

    .status-ok {
        color: #22c55e;
        font-weight: 700;
    }

    .status-warn {
        color: #f59e0b;
        font-weight: 700;
    }

    .status-danger {
        color: #ef4444;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 12px;
        padding: 0.65rem;
        background: rgba(15,23,42,0.38);
    }

    .wafer-card {
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 14px;
        padding: 0.9rem;
        background: rgba(15,23,42,0.35);
    }

    /* Keep navigation out of the sidebar. */
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Navigation
# ============================================================
# These are the only genuine analysis modules. Yield, severity, AI insight,
# and recommendations are part of the Single Wafer result and are therefore
# intentionally not duplicated as separate navigation pages.
NAVIGATION = [
    "2. Dataset Analysis",
    "3. Single Wafer Prediction",
    "5. Grad-CAM Explainability",
    "6. Similar Wafer Search",
    "7. Lot / Batch Analysis",
    "8. Anomaly & Unknown Detection",
    "9. Human-in-the-Loop Review",
    "12. Model Performance",
]

NAV_DESCRIPTIONS = {
    "1. Main Dashboard": "Project overview, results and project documentation",
    "2. Dataset Analysis": "WM-811K composition and classes",
    "3. Single Wafer Prediction": "Upload once and complete wafer inspection",
    "5. Grad-CAM Explainability": "Visual model explanation",
    "6. Similar Wafer Search": "Find historical matches",
    "7. Lot / Batch Analysis": "Lot-level defect patterns",
    "8. Anomaly & Unknown Detection": "Unusual-pattern screening",
    "9. Human-in-the-Loop Review": "Engineer validation workflow",
    "12. Model Performance": "Official evaluation metrics",
}


# ============================================================
# Stable project facts
# ============================================================
DATASET_STATS = {
    "total": 811_457,
    "labeled": 172_950,
    "unlabeled": 638_507,
    "lots": 10_762,
    "classes": {
        "none": 147_431,
        "Edge-Ring": 9_680,
        "Edge-Loc": 5_189,
        "Center": 4_294,
        "Loc": 3_593,
        "Scratch": 1_193,
        "Random": 866,
        "Donut": 555,
        "Near-full": 149,
    },
}

RECOMMENDATIONS = {
    "Center": (
        "Review center-region process uniformity and compare this wafer "
        "with neighboring lots and similar historical maps."
    ),
    "Donut": (
        "Review radial/annular process behavior and compare the pattern "
        "with historical lots before investigating a specific process cause."
    ),
    "Edge-Loc": (
        "Review wafer-edge process conditions and edge-uniformity measurements; "
        "compare with neighboring wafers."
    ),
    "Edge-Ring": (
        "Review radial edge behavior and edge-exclusion/process-uniformity "
        "records across the lot."
    ),
    "Loc": (
        "Check localized cluster behavior against lot, tool and process-step "
        "records; use similar-wafer search for comparison."
    ),
    "Near-full": (
        "Prioritize engineering inspection and containment review because "
        "the predicted pattern covers a large wafer region."
    ),
    "Random": (
        "Review contamination/process-event records and compare the pattern "
        "with other wafers from the same lot."
    ),
    "Scratch": (
        "Review wafer handling, transport and surface-inspection records "
        "for the affected lot."
    ),
    "none": (
        "No known defect class is predicted. Continue normal monitoring "
        "and verify confidence before closing the case."
    ),
}

INSIGHTS = {
    "Center": "A center-region pattern was identified. Compare center behavior across nearby wafers and lots.",
    "Donut": "An annular pattern was identified. Compare radial distribution across the lot and similar historical wafers.",
    "Edge-Loc": "An edge-localized pattern was identified. Check whether the behavior repeats at the wafer edge across the lot.",
    "Edge-Ring": "An edge-ring pattern was identified. Review radial/edge uniformity trends and lot-level recurrence.",
    "Loc": "A localized cluster was identified. Check lot and tool context and compare similar wafers.",
    "Near-full": "A near-full pattern was identified. The defect distribution covers a large wafer region and warrants detailed review.",
    "Random": "A random pattern was identified. Compare with other wafers and process-event records rather than assuming a single cause.",
    "Scratch": "A scratch-like pattern was identified. Review handling and surface-related records alongside the map.",
    "none": "No known defect pattern was predicted. This does not guarantee a defect-free wafer; confidence and map quality should still be reviewed.",
}

CLASS_DESCRIPTION = {
    "Center": "Center-region defect pattern",
    "Donut": "Annular/ring-shaped defect pattern",
    "Edge-Loc": "Localized edge defect pattern",
    "Edge-Ring": "Edge-ring defect pattern",
    "Loc": "Localized defect cluster",
    "Near-full": "Very broad defect coverage",
    "Random": "Scattered/random defect pattern",
    "Scratch": "Scratch-like defect pattern",
    "none": "Normal / no labeled defect pattern",
}


# ============================================================
# Session state
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "1. Main Dashboard"

if "wafer" not in st.session_state:
    st.session_state.wafer = None

if "wafer_source" not in st.session_state:
    st.session_state.wafer_source = None

if "upload_hash" not in st.session_state:
    st.session_state.upload_hash = None

if "upload_preview" not in st.session_state:
    st.session_state.upload_preview = None

if "upload_has_defect_channel" not in st.session_state:
    st.session_state.upload_has_defect_channel = True

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "analysis_message" not in st.session_state:
    st.session_state.analysis_message = None


# ============================================================
# Cached resources - deliberately lazy
# ============================================================
@st.cache_resource
def get_predictor():
    # The locked controlled checkpoint is selected inside WaferPredictor.
    return WaferPredictor()


@st.cache_data
def get_official_results():
    path = OUTPUTS_DIR / "controlled_official_test_results.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "accuracy": 95.55,
        "error_rate": 4.45,
        "balanced_accuracy": 73.39,
        "macro_f1": 0.7049,
        "weighted_f1": 0.9574,
        "test_samples": 118595,
    }


@st.cache_data
def get_metadata():
    # Loaded only when a page actually needs lot/class metadata.
    return parse_wm811k_metadata()


@st.cache_resource
def get_embedding_index():
    predictor = get_predictor()
    return build_embedding_index(
        predictor.model,
        sample_size_per_class=200,
        force_rebuild=False,
    )


# ============================================================
# Utility functions
# ============================================================
def percent_value(value, default=0.0):
    """Convert either 0.9555 or 95.55 to a percentage."""
    try:
        x = float(value)
    except Exception:
        return float(default)
    if 0 <= x <= 1:
        return x * 100.0
    return x


def f1_percent(value):
    return percent_value(value)


def normalize_label(value, default="none"):
    """Normalize metadata labels so lot analysis never displays blanks."""
    if value is None:
        return default

    if isinstance(value, float) and np.isnan(value):
        return default

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]

    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]

    text = str(value).strip().strip("[]").strip("'\"")

    if not text or text.lower() in {"nan", "null", "none/normal"}:
        return "none"

    # Common serialized forms.
    text = text.replace('"', "").replace("'", "").strip()
    if text.lower() in {"none", "normal"}:
        return "none"

    return text


def normalize_metadata(meta):
    work = meta.copy()

    if "failureType" in work.columns:
        work["failureType"] = work["failureType"].apply(normalize_label)
    else:
        work["failureType"] = "none"

    if "lotName" in work.columns:
        work["lotName"] = work["lotName"].apply(lambda x: normalize_label(x, "Unknown Lot"))
    else:
        work["lotName"] = "Unknown Lot"

    return work


def calculate_yield(wafer):
    arr = np.asarray(wafer)
    normal = int(np.sum(arr == 1))
    defective = int(np.sum(arr == 2))
    total = normal + defective

    if total == 0:
        return {
            "total_dies": 0,
            "normal_dies": 0,
            "defective_dies": 0,
            "yield_pct": None,
            "defect_pct": None,
        }

    return {
        "total_dies": total,
        "normal_dies": normal,
        "defective_dies": defective,
        "yield_pct": round(normal / total * 100.0, 2),
        "defect_pct": round(defective / total * 100.0, 2),
    }


def severity_from_defect_pct(defect_pct):
    if defect_pct is None:
        return "Unavailable"
    if defect_pct < 2:
        return "Low"
    if defect_pct < 10:
        return "Moderate"
    if defect_pct < 25:
        return "High"
    return "Critical"


def stable_hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def grayscale_from_array(arr):
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[..., 0].astype(np.float32)

        rgb = arr[..., :3].astype(np.float32)
        spread = rgb.max(axis=2) - rgb.min(axis=2)

        # Reject ordinary colorful photographs.
        if np.percentile(spread, 95) > 35:
            raise ValueError(
                "This looks like a color photograph rather than a wafer map."
            )

        return rgb.mean(axis=2).astype(np.float32)

    raise ValueError("The input must be a 2-D wafer map or a grayscale image.")


def quantize_image_to_wafer(gray):
    """
    Convert common WM-811K-style grayscale renderings to 0/1/2.
    0 = background, 1 = normal die, 2 = defective die.
    """
    gray = np.nan_to_num(
        gray.astype(np.float32),
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    if gray.ndim != 2 or min(gray.shape) < 16:
        raise ValueError("The uploaded image is too small to be a wafer map.")

    rounded_unique = np.unique(np.rint(gray).astype(np.int32))

    # Already encoded as WM-811K values.
    if set(rounded_unique.tolist()).issubset({0, 1, 2}):
        return np.rint(gray).astype(np.int8), True

    values, counts = np.unique(gray, return_counts=True)

    # Dataset PNGs often contain exactly three grayscale levels.
    if len(values) <= 8:
        order = np.argsort(values)
        values = values[order]
        counts = counts[order]

        # Background is normally the most frequent level.
        bg_index = int(np.argmax(counts))
        bg_value = float(values[bg_index])

        other = [float(v) for v in values if float(v) != bg_value]

        if not other:
            raise ValueError("No wafer structure was found in the image.")

        out = np.zeros(gray.shape, dtype=np.int8)

        if len(other) == 1:
            # Two-tone map: model classification can still run, but
            # die-level yield/defect density cannot be trusted.
            out[gray != bg_value] = 1
            return out, False

        other = sorted(other)
        split = (other[0] + other[-1]) / 2.0

        out[(gray != bg_value) & (gray <= split)] = 1
        out[(gray != bg_value) & (gray > split)] = 2

        return out, True

    # Fallback for anti-aliased grayscale images.
    q1, q2 = np.percentile(gray.reshape(-1), [33, 66])
    out = np.zeros(gray.shape, dtype=np.int8)
    out[(gray > q1) & (gray <= q2)] = 1
    out[gray > q2] = 2

    border = np.concatenate(
        [
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ]
    )

    # If the border is predominantly dark, force it to background.
    if np.mean(border <= q1) >= 0.65:
        out[gray <= q1] = 0

    return out, True


def looks_like_wafer_map(wafer):
    """
    Lightweight structural validation.
    It is intentionally conservative so ordinary screenshots/photos
    are rejected before the model is called.
    """
    arr = np.asarray(wafer)

    if arr.ndim != 2:
        return False, "Input is not a 2-D wafer map."

    if min(arr.shape) < 16:
        return False, "Image is too small."

    occupied = arr > 0
    occupancy = float(occupied.mean())

    if occupancy < 0.015:
        return False, "The image contains almost no wafer/die information."

    if occupancy > 0.98:
        return False, "The image does not contain a recognizable wafer-map background."

    ys, xs = np.where(occupied)
    if len(xs) == 0:
        return False, "No wafer structure was detected."

    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)

    ratio = bbox_w / max(bbox_h, 1)

    if ratio < 0.45 or ratio > 2.2:
        return False, "The occupied region does not resemble a wafer map."

    border = np.concatenate(
        [
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1],
        ]
    )

    border_background = float(np.mean(border == 0))

    if border_background < 0.05:
        return False, "The image does not have enough wafer-map background."

    return True, ""


def parse_uploaded_file(uploaded_file):
    """Read one uploaded file and return a normalized wafer map."""
    raw_bytes = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".npy"):
        raw = np.load(uploaded_file, allow_pickle=False)
        gray_or_map = np.asarray(raw)
        has_defect_channel = True

        # Exact WM-811K matrix.
        unique = np.unique(gray_or_map)
        if gray_or_map.ndim == 2 and set(unique.tolist()).issubset({0, 1, 2}):
            wafer = gray_or_map.astype(np.int8)
        else:
            gray = grayscale_from_array(gray_or_map)
            wafer, has_defect_channel = quantize_image_to_wafer(gray)

        source_preview = gray_or_map
    else:
        image = Image.open(uploaded_file)
        source_preview = np.asarray(image.convert("L"))
        gray = grayscale_from_array(np.asarray(image))
        wafer, has_defect_channel = quantize_image_to_wafer(gray)

    valid, reason = looks_like_wafer_map(wafer)

    if not valid:
        return {
            "valid": False,
            "reason": reason,
            "wafer": None,
            "preview": source_preview,
            "has_defect_channel": has_defect_channel,
            "file_hash": stable_hash_bytes(raw_bytes),
        }

    return {
        "valid": True,
        "reason": "",
        "wafer": wafer,
        "preview": source_preview,
        "has_defect_channel": has_defect_channel,
        "file_hash": stable_hash_bytes(raw_bytes),
    }


def colored_wafer_figure(wafer):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    cmap = ListedColormap(["#070b12", "#dbe4ee", "#ef4444"])
    ax.imshow(wafer, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title("Inspection map: background / normal / defect")
    ax.axis("off")
    fig.tight_layout()
    return fig



def _extract_logits(model_output):
    """Extract classification logits from tensor/tuple/list/dict model outputs."""
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, dict):
        for key in ("logits", "output", "predictions"):
            value = model_output.get(key)
            if torch.is_tensor(value):
                return value
        for value in model_output.values():
            if torch.is_tensor(value) and value.ndim >= 2:
                return value
    if isinstance(model_output, (tuple, list)):
        for value in model_output:
            if torch.is_tensor(value) and value.ndim >= 2:
                return value
    raise RuntimeError("Could not extract classification logits from the model output.")


def generate_gradcam_overlay_local(wafer, model, target_idx):
    """Generate Grad-CAM directly from the locked ResNet.

    This intentionally does not use the older src.explainability.gradcam wrapper,
    because that wrapper expects a different model-output structure in this project.
    """
    wafer = np.asarray(wafer, dtype=np.int8)
    model.eval()

    # Find the last convolutional layer automatically.
    target_layer = None
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        raise RuntimeError("No Conv2d layer was found for Grad-CAM.")

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        if torch.is_tensor(output):
            activations["value"] = output
            output.retain_grad()

    def backward_hook(module, grad_input, grad_output):
        if grad_output and torch.is_tensor(grad_output[0]):
            gradients["value"] = grad_output[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        processed = preprocess_wafer_matrix(wafer)
        x = torch.from_numpy(np.asarray(processed, dtype=np.float32)).unsqueeze(0)
        x.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        output = model(x)
        logits = _extract_logits(output)
        if logits.ndim != 2:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        target_idx = int(target_idx)
        if target_idx < 0 or target_idx >= logits.shape[1]:
            raise RuntimeError("The predicted class index is outside the model output range.")

        score = logits[0, target_idx]
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        activation = activations["value"][0].detach().cpu().numpy()
        gradient = gradients["value"][0].detach().cpu().numpy()

        weights = gradient.mean(axis=(1, 2))
        cam = np.sum(weights[:, None, None] * activation, axis=0)
        cam = np.maximum(cam, 0)

        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        from PIL import Image
        from matplotlib import cm

        h, w = wafer.shape
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (w, h), Image.Resampling.BILINEAR
        )
        cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0

        # Original wafer map as grayscale RGB.
        base = wafer.astype(np.float32)
        base = np.clip(base / 2.0, 0.0, 1.0)
        orig_rgb = np.stack([base, base, base], axis=-1)

        # Standard scientific-style heatmap.
        heatmap_rgb = cm.jet(cam_resized)[..., :3]
        overlay_rgb = np.clip(0.55 * orig_rgb + 0.45 * heatmap_rgb, 0.0, 1.0)

        return {
            "orig_rgb": orig_rgb,
            "heatmap_rgb": heatmap_rgb,
            "overlay_rgb": overlay_rgb,
        }
    finally:
        h1.remove()
        h2.remove()
        model.zero_grad(set_to_none=True)


def set_page(page):
    st.session_state.page = page
    st.rerun()


def current_result():
    return st.session_state.get("prediction")


def require_current_wafer():
    wafer = st.session_state.get("wafer")
    result = st.session_state.get("prediction")

    if wafer is None or not result or not result.get("is_valid", False):
        st.warning(
            "No analyzed wafer is available in this session. Open Single Wafer Prediction, "
            "upload the wafer once, and wait until the result card shows a prediction."
        )
        if st.button("Open Single Wafer Prediction", key="go_upload"):
            set_page("3. Single Wafer Prediction")
        return None, None

    return wafer, result


def run_model_on_wafer(wafer):
    """Run the locked ResNet directly on the validated NumPy wafer map.

    The dashboard previously routed this through WaferPredictor.analyze(),
    which in the current project can return an internal Tensor/NumPy type
    mismatch for image uploads.  Here we keep the dashboard input as a NumPy
    0/1/2 wafer map, use the project's official preprocessing function, and
    call the already-loaded locked model directly.  This does not retrain or
    change the model.
    """
    predictor = get_predictor()

    arr = np.asarray(wafer, dtype=np.int8)
    valid, reason = looks_like_wafer_map(arr)
    if not valid:
        return {"is_valid": False, "message": reason}

    try:
        # Use the exact preprocessing used by the project model.
        model_input = preprocess_wafer_matrix(arr)
        tensor = torch.from_numpy(np.asarray(model_input, dtype=np.float32)).unsqueeze(0)

        model = predictor.model
        model.eval()

        with torch.no_grad():
            output = model(tensor)

        # Be tolerant if the architecture returns a tuple/dict instead of raw logits.
        if isinstance(output, dict):
            logits = output.get("logits", output.get("output"))
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        if logits is None:
            raise RuntimeError("The model did not return classification logits.")

        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)

        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = DEFECT_CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx] * 100.0)

        order = np.argsort(probabilities)[::-1][:3]
        top3 = [
            {
                "class": DEFECT_CLASSES[int(i)],
                "confidence": round(float(probabilities[int(i)] * 100.0), 2),
            }
            for i in order
        ]

        stats = calculate_yield(arr)
        result = {
            "is_valid": True,
            "predicted_class": predicted_class,
            "confidence_pct": round(confidence, 2),
            "top3": top3,
            "yield_stats": stats,
            "severity": severity_from_defect_pct(stats["defect_pct"]),
            "review_required": confidence < 70.0,
        }

        result["insight"] = INSIGHTS.get(
            predicted_class,
            "Review the prediction and compare it with historical wafers.",
        )
        result["recommendation"] = RECOMMENDATIONS.get(
            predicted_class,
            "Review the wafer manually and compare it with similar historical cases.",
        )

        return result

    except Exception as exc:
        return {
            "is_valid": False,
            "message": f"Model analysis failed: {exc}",
        }

def show_prediction_summary(result):
    if not result or not result.get("is_valid", False):
        st.error(
            result.get("message", "This input is not a valid wafer map.")
            if result
            else "No prediction is available."
        )
        return

    pred = result.get("predicted_class", "Unknown")
    conf = float(result.get("confidence_pct", 0.0))
    stats = result.get("yield_stats", {})
    severity = result.get("severity", "Unavailable")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Pattern", pred)
    c2.metric("Model Confidence", f"{conf:.2f}%")

    if stats.get("yield_pct") is None:
        c3.metric("Yield", "Unavailable")
    else:
        c3.metric("Yield", f"{stats['yield_pct']:.2f}%")

    c4.metric("Severity", severity)

    if result.get("review_required", False):
        st.warning(
            "Low-confidence result: human review is recommended before using "
            "the prediction for an engineering decision."
        )
    else:
        st.success("Prediction confidence is above the review threshold.")

    if result.get("top3"):
        top3 = pd.DataFrame(result["top3"])
        st.subheader("Top model predictions")
        st.dataframe(top3, width="stretch", hide_index=True)

    if result.get("insight"):
        st.info("AI insight: " + str(result["insight"]))

    if result.get("recommendation"):
        st.success("Next step: " + str(result["recommendation"]))


def build_lot_summary(meta):
    work = normalize_metadata(meta)

    rows = []

    for lot, group in work.groupby("lotName", dropna=False):
        defect_counts = group.loc[
            group["failureType"] != "none",
            "failureType",
        ].value_counts()

        dominant = (
            defect_counts.index[0]
            if not defect_counts.empty
            else "none"
        )

        defective_count = int((group["failureType"] != "none").sum())

        rows.append(
            {
                "lotName": str(lot),
                "wafer_count": int(len(group)),
                "dominant_defect": dominant,
                "defective_wafer_count": defective_count,
                "defective_wafer_pct": round(
                    defective_count / max(len(group), 1) * 100,
                    2,
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["defective_wafer_pct", "wafer_count"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def metadata_item(metadata_obj, i):
    """Read either list-style or dict-of-arrays index metadata."""
    if isinstance(metadata_obj, (list, tuple)):
        if i < len(metadata_obj) and isinstance(metadata_obj[i], dict):
            return metadata_obj[i]
        return {}

    if isinstance(metadata_obj, np.ndarray):
        if metadata_obj.ndim == 1 and i < len(metadata_obj):
            item = metadata_obj[i]
            return item if isinstance(item, dict) else {}
        return {}

    if isinstance(metadata_obj, dict):
        item = {}
        for key, values in metadata_obj.items():
            try:
                if isinstance(values, np.ndarray):
                    value = values[i] if values.ndim > 0 else values
                elif isinstance(values, (list, tuple)):
                    value = values[i] if i < len(values) else None
                else:
                    value = values

                if isinstance(value, np.generic):
                    value = value.item()

                item[key] = value
            except (IndexError, TypeError, KeyError):
                continue
        return item

    return {}


def calculate_anomaly_scores(index_data):
    """
    Calculate anomaly scores using the existing 1,749-wafer embedding index.

    Metadata format in indexed_embeddings.npz:
        {
            "records": [
                {
                    "original_idx": ...,
                    "failureType": ...,
                    "lotName": ...,
                    "waferIndex": ...,
                    "yield_pct": ...,
                    "defect_density": ...,
                    "total_dies": ...
                },
                ...
            ]
        }
    """

    embeddings = index_data.get("embeddings")

    if embeddings is None:
        return pd.DataFrame()

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or len(embeddings) == 0:
        return pd.DataFrame()

    # ---------------------------------------------------------
    # Calculate centroid of indexed embeddings
    # ---------------------------------------------------------
    centroid = embeddings.mean(axis=0)

    centroid_norm = np.linalg.norm(centroid)

    if centroid_norm < 1e-8:
        return pd.DataFrame()

    centroid = centroid / centroid_norm

    # ---------------------------------------------------------
    # Anomaly score
    # Higher score = more unusual
    # ---------------------------------------------------------
    scores = 1.0 - np.dot(embeddings, centroid)

    # ---------------------------------------------------------
    # Read metadata
    # ---------------------------------------------------------
    metadata = index_data.get("metadata", {})

    # Handle numpy 0-dimensional object array
    if isinstance(metadata, np.ndarray):
        try:
            metadata = metadata.item()
        except Exception:
            metadata = {}

    # Your actual format:
    # metadata = {"records": [...]}
    if isinstance(metadata, dict):
        records = metadata.get("records", [])
    else:
        records = []

    rows = []

    # ---------------------------------------------------------
    # Build anomaly table
    # ---------------------------------------------------------
    for i, score in enumerate(scores):

        if i < len(records) and isinstance(records[i], dict):
            item = records[i]
        else:
            item = {}

        rows.append(
            {
                "original_idx": item.get("original_idx", i),

                "class": item.get(
                    "failureType",
                    "unknown"
                ),

                "lotName": item.get(
                    "lotName",
                    "unknown"
                ),

                "waferIndex": item.get(
                    "waferIndex",
                    "unknown"
                ),

                "yield_pct": item.get(
                    "yield_pct",
                    np.nan
                ),

                "defect_density": item.get(
                    "defect_density",
                    np.nan
                ),

                "total_dies": item.get(
                    "total_dies",
                    np.nan
                ),

                "anomaly_score": round(
                    float(score),
                    4
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    anomaly_df = pd.DataFrame(rows)

    # Highest anomaly first
    anomaly_df = anomaly_df.sort_values(
        "anomaly_score",
        ascending=False
    ).reset_index(drop=True)

    return anomaly_df


def show_navigation_grid():
    st.markdown("### Explore the project")
    st.caption(
        "Eight modules cover the complete workflow. Single Wafer Prediction is the central result page; its yield, severity, insight and recommendation are not duplicated elsewhere."
    )

    for start_idx in range(0, len(NAVIGATION), 3):
        cols = st.columns(3, gap="medium")
        for local_idx, (col, page_name) in enumerate(
            zip(cols, NAVIGATION[start_idx:start_idx + 3])
        ):
            module_number = start_idx + local_idx + 1
            with col:
                _, title = page_name.split(". ", 1)
                st.markdown(
                    f"<div class='nav-module-number'>MODULE {module_number}</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    title,
                    key=f"nav_{page_name}",
                    width="stretch",
                    type="secondary",
                ):
                    set_page(page_name)
                st.caption(NAV_DESCRIPTIONS.get(page_name, "Open module"))
        st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)


def show_current_wafer_status():
    if st.session_state.get("wafer") is not None and st.session_state.get("prediction"):
        result = st.session_state.prediction
        if result.get("is_valid"):
            st.markdown(
                f"<div class='status-strip'>Current wafer: <b>{result.get('predicted_class', 'Unknown')}</b> "
                f"· confidence <b>{float(result.get('confidence_pct', 0)):.2f}%</b> · "
                f"ID <b>UPLOAD-{st.session_state.upload_hash[:10]}</b></div>",
                unsafe_allow_html=True,
            )


def show_back_button():
    """Return to the connected inspection workflow by default.

    The dashboard remains an explicit second choice so users do not lose their
    current wafer analysis when moving between modules.
    """
    c1, c2 = st.columns([1.25, 1.0])
    with c1:
        if st.button("← Back to Single Wafer Prediction", key=f"back_single_{st.session_state.page}", width="content"):
            set_page("3. Single Wafer Prediction")
    with c2:
        if st.button("⌂ Project Dashboard", key=f"back_dashboard_{st.session_state.page}", width="content"):
            set_page("1. Main Dashboard")


# ============================================================
# Header
# ============================================================
# ============================================================
# Main dashboard
# ============================================================
page = st.session_state.page

# Migrate stale sessions from the previous separate About/Future Scope page.
if page == "13. About / Future Scope":
    page = "1. Main Dashboard"
    st.session_state.page = page

# Older dashboard versions had separate Yield/Severity, AI Insights and
# Recommendations pages. They are now consolidated into Single Wafer Prediction.
if page in {"4. Yield & Severity", "10. AI Insights", "11. Recommendations"}:
    page = "3. Single Wafer Prediction"
    st.session_state.page = page

if page != "1. Main Dashboard":
    show_current_wafer_status()

# ============================================================
# Human review edit/delete helpers
# ============================================================
def _read_review_store():
    """Read the persistent human-review JSON without changing its schema."""
    path = Path(HUMAN_REVIEWS_PATH)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Support either {"reviews": [...]} or a single review object.
            if isinstance(data.get("reviews"), list):
                return data["reviews"]
            return [data]
    except Exception:
        return []
    return []


def _write_review_store(reviews):
    path = Path(HUMAN_REVIEWS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)


def _review_id(review):
    return str(review.get("review_id", review.get("id", "")))


def _review_value(review, *keys, default=""):
    for key in keys:
        if key in review and review[key] is not None:
            return review[key]
    return default


if page == "1. Main Dashboard":
    st.markdown(
        "<div class='project-title'>AI-Based Wafer Defect Classification Using Deep Learning</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='project-subtitle'>AI-powered wafer-map classification, yield estimation, explainability, similarity search, lot intelligence and engineering decision support using WM-811K.</div>",
        unsafe_allow_html=True,
    )

    official = get_official_results()
    accuracy = percent_value(official.get("accuracy", 95.55))
    balanced = percent_value(official.get("balanced_accuracy", 73.39))
    error_rate = percent_value(official.get("error_rate", 4.45))
    macro_f1 = f1_percent(official.get("macro_f1", 0.7049))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("WM-811K Wafers", f"{DATASET_STATS['total']:,}")
    c2.metric("Labeled Wafers", f"{DATASET_STATS['labeled']:,}")
    c3.metric("Official Test Accuracy", f"{accuracy:.2f}%")
    c4.metric("Balanced Accuracy", f"{balanced:.2f}%")
    c5.metric("Macro F1", f"{macro_f1:.2f}%")

    st.caption(
        f"Official test set: {int(official.get('test_samples', 118595)):,} samples. "
        f"Reported error rate: {error_rate:.2f}%. "
        "Overall accuracy is influenced by class imbalance, so balanced accuracy "
        "and macro-F1 are also shown."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("WM-811K labeled class distribution")
        class_df = (
            pd.DataFrame(
                [
                    {"Class": k, "Wafers": v}
                    for k, v in DATASET_STATS["classes"].items()
                ]
            )
            .sort_values("Wafers", ascending=False)
            .set_index("Class")
        )
        st.bar_chart(class_df, height=350)

    with right:
        st.subheader("Locked model evaluation")
        metrics_df = pd.DataFrame(
            {
                "Metric": [
                    "Original baseline",
                    "Official test accuracy",
                    "Balanced accuracy",
                    "Macro F1",
                    "Weighted F1",
                ],
                "Percentage": [
                    float(BASELINE_ACCURACY),
                    accuracy,
                    balanced,
                    macro_f1,
                    f1_percent(official.get("weighted_f1", 0.9574)),
                ],
            }
        ).set_index("Metric")
        st.bar_chart(metrics_df, height=350)

    st.info(
        "The locked controlled ResNet achieved 95.55% official test accuracy "
        "and 73.39% balanced accuracy on 118,595 official test samples. "
        "The model is not retrained by the dashboard."
    )

    if st.session_state.wafer is not None:
        st.success(
            "Current wafer is loaded. Open **Single Wafer Prediction** to view "
            "the current result or use the other modules."
        )
    else:
        st.warning(
            "Start with **Single Wafer Prediction**. You upload the wafer only once; "
            "the rest of the dashboard reuses that same analysis."
        )

    st.divider()
    show_navigation_grid()

    st.divider()
    st.markdown("## About the project")
    st.caption("Project documentation and future extensions — kept on the main dashboard, not as a separate module.")
    st.write(
        "**AI-Based Wafer Defect Classification Using Deep Learning** is an ECE/VLSI-oriented "
        "AI project that uses the WM-811K wafer-map dataset and a controlled ResNet model "
        "to classify known wafer defect patterns."
    )

    about_left, about_right = st.columns(2, gap="large")
    with about_left:
        st.markdown("**Completed capabilities**")
        st.markdown(
            "- Wafer-map validation and classification\n"
            "- Yield and project-defined severity analysis\n"
            "- Grad-CAM visual explanation\n"
            "- Similar-wafer search\n"
            "- Lot/batch intelligence\n"
            "- Anomaly screening\n"
            "- Human-in-the-loop review\n"
            "- Prediction-specific insights and recommendations"
        )

    with about_right:
        st.markdown("**Future scope**")
        st.markdown(
            "- Unknown-defect discovery with stronger open-set methods\n"
            "- Process/tool correlation using additional manufacturing data\n"
            "- Outcome-based recovery/cure prediction when intervention data is available\n"
            "- API and production deployment\n"
            "- Continuous learning from verified engineer feedback"
        )

    st.caption(
        "Note: WM-811K does not contain treatment/intervention outcomes, so recovery/cure probability "
        "is future scope rather than a current model output."
    )


# ============================================================
# Dataset analysis
# ============================================================
elif page == "2. Dataset Analysis":
    show_back_button()
    st.title("WM-811K Dataset Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total wafers", f"{DATASET_STATS['total']:,}")
    c2.metric("Labeled", f"{DATASET_STATS['labeled']:,}")
    c3.metric("Unlabeled", f"{DATASET_STATS['unlabeled']:,}")
    c4.metric("Unique lots", f"{DATASET_STATS['lots']:,}")

    st.subheader("What the dataset contains")
    st.write(
        "WM-811K contains wafer maps derived from die-level semiconductor "
        "test results. The maps are not microscope or SEM photographs."
    )

    class_df = pd.DataFrame(
        [
            {
                "Defect class": cls,
                "Count": count,
                "Share of labeled": round(count / DATASET_STATS["labeled"] * 100, 3),
            }
            for cls, count in DATASET_STATS["classes"].items()
        ]
    ).sort_values("Count", ascending=False)

    st.dataframe(class_df, width="stretch", hide_index=True)

    st.subheader("Map encoding")
    st.markdown(
        """
        - **0** → background / no die
        - **1** → normal/pass die
        - **2** → defective/fail die
        """
    )

    st.info(
        "The dataset is highly imbalanced: the normal class is much larger "
        "than the minority defect classes. This is why the project reports "
        "balanced accuracy and macro-F1 in addition to overall accuracy."
    )


# ============================================================
# Single wafer prediction
# ============================================================
elif page == "3. Single Wafer Prediction":
    show_back_button()
    st.title("Single Wafer Prediction & Inspection")
    st.caption(
        "Upload one wafer image or .npy wafer map. The application automatically "
        "validates whether the input looks like a wafer map, converts it to the "
        "model format, classifies it and stores the result for the other pages."
    )

    uploaded = st.file_uploader(
        "Upload wafer image",
        type=["png", "jpg", "jpeg", "npy"],
        key="main_wafer_upload",
        help="For best results, upload a WM-811K wafer-map image or the original 0/1/2 .npy matrix.",
    )

    if uploaded is not None:
        file_hash = stable_hash_bytes(uploaded.getvalue())

        if file_hash != st.session_state.upload_hash:
            try:
                parsed = parse_uploaded_file(uploaded)

                st.session_state.upload_hash = file_hash
                st.session_state.wafer = parsed["wafer"]
                st.session_state.wafer_source = uploaded.name
                st.session_state.upload_preview = parsed["preview"]
                st.session_state.upload_has_defect_channel = parsed["has_defect_channel"]
                st.session_state.prediction = None
                st.session_state.analysis_message = parsed["reason"]

                if not parsed["valid"]:
                    st.session_state.wafer = None
            except Exception as exc:
                st.session_state.wafer = None
                st.session_state.prediction = None
                st.session_state.analysis_message = str(exc)

    if st.session_state.wafer is None:
        if st.session_state.analysis_message:
            st.error(
                "Unknown / unsupported input: "
                + str(st.session_state.analysis_message)
            )

        st.info(
            "The app accepts a wafer map only. A normal photograph, screenshot "
            "or unrelated black/white image is treated as an unknown input."
        )

        st.markdown("### What happens after upload?")
        st.markdown(
            "Upload → validate wafer structure → convert to 0/1/2 → classify defect → "
            "calculate yield when the image preserves die classes → show severity → "
            "generate input-specific insight and recommendation."
        )

    else:
        wafer = st.session_state.wafer

        st.subheader("Uploaded wafer")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            preview = st.session_state.upload_preview
            if preview is not None:
                preview_arr = np.asarray(preview)
                if preview_arr.ndim == 2:
                    lo, hi = np.percentile(preview_arr.astype(np.float32), [1, 99])
                    if hi > lo:
                        preview_display = np.clip(
                            (preview_arr.astype(np.float32) - lo) / (hi - lo), 0, 1
                        )
                    else:
                        preview_display = preview_arr
                else:
                    preview_display = preview_arr

                st.image(
                    preview_display,
                    caption=f"Uploaded input: {st.session_state.wafer_source}",
                    width="stretch",
                    clamp=True,
                )

        with c2:
            fig = colored_wafer_figure(wafer)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            st.caption(
                "Inspection view: dark = background, light = normal die, "
                "red = defective die."
            )

        if not st.session_state.upload_has_defect_channel:
            st.warning(
                "This upload appears to contain only two intensity levels. "
                "Classification can still run, but die-level yield/defect percentage "
                "is not treated as reliable unless the image preserves normal and "
                "defective die levels."
            )

        if st.session_state.prediction is None:
            with st.spinner("Analyzing wafer..."):
                try:
                    st.session_state.prediction = run_model_on_wafer(wafer)
                except Exception as exc:
                    st.session_state.prediction = {
                        "is_valid": False,
                        "message": f"Model analysis failed: {exc}",
                    }

        result = st.session_state.prediction

        if not result.get("is_valid", False):
            st.error(
                "Unknown / unsupported wafer input: "
                + result.get("message", "The uploaded input could not be analyzed."),
            )
            st.caption(
                "Try a WM-811K wafer-map image or the original 0/1/2 .npy wafer matrix. "
                "The model does not accept ordinary photographs or arbitrary screenshots."
            )
        else:
            st.success("Wafer validated and analyzed successfully.")
            st.caption(
                f"Input ID: UPLOAD-{st.session_state.upload_hash[:10]}  |  "
                f"Source: {st.session_state.wafer_source}"
            )
            show_prediction_summary(result)

            st.subheader("Die-level statistics")
            stats = result.get("yield_stats", {})
            d1, d2, d3 = st.columns(3)
            d1.metric("Total dies", stats.get("total_dies", 0))
            d2.metric("Normal dies", stats.get("normal_dies", 0))
            d3.metric("Defective dies", stats.get("defective_dies", 0))

            st.caption(
                "For an uploaded image, this is an application-generated input ID. "
                "It is not a WM-811K waferIndex. In WM-811K, waferIndex identifies "
                "a wafer's position within its lot."
            )

            st.divider()
            st.markdown("### Detailed analysis")
            st.caption(
                "The summary above already contains yield, severity, AI insight and recommendation. "
                "Use the modules below only when you need deeper analysis."
            )
            n1, n2, n3, n4 = st.columns(4)
            if n1.button("Grad-CAM", width="stretch"):
                set_page("5. Grad-CAM Explainability")
            if n2.button("Similar Wafers", width="stretch"):
                set_page("6. Similar Wafer Search")
            if n3.button("Anomaly Detection", width="stretch"):
                set_page("8. Anomaly & Unknown Detection")
            if n4.button("Human Review", width="stretch"):
                set_page("9. Human-in-the-Loop Review")


# ============================================================
# Yield and severity
# ============================================================
elif page == "4. Yield & Severity":
    show_back_button()
    st.title("Yield & Severity Analysis")
    st.caption(
        "This page uses the same wafer uploaded in Single Wafer Prediction. "
        "It does not ask you to select another hidden wafer."
    )

    wafer, result = require_current_wafer()

    if wafer is not None:
        st.success(
            f"Analyzed wafer loaded: UPLOAD-{st.session_state.upload_hash[:10]} · "
            f"prediction: {result.get('predicted_class', 'Unknown')}"
        )
        stats = result.get("yield_stats", {})
        pred = result.get("predicted_class", "Unknown")

        c1, c2, c3, c4 = st.columns(4)

        if stats.get("yield_pct") is None:
            c1.metric("Yield", "Unavailable")
            c2.metric("Defect %", "Unavailable")
        else:
            c1.metric("Yield", f"{stats['yield_pct']:.2f}%")
            c2.metric("Defect %", f"{stats['defect_pct']:.2f}%")

        c3.metric("Predicted defect", pred)
        c4.metric("Severity", result.get("severity", "Unavailable"))

        st.subheader("What these numbers mean")
        st.markdown(
            f"""
            - **Yield %** = normal dies ÷ (normal dies + defective dies) × 100
            - **Defect %** = defective dies ÷ (normal dies + defective dies) × 100
            - **Severity** is a project-defined indicator derived from defect percentage.
            - **Prediction** is the model's wafer-map class, not a severity grade.
            """
        )

        st.subheader("Severity bands used by this project")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Defect %": "< 2%", "Severity": "Low"},
                    {"Defect %": "2% – <10%", "Severity": "Moderate"},
                    {"Defect %": "10% – <25%", "Severity": "High"},
                    {"Defect %": "≥ 25%", "Severity": "Critical"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        st.info(
            "Severity is a project-defined screening indicator. It is not an "
            "industry-standard severity certification."
        )


# ============================================================
# Grad-CAM
# ============================================================
elif page == "5. Grad-CAM Explainability":
    show_back_button()
    st.title("Grad-CAM Explainability")
    st.caption(
        "Grad-CAM highlights the image regions that contributed most strongly "
        "to the model's selected class. It is an explanation aid, not a proof "
        "of physical root cause."
    )

    wafer, result = require_current_wafer()

    if wafer is not None:
        try:
            predicted_class = result.get("predicted_class", "none")
            target_idx = DEFECT_CLASSES.index(predicted_class) if predicted_class in DEFECT_CLASSES else 0
            gradcam = generate_gradcam_overlay_local(
                wafer,
                get_predictor().model,
                target_idx,
            )

            st.subheader(
                f"Why the model predicted: {result.get('predicted_class', 'Unknown')}"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.image(
                    gradcam["orig_rgb"],
                    caption="1. Original wafer map",
                    width="stretch",
                )

            with c2:
                st.image(
                    gradcam["heatmap_rgb"],
                    caption="2. Colored Grad-CAM heatmap",
                    width="stretch",
                )

            with c3:
                st.image(
                    gradcam["overlay_rgb"],
                    caption="3. Grad-CAM overlay",
                    width="stretch",
                )

            st.success(
                "The colored regions show where the network focused when producing "
                "the selected prediction."
            )

        except Exception as exc:
            st.error(f"Grad-CAM could not be generated: {exc}")


# ============================================================
# Similar wafer search
# ============================================================
elif page == "6. Similar Wafer Search":
    show_back_button()
    st.title("Similar Wafer Search")
    st.caption(
        "The uploaded wafer is converted to the model's 128-D embedding and "
        "compared with the existing representative WM-811K index."
    )

    wafer, result = require_current_wafer()

    if wafer is not None:
        try:
            with st.spinner("Searching the cached historical wafer index..."):
                matches = search_similar_wafers(
                    wafer,
                    get_predictor().model,
                    get_embedding_index(),
                    top_k=10,
                )

            if matches:
                st.dataframe(
                    pd.DataFrame(matches),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No similar historical wafers were returned.")

            st.caption(
                "The current index contains 1,749 representative labeled wafers "
                "(up to 200 per class). Similarity is embedding-based, not a "
                "causal or manufacturing-root-cause match."
            )

        except Exception as exc:
            st.error(f"Similarity search failed: {exc}")


# ============================================================
# Lot / batch analysis
# ============================================================
elif page == "7. Lot / Batch Analysis":
    show_back_button()
    st.title("Lot / Batch Analysis")
    st.caption(
        "This page loads WM-811K metadata only when requested. It does not load "
        "the 2 GB raw pickle just to navigate the dashboard."
    )

    try:
        meta = normalize_metadata(get_metadata())
        lot_summary = build_lot_summary(meta)

        if lot_summary.empty:
            st.warning("No usable lot metadata was found.")
        else:
            st.subheader("Problematic-lot screening")
            st.dataframe(
                lot_summary.head(25),
                width="stretch",
                hide_index=True,
            )

            selected_lot = st.selectbox(
                "Select a lot",
                lot_summary["lotName"].astype(str).tolist(),
            )

            group = meta[meta["lotName"].astype(str) == str(selected_lot)].copy()

            defect_counts = (
                group["failureType"]
                .value_counts()
                .rename_axis("Defect")
                .reset_index(name="Count")
            )

            defective_count = int((group["failureType"] != "none").sum())
            defective_pct = defective_count / max(len(group), 1) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Wafers in lot", len(group))
            c2.metric("Defective wafers", defective_count)
            c3.metric("Defective wafer %", f"{defective_pct:.2f}%")

            st.subheader("Defect distribution")
            st.dataframe(
                defect_counts,
                width="stretch",
                hide_index=True,
            )

            defect_only = defect_counts[defect_counts["Defect"] != "none"]
            if not defect_only.empty:
                st.bar_chart(
                    defect_only.set_index("Defect")["Count"],
                    height=300,
                )
            else:
                st.info("This selected lot contains no labeled defect classes.")

            dominant = (
                defect_only.iloc[0]["Defect"]
                if not defect_only.empty
                else "none"
            )
            st.info(
                f"Dominant labeled defect in this lot: **{dominant}**. "
                "Dominant defect is calculated from defect classes only; the "
                "normal class is not allowed to hide the defect distribution."
            )

            st.warning(
                "Lot-level statistics identify patterns in the dataset. They do "
                "not establish a confirmed manufacturing root cause."
            )

    except Exception as exc:
        st.error(f"Lot analysis could not be loaded: {exc}")


# ============================================================
# Anomaly / unknown detection
# ============================================================
elif page == "8. Anomaly & Unknown Detection":
    show_back_button()
    st.title("Anomaly & Unknown Pattern Detection")
    st.caption(
        "This module uses embedding distance as a project-level unusualness "
        "indicator. It is not a certified unknown-defect detector."
    )

    try:
        with st.spinner("Loading the cached embedding index..."):
            anomaly_df = calculate_anomaly_scores(get_embedding_index())

        if anomaly_df.empty:
            st.warning("No embedding index is available.")
        else:
            st.subheader("Most unusual indexed wafers")
            st.dataframe(
                anomaly_df.head(20),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Anomaly scores are calculated on the representative indexed "
                "subset, not all 811,457 WM-811K wafers."
            )
    except Exception as exc:
        st.error(f"Anomaly analysis could not be loaded: {exc}")

    st.divider()
    st.subheader("Current uploaded wafer")
    wafer, result = require_current_wafer()

    if wafer is not None:
        confidence = float(result.get("confidence_pct", 0.0))

        if result.get("review_required", False):
            st.warning(
                f"The current wafer is a known-class prediction with low confidence "
                f"({confidence:.2f}%). Treat it as uncertain and send it to human review."
            )
        else:
            st.success(
                f"The current wafer was classified as {result.get('predicted_class', 'Unknown')} "
                f"with {confidence:.2f}% confidence."
            )


# ============================================================
# Human in the loop
# ============================================================
elif page == "9. Human-in-the-Loop Review":
    show_back_button()
    st.title("Human-in-the-Loop Review")
    st.caption(
        "Validate the current prediction, add engineering notes, and manage previously saved reviews. "
        "Editing or deleting a review changes the audit record only; it does not retrain the locked model."
    )

    wafer, result = require_current_wafer()

    if wafer is not None:
        predicted = result.get("predicted_class", "Unknown")
        confidence = float(result.get("confidence_pct", 0.0))

        c1, c2, c3 = st.columns(3)
        c1.metric("Current prediction", predicted)
        c2.metric("Confidence", f"{confidence:.2f}%")
        c3.metric("Review status", "Required" if result.get("review_required") else "Optional")

        decision = st.selectbox(
            "Reviewer decision",
            ["Agree", "Override", "Flagged for Inspection"],
            key="new_review_decision",
        )

        corrected = predicted
        if decision == "Override":
            corrected = st.selectbox(
                "Corrected defect class",
                DEFECT_CLASSES,
                key="new_review_corrected",
            )

        notes = st.text_area(
            "Engineering notes",
            placeholder="Describe what the engineer observed, checked or wants inspected.",
            height=140,
            key="new_review_notes",
        )

        if st.button("Submit Human Review", type="primary", key="submit_human_review"):
            if not notes.strip():
                st.warning("Please add engineering notes before submitting the review.")
            else:
                try:
                    wafer_id = f"UPLOAD-{st.session_state.upload_hash[:10]}"
                    saved = save_human_review(
                        wafer_id,
                        predicted,
                        confidence,
                        decision,
                        corrected,
                        notes.strip(),
                    )
                    st.success(
                        f"Review saved successfully. Review ID: "
                        f"{saved.get('review_id', 'created')}"
                    )
                    st.info(
                        "The review is stored as human feedback for future audit/model-improvement workflows. "
                        "The current locked model is not retrained automatically."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save the review: {exc}")

        # --------------------------------------------------------
        # Saved review history + edit/delete controls
        # --------------------------------------------------------
        reviews = load_human_reviews()
        if not isinstance(reviews, list):
            reviews = _read_review_store()

        st.divider()
        st.subheader("Saved review history")

        if reviews:
            display_rows = []
            for review in reviews:
                display_rows.append(
                    {
                        "Review ID": _review_id(review),
                        "Wafer ID": _review_value(review, "wafer_id", "waferId"),
                        "Prediction": _review_value(review, "predicted_class", "prediction"),
                        "Decision": _review_value(review, "decision"),
                        "Corrected class": _review_value(review, "corrected_class", "corrected"),
                        "Notes": _review_value(review, "engineering_notes", "notes"),
                        "Created": _review_value(review, "created_at", "timestamp"),
                    }
                )

            st.dataframe(
                pd.DataFrame(display_rows),
                width="stretch",
                hide_index=True,
            )

            valid_reviews = [r for r in reviews if _review_id(r)]
            review_options = [_review_id(r) for r in valid_reviews]

            if review_options:
                selected_review_id = st.selectbox(
                    "Select a saved review to edit or delete",
                    review_options,
                    key="selected_review_id",
                )
                selected = next(
                    r for r in valid_reviews if _review_id(r) == selected_review_id
                )

                current_decision = _review_value(
                    selected, "decision", default="Agree"
                )
                if current_decision not in [
                    "Agree",
                    "Override",
                    "Flagged for Inspection",
                ]:
                    current_decision = "Agree"

                st.markdown("#### Edit selected review")
                edit_decision = st.selectbox(
                    "Reviewer decision",
                    ["Agree", "Override", "Flagged for Inspection"],
                    index=["Agree", "Override", "Flagged for Inspection"].index(current_decision),
                    key=f"edit_decision_{selected_review_id}",
                )

                current_corrected = _review_value(
                    selected,
                    "corrected_class",
                    "corrected",
                    default=_review_value(selected, "predicted_class", "prediction", default="Unknown"),
                )
                if current_corrected not in DEFECT_CLASSES:
                    current_corrected = DEFECT_CLASSES[0]

                edit_corrected = current_corrected
                if edit_decision == "Override":
                    edit_corrected = st.selectbox(
                        "Corrected defect class",
                        DEFECT_CLASSES,
                        index=DEFECT_CLASSES.index(current_corrected),
                        key=f"edit_corrected_{selected_review_id}",
                    )

                current_notes = _review_value(
                    selected,
                    "engineering_notes",
                    "notes",
                    default="",
                )
                edit_notes = st.text_area(
                    "Engineering notes",
                    value=str(current_notes),
                    height=150,
                    key=f"edit_notes_{selected_review_id}",
                )

                ec1, ec2 = st.columns(2)
                with ec1:
                    if st.button("Save Changes", type="primary", key=f"save_edit_{selected_review_id}"):
                        if not edit_notes.strip():
                            st.warning("Engineering notes cannot be empty.")
                        else:
                            updated = False
                            for review in reviews:
                                if _review_id(review) == selected_review_id:
                                    review["decision"] = edit_decision
                                    if "corrected_class" in review or "corrected" not in review:
                                        review["corrected_class"] = edit_corrected
                                    else:
                                        review["corrected"] = edit_corrected
                                    if "engineering_notes" in review or "notes" not in review:
                                        review["engineering_notes"] = edit_notes.strip()
                                    else:
                                        review["notes"] = edit_notes.strip()
                                    review["updated_at"] = pd.Timestamp.now().isoformat()
                                    updated = True
                                    break
                            if updated:
                                try:
                                    _write_review_store(reviews)
                                    st.success("Review updated successfully.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Could not update the review: {exc}")

                with ec2:
                    if st.button("Delete Review", key=f"delete_review_{selected_review_id}"):
                        st.session_state[f"confirm_delete_{selected_review_id}"] = True

                if st.session_state.get(f"confirm_delete_{selected_review_id}", False):
                    st.warning(
                        "This permanently deletes the selected review record. "
                        "It does not delete the uploaded wafer or model."
                    )
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        if st.button("Confirm Delete", type="primary", key=f"confirm_delete_btn_{selected_review_id}"):
                            remaining = [
                                r for r in reviews if _review_id(r) != selected_review_id
                            ]
                            try:
                                _write_review_store(remaining)
                                st.session_state.pop(f"confirm_delete_{selected_review_id}", None)
                                st.success("Review deleted successfully.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not delete the review: {exc}")
                    with dc2:
                        if st.button("Cancel", key=f"cancel_delete_{selected_review_id}"):
                            st.session_state.pop(f"confirm_delete_{selected_review_id}", None)
                            st.rerun()
        else:
            st.info("No human reviews have been saved yet.")


# ============================================================
# AI insights
# ============================================================
elif page == "10. AI Insights":
    show_back_button()
    st.title("AI Insights")

    wafer, result = require_current_wafer()

    if wafer is not None:
        pred = result.get("predicted_class", "Unknown")
        conf = float(result.get("confidence_pct", 0.0))
        stats = result.get("yield_stats", {})

        st.subheader("Current wafer interpretation")

        st.info(
            INSIGHTS.get(
                pred,
                "No class-specific insight is available for this prediction.",
            )
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted pattern", pred)
        c2.metric("Confidence", f"{conf:.2f}%")

        if stats.get("defect_pct") is None:
            c3.metric("Defect %", "Unavailable")
        else:
            c3.metric("Defect %", f"{stats['defect_pct']:.2f}%")

        if result.get("review_required"):
            st.warning(
                "Because confidence is low, this insight should be treated as "
                "a hypothesis for review, not a final engineering conclusion."
            )

        st.caption(
            "These insights are deterministic project logic based on the model "
            "class, confidence and wafer statistics. They are not causal diagnosis."
        )


# ============================================================
# Recommendations
# ============================================================
elif page == "11. Recommendations":
    show_back_button()
    st.title("Recommendations")
    st.caption(
        "Recommendations are tied to the currently uploaded wafer's predicted "
        "pattern. They are investigation steps, not confirmed root causes."
    )

    wafer, result = require_current_wafer()

    if wafer is not None:
        pred = result.get("predicted_class", "Unknown")
        conf = float(result.get("confidence_pct", 0.0))

        st.subheader(f"Recommended next steps for: {pred}")
        st.success(
            RECOMMENDATIONS.get(
                pred,
                "Review the wafer manually and compare it with similar historical cases.",
            )
        )

        st.markdown("### Suggested workflow")
        st.markdown(
            """
            1. Confirm the model prediction and confidence.
            2. Review the Grad-CAM regions.
            3. Compare similar historical wafers.
            4. Check lot-level recurrence.
            5. Review relevant process/tool/handling records.
            6. Record the engineer's final decision in Human-in-the-Loop Review.
            """
        )

        if conf < 70:
            st.warning(
                "Confidence is below 70%, so human review should be prioritized."
            )


# ============================================================
# Model performance
# ============================================================
elif page == "12. Model Performance":
    show_back_button()
    st.title("Official Model Performance")

    official = get_official_results()

    accuracy = percent_value(official.get("accuracy", 95.55))
    error_rate = percent_value(official.get("error_rate", 4.45))
    balanced = percent_value(official.get("balanced_accuracy", 73.39))
    macro_f1 = f1_percent(official.get("macro_f1", 0.7049))
    weighted_f1 = f1_percent(official.get("weighted_f1", 0.9574))
    test_samples = int(official.get("test_samples", 118595))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test samples", f"{test_samples:,}")
    c2.metric("Accuracy", f"{accuracy:.2f}%")
    c3.metric("Error rate", f"{error_rate:.2f}%")
    c4.metric("Balanced accuracy", f"{balanced:.2f}%")
    c5.metric("Macro F1", f"{macro_f1:.2f}%")

    st.metric("Weighted F1", f"{weighted_f1:.2f}%")

    st.divider()

    result_txt = OUTPUTS_DIR / "controlled_official_test_results.txt"
    if result_txt.exists():
        with open(result_txt, "r", encoding="utf-8") as f:
            st.subheader("Classification report")
            st.code(f.read())

    # Prefer the locked controlled confusion matrix.
    cm_path = OUTPUTS_DIR / "controlled_confusion_matrix.npy"
    if cm_path.exists():
        try:
            cm = np.load(cm_path)

            if cm.ndim == 2 and cm.shape[0] == len(DEFECT_CLASSES):
                fig, ax = plt.subplots(figsize=(8.5, 7))
                im = ax.imshow(cm, aspect="auto", cmap="Blues")
                ax.set_xticks(range(len(DEFECT_CLASSES)))
                ax.set_yticks(range(len(DEFECT_CLASSES)))
                ax.set_xticklabels(DEFECT_CLASSES, rotation=45, ha="right")
                ax.set_yticklabels(DEFECT_CLASSES)
                ax.set_xlabel("Predicted class")
                ax.set_ylabel("True class")
                ax.set_title("Controlled model — official test confusion matrix")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
        except Exception as exc:
            st.warning(f"Could not display the controlled confusion matrix: {exc}")

    st.info(
        "The official test partition was kept separate from model development. "
        "The 95.55% overall accuracy should not be interpreted as 95.55% performance "
        "for every defect class because WM-811K is highly imbalanced. Balanced accuracy "
        "and macro-F1 provide additional class-level context."
    )


# ============================================================
# About / future scope
# ============================================================
elif page == "13. About / Future Scope":
    show_back_button()
    st.title("About Project & Future Scope")

    st.subheader("Project title")
    st.markdown(
        "**AI-Based Semiconductor Wafer Defect Inspection, Severity Analysis "
        "and Intelligent Decision-Support System**"
    )

    st.subheader("Project subline")
    st.write(
        "AI-powered wafer-map classification, yield estimation, explainability, "
        "similarity search, lot intelligence and engineering decision support using WM-811K."
    )

    st.subheader("End-to-end workflow")
    st.code(
        "WM-811K\n"
        "→ preprocessing\n"
        "→ controlled ResNet\n"
        "→ defect classification\n"
        "→ yield / severity\n"
        "→ Grad-CAM\n"
        "→ similar-wafer search\n"
        "→ lot analysis\n"
        "→ anomaly screening\n"
        "→ human review\n"
        "→ recommendations",
        language="text",
    )

    st.subheader("Current completed capabilities")
    st.markdown(
        """
        - Wafer-map validation and defect classification
        - Official test evaluation
        - Yield and defect-percentage calculation
        - Project-defined severity screening
        - Grad-CAM explainability
        - Similar-wafer retrieval
        - Lot/batch analysis
        - Dataset-level anomaly screening
        - Human-in-the-loop review logging
        - Prediction-specific AI insights
        - Prediction-specific recommendations
        - Streamlit dashboard
        """
    )

    st.subheader("Future scope")
    st.markdown(
        """
        - Recommendation effectiveness using real intervention/outcome logs
        - Recovery/cure probability modeling
        - Engineer-feedback-driven continuous learning
        - API deployment
        - Cloud deployment
        - Real-time manufacturing integration
        """
    )

    st.warning(
        "WM-811K does not contain intervention records or post-intervention "
        "recovery outcomes. Therefore recovery probability and recommendation "
        "effectiveness are future extensions, not current measured outputs."
    )
