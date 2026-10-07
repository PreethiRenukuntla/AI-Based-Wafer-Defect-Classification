import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
WM811K_PKL_PATH = DATA_RAW_DIR / "wm811k" / "LSWMD.pkl"

# Model Directories
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "wafer_resnet_best.pth"
BASELINE_MODEL_PATH = MODELS_DIR / "wafer_cnn_baseline_best.pth"

# Output Directories
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
HUMAN_REVIEWS_PATH = PREDICTIONS_DIR / "human_reviews.json"
INDEXED_EMBEDDINGS_PATH = DATA_PROCESSED_DIR / "indexed_embeddings.npz"

# Ensure directories exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Defect Classes (WM-811K Official Classes)
DEFECT_CLASSES = [
    'Center',
    'Donut',
    'Edge-Loc',
    'Edge-Ring',
    'Loc',
    'Near-full',
    'Random',
    'Scratch',
    'none'  # Normal
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(DEFECT_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(DEFECT_CLASSES)}

# Baseline Performance
BASELINE_ACCURACY = 53.66  # %

# Model & Preprocessing Parameters
IMAGE_SIZE = 56  # 56x56 is fast and captures wafer map resolution cleanly
NUM_CLASSES = len(DEFECT_CLASSES)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 15
SEED = 42
DEVICE = "cpu"
