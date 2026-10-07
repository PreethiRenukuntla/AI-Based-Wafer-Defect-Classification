import sys
import numpy as np
import pandas as pd
from pathlib import Path

import pandas.core.indexes.base
import pandas.core.indexes.range
import pandas.core.indexes.category

# Workaround for legacy pickle files saved with older pandas versions
sys.modules["pandas.indexes"] = pd.core.indexes
sys.modules["pandas.indexes.base"] = pandas.core.indexes.base
sys.modules["pandas.indexes.range"] = pandas.core.indexes.range
sys.modules["pandas.indexes.category"] = pandas.core.indexes.category

from src.config import WM811K_PKL_PATH, DATA_PROCESSED_DIR, DEFECT_CLASSES, CLASS_TO_IDX

PROCESSED_META_PATH = DATA_PROCESSED_DIR / "wm811k_metadata.pkl"


def _clean_val(val):
    if isinstance(val, (list, np.ndarray)):
        if len(val) > 0:
            if isinstance(val[0], (list, np.ndarray)):
                if len(val[0]) > 0:
                    return str(val[0][0]).strip()
            return str(val[0]).strip()
        return ""
    return str(val).strip()


def load_raw_df(pkl_path=WM811K_PKL_PATH):
    if not Path(pkl_path).exists():
        raise FileNotFoundError(f"WM-811K pickle file not found at {pkl_path}")
    df = pd.read_pickle(pkl_path)
    return df


def parse_wm811k_metadata(pkl_path=WM811K_PKL_PATH, force_reparse=False):
    """
    Parses and caches metadata DataFrame with column indexing and clean types.
    """
    if PROCESSED_META_PATH.exists() and not force_reparse:
        print(f"Loading cached metadata from {PROCESSED_META_PATH}...")
        meta_df = pd.read_pickle(PROCESSED_META_PATH)
        return meta_df

    print(f"Parsing metadata from raw pickle {pkl_path}...")
    df = load_raw_df(pkl_path)
    total_count = len(df)

    # Vectorized / fast extraction
    raw_failures = df['failureType'].values
    raw_splits = df['trianTestLabel'].values
    raw_lots = df['lotName'].values
    raw_w_idx = df['waferIndex'].values
    raw_maps = df['waferMap'].values

    clean_failures = [_clean_val(v) for v in raw_failures]
    clean_splits = [_clean_val(v) for v in raw_splits]
    clean_lots = [_clean_val(v) for v in raw_lots]

    wafer_indices = []
    for v in raw_w_idx:
        try:
            if isinstance(v, (list, np.ndarray)) and len(v) > 0 and len(v[0]) > 0:
                wafer_indices.append(int(v[0][0]))
            elif isinstance(v, (int, float, np.integer)):
                wafer_indices.append(int(v))
            else:
                wafer_indices.append(0)
        except Exception:
            wafer_indices.append(0)

    dim_rows = []
    dim_cols = []
    for wm in raw_maps:
        if isinstance(wm, np.ndarray) and wm.ndim == 2:
            r, c = wm.shape
            dim_rows.append(r)
            dim_cols.append(c)
        else:
            dim_rows.append(0)
            dim_cols.append(0)

    meta_df = pd.DataFrame({
        'original_idx': np.arange(total_count, dtype=np.int32),
        'lotName': clean_lots,
        'waferIndex': np.array(wafer_indices, dtype=np.int16),
        'trianTestLabel': clean_splits,
        'failureType': clean_failures,
        'dim_rows': np.array(dim_rows, dtype=np.int16),
        'dim_cols': np.array(dim_cols, dtype=np.int16)
    })

    meta_df['is_labeled'] = meta_df['failureType'].isin(DEFECT_CLASSES)
    meta_df['class_idx'] = meta_df['failureType'].map(lambda x: CLASS_TO_IDX.get(x, -1)).astype(np.int8)

    print(f"Caching processed metadata to {PROCESSED_META_PATH}...")
    PROCESSED_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_pickle(PROCESSED_META_PATH)

    return meta_df


def get_dataset_analysis_summary(meta_df):
    total_wafers = len(meta_df)
    labeled_df = meta_df[meta_df['is_labeled']].copy()
    unlabeled_df = meta_df[~meta_df['is_labeled']].copy()

    total_labeled = len(labeled_df)
    total_unlabeled = len(unlabeled_df)

    class_counts = labeled_df['failureType'].value_counts().to_dict()
    class_percentages = {k: round((v / total_labeled) * 100, 2) for k, v in class_counts.items()}

    train_official_count = int((labeled_df['trianTestLabel'] == 'Training').sum())
    test_official_count = int((labeled_df['trianTestLabel'] == 'Test').sum())

    unique_lots = int(meta_df['lotName'].nunique())
    dim_distributions = {
        'min_shape': (int(meta_df['dim_rows'].min()), int(meta_df['dim_cols'].min())),
        'max_shape': (int(meta_df['dim_rows'].max()), int(meta_df['dim_cols'].max())),
        'mean_rows': round(float(meta_df['dim_rows'].mean()), 2),
        'mean_cols': round(float(meta_df['dim_cols'].mean()), 2),
    }

    summary = {
        "total_wafers": total_wafers,
        "total_labeled": total_labeled,
        "total_unlabeled": total_unlabeled,
        "labeled_percentage": round((total_labeled / total_wafers) * 100, 2),
        "unlabeled_percentage": round((total_unlabeled / total_wafers) * 100, 2),
        "unique_lots": unique_lots,
        "num_classes": len(DEFECT_CLASSES),
        "class_distribution": class_counts,
        "class_percentages": class_percentages,
        "train_official_count": train_official_count,
        "test_official_count": test_official_count,
        "dimension_stats": dim_distributions
    }

    return summary


if __name__ == "__main__":
    meta_df = parse_wm811k_metadata(force_reparse=True)
    summary = get_dataset_analysis_summary(meta_df)
    print("\nWM-811K Dataset Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
