import numpy as np
import pandas as pd

from src.data.wm811k_loader import parse_wm811k_metadata, load_raw_df
from src.analysis.yield_severity import calculate_wafer_yield_stats


def analyze_lot_batch(lot_name, meta_df=None, raw_df=None):
    """
    Analyzes all wafers belonging to a specific manufacturing lot.
    """
    if meta_df is None:
        meta_df = parse_wm811k_metadata()

    lot_meta = meta_df[meta_df['lotName'] == lot_name].copy()
    if len(lot_meta) == 0:
        return {
            'found': False,
            'message': f"Lot '{lot_name}' not found in WM-811K metadata."
        }

    total_wafers_in_lot = len(lot_meta)
    labeled_in_lot = lot_meta[lot_meta['is_labeled']]
    defect_counts = labeled_in_lot['failureType'].value_counts().to_dict()

    if raw_df is None:
        raw_df = load_raw_df()

    wafer_details = []
    accumulated_defects = None
    accumulated_normal = None
    stack_count = 0

    yields = []

    for _, row in lot_meta.iterrows():
        orig_idx = row['original_idx']
        wm = raw_df.iloc[orig_idx]['waferMap']

        stats = calculate_wafer_yield_stats(wm)
        yields.append(stats['yield_pct'])

        rec = {
            'wafer_index': int(row['waferIndex']),
            'original_idx': int(orig_idx),
            'failureType': str(row['failureType']),
            'trianTestLabel': str(row['trianTestLabel']),
            'yield_pct': stats['yield_pct'],
            'defect_pct': stats['defect_pct'],
            'total_dies': stats['total_dies'],
            'defective_dies': stats['defective_dies']
        }
        wafer_details.append(rec)

        # Spatial accumulation matrix if dimensions match
        if isinstance(wm, np.ndarray) and wm.ndim == 2:
            defects = (wm == 2).astype(np.float32)
            if accumulated_defects is None:
                accumulated_defects = defects
                stack_count = 1
            elif accumulated_defects.shape == defects.shape:
                accumulated_defects += defects
                stack_count += 1

    avg_lot_yield = round(float(np.mean(yields)), 2) if len(yields) > 0 else 0.0
    min_lot_yield = round(float(np.min(yields)), 2) if len(yields) > 0 else 0.0
    max_lot_yield = round(float(np.max(yields)), 2) if len(yields) > 0 else 0.0

    return {
        'found': True,
        'lotName': lot_name,
        'total_wafers': total_wafers_in_lot,
        'labeled_wafers': len(labeled_in_lot),
        'avg_lot_yield': avg_lot_yield,
        'min_lot_yield': min_lot_yield,
        'max_lot_yield': max_lot_yield,
        'defect_class_distribution': defect_counts,
        'wafer_details': wafer_details,
        'defect_stack_heatmap': accumulated_defects,
        'stack_count': stack_count
    }


def list_sample_lots(meta_df=None, min_wafers=25, num_lots=10):
    """Returns a list of sample lots with complete wafer counts."""
    if meta_df is None:
        meta_df = parse_wm811k_metadata()

    lot_counts = meta_df['lotName'].value_counts()
    full_lots = lot_counts[lot_counts >= min_wafers].index.tolist()
    return full_lots[:num_lots]
