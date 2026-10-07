import numpy as np
import cv2

# Class inherent severity weight (0.0 to 1.0)
CLASS_SEVERITY_WEIGHTS = {
    'none': 0.0,
    'Random': 0.40,
    'Scratch': 0.55,
    'Loc': 0.65,
    'Edge-Loc': 0.70,
    'Center': 0.75,
    'Donut': 0.80,
    'Edge-Ring': 0.85,
    'Near-full': 1.00
}


def calculate_wafer_yield_stats(matrix):
    """
    Calculates die-level yield statistics for a valid wafer matrix.
    0: background, 1: normal die, 2: defective die.
    """
    if matrix is None or not isinstance(matrix, np.ndarray):
        return {
            'normal_dies': 0,
            'defective_dies': 0,
            'total_dies': 0,
            'yield_pct': 0.0,
            'defect_pct': 0.0,
            'defect_density': 0.0
        }

    normal_dies = int(np.sum(matrix == 1))
    defective_dies = int(np.sum(matrix == 2))
    total_dies = normal_dies + defective_dies

    if total_dies == 0:
        yield_pct = 0.0
        defect_pct = 0.0
        defect_density = 0.0
    else:
        yield_pct = round(float((normal_dies / total_dies) * 100.0), 2)
        defect_pct = round(float((defective_dies / total_dies) * 100.0), 2)
        defect_density = round(float(defective_dies / total_dies), 4)

    return {
        'normal_dies': normal_dies,
        'defective_dies': defective_dies,
        'total_dies': total_dies,
        'yield_pct': yield_pct,
        'defect_pct': defect_pct,
        'defect_density': defect_density
    }


def compute_spatial_clustering_score(matrix):
    """
    Computes spatial concentration of defects using connected components ratio.
    Returns float score between 0.0 (scattered random) and 1.0 (highly localized cluster).
    """
    defect_mask = (matrix == 2).astype(np.uint8)
    total_defects = np.sum(defect_mask)

    if total_defects <= 1:
        return 0.0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    num_clusters = num_labels - 1  # Exclude background

    if num_clusters == 0:
        return 0.0

    # Cluster density ratio
    avg_cluster_size = total_defects / float(num_clusters)
    clustering_score = min(1.0, (avg_cluster_size - 1.0) / 15.0)
    return round(float(clustering_score), 4)


def analyze_wafer_severity(matrix, predicted_class, confidence):
    """
    Computes transparent analytical / model-derived severity indicator.
    Returns composite score (0-100), rating level, breakdown components, and disclaimer.
    """
    yield_stats = calculate_wafer_yield_stats(matrix)
    defect_density = yield_stats['defect_density']

    clustering_score = compute_spatial_clustering_score(matrix)
    class_weight = CLASS_SEVERITY_WEIGHTS.get(predicted_class, 0.5)

    # Composite Severity Score formula (0 to 100)
    # 50% defect density weight, 25% spatial clustering weight, 25% defect class weight
    raw_score = (0.50 * min(1.0, defect_density * 4.0) +
                 0.25 * clustering_score +
                 0.25 * class_weight) * 100.0

    severity_score = round(float(min(100.0, max(0.0, raw_score))), 2)

    if predicted_class == 'none' and yield_stats['defective_dies'] == 0:
        severity_score = 0.0

    # Categorize severity rating level
    if severity_score < 15.0:
        rating = "Negligible / Normal"
        color = "green"
    elif severity_score < 35.0:
        rating = "Minor Severity"
        color = "blue"
    elif severity_score < 65.0:
        rating = "Moderate Severity"
        color = "orange"
    elif severity_score < 85.0:
        rating = "Severe Defect Impact"
        color = "crimson"
    else:
        rating = "Critical Wafer Failure"
        color = "purple"

    return {
        'severity_score': severity_score,
        'severity_rating': rating,
        'color': color,
        'defect_density': defect_density,
        'clustering_score': clustering_score,
        'class_severity_weight': class_weight,
        'yield_stats': yield_stats,
        'disclaimer': 'Analytical / model-derived severity indicator (Not an official semiconductor manufacturing standard).'
    }
