import numpy as np
from pathlib import Path

from src.config import INDEXED_EMBEDDINGS_PATH, DEFECT_CLASSES


def load_embedding_index():
    """Load the existing similar-wafer embedding index."""

    if not INDEXED_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding index not found:\n"
            f"{INDEXED_EMBEDDINGS_PATH}\n\n"
            f"Run the similar-wafer search module first."
        )

    data = np.load(
        INDEXED_EMBEDDINGS_PATH,
        allow_pickle=True
    )

    return {
        "embeddings": data["embeddings"],
        "metadata": data["metadata"].item(),
    }


def calculate_dataset_statistics(index_data):
    """
    Calculate dataset-level statistics from the indexed
    representative wafer population.
    """

    records = index_data["metadata"]["records"]

    if not records:
        return {}

    classes = [
        record["failureType"]
        for record in records
    ]

    total = len(records)

    class_counts = {}

    for cls in DEFECT_CLASSES:
        class_counts[cls] = classes.count(cls)

    defect_count = sum(
        1
        for cls in classes
        if cls != "none"
    )

    normal_count = classes.count("none")

    defect_percentage = (
        defect_count / total * 100
        if total > 0
        else 0
    )

    normal_percentage = (
        normal_count / total * 100
        if total > 0
        else 0
    )

    yields = np.array(
        [
            record["yield_pct"]
            for record in records
        ],
        dtype=np.float32
    )

    defect_densities = np.array(
        [
            record["defect_density"]
            for record in records
        ],
        dtype=np.float32
    )

    statistics = {
        "total_indexed_wafers": total,
        "normal_wafers": normal_count,
        "defective_wafers": defect_count,
        "normal_percentage": round(
            normal_percentage,
            2
        ),
        "defective_percentage": round(
            defect_percentage,
            2
        ),
        "average_yield": round(
            float(np.mean(yields)),
            2
        ),
        "minimum_yield": round(
            float(np.min(yields)),
            2
        ),
        "maximum_yield": round(
            float(np.max(yields)),
            2
        ),
        "average_defect_density": round(
            float(np.mean(defect_densities)),
            4
        ),
        "class_distribution": class_counts,
    }

    return statistics


def calculate_embedding_centroid(index_data):
    """
    Calculate the overall embedding centroid.
    """

    embeddings = index_data["embeddings"]

    if len(embeddings) == 0:
        return None

    centroid = np.mean(
        embeddings,
        axis=0
    )

    norm = np.linalg.norm(centroid)

    if norm > 0:
        centroid = centroid / norm

    return centroid.astype(np.float32)


def calculate_anomaly_scores(index_data):
    """
    Calculate anomaly scores based on distance from
    the dataset embedding centroid.

    Higher score = more unusual representation.

    This is an unsupervised project-level anomaly indicator,
    not a certified semiconductor anomaly metric.
    """

    embeddings = index_data["embeddings"]
    records = index_data["metadata"]["records"]

    centroid = calculate_embedding_centroid(
        index_data
    )

    if centroid is None:
        return []

    similarities = np.dot(
        embeddings,
        centroid
    )

    # Convert similarity to anomaly score.
    # Higher distance from centroid = more unusual.
    anomaly_scores = 1.0 - similarities

    results = []

    for i, record in enumerate(records):

        result = record.copy()

        result["anomaly_score"] = round(
            float(anomaly_scores[i]),
            4
        )

        result["centroid_similarity"] = round(
            float(similarities[i]),
            4
        )

        results.append(result)

    results.sort(
        key=lambda x: x["anomaly_score"],
        reverse=True
    )

    return results


def classify_anomaly_level(score):
    """
    Project-defined anomaly interpretation.
    """

    if score < 0.10:
        return "Normal"

    if score < 0.20:
        return "Watch"

    if score < 0.30:
        return "Unusual"

    return "Highly unusual"


def generate_dataset_insight(statistics):
    """
    Generate deterministic dataset-level insight.
    """

    if not statistics:
        return "No dataset statistics available."

    total = statistics["total_indexed_wafers"]
    defective = statistics["defective_wafers"]
    defective_pct = statistics["defective_percentage"]
    avg_yield = statistics["average_yield"]

    distribution = statistics[
        "class_distribution"
    ]

    defect_classes = {
        cls: count
        for cls, count in distribution.items()
        if cls != "none"
    }

    if defect_classes:
        dominant_defect = max(
            defect_classes,
            key=defect_classes.get
        )
    else:
        dominant_defect = "none"

    return (
        f"The indexed dataset contains {total:,} representative "
        f"wafers, with {defective:,} labeled as defective "
        f"({defective_pct:.2f}%). Average estimated yield is "
        f"{avg_yield:.2f}%. Among the defect classes in the "
        f"index, {dominant_defect} is the most represented "
        f"defect pattern."
    )


def generate_anomaly_insight(anomaly_results, top_n=10):
    """
    Summarize the most unusual indexed wafers.
    """

    if not anomaly_results:
        return "No anomaly results available."

    top_results = anomaly_results[:top_n]

    classes = [
        result["failureType"]
        for result in top_results
    ]

    counts = {}

    for cls in classes:
        counts[cls] = counts.get(cls, 0) + 1

    dominant_class = max(
        counts,
        key=counts.get
    )

    highest_score = top_results[0][
        "anomaly_score"
    ]

    return (
        f"The most unusual indexed wafer has an anomaly "
        f"score of {highest_score:.4f}. Within the top "
        f"{top_n} unusual wafers, the most frequent labeled "
        f"pattern is {dominant_class}. These wafers should "
        f"be reviewed against similar historical patterns "
        f"before engineering decisions are made."
    )


def print_dataset_statistics(statistics):
    """Print dataset-level statistics."""

    print("\n" + "=" * 70)
    print("DATASET-LEVEL INTELLIGENCE")
    print("=" * 70)

    print(
        f"Indexed wafers       : "
        f"{statistics['total_indexed_wafers']:,}"
    )

    print(
        f"Normal wafers        : "
        f"{statistics['normal_wafers']:,}"
    )

    print(
        f"Defective wafers     : "
        f"{statistics['defective_wafers']:,}"
    )

    print(
        f"Normal percentage    : "
        f"{statistics['normal_percentage']:.2f}%"
    )

    print(
        f"Defective percentage : "
        f"{statistics['defective_percentage']:.2f}%"
    )

    print(
        f"Average yield        : "
        f"{statistics['average_yield']:.2f}%"
    )

    print(
        f"Minimum yield        : "
        f"{statistics['minimum_yield']:.2f}%"
    )

    print(
        f"Maximum yield        : "
        f"{statistics['maximum_yield']:.2f}%"
    )

    print(
        f"Average defect density: "
        f"{statistics['average_defect_density']:.4f}%"
    )

    print("\nClass distribution:")

    for cls, count in (
        statistics["class_distribution"].items()
    ):
        print(
            f"  {cls:<12}: {count:,}"
        )


def print_top_anomalies(
    anomaly_results,
    top_n=10
):
    """Print the most unusual wafers."""

    print("\n" + "=" * 70)
    print("TOP ANOMALOUS / UNUSUAL WAFERS")
    print("=" * 70)

    if not anomaly_results:
        print("No anomaly results available.")
        return

    for result in anomaly_results[:top_n]:

        level = classify_anomaly_level(
            result["anomaly_score"]
        )

        print(
            f"\n#{result.get('rank', '')}"
        )

        print(
            f"  Original index : "
            f"{result['original_idx']}"
        )

        print(
            f"  Lot            : "
            f"{result['lotName']}"
        )

        print(
            f"  Wafer index    : "
            f"{result['waferIndex']}"
        )

        print(
            f"  Class          : "
            f"{result['failureType']}"
        )

        print(
            f"  Anomaly score  : "
            f"{result['anomaly_score']:.4f}"
        )

        print(
            f"  Level          : "
            f"{level}"
        )

        print(
            f"  Yield          : "
            f"{result['yield_pct']:.2f}%"
        )


def main():

    print("\n" + "=" * 70)
    print("ANOMALY / DATASET INTELLIGENCE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD EXISTING INDEX
    # ---------------------------------------------------------

    print("\nLoading existing embedding index...")

    index_data = load_embedding_index()

    print(
        f"Loaded "
        f"{len(index_data['embeddings']):,} "
        f"indexed wafers."
    )

    # ---------------------------------------------------------
    # DATASET STATISTICS
    # ---------------------------------------------------------

    statistics = calculate_dataset_statistics(
        index_data
    )

    print_dataset_statistics(
        statistics
    )

    # ---------------------------------------------------------
    # DATASET INSIGHT
    # ---------------------------------------------------------

    print("\nAI Dataset Insight:")

    print(
        generate_dataset_insight(
            statistics
        )
    )

    # ---------------------------------------------------------
    # ANOMALY DETECTION
    # ---------------------------------------------------------

    print("\nCalculating anomaly scores...")

    anomaly_results = calculate_anomaly_scores(
        index_data
    )

    # Add rank
    for rank, result in enumerate(
        anomaly_results,
        start=1
    ):
        result["rank"] = rank

    print_top_anomalies(
        anomaly_results,
        top_n=10
    )

    # ---------------------------------------------------------
    # ANOMALY INSIGHT
    # ---------------------------------------------------------

    print("\nAI Anomaly Insight:")

    print(
        generate_anomaly_insight(
            anomaly_results,
            top_n=10
        )
    )

    print("\nAnomaly/dataset intelligence test completed.")


if __name__ == "__main__":
    main()