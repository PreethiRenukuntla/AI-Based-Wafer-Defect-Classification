import numpy as np
import torch
from pathlib import Path

from src.config import (
    INDEXED_EMBEDDINGS_PATH,
    DEVICE,
    DEFECT_CLASSES,
    IMAGE_SIZE,
)
from src.data.dataset import preprocess_wafer_matrix
from src.data.wm811k_loader import parse_wm811k_metadata, load_raw_df
from src.analysis.yield_severity import calculate_wafer_yield_stats


def build_embedding_index(
    model,
    sample_size_per_class=200,
    force_rebuild=False
):
    """
    Build and cache a representative embedding index from labeled
    WM-811K wafers.

    The index is stratified across the 9 defect classes and uses the
    final trained ResNet's 128-D feature embeddings.
    """

    # ---------------------------------------------------------
    # LOAD EXISTING INDEX
    # ---------------------------------------------------------
    if INDEXED_EMBEDDINGS_PATH.exists() and not force_rebuild:
        print(f"Loading indexed embeddings from:")
        print(f"  {INDEXED_EMBEDDINGS_PATH}")

        data = np.load(
            INDEXED_EMBEDDINGS_PATH,
            allow_pickle=True
        )

        return {
            "embeddings": data["embeddings"],
            "metadata": data["metadata"].item(),
        }

    # ---------------------------------------------------------
    # START BUILD
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILDING SIMILAR-WAFER EMBEDDING INDEX")
    print("=" * 60)

    model.eval()

    # ---------------------------------------------------------
    # LOAD WM-811K DATA
    # ---------------------------------------------------------
    print("\nLoading WM-811K metadata...")
    meta_df = parse_wm811k_metadata()

    print("Loading WM-811K wafer maps...")
    raw_df = load_raw_df()

    labeled_meta = meta_df[
        meta_df["is_labeled"]
    ].copy()

    print(
        f"Labeled wafers available: "
        f"{len(labeled_meta):,}"
    )

    # ---------------------------------------------------------
    # STRATIFIED SAMPLING
    # ---------------------------------------------------------
    selected_indices = []

    print("\nSelecting representative wafers:")

    for cls_name in DEFECT_CLASSES:

        cls_sub = labeled_meta[
            labeled_meta["failureType"] == cls_name
        ]

        n_samples = min(
            len(cls_sub),
            sample_size_per_class
        )

        if n_samples == 0:
            print(f"  {cls_name:<12}: 0")
            continue

        sub_sample = cls_sub.sample(
            n=n_samples,
            random_state=42
        )

        selected_indices.extend(
            sub_sample["original_idx"].astype(int).tolist()
        )

        print(
            f"  {cls_name:<12}: "
            f"{n_samples:>4} wafers"
        )

    selected_indices = np.array(
        selected_indices,
        dtype=np.int64
    )

    print(
        f"\nTotal wafers selected: "
        f"{len(selected_indices):,}"
    )

    # ---------------------------------------------------------
    # CREATE LOOKUP FOR METADATA
    # ---------------------------------------------------------
    metadata_lookup = {
        int(row["original_idx"]): row
        for _, row in meta_df.iterrows()
    }

    # ---------------------------------------------------------
    # EXTRACT EMBEDDINGS
    # ---------------------------------------------------------
    embeddings_list = []
    records = []

    print("\nExtracting 128-D embeddings...")
    print("This may take some time on CPU.\n")

    with torch.no_grad():

        for count, orig_idx in enumerate(
            selected_indices,
            start=1
        ):

            # Progress
            if count == 1 or count % 100 == 0:
                print(
                    f"  Processing "
                    f"{count:,}/{len(selected_indices):,}"
                )

            # -------------------------------------------------
            # GET WAFER MAP
            # -------------------------------------------------
            row = raw_df.iloc[int(orig_idx)]
            wm = row["waferMap"]

            # -------------------------------------------------
            # PREPROCESS
            # -------------------------------------------------
            tensor = preprocess_wafer_matrix(
                wm,
                target_size=IMAGE_SIZE
            ).unsqueeze(0).to(DEVICE)

            # -------------------------------------------------
            # FEATURE EMBEDDING
            # -------------------------------------------------
            emb = model.extract_features(
                tensor
            )

            emb = (
                emb
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )

            # -------------------------------------------------
            # NORMALIZE
            # -------------------------------------------------
            norm = np.linalg.norm(emb)

            if norm > 0:
                emb = emb / norm

            embeddings_list.append(emb)

            # -------------------------------------------------
            # METADATA
            # -------------------------------------------------
            meta_row = metadata_lookup[int(orig_idx)]

            yield_stats = calculate_wafer_yield_stats(wm)

            records.append(
                {
                    "original_idx": int(orig_idx),

                    "failureType": str(
                        meta_row["failureType"]
                    ),

                    "lotName": str(
                        meta_row["lotName"]
                    ),

                    "waferIndex": int(
                        meta_row["waferIndex"]
                    ),

                    "yield_pct": float(
                        yield_stats["yield_pct"]
                    ),

                    "defect_density": float(
                        yield_stats["defect_density"]
                    ),

                    "total_dies": int(
                        yield_stats["total_dies"]
                    ),
                }
            )

    # ---------------------------------------------------------
    # CONVERT TO NUMPY
    # ---------------------------------------------------------
    embeddings_arr = np.asarray(
        embeddings_list,
        dtype=np.float32
    )

    metadata = {
        "records": records,
        "embedding_dimension": int(
            embeddings_arr.shape[1]
        ),
        "num_wafers": int(
            embeddings_arr.shape[0]
        ),
        "sample_size_per_class": int(
            sample_size_per_class
        ),
    }

    # ---------------------------------------------------------
    # SAVE INDEX
    # ---------------------------------------------------------
    INDEXED_EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        INDEXED_EMBEDDINGS_PATH,
        embeddings=embeddings_arr,
        metadata=metadata,
    )

    print("\n" + "=" * 60)
    print("EMBEDDING INDEX BUILT SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Indexed wafers : "
        f"{len(embeddings_arr):,}"
    )

    print(
        f"Embedding size : "
        f"{embeddings_arr.shape[1]}"
    )

    print(
        f"Saved to       : "
        f"{INDEXED_EMBEDDINGS_PATH}"
    )

    return {
        "embeddings": embeddings_arr,
        "metadata": metadata,
    }


def search_similar_wafers(
    query_matrix,
    model,
    index_data,
    top_k=5
):
    """
    Find the most similar historical wafers using
    cosine similarity between 128-D embeddings.
    """

    if query_matrix is None:
        return []

    if index_data is None:
        return []

    if "embeddings" not in index_data:
        return []

    model.eval()

    # ---------------------------------------------------------
    # PREPROCESS QUERY WAFER
    # ---------------------------------------------------------
    tensor = preprocess_wafer_matrix(
        query_matrix,
        target_size=IMAGE_SIZE
    ).unsqueeze(0).to(DEVICE)

    # ---------------------------------------------------------
    # EXTRACT QUERY EMBEDDING
    # ---------------------------------------------------------
    with torch.no_grad():

        query_emb = model.extract_features(
            tensor
        )

        query_emb = (
            query_emb
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )

    # ---------------------------------------------------------
    # NORMALIZE QUERY
    # ---------------------------------------------------------
    norm = np.linalg.norm(query_emb)

    if norm > 0:
        query_emb = query_emb / norm

    # ---------------------------------------------------------
    # STORED EMBEDDINGS
    # ---------------------------------------------------------
    stored_embeddings = index_data[
        "embeddings"
    ]

    records = index_data[
        "metadata"
    ]["records"]

    if len(stored_embeddings) == 0:
        return []

    # ---------------------------------------------------------
    # COSINE SIMILARITY
    # ---------------------------------------------------------
    similarities = np.dot(
        stored_embeddings,
        query_emb
    )

    # ---------------------------------------------------------
    # TOP-K
    # ---------------------------------------------------------
    top_k = min(
        int(top_k),
        len(similarities)
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    results = []

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        rec = records[int(idx)].copy()

        similarity = float(
            similarities[int(idx)]
        )

        rec["rank"] = rank

        rec["similarity_pct"] = round(
            similarity * 100.0,
            2
        )

        results.append(rec)

    return results


def print_search_results(results):
    """
    Pretty-print similar wafer results.
    """

    print("\n" + "=" * 70)
    print("SIMILAR WAFERS")
    print("=" * 70)

    if not results:
        print("No similar wafers found.")
        return

    for result in results:

        print(
            f"\n#{result['rank']} "
            f"Similarity: "
            f"{result['similarity_pct']:.2f}%"
        )

        print(
            f"   Defect class : "
            f"{result['failureType']}"
        )

        print(
            f"   Lot          : "
            f"{result['lotName']}"
        )

        print(
            f"   Wafer index  : "
            f"{result['waferIndex']}"
        )

        print(
            f"   Yield        : "
            f"{result['yield_pct']:.2f}%"
        )

        print(
            f"   Defect       : "
            f"{result['defect_density']:.2f}%"
        )

        print(
            f"   Total dies   : "
            f"{result['total_dies']:,}"
        )


def main():
    """
    Smoke test for the similarity-search pipeline.
    """

    from src.models.architecture import (
        WaferDefectResNet
    )

    MODEL_PATH = Path(
        "models/wafer_resnet_controlled_best.pth"
    )

    print("\n" + "=" * 60)
    print("SIMILAR-WAFER SEARCH TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------
    model = WaferDefectResNet(
        num_classes=len(DEFECT_CLASSES)
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        f"Model loaded: {MODEL_PATH}"
    )

    # ---------------------------------------------------------
    # BUILD / LOAD INDEX
    # ---------------------------------------------------------
    index_data = build_embedding_index(
        model,
        sample_size_per_class=200,
        force_rebuild=False
    )

    # ---------------------------------------------------------
    # SYNTHETIC QUERY WAFER
    # ---------------------------------------------------------
    query = np.ones(
        (64, 64),
        dtype=np.uint8
    )

    query[0:5, :] = 0
    query[-5:, :] = 0
    query[:, 0:5] = 0
    query[:, -5:] = 0

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------
    results = search_similar_wafers(
        query,
        model,
        index_data,
        top_k=5
    )

    print_search_results(results)

    print("\nSimilarity search test completed.")


if __name__ == "__main__":
    main()