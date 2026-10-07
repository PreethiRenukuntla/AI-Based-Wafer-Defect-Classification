import numpy as np
import pandas as pd

from src.data.wm811k_loader import parse_wm811k_metadata
#from src.data.yield_severity import calculate_wafer_yield_stats


def build_lot_analysis():
    """
    Analyze WM-811K labeled wafers at lot/batch level.

    Returns one row per lot with:
        - number of wafers
        - average yield
        - average defect density
        - defective wafer count
        - dominant defect pattern
        - defect distribution
    """

    print("\n" + "=" * 65)
    print("BUILDING LOT / BATCH ANALYSIS")
    print("=" * 65)

    print("\nLoading WM-811K metadata...")

    meta_df = parse_wm811k_metadata()

    labeled_meta = meta_df[
        meta_df["is_labeled"]
    ].copy()

    print(
        f"Labeled wafers available: "
        f"{len(labeled_meta):,}"
    )

    # ---------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # ---------------------------------------------------------

    required_columns = [
        "original_idx",
        "lotName",
        "waferIndex",
        "failureType",
    ]

    missing = [
        col for col in required_columns
        if col not in labeled_meta.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required metadata columns: {missing}"
        )

    # ---------------------------------------------------------
    # LOT-LEVEL DEFECT DISTRIBUTION
    # ---------------------------------------------------------

    print("\nCalculating lot-level statistics...")

    lot_summary = (
        labeled_meta
        .groupby("lotName")
        .agg(
            wafer_count=("original_idx", "count"),
            dominant_defect=("failureType", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # DEFECT COUNTS PER LOT
    # ---------------------------------------------------------

    defect_counts = pd.crosstab(
        labeled_meta["lotName"],
        labeled_meta["failureType"]
    ).reset_index()

    lot_summary = lot_summary.merge(
        defect_counts,
        on="lotName",
        how="left"
    )

    # ---------------------------------------------------------
    # DEFECT RATE
    # ---------------------------------------------------------

    if "none" in lot_summary.columns:

        lot_summary["defective_wafer_count"] = (
            lot_summary["wafer_count"]
            - lot_summary["none"]
        )

        lot_summary["defective_wafer_pct"] = (
            lot_summary["defective_wafer_count"]
            / lot_summary["wafer_count"]
            * 100.0
        )

    # ---------------------------------------------------------
    # SORT
    # ---------------------------------------------------------

    lot_summary = lot_summary.sort_values(
        "wafer_count",
        ascending=False
    ).reset_index(drop=True)

    print(
        f"Unique lots analyzed: "
        f"{len(lot_summary):,}"
    )

    return lot_summary


def analyze_specific_lot(lot_name):
    """
    Return detailed information for one specific lot.
    """

    meta_df = parse_wm811k_metadata()

    lot_df = meta_df[
        (meta_df["is_labeled"])
        & (meta_df["lotName"] == lot_name)
    ].copy()

    if lot_df.empty:
        return None

    result = {
        "lotName": lot_name,
        "wafer_count": int(len(lot_df)),
        "defect_distribution": (
            lot_df["failureType"]
            .value_counts()
            .to_dict()
        ),
        "wafer_indices": (
            lot_df["waferIndex"]
            .astype(int)
            .tolist()
        ),
    }

    # Dominant defect
    result["dominant_defect"] = (
        lot_df["failureType"]
        .value_counts()
        .idxmax()
    )

    # Defective wafers
    result["defective_wafer_count"] = int(
        (lot_df["failureType"] != "none").sum()
    )

    result["defective_wafer_pct"] = round(
        result["defective_wafer_count"]
        / result["wafer_count"]
        * 100.0,
        2
    )

    return result


def find_problematic_lots(
    lot_summary,
    min_wafers=3,
    defect_threshold=20.0
):
    """
    Identify lots with a high proportion of defective wafers.

    This is a project-defined screening rule, not a
    semiconductor manufacturing standard.
    """

    if lot_summary is None or lot_summary.empty:
        return pd.DataFrame()

    problematic = lot_summary[
        (lot_summary["wafer_count"] >= min_wafers)
        & (
            lot_summary["defective_wafer_pct"]
            >= defect_threshold
        )
    ].copy()

    problematic = problematic.sort_values(
        "defective_wafer_pct",
        ascending=False
    )

    return problematic.reset_index(drop=True)


def generate_lot_insight(lot_result):
    """
    Generate a deterministic project-level insight
    from lot statistics.
    """

    if lot_result is None:
        return "No lot information available."

    lot_name = lot_result["lotName"]
    wafer_count = lot_result["wafer_count"]
    defective_count = lot_result["defective_wafer_count"]
    defective_pct = lot_result["defective_wafer_pct"]
    dominant_defect = lot_result["dominant_defect"]

    if defective_pct < 5:
        severity = "low"
    elif defective_pct < 15:
        severity = "moderate"
    elif defective_pct < 30:
        severity = "high"
    else:
        severity = "critical"

    return (
        f"Lot {lot_name} contains {wafer_count} labeled wafers. "
        f"{defective_count} wafers are associated with labeled "
        f"defect patterns ({defective_pct:.1f}%). "
        f"The dominant observed pattern is {dominant_defect}. "
        f"The lot-level defect indicator is {severity}."
    )


def generate_lot_recommendation(lot_result):
    """
    Generate engineering-oriented screening recommendations.
    These are decision-support suggestions, not process prescriptions.
    """

    if lot_result is None:
        return (
            "No lot data available. "
            "Review the individual wafer first."
        )

    defective_pct = lot_result["defective_wafer_pct"]
    dominant_defect = lot_result["dominant_defect"]

    if defective_pct < 5:
        return (
            "Continue monitoring the lot and compare new wafers "
            "against historical similar wafers."
        )

    if defective_pct < 15:
        return (
            f"Review wafers associated with the {dominant_defect} "
            f"pattern and compare their spatial patterns with "
            f"historical wafers from related lots."
        )

    if defective_pct < 30:
        return (
            f"Prioritize engineering review of the lot. "
            f"Investigate the {dominant_defect} pattern, compare "
            f"affected wafers with similar historical wafers, "
            f"and review relevant process/test information."
        )

    return (
        f"Flag the lot for detailed engineering review. "
        f"The {dominant_defect} pattern should be investigated "
        f"across affected wafers and compared with historical "
        f"lot-level behavior before process decisions are made."
    )


def print_lot_summary(lot_summary, top_n=10):
    """
    Print the largest lots.
    """

    print("\n" + "=" * 65)
    print("LOT / BATCH SUMMARY")
    print("=" * 65)

    if lot_summary.empty:
        print("No lot information found.")
        return

    display_columns = [
        "lotName",
        "wafer_count",
        "dominant_defect",
        "defective_wafer_count",
        "defective_wafer_pct",
    ]

    available_columns = [
        col
        for col in display_columns
        if col in lot_summary.columns
    ]

    print(
        lot_summary[
            available_columns
        ]
        .head(top_n)
        .to_string(index=False)
    )


def main():

    lot_summary = build_lot_analysis()

    print_lot_summary(
        lot_summary,
        top_n=10
    )

    # ---------------------------------------------------------
    # PROBLEMATIC LOTS
    # ---------------------------------------------------------

    problematic = find_problematic_lots(
        lot_summary,
        min_wafers=3,
        defect_threshold=20.0
    )

    print("\n" + "=" * 65)
    print("PROBLEMATIC LOT SCREENING")
    print("=" * 65)

    print(
        f"Lots meeting screening threshold: "
        f"{len(problematic):,}"
    )

    if not problematic.empty:

        print(
            problematic[
                [
                    "lotName",
                    "wafer_count",
                    "dominant_defect",
                    "defective_wafer_count",
                    "defective_wafer_pct",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        # Analyze first problematic lot
        first_lot = problematic.iloc[0]["lotName"]

        print("\n" + "=" * 65)
        print(
            f"DETAILED ANALYSIS: {first_lot}"
        )
        print("=" * 65)

        lot_result = analyze_specific_lot(
            first_lot
        )

        print(
            f"Lot              : "
            f"{lot_result['lotName']}"
        )

        print(
            f"Wafers           : "
            f"{lot_result['wafer_count']}"
        )

        print(
            f"Defective wafers : "
            f"{lot_result['defective_wafer_count']}"
        )

        print(
            f"Defective %      : "
            f"{lot_result['defective_wafer_pct']:.2f}%"
        )

        print(
            f"Dominant defect  : "
            f"{lot_result['dominant_defect']}"
        )

        print("\nDefect distribution:")

        for defect, count in (
            lot_result["defect_distribution"]
            .items()
        ):
            print(
                f"  {defect:<12}: {count}"
            )

        print("\nAI Insight:")
        print(
            generate_lot_insight(
                lot_result
            )
        )

        print("\nEngineering Recommendation:")
        print(
            generate_lot_recommendation(
                lot_result
            )
        )

    else:

        print(
            "No lots met the current screening threshold."
        )

    print("\nLot/batch analysis test completed.")


if __name__ == "__main__":
    main()