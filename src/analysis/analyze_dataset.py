import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import FIGURES_DIR, REPORTS_DIR, DEFECT_CLASSES
from src.data.wm811k_loader import parse_wm811k_metadata, get_dataset_analysis_summary, load_raw_df

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_dataset_analysis():
    print("Running Full Dataset Analysis...")

    # Load metadata (from cache)
    meta_df = parse_wm811k_metadata()
    summary = get_dataset_analysis_summary(meta_df)

    # Save dataset_report.json
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved dataset report to {report_path}")

    # Figure 1: Class Distribution
    labeled_meta = meta_df[meta_df['is_labeled']].copy()
    class_counts = labeled_meta['failureType'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis', ax=ax)
    ax.set_title("WM-811K Defect Class Distribution (172,950 Labeled Wafers)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Failure Pattern Class", fontsize=12)
    ax.set_ylabel("Wafer Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        pct = (height / summary['total_labeled']) * 100
        ax.annotate(f"{int(height):,}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                    textcoords='offset points')
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close()

    # Figure 2: Split Distribution
    fig, ax = plt.subplots(figsize=(6, 6))
    splits = ['Official Test Set\n(118,595)', 'Official Train Set\n(54,355)', 'Unlabeled Set\n(638,507)']
    counts = [summary['test_official_count'], summary['train_official_count'], summary['total_unlabeled']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax.pie(counts, labels=splits, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.05, 0.05, 0.05))
    ax.set_title("WM-811K Complete Dataset Split Breakdown (811,457 Wafers)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "split_distribution.png", dpi=300)
    plt.close()

    # Figure 3: Dimension Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(meta_df['dim_rows'], bins=30, color='teal', ax=ax1, kde=True)
    ax1.set_title("Wafer Height (Rows) Distribution")
    ax1.set_xlabel("Rows")

    sns.histplot(meta_df['dim_cols'], bins=30, color='darkorange', ax=ax2, kde=True)
    ax2.set_title("Wafer Width (Columns) Distribution")
    ax2.set_xlabel("Columns")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "wafer_dimension_hist.png", dpi=300)
    plt.close()

    print("Generating representative wafer maps visualization...")
    df_raw = load_raw_df()
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()

    for idx, cls_name in enumerate(DEFECT_CLASSES):
        sample_indices = meta_df[meta_df['failureType'] == cls_name]['original_idx'].values
        if len(sample_indices) > 0:
            sample_idx = sample_indices[0]
            wm = df_raw.iloc[sample_idx]['waferMap']
            ax = axes[idx]
            ax.imshow(wm, cmap='coolwarm', interpolation='nearest')
            ax.set_title(f"{cls_name}\n({wm.shape[0]}x{wm.shape[1]})", fontsize=11, fontweight='bold')
            ax.axis('off')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "representative_wafers.png", dpi=300)
    plt.close()

    print("Dataset Analysis Complete! Figures saved to outputs/figures/")

if __name__ == "__main__":
    generate_dataset_analysis()
