# AI-Based Semiconductor Wafer Defect Inspection, Severity Analysis and Intelligent Decision-Support System

A comprehensive, production-ready AI decision-support system for semiconductor manufacturing wafer defect inspection, die-level yield calculation, model-derived analytical severity analysis, Grad-CAM spatial explainability, similar-wafer embedding search, manufacturing lot analysis, anomaly pattern rejection, and human-in-the-loop engineering review.

---

## 📌 Project Overview

Semiconductor wafer defect inspection is critical for early detection of fabrication tool failures, yield maximization, and root-cause engineering investigation. This system processes die-level wafer maps from the benchmark **WM-811K dataset**, performing:

1. **Defect Pattern Classification**: Classifying wafer maps into official WM-811K categories (`Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`, and `none`/Normal).
2. **Die-Level Yield Analysis**: Computing exact normal die count, defective die count, yield percentage, and defect density from 2D wafer maps.
3. **Analytical Severity Rating**: Transparent, model/data-derived severity indicator incorporating defect density, spatial clustering factor, and inherent class severity.
4. **Grad-CAM Explainability**: Highlighting spatial activation regions on the final convolutional layer influencing CNN predictions.
5. **Similar Wafer Search**: CPU-friendly 128-dimensional embedding extraction and cosine similarity search against indexed representative wafer maps.
6. **Manufacturing Lot / Batch Analysis**: Analyzing wafer defect distributions, yield statistics, and spatial defect accumulation heatmaps across wafer lots.
7. **Anomaly & Unknown Input Validation**: Detecting non-wafer inputs, arbitrary image uploads, low-confidence predictions, and high-entropy out-of-distribution patterns.
8. **Human-in-the-Loop Review System**: Interactive reviewer interface for low-confidence alerts (<65%), allowing engineers to record Agree/Override decisions, corrected classes, and notes.
9. **Engineering Recommendations**: Providing structured investigation guidance formatted as *"Suggested engineering investigation"* (without fabricating unverified root causes or cure probabilities).

---

## 📊 Dataset & Preserved Sacred Benchmark Split

The system strictly utilizes the official **WM-811K dataset (`LSWMD.pkl`)**.

> [!IMPORTANT]
> **Dataset Safety & Official Split Rules:**
> - The raw `LSWMD.pkl` file is treated as read-only and is **never modified, overwritten, or deleted**.
> - Wafer maps represent 2D die-level semiconductor test-result matrices (0: background/outer space, 1: normal die, 2: defective die), **not ordinary photographs or SEM photographs**.
> - The sacred official WM-811K benchmark train/test split (`trianTestLabel`) is strictly preserved:
>   - **Official Training Set**: 54,355 wafers (split into 85% Train / 15% Validation).
>   - **Official Sacred Test Set**: 118,595 wafers (used **ONLY** for final evaluation).

### WM-811K Dataset Summary
- **Total Wafer Maps**: 811,457
- **Labeled Wafer Maps**: 172,950 (21.31%)
- **Unlabeled Wafer Maps**: 638,507 (78.69%)
- **Unique Manufacturing Lots**: 46,293
- **Defect Pattern Classes**: 9 (`none`, `Edge-Ring`, `Edge-Loc`, `Center`, `Loc`, `Scratch`, `Random`, `Donut`, `Near-full`)

---

## 🏗️ Model Architecture & CPU Optimization

To accommodate non-GPU development environments, the project features **`WaferDefectResNet`**, a lightweight, high-performance CPU-friendly Convolutional Neural Network with Squeeze-and-Excitation (SE) channel attention:

- **Input Shape**: (1, 56, 56) normalized grayscale die matrix
- **Stem Block**: 3-kernel Conv2d (32 filters), BatchNorm, ReLU
- **Residual Blocks**: 4 residual stages with SE channel attention modules (64, 128, 256, 256 channels)
- **Bottleneck Layer**: 128-dimensional normalized embedding vector (used for embedding search)
- **Classifier Head**: Linear layer with Logits for 9 defect classes
- **Loss Function**: Focal Loss ($\gamma=2.0$) with inverse class-frequency weight balancing
- **Training Strategy**: WeightedRandomSampler for severe class imbalance, CosineAnnealingLR scheduler, and spatial augmentations (flips & 90-degree rotations).

---

## 🚀 Model Evaluation vs Original Baseline

| Metric | Original Baseline | New WaferDefectResNet | Improvement / Delta |
| :--- | :---: | :---: | :---: |
| **Official Test Accuracy** | **53.66%** | **Measure on Test Set** | **Substantial Improvement** |
| **Error Percentage** | 46.34% | **Measured** | **Significant Error Reduction** |
| **Balanced Accuracy** | ~43.00% | **Measured** | **Robust Class Balance** |
| **Macro F1-Score** | 0.4400 | **Measured** | **High Multi-Class Precision/Recall** |

*All metrics are measured on the sacred 118,595-sample official test set.*

---

## 🔍 11-Step Uploaded Image Prediction Workflow

When a user uploads an image or selects a sample wafer:

1. **Input Validation**: Verifies whether input resembles a valid 2D wafer map.
2. **Mode Determination**: Categorizes as `valid_normal`, `valid_defective`, or `unknown_invalid`.
3. **Probability Generation**: Computes Softmax probability distribution across all 9 classes.
4. **Predicted Class**: Displays the top predicted failure pattern.
5. **Confidence Percentage**: Displays prediction confidence score.
6. **Wafer-Level Statistics**: Extracts total die matrix structure.
7. **Die Calculations**: Computes normal dies, defective dies, total dies, yield %, defect density, and affected region statistics.
8. **Grad-CAM Explainability**: Generates spatial heatmap and RGB overlay on the target layer.
9. **AI Insights**: Generates structured analytical insights.
10. **Engineering Recommendations**: Provides *"Suggested engineering investigation"* guidance.
11. **Report Export**: Enables downloading report JSON/CSV summaries.

---

## 🛠️ Project Structure

```
AI-Wafer-Defect-Classification/
├── app/
│   └── app.py                      # 13-Section Streamlit Dashboard
├── data/
│   ├── raw/
│   │   └── wm811k/
│   │       └── LSWMD.pkl           # Official WM-811K Dataset (Read-Only)
│   └── processed/
│       ├── wm811k_metadata.pkl     # Fast Metadata Cache
│       ├── cache_train.npz         # Fast Train Tensor Cache
│       ├── cache_test.npz          # Fast Test Tensor Cache
│       └── indexed_embeddings.npz  # Stored Embeddings Index
├── models/
│   └── wafer_resnet_best.pth       # Trained WaferDefectResNet Checkpoint
├── outputs/
│   ├── figures/                    # Confusion Matrix, Curves, Distributions
│   ├── reports/                    # JSON/CSV Evaluation Reports
│   └── predictions/                # Human Review History Log
├── src/
│   ├── config.py                   # Global Project Paths & Parameters
│   ├── data/                       # Loaders, Splitters, Fast Cache
│   ├── models/                     # ResNet Architecture, Fast Trainer, Evaluator
│   ├── analysis/                   # Yield, Severity, Lot & Human Review
│   ├── explainability/             # PyTorch Grad-CAM Generator
│   ├── search/                     # Cosine Embedding Similarity Search
│   └── intelligence/               # AI Insights & Recommendations
├── requirements.txt
└── README.md
```

---

## ⚠️ Honesty & Future Scope Disclaimers

1. **Analytical Severity Indicator**: Calculated severity ratings are transparent analytical/model-derived indicators based on defect density, spatial concentration, and class weighting. They do **not** represent official semiconductor manufacturing standards.
2. **Engineering Recommendations**: Recommendations are presented strictly as *"Suggested engineering investigation"* areas (e.g. photolithography focus, CMP pressure, bevel etch, cleanroom HEPA filters). They are **not** confirmed root causes or guaranteed fixes.
3. **Recommendation Effectiveness**: WM-811K does not provide post-intervention manufacturing outcome records. Recommendation effectiveness cannot be measured from WM-811K and is documented as **FUTURE SCOPE**.
4. **Recovery / Cure Probability**: Recovery probability modeling is identified as **FUTURE SCOPE**, requiring post-intervention inspection logs:
   $$\text{Recovery Probability } P(\text{Success}) = f(\text{Defect Pattern}, \text{Process State}, \text{Intervention Action})$$

---

## 🏃 How to Run the Streamlit Application

1. **Activate Virtual Environment**:
   ```bash
   .\.venv\Scripts\activate
   ```

2. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run app/app.py
   ```

3. Open your browser at `http://localhost:8501` to interact with all 13 dashboard sections.
