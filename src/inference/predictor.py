"""
Central inference engine for the AI-Based Wafer Defect Inspection System.

Workflow:
    Wafer matrix
        ↓
    Input validation
        ↓
    Preprocessing
        ↓
    Controlled ResNet
        ↓
    Defect classification
        ↓
    Confidence
        ↓
    Yield calculation
        ↓
    Severity analysis
        ↓
    Engineering insight

This module does NOT train the model.
It only performs inference using the final locked model.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.config import (
    DEVICE,
    DEFECT_CLASSES,
    NUM_CLASSES,
    IMAGE_SIZE,
)

from src.data.dataset import preprocess_wafer_matrix

from src.models.architecture import WaferDefectResNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/wafer_resnet_controlled_best.pth"
)

# Confidence below this threshold is sent for human review.
LOW_CONFIDENCE_THRESHOLD = 0.70

# Confidence below this level is treated as highly uncertain.
VERY_LOW_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# MODEL LOADER
# ============================================================

class WaferPredictor:
    """
    Final model inference class.

    The model is loaded once and reused for predictions.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = DEVICE,
    ):

        self.device = torch.device(device)

        if model_path is None:
            model_path = MODEL_PATH

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found:\n"
                f"{self.model_path}"
            )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        self.model = WaferDefectResNet(
            num_classes=NUM_CLASSES
        )

        self.model = self.model.to(
            self.device
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
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

        # Remove DataParallel prefix if present.
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):
                key = key[len("module."):]

            cleaned_state_dict[key] = value

        self.model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        self.model.eval()

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def validate_wafer_matrix(
        wafer_matrix: Any,
    ) -> Dict[str, Any]:
        """
        Validate an input wafer map.

        Expected:
            2D NumPy array

        Expected WM-811K values:
            0 = background
            1 = normal/pass die
            2 = defective/fail die
        """

        result = {
            "valid": False,
            "message": "",
            "shape": None,
            "unique_values": [],
        }

        # ----------------------------------------------------
        # Convert to NumPy array
        # ----------------------------------------------------

        try:

            array = np.asarray(
                wafer_matrix
            )

        except Exception as exc:

            result["message"] = (
                f"Could not convert input to NumPy array: {exc}"
            )

            return result

        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        if array.ndim != 2:

            result["message"] = (
                "Invalid wafer map. "
                "Expected a 2D wafer matrix."
            )

            return result

        result["shape"] = tuple(
            array.shape
        )

        # ----------------------------------------------------
        # Check empty input
        # ----------------------------------------------------

        if array.size == 0:

            result["message"] = (
                "Invalid wafer map. "
                "The input is empty."
            )

            return result

        # ----------------------------------------------------
        # Check NaN / infinity
        # ----------------------------------------------------

        if np.issubdtype(
            array.dtype,
            np.floating,
        ):

            if not np.all(
                np.isfinite(array)
            ):

                result["message"] = (
                    "Invalid wafer map. "
                    "Input contains NaN or infinity."
                )

                return result

        # ----------------------------------------------------
        # Unique values
        # ----------------------------------------------------

        unique_values = np.unique(
            array
        )

        result["unique_values"] = [
            int(value)
            for value in unique_values
        ]

        # ----------------------------------------------------
        # Check WM-811K encoding
        # ----------------------------------------------------

        allowed_values = {
            0,
            1,
            2,
        }

        invalid_values = [
            int(value)
            for value in unique_values
            if int(value) not in allowed_values
        ]

        if invalid_values:

            result["message"] = (
                "Invalid wafer map values detected: "
                f"{invalid_values}. "
                "Expected only 0, 1 and 2."
            )

            return result

        # ----------------------------------------------------
        # Check that wafer contains die information
        # ----------------------------------------------------

        die_pixels = np.sum(
            array > 0
        )

        if die_pixels == 0:

            result["message"] = (
                "Invalid wafer map. "
                "No die information was detected."
            )

            return result

        # ----------------------------------------------------
        # Valid
        # ----------------------------------------------------

        result["valid"] = True

        result["message"] = (
            "Valid wafer map."
        )

        return result

    # ========================================================
    # PREPROCESSING
    # ========================================================

    @staticmethod
    def preprocess(
        wafer_matrix: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert raw wafer matrix into model input.

        Output:
            Tensor shape = (1, 1, IMAGE_SIZE, IMAGE_SIZE)
        """

        tensor = preprocess_wafer_matrix(
            wafer_matrix,
            target_size=IMAGE_SIZE,
        )

        # Dataset preprocessing returns:
        # (1, H, W)

        if tensor.ndim == 3:

            tensor = tensor.unsqueeze(0)

        elif tensor.ndim != 4:

            raise ValueError(
                "Unexpected preprocessed tensor shape: "
                f"{tensor.shape}"
            )

        return tensor.float()

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def classify(
        self,
        wafer_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Classify one wafer map.
        """

        validation = self.validate_wafer_matrix(
            wafer_matrix
        )

        if not validation["valid"]:

            raise ValueError(
                validation["message"]
            )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        tensor = self.preprocess(
            wafer_matrix
        )

        tensor = tensor.to(
            self.device
        )

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        predicted_index = int(
            prediction.item()
        )

        confidence_value = float(
            confidence.item()
        )

        predicted_class = DEFECT_CLASSES[
            predicted_index
        ]

        # ----------------------------------------------------
        # All class probabilities
        # ----------------------------------------------------

        class_probabilities = {}

        probability_array = (
            probabilities[0]
            .cpu()
            .numpy()
        )

        for index, probability in enumerate(
            probability_array
        ):

            class_name = DEFECT_CLASSES[
                index
            ]

            class_probabilities[
                class_name
            ] = float(
                probability
            )

        # ----------------------------------------------------
        # Top 3 predictions
        # ----------------------------------------------------

        top_k = min(
            3,
            NUM_CLASSES
        )

        top_probabilities, top_indices = (
            torch.topk(
                probabilities[0],
                k=top_k,
            )
        )

        top_predictions = []

        for probability, index in zip(
            top_probabilities.cpu().numpy(),
            top_indices.cpu().numpy(),
        ):

            top_predictions.append(
                {
                    "class": DEFECT_CLASSES[
                        int(index)
                    ],
                    "confidence": float(
                        probability
                    ),
                    "confidence_percent": float(
                        probability * 100
                    ),
                }
            )

        # ----------------------------------------------------
        # Confidence status
        # ----------------------------------------------------

        if (
            confidence_value
            < VERY_LOW_CONFIDENCE_THRESHOLD
        ):

            confidence_status = (
                "very_low"
            )

        elif (
            confidence_value
            < LOW_CONFIDENCE_THRESHOLD
        ):

            confidence_status = (
                "low"
            )

        else:

            confidence_status = (
                "high"
            )

        return {
            "predicted_class": predicted_class,
            "predicted_index": predicted_index,

            "confidence": confidence_value,

            "confidence_percent": (
                confidence_value * 100
            ),

            "confidence_status": (
                confidence_status
            ),

            "requires_human_review": (
                confidence_value
                < LOW_CONFIDENCE_THRESHOLD
            ),

            "class_probabilities": (
                class_probabilities
            ),

            "top_predictions": (
                top_predictions
            ),
        }

    # ========================================================
    # YIELD CALCULATION
    # ========================================================

    @staticmethod
    def calculate_yield(
        wafer_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate wafer-level die statistics.

        Encoding:
            0 = background
            1 = normal/pass
            2 = defective/fail
        """

        array = np.asarray(
            wafer_matrix
        )

        total_positions = int(
            array.size
        )

        normal_dies = int(
            np.sum(array == 1)
        )

        defective_dies = int(
            np.sum(array == 2)
        )

        total_dies = (
            normal_dies
            + defective_dies
        )

        background = int(
            np.sum(array == 0)
        )

        if total_dies > 0:

            yield_percent = (
                normal_dies
                / total_dies
                * 100
            )

            defect_percent = (
                defective_dies
                / total_dies
                * 100
            )

        else:

            yield_percent = 0.0
            defect_percent = 0.0

        return {
            "total_positions": (
                total_positions
            ),

            "background_positions": (
                background
            ),

            "total_dies": (
                total_dies
            ),

            "normal_dies": (
                normal_dies
            ),

            "defective_dies": (
                defective_dies
            ),

            "yield_percent": (
                float(yield_percent)
            ),

            "defect_percent": (
                float(defect_percent)
            ),
        }

    # ========================================================
    # SEVERITY ANALYSIS
    # ========================================================

    @staticmethod
    def calculate_severity(
        wafer_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Estimate defect severity from the percentage of
        defective dies.

        This is a project-defined severity indicator,
        not a medical/industrial standard.

        Categories:
            < 2%       → Low
            2–10%      → Moderate
            10–25%     → High
            > 25%      → Critical
        """

        yield_data = (
            WaferPredictor.calculate_yield(
                wafer_matrix
            )
        )

        defect_percent = (
            yield_data["defect_percent"]
        )

        if defect_percent < 2.0:

            severity = "Low"

        elif defect_percent < 10.0:

            severity = "Moderate"

        elif defect_percent < 25.0:

            severity = "High"

        else:

            severity = "Critical"

        return {
            "severity": severity,

            "severity_score": float(
                min(
                    defect_percent,
                    100.0,
                )
            ),

            "defect_percent": float(
                defect_percent
            ),

            "basis": (
                "Estimated from defective-die "
                "percentage in the wafer map."
            ),
        }

    # ========================================================
    # INSIGHT GENERATION
    # ========================================================

    @staticmethod
    def generate_insight(
        prediction: Dict[str, Any],
        yield_data: Dict[str, Any],
        severity_data: Dict[str, Any],
    ) -> str:
        """
        Generate a simple deterministic engineering insight.

        This is intentionally rule-based.
        It does not claim to be an LLM-generated diagnosis.
        """

        defect_class = (
            prediction["predicted_class"]
        )

        confidence = (
            prediction["confidence_percent"]
        )

        yield_percent = (
            yield_data["yield_percent"]
        )

        severity = (
            severity_data["severity"]
        )

        if defect_class == "none":

            insight = (
                f"The wafer is classified as Normal "
                f"with {confidence:.1f}% model confidence. "
                f"Estimated die yield is "
                f"{yield_percent:.1f}%."
            )

        else:

            insight = (
                f"The wafer is classified as "
                f"{defect_class} with "
                f"{confidence:.1f}% model confidence. "
                f"Estimated die yield is "
                f"{yield_percent:.1f}% and the "
                f"defect severity indicator is "
                f"{severity}."
            )

        if prediction[
            "requires_human_review"
        ]:

            insight += (
                " Model confidence is below the "
                "review threshold, so human "
                "engineering review is recommended."
            )

        return insight

    # ========================================================
    # ENGINEERING RECOMMENDATION
    # ========================================================

    @staticmethod
    def generate_recommendation(
        prediction: Dict[str, Any],
        severity_data: Dict[str, Any],
    ) -> str:
        """
        Generate a project-level next-step recommendation.

        These are engineering investigation suggestions,
        not claims about the physical root cause.
        """

        defect_class = (
            prediction["predicted_class"]
        )

        severity = (
            severity_data["severity"]
        )

        if (
            prediction[
                "requires_human_review"
            ]
        ):

            return (
                "Review the wafer manually and "
                "compare it with similar historical "
                "wafer maps before making process decisions."
            )

        if defect_class == "none":

            return (
                "Continue routine monitoring and "
                "compare the wafer with historical "
                "lot-level yield statistics."
            )

        recommendations = {

            "Center": (
                "Inspect center-region process conditions "
                "and compare against neighboring wafers "
                "from the same lot."
            ),

            "Donut": (
                "Investigate annular/ring-shaped spatial "
                "patterns and compare process conditions "
                "across the affected radial region."
            ),

            "Edge-Loc": (
                "Inspect edge-region process behavior and "
                "compare edge-related patterns across the lot."
            ),

            "Edge-Ring": (
                "Investigate radial edge effects and "
                "compare the affected ring region with "
                "historical wafer maps."
            ),

            "Loc": (
                "Inspect the localized defect region and "
                "compare its spatial position across "
                "neighboring wafers."
            ),

            "Near-full": (
                "Perform detailed review because the "
                "defective region covers a large portion "
                "of the wafer."
            ),

            "Random": (
                "Investigate possible random/distributed "
                "defect mechanisms and compare with "
                "historical lot statistics."
            ),

            "Scratch": (
                "Inspect wafer handling and process history "
                "for possible directional defect patterns."
            ),
        }

        recommendation = recommendations.get(
            defect_class,
            (
                "Perform engineering review and compare "
                "the wafer with similar historical samples."
            ),
        )

        if severity in {
            "High",
            "Critical",
        }:

            recommendation += (
                " Due to the elevated severity indicator, "
                "prioritize engineering review."
            )

        return recommendation

    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    def analyze(
        self,
        wafer_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end analysis.

        Workflow:
            Validate
            ↓
            Classify
            ↓
            Yield
            ↓
            Severity
            ↓
            Insight
            ↓
            Recommendation
        """

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation = (
            self.validate_wafer_matrix(
                wafer_matrix
            )
        )

        if not validation["valid"]:

            return {
                "success": False,
                "validation": validation,
            }

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        prediction = self.classify(
            wafer_matrix
        )

        # ----------------------------------------------------
        # Yield
        # ----------------------------------------------------

        yield_data = (
            self.calculate_yield(
                wafer_matrix
            )
        )

        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        severity_data = (
            self.calculate_severity(
                wafer_matrix
            )
        )

        # ----------------------------------------------------
        # Insight
        # ----------------------------------------------------

        insight = (
            self.generate_insight(
                prediction,
                yield_data,
                severity_data,
            )
        )

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        recommendation = (
            self.generate_recommendation(
                prediction,
                severity_data,
            )
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return {
            "success": True,

            "validation": validation,

            "prediction": prediction,

            "yield": yield_data,

            "severity": severity_data,

            "insight": insight,

            "recommendation": recommendation,

            "model": str(
                self.model_path
            ),
        }


# ============================================================
# SIMPLE TEST / COMMAND LINE USAGE
# ============================================================

def main():
    """
    Basic smoke test using a small synthetic wafer map.

    This does NOT retrain the model.
    """

    print("=" * 70)
    print("WAFER PREDICTION ENGINE")
    print("=" * 70)

    print(
        f"\nModel: {MODEL_PATH}"
    )

    predictor = WaferPredictor()

    # --------------------------------------------------------
    # Create a simple valid wafer-shaped test input.
    #
    # This is only a software pipeline test.
    # It is NOT used as a scientific evaluation sample.
    # --------------------------------------------------------

    wafer = np.zeros(
        (56, 56),
        dtype=np.uint8,
    )

    # Normal die region
    wafer[10:46, 10:46] = 1

    # Small defective region
    wafer[25:30, 25:30] = 2

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    result = predictor.analyze(
        wafer
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    if not result["success"]:

        print(
            "\nValidation failed:"
        )

        print(
            result["validation"]
        )

        return

    prediction = (
        result["prediction"]
    )

    yield_data = (
        result["yield"]
    )

    severity = (
        result["severity"]
    )

    print(
        "\nPrediction:"
    )

    print(
        f"  Class: "
        f"{prediction['predicted_class']}"
    )

    print(
        f"  Confidence: "
        f"{prediction['confidence_percent']:.2f}%"
    )

    print(
        f"  Review required: "
        f"{prediction['requires_human_review']}"
    )

    print(
        "\nYield:"
    )

    print(
        f"  Total dies: "
        f"{yield_data['total_dies']:,}"
    )

    print(
        f"  Normal dies: "
        f"{yield_data['normal_dies']:,}"
    )

    print(
        f"  Defective dies: "
        f"{yield_data['defective_dies']:,}"
    )

    print(
        f"  Yield: "
        f"{yield_data['yield_percent']:.2f}%"
    )

    print(
        f"  Defect: "
        f"{yield_data['defect_percent']:.2f}%"
    )

    print(
        "\nSeverity:"
    )

    print(
        f"  {severity['severity']}"
    )

    print(
        "\nAI Insight:"
    )

    print(
        f"  {result['insight']}"
    )

    print(
        "\nEngineering Recommendation:"
    )

    print(
        f"  {result['recommendation']}"
    )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()