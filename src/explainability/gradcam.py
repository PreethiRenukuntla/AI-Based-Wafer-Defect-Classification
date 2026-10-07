"""
Grad-CAM explainability for the final controlled WaferDefectResNet.

Purpose:
    Show which spatial regions influenced the model prediction.

Workflow:
    Wafer map
        ↓
    Preprocessing
        ↓
    Controlled ResNet
        ↓
    Prediction
        ↓
    Grad-CAM
        ↓
    Heatmap
        ↓
    Overlay
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    DEVICE,
    DEFECT_CLASSES,
    IMAGE_SIZE,
)

from src.data.dataset import preprocess_wafer_matrix


# ============================================================
# MODEL CHECKPOINT
# ============================================================

MODEL_PATH = Path(
    "models/wafer_resnet_controlled_best.pth"
)


# ============================================================
# GRAD-CAM
# ============================================================

class WaferGradCAM:
    """
    Grad-CAM implementation for WaferDefectResNet.

    The final convolutional layer is used as the target layer.
    """

    def __init__(
        self,
        model,
        target_layer=None,
    ):

        self.model = model
        self.model.eval()

        # ----------------------------------------------------
        # Automatically find target layer
        # ----------------------------------------------------

        if target_layer is None:

            target_layer = self._find_target_layer()

        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_handle = None
        self.backward_handle = None

        self._register_hooks()

    # ========================================================
    # FIND TARGET LAYER
    # ========================================================

    def _find_target_layer(self):
        """
        Find the last Conv2d layer in the model.

        This is safer than assuming a particular layer
        structure such as model.layer4.conv2.
        """

        last_conv = None

        for module in self.model.modules():

            if isinstance(
                module,
                torch.nn.Conv2d,
            ):

                last_conv = module

        if last_conv is None:

            raise RuntimeError(
                "No Conv2d layer was found in the model."
            )

        return last_conv

    # ========================================================
    # HOOKS
    # ========================================================

    def _register_hooks(self):

        def forward_hook(
            module,
            inputs,
            output,
        ):

            self.activations = output

        def backward_hook(
            module,
            grad_input,
            grad_output,
        ):

            self.gradients = grad_output[0]

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                backward_hook
            )
        )

    # ========================================================
    # REMOVE HOOKS
    # ========================================================

    def remove_hooks(self):

        if self.forward_handle is not None:

            self.forward_handle.remove()

            self.forward_handle = None

        if self.backward_handle is not None:

            self.backward_handle.remove()

            self.backward_handle = None

    # ========================================================
    # GENERATE HEATMAP
    # ========================================================

    def generate_heatmap(
        self,
        input_tensor,
        target_class_idx=None,
    ):
        """
        Generate a Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor:
            Tensor of shape (1, 1, H, W)

        target_class_idx:
            Class to explain.
            If None, explains predicted class.

        Returns
        -------
        heatmap:
            NumPy array [H, W], values [0, 1]

        predicted_class:
            Predicted defect class

        confidence:
            Confidence percentage

        probabilities:
            All class probabilities
        """

        # ----------------------------------------------------
        # Ensure correct shape
        # ----------------------------------------------------

        if input_tensor.ndim == 3:

            input_tensor = (
                input_tensor.unsqueeze(0)
            )

        if input_tensor.ndim != 4:

            raise ValueError(
                "Expected input tensor with shape "
                "(1, 1, H, W). "
                f"Received: {input_tensor.shape}"
            )

        input_tensor = input_tensor.to(
            DEVICE
        )

        input_tensor = input_tensor.clone()

        input_tensor.requires_grad_(True)

        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        self.model.zero_grad(
            set_to_none=True
        )

        self.gradients = None
        self.activations = None

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = self.model(
            input_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1,
        )

        # ----------------------------------------------------
        # Select class
        # ----------------------------------------------------

        predicted_index = int(
            outputs.argmax(
                dim=1
            ).item()
        )

        if target_class_idx is None:

            target_class_idx = (
                predicted_index
            )

        target_class_idx = int(
            target_class_idx
        )

        if not (
            0
            <= target_class_idx
            < len(DEFECT_CLASSES)
        ):

            raise ValueError(
                "Invalid target class index: "
                f"{target_class_idx}"
            )

        predicted_class = (
            DEFECT_CLASSES[
                predicted_index
            ]
        )

        confidence = float(
            probabilities[
                0,
                predicted_index,
            ].item()
            * 100.0
        )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        target_score = outputs[
            0,
            target_class_idx,
        ]

        target_score.backward()

        # ----------------------------------------------------
        # Verify hooks
        # ----------------------------------------------------

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ----------------------------------------------------
        # Grad-CAM weights
        # ----------------------------------------------------

        weights = torch.mean(
            self.gradients,
            dim=(2, 3),
            keepdim=True,
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = torch.sum(
            weights
            * self.activations,
            dim=1,
            keepdim=True,
        )

        # Only positive influence
        cam = F.relu(
            cam
        )

        cam = cam.squeeze()

        cam = cam.detach().cpu().numpy()

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam_min = float(
            np.min(cam)
        )

        cam_max = float(
            np.max(cam)
        )

        if cam_max > cam_min:

            cam = (
                cam - cam_min
            ) / (
                cam_max
                - cam_min
                + 1e-8
            )

        else:

            cam = np.zeros_like(
                cam,
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # Resize to model input size
        # ----------------------------------------------------

        target_h = int(
            input_tensor.shape[2]
        )

        target_w = int(
            input_tensor.shape[3]
        )

        cam = cv2.resize(
            cam,
            (
                target_w,
                target_h,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        cam = np.clip(
            cam,
            0.0,
            1.0,
        )

        probabilities_np = (
            probabilities[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        return (
            cam,
            predicted_class,
            confidence,
            probabilities_np,
        )


# ============================================================
# CREATE ORIGINAL RGB IMAGE
# ============================================================

def wafer_matrix_to_rgb(
    wafer_matrix,
):
    """
    Convert WM-811K encoded wafer map into RGB.

    Encoding:
        0 = background
        1 = normal die
        2 = defective die
    """

    wafer_matrix = np.asarray(
        wafer_matrix
    )

    if wafer_matrix.ndim != 2:

        raise ValueError(
            "Wafer matrix must be 2D."
        )

    height, width = (
        wafer_matrix.shape
    )

    rgb = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    # Background
    rgb[
        wafer_matrix == 0
    ] = [
        20,
        20,
        20,
    ]

    # Normal die
    rgb[
        wafer_matrix == 1
    ] = [
        180,
        180,
        180,
    ]

    # Defective die
    rgb[
        wafer_matrix == 2
    ] = [
        255,
        60,
        60,
    ]

    return rgb


# ============================================================
# GENERATE GRAD-CAM OVERLAY
# ============================================================

def generate_gradcam_overlay(
    wafer_matrix,
    model,
    target_class_idx=None,
):
    """
    Generate:

        original RGB
        heatmap RGB
        overlay RGB
        predicted class
        confidence
        probabilities

    """

    wafer_matrix = np.asarray(
        wafer_matrix
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if wafer_matrix.ndim != 2:

        raise ValueError(
            "Wafer matrix must be 2D."
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    input_tensor = (
        preprocess_wafer_matrix(
            wafer_matrix,
            target_size=IMAGE_SIZE,
        )
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gradcam = WaferGradCAM(
        model
    )

    try:

        (
            cam_heatmap,
            predicted_class,
            confidence,
            probabilities,
        ) = gradcam.generate_heatmap(
            input_tensor,
            target_class_idx,
        )

    finally:

        gradcam.remove_hooks()

    # --------------------------------------------------------
    # Original wafer image
    # --------------------------------------------------------

    original_rgb = (
        wafer_matrix_to_rgb(
            wafer_matrix
        )
    )

    original_height, original_width = (
        wafer_matrix.shape
    )

    # --------------------------------------------------------
    # Resize heatmap to original
    # wafer dimensions
    # --------------------------------------------------------

    heatmap_resized = cv2.resize(
        cam_heatmap,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap_resized = np.clip(
        heatmap_resized,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Convert heatmap to uint8
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        heatmap_resized * 255
    )

    # --------------------------------------------------------
    # Apply OpenCV color map
    # --------------------------------------------------------

    heatmap_bgr = (
        cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )
    )

    heatmap_rgb = (
        cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB,
        )
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay_rgb = cv2.addWeighted(
        original_rgb,
        0.50,
        heatmap_rgb,
        0.50,
        0,
    )

    return (
        original_rgb,
        heatmap_rgb,
        overlay_rgb,
        predicted_class,
        confidence,
        probabilities,
    )


# ============================================================
# SAVE GRAD-CAM IMAGES
# ============================================================

def save_gradcam_images(
    original_rgb,
    heatmap_rgb,
    overlay_rgb,
    output_dir="outputs/gradcam",
    prefix="wafer",
):
    """
    Save Grad-CAM visualizations.
    """

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_path = (
        output_path
        / f"{prefix}_original.png"
    )

    heatmap_path = (
        output_path
        / f"{prefix}_heatmap.png"
    )

    overlay_path = (
        output_path
        / f"{prefix}_overlay.png"
    )

    cv2.imwrite(
        str(original_path),
        cv2.cvtColor(
            original_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    cv2.imwrite(
        str(heatmap_path),
        cv2.cvtColor(
            heatmap_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(
            overlay_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    return {
        "original": str(
            original_path
        ),

        "heatmap": str(
            heatmap_path
        ),

        "overlay": str(
            overlay_path
        ),
    }


# ============================================================
# LOAD FINAL MODEL
# ============================================================

def load_final_model():
    """
    Load the final controlled ResNet.
    """

    from src.models.architecture import (
        WaferDefectResNet,
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Final controlled model not found:\n"
            f"{MODEL_PATH}"
        )

    model = WaferDefectResNet(
        num_classes=len(
            DEFECT_CLASSES
        )
    )

    model = model.to(
        DEVICE
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main():

    print("=" * 70)
    print("GRAD-CAM EXPLAINABILITY TEST")
    print("=" * 70)

    print(
        f"\nModel: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Load final model
    # --------------------------------------------------------

    model = load_final_model()

    print(
        "Final controlled ResNet loaded."
    )

    # --------------------------------------------------------
    # Synthetic smoke-test wafer
    #
    # This only verifies the Grad-CAM pipeline.
    # It is NOT a scientific evaluation sample.
    # --------------------------------------------------------

    wafer = np.zeros(
        (56, 56),
        dtype=np.uint8,
    )

    wafer[10:46, 10:46] = 1

    wafer[24:31, 24:31] = 2

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    print(
        "\nGenerating Grad-CAM..."
    )

    (
        original_rgb,
        heatmap_rgb,
        overlay_rgb,
        predicted_class,
        confidence,
        probabilities,
    ) = generate_gradcam_overlay(
        wafer,
        model,
    )

    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    paths = save_gradcam_images(
        original_rgb,
        heatmap_rgb,
        overlay_rgb,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\nPrediction:"
    )

    print(
        f"  Class: {predicted_class}"
    )

    print(
        f"  Confidence: {confidence:.2f}%"
    )

    print(
        "\nSaved Grad-CAM files:"
    )

    print(
        f"  Original: {paths['original']}"
    )

    print(
        f"  Heatmap:  {paths['heatmap']}"
    )

    print(
        f"  Overlay:  {paths['overlay']}"
    )

    print(
        "\nGrad-CAM test completed."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()