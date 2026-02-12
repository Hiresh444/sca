"""
app.py — Streamlit Deployment Application
===========================================

Interactive web application for skin lesion classification with:
  - Image upload and preprocessing visualization
  - Spectral Skin Decomposition (SSD) channel display
  - 7-class prediction with confidence scores
  - GradCAM visual explanation overlay
  - Prototype similarity visualization (PAEC)
  - Uncertainty-based risk level assessment
  - Medical disclaimer

Usage:
  streamlit run app.py -- --checkpoint checkpoints/best_model.pth

Authors: [Your Name]
License: MIT
"""

import sys
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image

# Append project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import (
    SpectralSkinDecomposition, CLASS_NAMES, NUM_CLASSES,
    IMAGENET_MEAN, IMAGENET_STD, FULL_MEAN, FULL_STD,
)
from models import SkinLesionClassifier, GradCAM


# ---------------------------------------------------------------------------
#  Class metadata for display
# ---------------------------------------------------------------------------
CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesion",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus (Mole)",
    "vasc":  "Vascular Lesion",
}

CLASS_RISK = {
    "akiec": ("High", "#FF4444"),
    "bcc":   ("High", "#FF4444"),
    "bkl":   ("Low", "#44BB44"),
    "df":    ("Low", "#44BB44"),
    "mel":   ("Critical", "#FF0000"),
    "nv":    ("Low", "#44BB44"),
    "vasc":  ("Low", "#44BB44"),
}


# ---------------------------------------------------------------------------
#  Model Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(checkpoint_path: str):
    """Load model from checkpoint (cached across Streamlit reruns)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    use_ssd = config.get("use_ssd", True)
    use_saa = config.get("use_saa", True)
    use_paec = config.get("use_paec", True)
    img_size = config.get("img_size", 300)

    model = SkinLesionClassifier(
        num_classes=config.get("num_classes", NUM_CLASSES),
        use_ssd=use_ssd, use_saa=use_saa, use_paec=use_paec,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, device, {"use_ssd": use_ssd, "use_saa": use_saa,
                            "use_paec": use_paec, "img_size": img_size}


def preprocess_image(image_rgb: np.ndarray, config: dict) -> torch.Tensor:
    """
    Preprocess a single RGB image for model inference.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        config: Dict with 'use_ssd' and 'img_size' keys.

    Returns:
        (1, C, H, W) normalized tensor ready for the model.
    """
    img_size = config.get("img_size", 300)
    use_ssd = config.get("use_ssd", True)

    # Resize
    image_resized = cv2.resize(image_rgb, (img_size, img_size),
                                interpolation=cv2.INTER_LINEAR)

    if use_ssd:
        ssd = SpectralSkinDecomposition()
        image_7ch = ssd(image_resized)  # (H, W, 7) float32 [0, 1]
        mean = np.array(FULL_MEAN, dtype=np.float32)
        std = np.array(FULL_STD, dtype=np.float32)
    else:
        image_7ch = image_resized.astype(np.float32) / 255.0  # (H, W, 3)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)

    # Normalize
    image_norm = (image_7ch - mean) / std

    # To tensor: (H, W, C) → (C, H, W) → (1, C, H, W)
    tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


# ---------------------------------------------------------------------------
#  Streamlit Application
# ---------------------------------------------------------------------------

def main():
    # ---- Page config ----
    st.set_page_config(
        page_title="Skin Lesion Classifier",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Skin Lesion Classification System")
    st.markdown(
        "EfficientNet-B3 with Spectral Skin Decomposition (SSD), "
        "Selective Amplification Activation (SAA), and "
        "Prototype-Anchored Evidential Classification (PAEC)"
    )

    # ---- Sidebar ----
    st.sidebar.header("Settings")

    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint Path",
        value="checkpoints/best_model.pth",
        help="Path to the trained model checkpoint (.pth file)",
    )

    show_ssd = st.sidebar.checkbox("Show SSD Channels", value=True,
                                    help="Display the 4 diagnostic channels from SSD")
    show_gradcam = st.sidebar.checkbox("Show GradCAM Overlay", value=True,
                                        help="Show attention heatmap on the image")
    show_prototypes = st.sidebar.checkbox("Show Prototype Distances", value=True,
                                          help="Show similarity to class prototypes (PAEC)")

    # ---- Load model ----
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.info("Train a model first using: `python train.py --csv_path <path> --img_dir <path>`")
        return

    try:
        model, device, config = load_model(checkpoint_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    st.sidebar.success(
        f"Model loaded. SSD: {config['use_ssd']}, "
        f"SAA: {config['use_saa']}, PAEC: {config['use_paec']}"
    )

    # ---- Image upload ----
    uploaded_file = st.file_uploader(
        "Upload a dermoscopy image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a skin lesion image for classification",
    )

    if uploaded_file is not None:
        # Load and display original image
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image_pil)

        # ---- Row 1: Original + SSD Channels ----
        st.subheader("Input Analysis")

        if show_ssd and config["use_ssd"]:
            cols = st.columns(5)
            cols[0].image(image_rgb, caption="Original Image", use_container_width=True)

            # Compute SSD channels for visualization
            ssd = SpectralSkinDecomposition()
            img_resized = cv2.resize(image_rgb, (config["img_size"], config["img_size"]))
            ssd_image = ssd(img_resized)  # (H, W, 7)

            channel_names = ["Melanin Density", "Erythema (Hemoglobin)",
                             "Boundary Enhancement", "Texture Entropy"]
            cmaps = ["hot", "Reds", "gray", "viridis"]

            for i in range(4):
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(ssd_image[:, :, 3 + i], cmap=cmaps[i])
                ax.axis("off")
                ax.set_title(channel_names[i], fontsize=8)
                plt.tight_layout()
                cols[i + 1].pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.image(image_rgb, caption="Original Image", width=400)

        # ---- Inference ----
        with st.spinner("Running classification..."):
            input_tensor = preprocess_image(image_rgb, config).to(device)

            with torch.no_grad():
                output = model(input_tensor)

            probs = output["probs"][0].cpu().numpy()  # (K,)
            pred_idx = int(probs.argmax())
            pred_class = CLASS_NAMES[pred_idx]
            pred_conf = float(probs[pred_idx])

        # ---- Row 2: Prediction Results ----
        st.subheader("Classification Results")

        col_pred, col_conf, col_risk = st.columns(3)

        with col_pred:
            st.metric("Predicted Class", pred_class.upper())
            st.caption(CLASS_FULL_NAMES[pred_class])

        with col_conf:
            st.metric("Confidence", f"{pred_conf * 100:.1f}%")

        with col_risk:
            risk_level, risk_color = CLASS_RISK[pred_class]
            if config["use_paec"] and "uncertainty" in output:
                uncertainty = float(output["uncertainty"][0].cpu())
                st.metric("Uncertainty", f"{uncertainty:.3f}")
                if uncertainty > 0.5:
                    risk_level = "Uncertain"
                    risk_color = "#FFaa00"
            st.markdown(
                f"<h3 style='color: {risk_color};'>Risk: {risk_level}</h3>",
                unsafe_allow_html=True,
            )

        # ---- Class probability bar chart ----
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        colors = ["#FF4444" if CLASS_RISK[c][0] in ("High", "Critical")
                  else "#44BB44" for c in CLASS_NAMES]
        bars = ax.barh(CLASS_NAMES, probs, color=colors, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Class Probabilities")
        for bar, prob in zip(bars, probs):
            if prob > 0.01:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{prob:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ---- Row 3: GradCAM ----
        if show_gradcam:
            st.subheader("GradCAM Explanation")
            st.caption(
                "Highlighted regions show which areas of the image most influenced "
                "the model's prediction."
            )

            gradcam = GradCAM(model)
            input_grad = preprocess_image(image_rgb, config).to(device)
            heatmap = gradcam.generate(input_grad, target_class=pred_idx)

            # Resize original image to match model input
            img_display = cv2.resize(image_rgb, (config["img_size"], config["img_size"]))
            overlay = GradCAM.overlay(img_display, heatmap, alpha=0.4)

            col_g1, col_g2 = st.columns(2)
            col_g1.image(img_display, caption="Input Image", use_container_width=True)
            col_g2.image(overlay, caption=f"GradCAM for {pred_class.upper()}",
                        use_container_width=True)

        # ---- Row 4: Prototype Distances (PAEC) ----
        if show_prototypes and config["use_paec"] and "proto_distances" in output:
            st.subheader("Prototype Similarity Analysis")
            st.caption(
                "Shows how similar the input is to each class prototype in the "
                "learned embedding space. Lower distance = higher similarity."
            )

            distances = output["proto_distances"][0].cpu().numpy()  # (K,)
            # Convert to similarity (negative distance → higher = better)
            similarity = np.exp(-distances / (2 * distances.std() + 1e-8))
            similarity = similarity / similarity.sum()

            fig, ax = plt.subplots(figsize=(10, 3))
            bars = ax.barh(CLASS_NAMES, similarity, color="#5599DD", alpha=0.8)
            ax.set_xlim(0, max(similarity) * 1.2)
            ax.set_xlabel("Relative Similarity")
            ax.set_title("Prototype Similarity (PAEC)")
            for bar, sim in zip(bars, similarity):
                if sim > 0.01:
                    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                            f"{sim:.3f}", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ---- Medical Disclaimer ----
        st.markdown("---")
        st.warning(
            "**DISCLAIMER**: This tool is for **research and educational purposes only**. "
            "It is NOT a substitute for professional medical diagnosis. Skin lesion "
            "classification by AI can be inaccurate. Always consult a qualified "
            "dermatologist for medical advice. Do not use this tool to make medical "
            "decisions."
        )


if __name__ == "__main__":
    main()
