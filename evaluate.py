"""
evaluate.py — Comprehensive Evaluation, Visualization, and Ablation Study
===========================================================================

This module implements:
  1. comprehensive_evaluation: Full test-set evaluation with per-class metrics,
     confusion matrix, ROC curves, uncertainty analysis, and GradCAM grids.

  2. run_ablation_study: Evaluates all 8 combinations of the three novelties
     (SSD, SAA, PAEC) and generates an ablation results table.

Usage:
  # Evaluate a single model
  python evaluate.py --checkpoint checkpoints/best_model.pth \\
                     --csv_path data/HAM10000_metadata.csv \\
                     --img_dir data/

  # Run ablation study (requires checkpoints for each configuration)
  python evaluate.py --ablation --checkpoint_dir checkpoints/ \\
                     --csv_path data/HAM10000_metadata.csv \\
                     --img_dir data/

Authors: [Your Name]
License: MIT
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc,
)
from tqdm import tqdm

from preprocess import (
    create_dataloaders, CLASS_NAMES, NUM_CLASSES,
    IMAGENET_MEAN, IMAGENET_STD, FULL_MEAN, FULL_STD,
)
from models import SkinLesionClassifier, GradCAM


# ============================================================================
#  Comprehensive Evaluation
# ============================================================================

def comprehensive_evaluation(
    model: SkinLesionClassifier,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: str,
    use_paec: bool = True,
    use_ssd: bool = True,
) -> Dict[str, float]:
    """
    Run comprehensive evaluation on the test set and generate all artifacts.

    Produces:
      - Per-class precision, recall, F1 (printed + saved)
      - Overall accuracy, macro F1, weighted F1
      - Normalized confusion matrix heatmap (PNG)
      - Per-class ROC curves with AUC (PNG)
      - Uncertainty distribution plot (if PAEC active) (PNG)
      - GradCAM visualization grid: 2 samples per class (PNG)
      - All metrics saved to evaluation_results.csv

    Args:
        model: Trained SkinLesionClassifier in eval mode.
        test_loader: Test set DataLoader.
        device: Torch device.
        class_names: List of class name strings.
        output_dir: Directory to save output artifacts.
        use_paec: Whether PAEC head is active (for uncertainty analysis).
        use_ssd: Whether SSD channels are active (for denormalization).

    Returns:
        Dict of overall metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    all_uncertainties = []
    all_images = []  # Store a few images for GradCAM
    all_image_labels = []
    images_per_class = {i: [] for i in range(len(class_names))}

    print("\n[Eval] Running inference on test set...")
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Evaluating", leave=False):
            images_d = images.to(device, non_blocking=True)
            labels_d = labels.to(device, non_blocking=True)

            output = model(images_d)

            probs = output["probs"].cpu().numpy()    # (B, K)
            preds = probs.argmax(axis=1)             # (B,)
            targets_np = labels.numpy()

            all_preds.extend(preds)
            all_targets.extend(targets_np)
            all_probs.append(probs)

            if use_paec and "uncertainty" in output:
                all_uncertainties.extend(output["uncertainty"].cpu().numpy())

            # Collect sample images for GradCAM (2 per class)
            for i in range(len(targets_np)):
                cls = int(targets_np[i])
                if len(images_per_class[cls]) < 2:
                    images_per_class[cls].append(images[i])
                    all_image_labels.append(cls)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.concatenate(all_probs, axis=0)

    # ---- 1. Classification Report ----
    print("\n" + "=" * 70)
    print("  CLASSIFICATION REPORT")
    print("=" * 70)
    report = classification_report(
        all_targets, all_preds, target_names=class_names, digits=4, zero_division=0,
    )
    print(report)

    # Overall metrics
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    per_class_f1 = f1_score(
        all_targets, all_preds, average=None,
        labels=list(range(len(class_names))), zero_division=0,
    )
    per_class_prec = precision_score(
        all_targets, all_preds, average=None,
        labels=list(range(len(class_names))), zero_division=0,
    )
    per_class_rec = recall_score(
        all_targets, all_preds, average=None,
        labels=list(range(len(class_names))), zero_division=0,
    )

    print(f"  Overall Accuracy:  {accuracy:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Weighted F1:       {weighted_f1:.4f}")
    print(f"  MEL F1:            {per_class_f1[class_names.index('mel')]:.4f}")
    print("=" * 70)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        "class": class_names,
        "precision": per_class_prec,
        "recall": per_class_rec,
        "f1": per_class_f1,
    })
    metrics_df.loc[len(metrics_df)] = ["macro_avg",
                                        precision_score(all_targets, all_preds, average="macro", zero_division=0),
                                        recall_score(all_targets, all_preds, average="macro", zero_division=0),
                                        macro_f1]
    metrics_df.loc[len(metrics_df)] = ["accuracy", accuracy, accuracy, accuracy]
    metrics_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)

    # ---- 2. Confusion Matrix ----
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    cm_normalized = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] Confusion matrix saved to: {cm_path}")

    # ---- 3. Per-class ROC Curves ----
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls_name in enumerate(class_names):
        # One-vs-rest binary labels
        y_true_bin = (all_targets == i).astype(int)
        y_score = all_probs[:, i]

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{cls_name} (AUC = {roc_auc:.3f})", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-Class ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] ROC curves saved to: {roc_path}")

    # ---- 4. Uncertainty Distribution (PAEC only) ----
    if use_paec and len(all_uncertainties) > 0:
        all_uncertainties = np.array(all_uncertainties)
        correct_mask = (all_preds == all_targets)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 4a: Uncertainty for correct vs incorrect predictions
        axes[0].hist(all_uncertainties[correct_mask], bins=50, alpha=0.6,
                     label="Correct", color="green", density=True)
        axes[0].hist(all_uncertainties[~correct_mask], bins=50, alpha=0.6,
                     label="Incorrect", color="red", density=True)
        axes[0].set_xlabel("Epistemic Uncertainty", fontsize=12)
        axes[0].set_ylabel("Density", fontsize=12)
        axes[0].set_title("Uncertainty: Correct vs Incorrect", fontsize=13)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # 4b: Uncertainty per class
        for i, cls_name in enumerate(class_names):
            mask = (all_targets == i)
            if mask.any():
                axes[1].hist(all_uncertainties[mask], bins=30, alpha=0.5,
                             label=cls_name, density=True)
        axes[1].set_xlabel("Epistemic Uncertainty", fontsize=12)
        axes[1].set_ylabel("Density", fontsize=12)
        axes[1].set_title("Uncertainty per Class", fontsize=13)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        unc_path = os.path.join(output_dir, "uncertainty_distribution.png")
        fig.savefig(unc_path, dpi=150)
        plt.close(fig)
        print(f"[Eval] Uncertainty distribution saved to: {unc_path}")

    # ---- 5. GradCAM Visualization Grid ----
    _generate_gradcam_grid(model, images_per_class, class_names, device,
                           output_dir, use_ssd)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": dict(zip(class_names, per_class_f1)),
    }


def _denormalize_image(tensor: torch.Tensor, use_ssd: bool) -> np.ndarray:
    """
    Denormalize a tensor back to a displayable RGB image (uint8).

    Only uses the first 3 channels (RGB), ignoring SSD channels.
    """
    mean = FULL_MEAN[:3] if use_ssd else IMAGENET_MEAN
    std = FULL_STD[:3] if use_ssd else IMAGENET_STD

    img = tensor[:3].clone()  # Take only RGB channels
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _generate_gradcam_grid(
    model: SkinLesionClassifier,
    images_per_class: Dict[int, List[torch.Tensor]],
    class_names: List[str],
    device: torch.device,
    output_dir: str,
    use_ssd: bool,
):
    """Generate a grid of GradCAM visualizations, 2 samples per class."""
    gradcam = GradCAM(model)
    n_classes = len(class_names)
    max_samples = 2

    fig, axes = plt.subplots(n_classes, max_samples * 2, figsize=(16, 4 * n_classes))
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for cls_idx in range(n_classes):
        samples = images_per_class.get(cls_idx, [])
        for s_idx in range(max_samples):
            col_orig = s_idx * 2
            col_cam = s_idx * 2 + 1

            if s_idx < len(samples):
                img_tensor = samples[s_idx].unsqueeze(0).to(device)

                # Original image
                orig = _denormalize_image(samples[s_idx], use_ssd)
                axes[cls_idx, col_orig].imshow(orig)
                axes[cls_idx, col_orig].set_title(
                    f"{class_names[cls_idx]} (orig)", fontsize=9,
                )

                # GradCAM
                heatmap = gradcam.generate(img_tensor, target_class=cls_idx)
                overlay = GradCAM.overlay(orig, heatmap, alpha=0.4)
                axes[cls_idx, col_cam].imshow(overlay)
                axes[cls_idx, col_cam].set_title(
                    f"{class_names[cls_idx]} (GradCAM)", fontsize=9,
                )
            else:
                axes[cls_idx, col_orig].axis("off")
                axes[cls_idx, col_cam].axis("off")

            axes[cls_idx, col_orig].axis("off")
            axes[cls_idx, col_cam].axis("off")

    plt.suptitle("GradCAM Visualization Grid (2 samples per class)", fontsize=14, y=1.01)
    plt.tight_layout()
    gradcam_path = os.path.join(output_dir, "gradcam_grid.png")
    fig.savefig(gradcam_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] GradCAM grid saved to: {gradcam_path}")


# ============================================================================
#  Ablation Study Runner
# ============================================================================

def run_ablation_study(
    csv_path: str,
    img_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    batch_size: int = 32,
    img_size: int = 300,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Run ablation study across all 8 combinations of (SSD, SAA, PAEC).

    Expects checkpoint files named:
      - best_model_ssd{0|1}_saa{0|1}_paec{0|1}.pth
      OR a single best_model.pth for the full model.

    Falls back to evaluating only existing checkpoints.

    Generates an ablation results table (CSV + printed).

    Args:
        csv_path: Path to HAM10000_metadata.csv.
        img_dir: Root image directory.
        checkpoint_dir: Directory containing model checkpoints.
        output_dir: Directory to save ablation results.
        batch_size: Evaluation batch size.
        img_size: Image spatial resolution.
        num_workers: DataLoader workers.
        seed: Random seed.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 8 ablation configurations
    configurations = [
        {"use_ssd": False, "use_saa": False, "use_paec": False, "label": "Baseline (EfficientNet-B3)"},
        {"use_ssd": True,  "use_saa": False, "use_paec": False, "label": "+SSD only"},
        {"use_ssd": False, "use_saa": True,  "use_paec": False, "label": "+SAA only"},
        {"use_ssd": False, "use_saa": False, "use_paec": True,  "label": "+PAEC only"},
        {"use_ssd": True,  "use_saa": True,  "use_paec": False, "label": "+SSD +SAA"},
        {"use_ssd": True,  "use_saa": False, "use_paec": True,  "label": "+SSD +PAEC"},
        {"use_ssd": False, "use_saa": True,  "use_paec": True,  "label": "+SAA +PAEC"},
        {"use_ssd": True,  "use_saa": True,  "use_paec": True,  "label": "Full Model (SSD+SAA+PAEC)"},
    ]

    results = []

    for config in configurations:
        ssd_flag = int(config["use_ssd"])
        saa_flag = int(config["use_saa"])
        paec_flag = int(config["use_paec"])
        label = config["label"]

        # Look for checkpoint file
        ckpt_name = f"best_model_ssd{ssd_flag}_saa{saa_flag}_paec{paec_flag}.pth"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"[Ablation] Checkpoint not found for {label}: {ckpt_path} — skipping.")
            results.append({
                "Configuration": label,
                "SSD": ssd_flag, "SAA": saa_flag, "PAEC": paec_flag,
                "Accuracy": "-", "Macro-F1": "-", "Weighted-F1": "-",
                "MEL-F1": "-",
            })
            continue

        print(f"\n[Ablation] Evaluating: {label}")

        # Create dataloaders for this configuration
        _, _, test_loader, _, _ = create_dataloaders(
            csv_path=csv_path, img_dir=img_dir,
            batch_size=batch_size, img_size=img_size,
            use_ssd=config["use_ssd"], num_workers=num_workers, seed=seed,
        )

        # Load model
        model = SkinLesionClassifier(
            num_classes=NUM_CLASSES,
            use_ssd=config["use_ssd"],
            use_saa=config["use_saa"],
            use_paec=config["use_paec"],
            pretrained=False,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        # Evaluate
        metrics = comprehensive_evaluation(
            model, test_loader, device, CLASS_NAMES,
            output_dir=os.path.join(output_dir, f"ssd{ssd_flag}_saa{saa_flag}_paec{paec_flag}"),
            use_paec=config["use_paec"],
            use_ssd=config["use_ssd"],
        )

        results.append({
            "Configuration": label,
            "SSD": ssd_flag, "SAA": saa_flag, "PAEC": paec_flag,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Macro-F1": f"{metrics['macro_f1']:.4f}",
            "Weighted-F1": f"{metrics['weighted_f1']:.4f}",
            "MEL-F1": f"{metrics['per_class_f1'].get('mel', 0):.4f}",
        })

    # ---- Print and save ablation table ----
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("  ABLATION STUDY RESULTS")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    ablation_csv = os.path.join(output_dir, "ablation_results.csv")
    results_df.to_csv(ablation_csv, index=False)
    print(f"[Ablation] Results saved to: {ablation_csv}")


# ============================================================================
#  Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Skin Lesion Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to HAM10000_metadata.csv")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Root directory containing image folders")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth) for single evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save evaluation outputs")

    # Ablation mode
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation study across all 8 configurations")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing ablation checkpoints")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ablation:
        # ---- Ablation study ----
        run_ablation_study(
            csv_path=args.csv_path, img_dir=args.img_dir,
            checkpoint_dir=args.checkpoint_dir, output_dir=args.output_dir,
            batch_size=args.batch_size, img_size=args.img_size,
            num_workers=args.num_workers, seed=args.seed,
        )
    else:
        # ---- Single model evaluation ----
        if args.checkpoint is None:
            parser.error("--checkpoint is required for single-model evaluation")

        # Load checkpoint and extract config
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        use_ssd = config.get("use_ssd", True)
        use_saa = config.get("use_saa", True)
        use_paec = config.get("use_paec", True)

        print(f"[Eval] Loaded checkpoint: {args.checkpoint}")
        print(f"[Eval] Config — SSD: {use_ssd}, SAA: {use_saa}, PAEC: {use_paec}")

        # Create test dataloader
        _, _, test_loader, _, _ = create_dataloaders(
            csv_path=args.csv_path, img_dir=args.img_dir,
            batch_size=args.batch_size, img_size=args.img_size,
            use_ssd=use_ssd, num_workers=args.num_workers, seed=args.seed,
        )

        # Load model
        model = SkinLesionClassifier(
            num_classes=NUM_CLASSES,
            use_ssd=use_ssd, use_saa=use_saa, use_paec=use_paec,
            pretrained=False,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Run evaluation
        comprehensive_evaluation(
            model, test_loader, device, CLASS_NAMES, args.output_dir,
            use_paec=use_paec, use_ssd=use_ssd,
        )


if __name__ == "__main__":
    main()
