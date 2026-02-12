"""
train.py — Training Loop, Evidential Focal Loss, and Scheduler
================================================================

This module implements:
  1. EvidentialFocalLoss:
     Custom loss function for PAEC (Novelty 3) combining Type-II maximum
     likelihood for Dirichlet distributions, KL divergence regularization
     with epoch-based annealing, focal modulation, and prototype separation.
     Falls back to standard focal loss when PAEC is disabled.

  2. FocalLoss:
     Standard focal loss with class weighting for the non-PAEC baseline.

  3. CosineAnnealingWarmupLR:
     Learning rate scheduler with linear warmup followed by cosine decay.

  4. Training and validation loops with mixed-precision (AMP) support.

  5. Main entry point with full argument parsing for hyperparameters and
     ablation toggle flags (--no_ssd, --no_saa, --no_paec).

Authors: [Your Name]
License: MIT
"""

import argparse
import os
import csv
import time
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from preprocess import create_dataloaders, CLASS_NAMES, NUM_CLASSES
from models import SkinLesionClassifier


# ============================================================================
#  FocalLoss — Standard baseline loss (used when PAEC is disabled)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class weighting.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Downweights easy examples (high p_t) and focuses on hard examples.
    Class weights α_t address the severe class imbalance in HAM10000.

    Args:
        gamma: Focusing parameter (γ=0 recovers standard CE).
        class_weights: Per-class weight tensor. If None, uniform weights.
        label_smoothing: Label smoothing factor ε.
    """

    def __init__(self, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # Register class_weights as buffer so it moves with .to(device)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, K) raw logits.
            targets: (B,) integer class labels.

        Returns:
            Scalar loss.
        """
        num_classes = logits.size(1)

        # Apply label smoothing to one-hot targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Softmax probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal modulation: (1 - p_t)^γ
        focal_weight = (1.0 - probs) ** self.gamma

        # Class weighting
        if self.class_weights is not None:
            cw = self.class_weights[targets].unsqueeze(1)  # (B, 1)
            focal_weight = focal_weight * cw

        # Cross-entropy with focal weighting
        loss = -focal_weight * smooth_targets * log_probs
        return loss.sum(dim=1).mean()


# ============================================================================
#  EvidentialFocalLoss — Custom loss for PAEC (Novelty 3)
# ============================================================================

class EvidentialFocalLoss(nn.Module):
    """
    Evidential Focal Loss for Prototype-Anchored Evidential Classification.

    Combines four loss components:

    1. Type-II Maximum Likelihood Loss (Dirichlet-based):
       L_mll = Σ_k y_k · [log(S) - log(α_k)]
       where S = Σα_k (Dirichlet strength), y_k is one-hot target.
       This is the negative log-likelihood of the target under the
       expected categorical distribution from the Dirichlet.

    2. KL Divergence Regularizer (with annealing):
       L_kl = KL[Dir(ã) || Dir(1)]
       where ã = y + (1-y)·α (removes evidence for correct class).
       Penalizes overconfidence by pulling non-target evidence toward 0.
       Annealing: λ_kl starts at 0 and linearly increases to kl_weight_max
       over the first `annealing_epochs` epochs. This prevents early
       regularization from suppressing evidence before the model learns.

    3. Focal Modulation:
       Multiplies the MLL loss by (1 - p_t)^γ where p_t is the predicted
       probability for the true class. Focuses training on hard examples.

    4. Prototype Separation Loss:
       L_sep = mean_{i≠j} max(0, margin - ||p_i - p_j||)
       Pushes class prototypes apart in embedding space to improve class
       discriminability. Comes from the PAEC head module.

    Total loss:
       L = L_mll_focal + λ_kl(epoch) · L_kl + λ_sep · L_sep

    Args:
        num_classes: Number of classes.
        gamma: Focal loss focusing parameter.
        kl_weight_max: Maximum KL regularization weight (reached after annealing).
        annealing_epochs: Number of epochs over which KL weight linearly increases.
        class_weights: Per-class weight tensor for imbalance handling.
        proto_sep_weight: Weight for prototype separation loss.
    """

    def __init__(self, num_classes: int = 7, gamma: float = 2.0,
                 kl_weight_max: float = 0.1, annealing_epochs: int = 30,
                 class_weights: Optional[torch.Tensor] = None,
                 proto_sep_weight: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.kl_weight_max = kl_weight_max
        self.annealing_epochs = annealing_epochs
        self.proto_sep_weight = proto_sep_weight

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def _kl_weight(self, epoch: int) -> float:
        """Compute annealed KL weight for current epoch."""
        return min(1.0, epoch / max(1, self.annealing_epochs)) * self.kl_weight_max

    def _type2_mll(self, alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Type-II Maximum Likelihood Loss for Dirichlet.

        L = Σ_k y_k · [log(S) - log(α_k)]

        Args:
            alpha: (B, K) Dirichlet concentration parameters (all > 1).
            targets: (B,) integer class labels.

        Returns:
            (B,) per-sample loss.
        """
        S = alpha.sum(dim=-1, keepdim=True)  # (B, 1)
        one_hot = F.one_hot(targets, self.num_classes).float()  # (B, K)

        # log(S) - log(α_k) for target class k
        loss = (one_hot * (torch.log(S) - torch.log(alpha))).sum(dim=-1)  # (B,)
        return loss

    def _kl_divergence(self, alpha: torch.Tensor,
                       targets: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between posterior Dir(ã) and uniform prior Dir(1).

        ã = y + (1-y)·α removes evidence for the correct class, so the KL
        only penalizes spurious evidence for incorrect classes.

        Args:
            alpha: (B, K) Dirichlet concentration parameters.
            targets: (B,) integer class labels.

        Returns:
            (B,) per-sample KL divergence.
        """
        one_hot = F.one_hot(targets, self.num_classes).float()

        # Remove evidence for correct class
        alpha_tilde = one_hot + (1.0 - one_hot) * alpha  # (B, K)
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

        # KL(Dir(ã) || Dir(1)) in closed form
        kl = (
            torch.lgamma(S_tilde.squeeze(-1))
            - torch.lgamma(torch.tensor(float(self.num_classes), device=alpha.device))
            - torch.lgamma(alpha_tilde).sum(dim=-1)
            + ((alpha_tilde - 1.0) * (
                torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
            )).sum(dim=-1)
        )
        return kl

    def forward(self, output_dict: Dict[str, torch.Tensor],
                targets: torch.Tensor, epoch: int,
                head_module: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Compute total evidential focal loss.

        Args:
            output_dict: Model output dictionary (must contain 'alpha', 'probs').
            targets: (B,) integer class labels.
            epoch: Current training epoch (for KL annealing).
            head_module: PAEC head module (to compute prototype separation loss).

        Returns:
            Scalar total loss.
        """
        alpha = output_dict["alpha"]  # (B, K)
        probs = output_dict["probs"]  # (B, K)

        # 1. Type-II MLL
        mll_loss = self._type2_mll(alpha, targets)  # (B,)

        # 2. Focal modulation: weight by (1 - p_true)^γ
        p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        focal_weight = (1.0 - p_true) ** self.gamma

        # 3. Class weighting
        if self.class_weights is not None:
            cw = self.class_weights[targets]  # (B,)
            focal_weight = focal_weight * cw

        mll_focal = (focal_weight * mll_loss).mean()

        # 4. KL divergence with annealing
        kl_weight = self._kl_weight(epoch)
        kl_loss = self._kl_divergence(alpha, targets).mean()

        total = mll_focal + kl_weight * kl_loss

        # 5. Prototype separation loss (if PAEC head provided)
        if head_module is not None and hasattr(head_module, "prototype_separation_loss"):
            sep_loss = head_module.prototype_separation_loss()
            total = total + self.proto_sep_weight * sep_loss

        return total


# ============================================================================
#  Learning Rate Scheduler — Cosine Annealing with Warmup
# ============================================================================

class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    During warmup (epochs 0 to warmup_epochs-1):
      lr = base_lr * (epoch + 1) / warmup_epochs

    After warmup (epochs warmup_epochs to total_epochs):
      lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
      where progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)

    Args:
        optimizer: Optimizer instance.
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs.
        min_lr: Minimum learning rate at end of cosine decay.
    """

    def __init__(self, optimizer, total_epochs: int, warmup_epochs: int = 5,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


# ============================================================================
#  Training and Validation Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    use_paec: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Train the model for one epoch with mixed-precision.

    Args:
        model: SkinLesionClassifier instance.
        loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function (EvidentialFocalLoss or FocalLoss).
        device: Torch device.
        epoch: Current epoch number.
        scaler: GradScaler for AMP.
        use_paec: Whether PAEC is active (determines loss computation).

    Returns:
        (avg_loss, metrics_dict) where metrics_dict has 'accuracy' and 'macro_f1'.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch + 1}", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            output = model(images)

            if use_paec and isinstance(criterion, EvidentialFocalLoss):
                # Evidential focal loss needs alpha, probs, and head module
                # Cast alpha to float32 for numerical stability in log/lgamma
                output_fp32 = {
                    k: v.float() if isinstance(v, torch.Tensor) else v
                    for k, v in output.items()
                }
                head_module = model.head if hasattr(model, "head") else None
                loss = criterion(output_fp32, labels, epoch, head_module)
            else:
                # Standard focal loss
                loss = criterion(output["logits"], labels)

        # Mixed-precision backward pass
        scaler.scale(loss).backward()
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Collect predictions
        preds = output["probs"].argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, {"accuracy": accuracy, "macro_f1": macro_f1}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_paec: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model on the validation set.

    Args:
        model: SkinLesionClassifier instance.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device.
        epoch: Current epoch number.
        use_paec: Whether PAEC is active.

    Returns:
        (avg_loss, metrics_dict) with 'accuracy', 'macro_f1', 'per_class_f1'.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)

        if use_paec and isinstance(criterion, EvidentialFocalLoss):
            output_fp32 = {
                k: v.float() if isinstance(v, torch.Tensor) else v
                for k, v in output.items()
            }
            head_module = model.head if hasattr(model, "head") else None
            loss = criterion(output_fp32, labels, epoch, head_module)
        else:
            loss = criterion(output["logits"], labels)

        running_loss += loss.item() * images.size(0)

        preds = output["probs"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_targets, all_preds, average=None, labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    return avg_loss, {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(CLASS_NAMES, per_class_f1)),
    }


# ============================================================================
#  Main Training Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Skin Lesion Classifier on HAM10000",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data arguments ----
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to HAM10000_metadata.csv")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Root directory containing image folders")

    # ---- Training hyperparameters ----
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=300)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--lr_saa", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal loss gamma")
    parser.add_argument("--kl_weight_max", type=float, default=0.1,
                        help="Max KL regularization weight (PAEC)")
    parser.add_argument("--kl_annealing_epochs", type=int, default=30,
                        help="Epochs for KL weight annealing (PAEC)")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # ---- Ablation toggles ----
    parser.add_argument("--no_ssd", action="store_true",
                        help="Disable Spectral Skin Decomposition (Novelty 1)")
    parser.add_argument("--no_saa", action="store_true",
                        help="Disable Selective Amplification Activation (Novelty 2)")
    parser.add_argument("--no_paec", action="store_true",
                        help="Disable Prototype-Anchored Evidential Classification (Novelty 3)")

    # ---- Output ----
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints and logs")

    args = parser.parse_args()

    # ---- Derived flags ----
    use_ssd = not args.no_ssd
    use_saa = not args.no_saa
    use_paec = not args.no_paec

    print("=" * 70)
    print("  Skin Lesion Classifier — Training Configuration")
    print("=" * 70)
    print(f"  Novelty 1 (SSD):  {'ENABLED' if use_ssd else 'DISABLED'}")
    print(f"  Novelty 2 (SAA):  {'ENABLED' if use_saa else 'DISABLED'}")
    print(f"  Novelty 3 (PAEC): {'ENABLED' if use_paec else 'DISABLED'}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, ImgSize: {args.img_size}")
    print(f"  LR backbone: {args.lr_backbone}, head: {args.lr_head}, SAA: {args.lr_saa}")
    print("=" * 70)

    # ---- Seed for reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ---- Data ----
    train_loader, val_loader, test_loader, class_weights, class_names = (
        create_dataloaders(
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            use_ssd=use_ssd,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    )

    # ---- Model ----
    model = SkinLesionClassifier(
        num_classes=NUM_CLASSES,
        use_ssd=use_ssd,
        use_saa=use_saa,
        use_paec=use_paec,
        pretrained=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # ---- Optimizer with differential learning rates ----
    param_groups = []

    # Backbone parameters (lower LR for pretrained weights)
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and "alpha_raw" not in n
                       and "beta_raw" not in n and "gamma_raw" not in n
                       and p.requires_grad]
    param_groups.append({"params": backbone_params, "lr": args.lr_backbone})

    # SAA parameters (higher LR to learn activation quickly)
    saa_params = [p for n, p in model.named_parameters()
                  if any(k in n for k in ["alpha_raw", "beta_raw", "gamma_raw"])
                  and p.requires_grad]
    if saa_params:
        param_groups.append({"params": saa_params, "lr": args.lr_saa})

    # Head parameters (higher LR for randomly initialized layers)
    head_params = [p for n, p in model.named_parameters()
                   if "head" in n and p.requires_grad]
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # ---- Scheduler ----
    scheduler = CosineAnnealingWarmupLR(
        optimizer, total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, min_lr=1e-6,
    )

    # ---- Loss function ----
    cw = class_weights.to(device)
    if use_paec:
        criterion = EvidentialFocalLoss(
            num_classes=NUM_CLASSES,
            gamma=args.gamma,
            kl_weight_max=args.kl_weight_max,
            annealing_epochs=args.kl_annealing_epochs,
            class_weights=cw,
            proto_sep_weight=0.01,
        ).to(device)
    else:
        criterion = FocalLoss(
            gamma=args.gamma,
            class_weights=cw,
            label_smoothing=args.label_smoothing,
        ).to(device)

    # ---- Mixed precision scaler ----
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ---- Output directory ----
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.csv")

    # ---- Training loop ----
    best_val_f1 = 0.0
    patience_counter = 0

    # CSV log header
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "train_f1",
            "val_loss", "val_acc", "val_f1", "lr",
        ] + [f"val_f1_{c}" for c in CLASS_NAMES])

    print(f"\n[Train] Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler, use_paec,
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, use_paec,
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # ---- Logging ----
        per_class = val_metrics["per_class_f1"]
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} "
            f"| Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} "
            f"F1: {train_metrics['macro_f1']:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['macro_f1']:.4f} "
            f"| LR: {current_lr:.2e} | {elapsed:.1f}s"
        )
        print(
            f"         Per-class F1: "
            + " | ".join(f"{c}: {per_class[c]:.3f}" for c in CLASS_NAMES)
        )

        # Write to CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{train_loss:.4f}",
                f"{train_metrics['accuracy']:.4f}",
                f"{train_metrics['macro_f1']:.4f}",
                f"{val_loss:.4f}",
                f"{val_metrics['accuracy']:.4f}",
                f"{val_metrics['macro_f1']:.4f}",
                f"{current_lr:.6f}",
            ] + [f"{per_class[c]:.4f}" for c in CLASS_NAMES])

        # ---- Checkpointing ----
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "config": {
                "use_ssd": use_ssd,
                "use_saa": use_saa,
                "use_paec": use_paec,
                "img_size": args.img_size,
                "num_classes": NUM_CLASSES,
            },
        }

        # Save last checkpoint
        torch.save(checkpoint, os.path.join(args.output_dir, "last_model.pth"))

        # Save best checkpoint (by validation macro F1)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  >>> New best val F1: {best_val_f1:.4f} — model saved.")
            patience_counter = 0
        else:
            patience_counter += 1

        # ---- Early stopping ----
        if patience_counter >= args.patience:
            print(f"\n[Train] Early stopping triggered after {args.patience} epochs "
                  f"without improvement. Best val F1: {best_val_f1:.4f}")
            break

    print(f"\n[Train] Training complete. Best validation macro-F1: {best_val_f1:.4f}")
    print(f"[Train] Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"[Train] Training log saved to: {log_path}")

    # ---- Final evaluation on test set ----
    print("\n" + "=" * 70)
    print("  Final Evaluation on Test Set")
    print("=" * 70)

    # Load best model
    best_ckpt = torch.load(
        os.path.join(args.output_dir, "best_model.pth"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_metrics = validate(
        model, test_loader, criterion, device, epoch=args.epochs, use_paec=use_paec,
    )

    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro-F1:  {test_metrics['macro_f1']:.4f}")
    print(f"  Per-class F1:")
    for c in CLASS_NAMES:
        print(f"    {c:>6s}: {test_metrics['per_class_f1'][c]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
