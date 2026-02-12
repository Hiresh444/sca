"""
models.py — Model Architecture, Novel Activation, Evidential Head, and GradCAM
================================================================================

This module implements:
  1. NOVELTY 2 — Selective Amplification Activation (SAA):
     A novel per-channel learnable activation function that blends a Swish-like
     pathway (strong feature amplification) with a tanh-bounded pathway (subtle
     feature preservation) via a learned gating parameter.

  2. NOVELTY 3 — Prototype-Anchored Evidential Classification (PAEC):
     A dual-pathway classification head that combines prototype-distance evidence
     with FC-based evidence, fusing them into a Dirichlet distribution for
     calibrated uncertainty estimation.

  3. SkinLesionClassifier:
     EfficientNet-B3 backbone with configurable novelties (SSD stem, SAA
     activations, PAEC head), each independently toggleable for ablation.

  4. GradCAM:
     Gradient-weighted Class Activation Mapping for visual explanations.

Authors: [Your Name]
License: MIT
"""

import math
from typing import Dict, Optional, List

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ============================================================================
#  NOVELTY 2 — Selective Amplification Activation (SAA)
# ============================================================================

class SelectiveAmplificationActivation(nn.Module):
    """
    Selective Amplification Activation (SAA) — Activation Function Novelty
    -----------------------------------------------------------------------
    A novel activation function with three learnable per-channel parameters
    that blends two complementary non-linear pathways:

      SAA(x) = α · x · σ(β · x) + (1 - α) · x · tanh(γ · x)

    Where:
      - α ∈ (0, 1): Learned gate that blends the two pathways.
        Parameterized as sigmoid(α_raw) to ensure it stays in (0, 1).
      - β > 0: Controls the Swish-like pathway slope.
        Parameterized as softplus(β_raw) to ensure positivity.
      - γ > 0: Controls the tanh-bounded pathway slope.
        Parameterized as softplus(γ_raw) to ensure positivity.

    Pathway 1 — Swish-like: x · σ(βx)
      - Unbounded for large positive x → amplifies strong, confident features.
      - Smooth gating preserves gradient flow.
      - Dominant when α → 1.

    Pathway 2 — Tanh-bounded: x · tanh(γx)
      - Output bounded in magnitude → preserves subtle, low-magnitude features
        that might otherwise be suppressed.
      - Particularly useful for minority-class features that produce weaker
        activations in early training.
      - Dominant when α → 0.

    The network learns per-channel α values, allowing it to decide which
    features benefit from amplification vs. preservation. This is especially
    valuable for imbalanced medical image classification, where minority-class
    features are initially weak and easily suppressed by unbounded activations.

    Initialization:
      α_raw=0 → α=0.5 (equal blend), β_raw≈0.54 → β=1.0, γ_raw≈0.54 → γ=1.0
      At init, SAA ≈ 0.5 * SiLU(x) + 0.5 * x * tanh(x), a smooth interpolation.

    Parameter count:
      3 parameters per channel — negligible overhead (e.g., 3 × 1536 = 4608
      extra params for the entire EfficientNet-B3 backbone).

    Ablation:
      Set use_saa=False to revert to standard SiLU activation.
    """

    def __init__(self, num_channels: int):
        """
        Args:
            num_channels: Number of channels in the layer this activation serves.
                          Each channel gets its own (α, β, γ) parameters.
        """
        super().__init__()
        self.num_channels = num_channels

        # Raw (unconstrained) learnable parameters
        # α_raw: initialized to 0 → sigmoid(0) = 0.5 (equal blend)
        self.alpha_raw = nn.Parameter(torch.zeros(num_channels))

        # β_raw: initialized so that softplus(β_raw) ≈ 1.0
        # softplus(x) = log(1 + exp(x)); softplus(0.541) ≈ 1.0
        self.beta_raw = nn.Parameter(torch.full((num_channels,), 0.541))

        # γ_raw: same initialization as β_raw
        self.gamma_raw = nn.Parameter(torch.full((num_channels,), 0.541))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SAA activation.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C).

        Returns:
            Activated tensor of same shape.
        """
        # Constrain parameters to valid ranges
        alpha = torch.sigmoid(self.alpha_raw)        # (C,) in (0, 1)
        beta = F.softplus(self.beta_raw)             # (C,) in (0, ∞)
        gamma = F.softplus(self.gamma_raw)            # (C,) in (0, ∞)

        # Reshape for broadcasting: (C,) → (1, C, 1, 1) for 4D or (1, C) for 2D
        if x.dim() == 4:
            alpha = alpha.view(1, -1, 1, 1)
            beta = beta.view(1, -1, 1, 1)
            gamma = gamma.view(1, -1, 1, 1)
        elif x.dim() == 2:
            alpha = alpha.view(1, -1)
            beta = beta.view(1, -1)
            gamma = gamma.view(1, -1)
        else:
            # For 3D or other shapes, reshape to broadcast over dim=1
            shape = [1] * x.dim()
            shape[1] = -1
            alpha = alpha.view(shape)
            beta = beta.view(shape)
            gamma = gamma.view(shape)

        # Pathway 1: Swish-like — amplifies strong features
        swish_path = x * torch.sigmoid(beta * x)

        # Pathway 2: Tanh-bounded — preserves subtle features
        tanh_path = x * torch.tanh(gamma * x)

        # Blend pathways via learned gate α
        return alpha * swish_path + (1.0 - alpha) * tanh_path

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}"


def _get_num_channels(module: nn.Module) -> Optional[int]:
    """
    Infer the number of output channels for a module that precedes an activation.

    Walks common patterns: Conv2d, BatchNorm2d, etc.
    Returns None if channel count cannot be determined.
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return module.out_channels
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        return module.num_features if hasattr(module, 'num_features') else module.num_channels
    elif isinstance(module, nn.Linear):
        return module.out_features
    return None


def replace_activations(model: nn.Module, parent_name: str = "") -> nn.Module:
    """
    Recursively replace all SiLU / Swish activations in a model with SAA.

    For each SiLU found, we look at the preceding layer in the parent's
    children to infer the channel count. If the preceding sibling doesn't
    provide channel info, we also search all siblings for any BatchNorm or
    Conv2d to infer the channel width.

    Args:
        model: The nn.Module to modify in-place.
        parent_name: Used for recursive naming (internal).

    Returns:
        The modified model (same object, modified in-place).
    """
    children = list(model.named_children())
    prev_module = None

    for i, (name, child) in enumerate(children):
        if isinstance(child, nn.SiLU):
            # Determine channel count from preceding layer
            num_ch = None
            if prev_module is not None:
                num_ch = _get_num_channels(prev_module)

            # Fallback: scan all sibling modules for BatchNorm/Conv to infer channels
            if num_ch is None:
                for sib_name, sib_module in children:
                    if sib_name == name:
                        continue
                    inferred = _get_num_channels(sib_module)
                    if inferred is not None and inferred > 1:
                        num_ch = inferred
                        break

            if num_ch is None or num_ch < 1:
                num_ch = 1  # scalar fallback — broadcasts correctly

            setattr(model, name, SelectiveAmplificationActivation(num_ch))
        elif len(list(child.children())) > 0:
            # Recurse into child modules
            replace_activations(child, parent_name=f"{parent_name}.{name}")
        prev_module = child

    return model


# ============================================================================
#  NOVELTY 3 — Prototype-Anchored Evidential Classification (PAEC)
# ============================================================================

class PrototypeAnchoredEvidentialHead(nn.Module):
    """
    Prototype-Anchored Evidential Classification (PAEC) — Output-Stage Novelty
    ---------------------------------------------------------------------------
    A dual-pathway classification head that produces calibrated uncertainty
    estimates by combining two sources of evidence into a Dirichlet distribution:

    Path A — Prototype Distance Evidence:
      - Projects backbone features into a 256-dim embedding space.
      - Maintains K learnable prototype vectors (one per class).
      - Computes negative squared L2 distance from embedding to each prototype.
      - Distances serve as proximity-based evidence for each class.

    Path B — FC Evidence:
      - Standard fully-connected layer producing logit-based evidence.
      - Captures non-metric, discriminative class boundaries.

    Fusion — Dirichlet Posterior:
      - Combines both evidence sources: e = softplus(w_p · d + w_f · z)
      - Dirichlet parameters: α_i = e_i + 1 (ensures α > 1)
      - Class probabilities: p_i = α_i / S  where S = Σα_i
      - Epistemic uncertainty: u = K / S  (higher S → more confident)

    Why it's novel:
      Standard evidential deep learning (Sensoy et al., 2018) uses a single FC
      layer for evidence. PAEC anchors evidence in a learned metric space via
      prototypes, providing:
        1. Better calibrated uncertainty (dual evidence is more robust)
        2. Interpretability (show which class prototype the sample is closest to)
        3. Improved minority-class separation (prototype separation loss)

    Training requires EvidentialFocalLoss (see train.py) which includes:
      - Type-II MLL for Dirichlet
      - KL divergence regularizer with annealing
      - Focal modulation for hard examples
      - Prototype separation loss (pushes prototypes apart in embedding space)

    Ablation:
      Set use_paec=False to use StandardHead (FC + softmax) instead.
    """

    def __init__(self, in_features: int = 1536, embed_dim: int = 256,
                 num_classes: int = 7, margin: float = 2.0):
        """
        Args:
            in_features: Dimensionality of backbone output (1536 for EfficientNet-B3).
            embed_dim: Dimensionality of the prototype embedding space.
            num_classes: Number of target classes.
            margin: Minimum desired distance between any two prototypes.
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.margin = margin

        # Projection network: backbone features → embedding space
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )

        # Learnable class prototypes in embedding space
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # FC pathway for additional logit-based evidence
        self.fc_evidence = nn.Linear(embed_dim, num_classes)

        # Learnable fusion weights (how much to trust each evidence pathway)
        self.w_proto = nn.Parameter(torch.tensor(1.0))
        self.w_fc = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing Dirichlet-based predictions with uncertainty.

        Args:
            features: (B, in_features) pooled backbone features.

        Returns:
            Dict with keys:
              - 'logits': (B, K) — fused evidence (for compatibility with standard loss)
              - 'evidence': (B, K) — non-negative evidence values
              - 'alpha': (B, K) — Dirichlet concentration parameters (evidence + 1)
              - 'uncertainty': (B,) — epistemic uncertainty in [0, 1]
              - 'probs': (B, K) — class probabilities (alpha / S)
              - 'embedding': (B, embed_dim) — sample embeddings (for visualization)
              - 'proto_distances': (B, K) — squared L2 distances to prototypes
        """
        # Project to embedding space
        embedding = self.projector(features)  # (B, embed_dim)

        # Path A: Prototype distances
        # (B, 1, D) - (1, K, D) → (B, K, D) → sum over D → (B, K)
        diff = embedding.unsqueeze(1) - self.prototypes.unsqueeze(0)
        sq_distances = (diff ** 2).sum(dim=-1)          # (B, K)
        proto_evidence = -sq_distances                   # negative distance as evidence

        # Path B: FC evidence
        fc_evidence = self.fc_evidence(embedding)       # (B, K)

        # Fusion: weighted combination → ensure non-negative via softplus
        fused = self.w_proto * proto_evidence + self.w_fc * fc_evidence
        evidence = F.softplus(fused)                    # (B, K), strictly positive

        # Dirichlet parameters: α_i = e_i + 1 (ensures α > 1, proper Dirichlet)
        alpha = evidence + 1.0                          # (B, K)
        strength = alpha.sum(dim=-1, keepdim=True)      # (B, 1) — Dirichlet strength S
        probs = alpha / strength                        # (B, K) — expected probabilities

        # Epistemic uncertainty: u = K / S  (range: [0, 1] when evidence ≥ 0)
        uncertainty = self.num_classes / strength.squeeze(-1)  # (B,)

        return {
            "logits": fused,              # raw fused logits (before softplus)
            "evidence": evidence,
            "alpha": alpha,
            "uncertainty": uncertainty,
            "probs": probs,
            "embedding": embedding,
            "proto_distances": sq_distances,
        }

    def prototype_separation_loss(self) -> torch.Tensor:
        """
        Compute prototype separation loss: pushes prototypes apart.

        L_sep = Σ_{i≠j} max(0, margin - ||p_i - p_j||_2)

        Returns:
            Scalar loss value.
        """
        # Pairwise distances between all prototypes
        diff = self.prototypes.unsqueeze(0) - self.prototypes.unsqueeze(1)  # (K, K, D)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)                  # (K, K)

        # Hinge loss: penalize pairs closer than margin
        # Only upper triangle (avoid double counting and self-distance)
        K = self.num_classes
        mask = torch.triu(torch.ones(K, K, device=dist.device), diagonal=1).bool()
        pairwise_dists = dist[mask]
        loss = F.relu(self.margin - pairwise_dists).mean()
        return loss


class StandardHead(nn.Module):
    """
    Standard classification head (FC + softmax) used when PAEC is disabled.

    This serves as the baseline for ablation: identical pooling and dropout,
    but no prototypes, no evidential output, no uncertainty estimation.
    """

    def __init__(self, in_features: int = 1536, num_classes: int = 7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, in_features) pooled backbone features.

        Returns:
            Dict with 'logits' and 'probs' keys.
        """
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probs": probs,
        }


# ============================================================================
#  Main Classifier — EfficientNet-B3 with all novelties
# ============================================================================

class SkinLesionClassifier(nn.Module):
    """
    Skin Lesion Classifier with three independently toggleable novelties.

    Architecture: EfficientNet-B3 (pretrained on ImageNet) as backbone with:
      - Novelty 1 (SSD): Modified stem conv (7→40 instead of 3→40 channels)
      - Novelty 2 (SAA): All SiLU activations replaced with SAA
      - Novelty 3 (PAEC): Evidential classification head with prototypes

    Each novelty can be disabled independently for ablation studies:
      use_ssd=False → standard 3-channel RGB input
      use_saa=False → standard SiLU activations
      use_paec=False → standard FC + softmax head

    Args:
        num_classes: Number of output classes (7 for HAM10000).
        use_ssd: Enable Spectral Skin Decomposition input (7-channel stem).
        use_saa: Enable Selective Amplification Activation.
        use_paec: Enable Prototype-Anchored Evidential Classification.
        pretrained: Use ImageNet pretrained weights for backbone.
    """

    def __init__(self, num_classes: int = 7, use_ssd: bool = True,
                 use_saa: bool = True, use_paec: bool = True,
                 pretrained: bool = True):
        super().__init__()
        self.use_ssd = use_ssd
        self.use_saa = use_saa
        self.use_paec = use_paec
        self.num_classes = num_classes

        # ---- Load EfficientNet-B3 backbone from timm ----
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,           # Remove original classifier head
            global_pool="avg",       # Global average pooling
        )
        # EfficientNet-B3 feature dimension after global pool
        self.feature_dim = self.backbone.num_features  # 1536

        # ---- Novelty 1: Modify stem for 7-channel SSD input ----
        if use_ssd:
            old_conv = self.backbone.conv_stem
            # Create new stem conv accepting 7 channels instead of 3
            new_conv = nn.Conv2d(
                in_channels=7,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize: copy pretrained weights for RGB channels (0:3),
            # initialize extra channels (3:7) with small random values
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
                # Initialize SSD channels from mean of RGB weights + small noise
                rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
                for ch in range(4):
                    new_conv.weight[:, 3 + ch, :, :] = (
                        rgb_mean.squeeze(1) * 0.25
                        + torch.randn_like(rgb_mean.squeeze(1)) * 0.02
                    )
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            self.backbone.conv_stem = new_conv

        # ---- Novelty 2: Replace activations with SAA ----
        if use_saa:
            replace_activations(self.backbone)

        # ---- Novelty 3: Classification head ----
        if use_paec:
            self.head = PrototypeAnchoredEvidentialHead(
                in_features=self.feature_dim,
                embed_dim=256,
                num_classes=num_classes,
                margin=2.0,
            )
        else:
            self.head = StandardHead(
                in_features=self.feature_dim,
                num_classes=num_classes,
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input tensor. C=7 if SSD enabled, C=3 otherwise.

        Returns:
            Dict with at minimum 'logits' and 'probs'. If PAEC is active,
            also includes 'evidence', 'alpha', 'uncertainty', 'embedding',
            'proto_distances'.
        """
        features = self.backbone(x)     # (B, 1536) — globally pooled features
        output = self.head(features)    # Dict of tensors
        return output

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pre-pooling feature maps for GradCAM visualization.

        Returns the output of the last convolutional layer before global pooling.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            (B, C_feat, H_feat, W_feat) feature map tensor.
        """
        # EfficientNet-B3 architecture: conv_stem → blocks → conv_head → bn2
        # We want the output after conv_head + bn2 (before global pool)
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        # In timm EfficientNet, the act1 may be SiLU or SAA
        if hasattr(self.backbone, 'act1'):
            x = self.backbone.act1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        if hasattr(self.backbone, 'act2'):
            x = self.backbone.act2(x)
        return x


# ============================================================================
#  GradCAM — Gradient-weighted Class Activation Mapping
# ============================================================================

class GradCAM:
    """
    GradCAM: Visual Explanations from Deep Networks.

    Generates heatmaps showing which spatial regions of the input image most
    influenced the model's prediction for a given class. Uses gradients flowing
    into the last convolutional layer to weight feature map activations.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.

    Usage:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate(input_tensor, target_class=4)  # mel
        overlay = gradcam.overlay(original_image, heatmap)
    """

    def __init__(self, model: SkinLesionClassifier):
        """
        Args:
            model: SkinLesionClassifier instance (must be in eval mode for inference).
        """
        self.model = model
        self.feature_maps = None
        self.gradients = None

        # Register hooks on the last conv layer of the backbone
        # In timm's EfficientNet, this is backbone.conv_head
        target_layer = self.model.backbone.conv_head
        target_layer.register_forward_hook(self._save_feature_maps)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        """Forward hook: save feature maps."""
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook: save gradients."""
        self.gradients = grad_output[0].detach()

    @torch.enable_grad()
    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate a GradCAM heatmap for a single input image.

        Args:
            input_tensor: (1, C, H, W) preprocessed input tensor (on correct device).
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            (H_input, W_input) float32 heatmap in [0, 1], same spatial size as input.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        logits = output["logits"]  # (1, K)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        # Backward pass: gradient of target logit w.r.t. feature maps
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=False)

        # GradCAM computation
        # Global average pool the gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        # Weighted combination of feature maps
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input spatial dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        return cam.astype(np.float32)

    @staticmethod
    def overlay(image: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.4) -> np.ndarray:
        """
        Overlay GradCAM heatmap on the original image.

        Args:
            image: (H, W, 3) uint8 RGB image.
            heatmap: (H, W) float32 heatmap in [0, 1].
            alpha: Blending factor (0 = only image, 1 = only heatmap).

        Returns:
            (H, W, 3) uint8 blended image.
        """
        # Convert heatmap to colormap (COLORMAP_JET)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Resize heatmap to match image if needed
        if heatmap_colored.shape[:2] != image.shape[:2]:
            heatmap_colored = cv2.resize(
                heatmap_colored, (image.shape[1], image.shape[0])
            )

        # Blend
        overlay = (
            (1 - alpha) * image.astype(np.float32)
            + alpha * heatmap_colored.astype(np.float32)
        )
        return np.clip(overlay, 0, 255).astype(np.uint8)
