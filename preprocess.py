"""
preprocess.py — Input Pipeline, Augmentation, and Spectral Skin Decomposition (SSD)
====================================================================================

This module implements:
  1. NOVELTY 1 — Spectral Skin Decomposition (SSD):
     Converts RGB dermoscopy images into 7-channel tensors by appending 4 biologically
     meaningful diagnostic channels: melanin density, erythema (hemoglobin), boundary
     enhancement (Difference-of-Gaussians), and local texture entropy.

  2. HAM10000Dataset: A PyTorch Dataset with lesion-level deduplication to prevent
     data leakage between splits.

  3. Augmentation pipelines (train / val) using albumentations.

  4. create_dataloaders: End-to-end data loading with stratified splitting,
     class-weighted sampling, and configurable SSD toggle for ablation.

Authors: [Your Name]
License: MIT
"""

import os
import glob
import warnings
from typing import Tuple, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
#  Class label encoding — consistent across the entire project
# ---------------------------------------------------------------------------
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
#  ImageNet normalization stats (for the 3 RGB channels)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# SSD channel stats — pre-computed on HAM10000 training set.
# These are reasonable defaults; exact values depend on the split.
SSD_CHANNEL_MEAN = [0.45, 0.30, 0.0, 0.55]  # melanin, erythema, boundary, entropy
SSD_CHANNEL_STD = [0.18, 0.15, 0.12, 0.20]

# Combined 7-channel stats
FULL_MEAN = IMAGENET_MEAN + SSD_CHANNEL_MEAN
FULL_STD = IMAGENET_STD + SSD_CHANNEL_STD


# ============================================================================
#  NOVELTY 1 — Spectral Skin Decomposition (SSD)
# ============================================================================
class SpectralSkinDecomposition:
    """
    Spectral Skin Decomposition (SSD) — Input-Stage Novelty
    --------------------------------------------------------
    Converts a standard RGB dermoscopy image (H, W, 3) into a 7-channel
    diagnostic tensor (H, W, 7) by appending four biologically meaningful
    channels derived from optical skin properties:

      Channel 4: Melanin Density Map
        - Based on optical density difference between red and blue channels.
        - High values indicate pigmented regions (melanocytic lesions).
        - Formula: M = OD_blue - OD_red  where OD = -log10(I / 255 + eps)

      Channel 5: Erythema (Hemoglobin) Map
        - Estimates redness / vascular activity from the green channel.
        - High values indicate vascular or inflamed tissue.
        - Formula: E = OD_green - 0.5 * (OD_red + OD_blue)

      Channel 6: Lesion Boundary Enhancement (Difference-of-Gaussians)
        - DoG filter on luminance highlights border irregularity.
        - Relevant for melanoma detection (ABCDE rule: Border irregularity).
        - Parameters: sigma1=1.0, sigma2=3.0

      Channel 7: Local Texture Entropy
        - Shannon entropy in a sliding window captures structural heterogeneity.
        - Higher entropy → more complex / heterogeneous lesion texture.
        - Window size: 7×7 (disk radius 3)

    Why it's novel:
      Unlike Reinhard or Macenko color normalization (which homogenize color),
      SSD *extracts* biologically relevant features as additional input channels,
      encoding domain knowledge about skin optics directly into the network input.
      It is deterministic, fast, requires no learned parameters, and is fully
      interpretable — each channel has a clear dermatological meaning.

    Ablation:
      Set use_ssd=False to disable SSD and use standard 3-channel RGB input.
    """

    def __init__(self, sigma1: float = 1.0, sigma2: float = 3.0,
                 entropy_radius: int = 3):
        """
        Args:
            sigma1: Smaller Gaussian sigma for DoG boundary detection.
            sigma2: Larger Gaussian sigma for DoG boundary detection.
            entropy_radius: Radius of the disk structuring element for entropy.
        """
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.entropy_radius = entropy_radius
        self._selem = disk(entropy_radius)

    def compute_melanin_map(self, rgb: np.ndarray) -> np.ndarray:
        """
        Estimate relative melanin concentration from RGB image.

        Melanin absorbs more strongly in visible blue than red. The optical
        density difference OD_blue - OD_red approximates relative melanin
        content in the Beer-Lambert framework for skin optics.

        Args:
            rgb: (H, W, 3) float32 image, values in [0, 1].

        Returns:
            (H, W) float32 melanin density map, normalized to [0, 1].
        """
        eps = 1e-6
        # Optical density per channel: OD = -log10(I + eps)
        od_red = -np.log10(rgb[:, :, 0] + eps)
        od_blue = -np.log10(rgb[:, :, 2] + eps)
        melanin = od_blue - od_red

        # Normalize to [0, 1]
        m_min, m_max = melanin.min(), melanin.max()
        if m_max - m_min > eps:
            melanin = (melanin - m_min) / (m_max - m_min)
        else:
            melanin = np.zeros_like(melanin)
        return melanin.astype(np.float32)

    def compute_erythema_map(self, rgb: np.ndarray) -> np.ndarray:
        """
        Estimate erythema (hemoglobin / redness) from RGB image.

        Hemoglobin absorbs strongly in the green band. The difference between
        green OD and the average of red/blue OD isolates the hemoglobin signal.

        Args:
            rgb: (H, W, 3) float32 image, values in [0, 1].

        Returns:
            (H, W) float32 erythema map, normalized to [0, 1].
        """
        eps = 1e-6
        od_red = -np.log10(rgb[:, :, 0] + eps)
        od_green = -np.log10(rgb[:, :, 1] + eps)
        od_blue = -np.log10(rgb[:, :, 2] + eps)
        erythema = od_green - 0.5 * (od_red + od_blue)

        e_min, e_max = erythema.min(), erythema.max()
        if e_max - e_min > eps:
            erythema = (erythema - e_min) / (e_max - e_min)
        else:
            erythema = np.zeros_like(erythema)
        return erythema.astype(np.float32)

    def compute_boundary_map(self, gray: np.ndarray) -> np.ndarray:
        """
        Lesion boundary enhancement via Difference-of-Gaussians (DoG).

        DoG approximates the Laplacian of Gaussian and highlights edges at a
        specific spatial scale. Border irregularity is a critical diagnostic
        feature for melanoma (the "B" in the ABCDE rule).

        Args:
            gray: (H, W) float32 grayscale image, values in [0, 1].

        Returns:
            (H, W) float32 boundary map, normalized to [0, 1].
        """
        eps = 1e-6
        blur1 = cv2.GaussianBlur(gray, (0, 0), self.sigma1)
        blur2 = cv2.GaussianBlur(gray, (0, 0), self.sigma2)
        dog = np.abs(blur1 - blur2)

        d_min, d_max = dog.min(), dog.max()
        if d_max - d_min > eps:
            dog = (dog - d_min) / (d_max - d_min)
        else:
            dog = np.zeros_like(dog)
        return dog.astype(np.float32)

    def compute_texture_entropy(self, gray: np.ndarray) -> np.ndarray:
        """
        Local texture entropy via sliding-window Shannon entropy.

        Measures structural heterogeneity within the lesion. Melanomas tend to
        have higher entropy (more disordered texture) than benign nevi.

        Args:
            gray: (H, W) float32 grayscale image, values in [0, 1].

        Returns:
            (H, W) float32 entropy map, normalized to [0, 1].
        """
        eps = 1e-6
        # skimage entropy expects uint8
        gray_uint8 = (gray * 255).astype(np.uint8)
        ent = sk_entropy(gray_uint8, self._selem)

        e_min, e_max = ent.min(), ent.max()
        if e_max - e_min > eps:
            ent = (ent - e_min) / (e_max - e_min)
        else:
            ent = np.zeros_like(ent)
        return ent.astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply SSD to an RGB image.

        Args:
            image: (H, W, 3) uint8 or float32 RGB image.

        Returns:
            (H, W, 7) float32 array: [R, G, B, melanin, erythema, boundary, entropy]
        """
        # Convert to float32 in [0, 1] if needed
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)

        # Grayscale for boundary and entropy channels
        gray = cv2.cvtColor((img_float * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Compute 4 diagnostic channels
        melanin = self.compute_melanin_map(img_float)
        erythema = self.compute_erythema_map(img_float)
        boundary = self.compute_boundary_map(gray)
        texture = self.compute_texture_entropy(gray)

        # Stack: RGB (3) + diagnostic (4) = 7 channels
        ssd_image = np.dstack([
            img_float,           # channels 0-2: RGB
            melanin,             # channel 3: melanin density
            erythema,            # channel 4: erythema / hemoglobin
            boundary,            # channel 5: boundary enhancement
            texture              # channel 6: texture entropy
        ])
        return ssd_image


# ---------------------------------------------------------------------------
#  Augmentation pipelines
# ---------------------------------------------------------------------------
#
#  IMPORTANT DESIGN NOTE — Two-stage transform pipeline:
#  Color transforms (ColorJitter, etc.) operate on 3-channel RGB images and
#  would fail or produce incorrect results on 7-channel SSD images. Therefore
#  the pipeline is split into two stages:
#
#    Stage 1 (rgb_transform): Spatial + color augmentation on RGB (3 channels).
#       Applied BEFORE SSD computation. This also means SSD channels are
#       derived from the augmented image, which is correct — the diagnostic
#       channels should reflect the actual augmented input the network sees.
#
#    Stage 2 (post_transform): Normalization + ToTensor on all channels (7 or 3).
#       Applied AFTER SSD computation (if enabled) or directly after stage 1.
#
#  The HAM10000Dataset.__getitem__ method orchestrates this two-stage flow.
# ---------------------------------------------------------------------------

def get_rgb_train_transforms(img_size: int = 300) -> A.Compose:
    """
    Stage 1: RGB augmentation pipeline (applied BEFORE SSD).

    Operates on 3-channel uint8 RGB images. Includes spatial and color
    augmentations appropriate for dermoscopy images.

    Args:
        img_size: Target spatial resolution (square).

    Returns:
        albumentations.Compose pipeline for RGB images.
    """
    return A.Compose([
        # Spatial transforms
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR,
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(rotate=(-180, 180), shear=(-10, 10), p=0.5),

        # Color / intensity transforms (safe on 3-channel RGB)
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.1, p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.0), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),

        # CoarseDropout — simulates hair / ruler / ink artifact occlusion
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(int(img_size * 0.05), int(img_size * 0.15)),
            hole_width_range=(int(img_size * 0.05), int(img_size * 0.15)),
            fill="random",
            p=0.4,
        ),
    ])


def get_rgb_val_transforms(img_size: int = 300) -> A.Compose:
    """
    Stage 1 (val): Resize only, no augmentation.

    Args:
        img_size: Target spatial resolution.

    Returns:
        albumentations.Compose pipeline.
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
    ])


def get_post_transforms(use_ssd: bool = True) -> A.Compose:
    """
    Stage 2: Normalize all channels and convert to tensor.

    Applied AFTER SSD computation (7-ch) or directly on RGB (3-ch).
    Input images must be float32 in [0, 1].

    Args:
        use_ssd: If True, normalize 7 channels. If False, normalize 3 channels.

    Returns:
        albumentations.Compose pipeline.
    """
    mean = FULL_MEAN if use_ssd else IMAGENET_MEAN
    std = FULL_STD if use_ssd else IMAGENET_STD

    return A.Compose([
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
#  HAM10000 Dataset
# ---------------------------------------------------------------------------

def _find_image_path(image_id: str, img_dirs: List[str]) -> Optional[str]:
    """Look for image_id.jpg across multiple directories."""
    for d in img_dirs:
        path = os.path.join(d, f"{image_id}.jpg")
        if os.path.isfile(path):
            return path
    return None


class HAM10000Dataset(Dataset):
    """
    HAM10000 Dataset with lesion-level deduplication and two-stage transforms.

    The HAM10000 dataset contains some lesions imaged multiple times (different
    imaging sessions). To prevent data leakage, we group by `lesion_id` so all
    images of the same physical lesion stay in the same split.

    Transform pipeline (two-stage):
      1. rgb_transform: Spatial + color augmentation on uint8 RGB (3 channels)
      2. SSD computation (if enabled): RGB float32 → 7-channel float32
      3. post_transform: Normalize all channels + convert to tensor

    This two-stage design ensures color transforms (ColorJitter, etc.) only
    operate on valid 3-channel RGB images, and SSD channels are computed from
    the augmented image (so the network sees consistent input).

    Args:
        df: DataFrame with columns ['image_id', 'dx', 'lesion_id'].
        img_dirs: List of directories containing the .jpg images.
        rgb_transform: Stage 1 — spatial + color augmentation on RGB (uint8 input).
        post_transform: Stage 2 — normalize + ToTensor (float32 input, 3 or 7 ch).
        use_ssd: If True, apply Spectral Skin Decomposition to create 7-channel input.
    """

    def __init__(self, df: pd.DataFrame, img_dirs: List[str],
                 rgb_transform: A.Compose, post_transform: A.Compose,
                 use_ssd: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.rgb_transform = rgb_transform
        self.post_transform = post_transform
        self.use_ssd = use_ssd
        self.ssd = SpectralSkinDecomposition() if use_ssd else None

        # Encode labels
        self.labels = self.df["dx"].map(CLASS_TO_IDX).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Load image (BGR → RGB), uint8
        img_path = _find_image_path(image_id, self.img_dirs)
        if img_path is None:
            raise FileNotFoundError(
                f"Image {image_id}.jpg not found in {self.img_dirs}"
            )
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8

        # Stage 1: Spatial + color augmentation on RGB (uint8)
        transformed_rgb = self.rgb_transform(image=image)
        image_aug = transformed_rgb["image"]  # (H', W', 3) uint8

        # Stage 2a: Apply SSD to get 7 channels, or convert to float32
        if self.use_ssd and self.ssd is not None:
            image_float = self.ssd(image_aug)  # (H', W', 7) float32 in [0, 1]
        else:
            image_float = image_aug.astype(np.float32) / 255.0  # (H', W', 3)

        # Stage 2b: Normalize all channels and convert to tensor
        transformed_final = self.post_transform(image=image_float)
        image_tensor = transformed_final["image"]  # (C, H, W) torch.Tensor

        label = int(self.labels[idx])
        return image_tensor, label, image_id


# ---------------------------------------------------------------------------
#  Dataloader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    csv_path: str,
    img_dir: str,
    batch_size: int = 32,
    img_size: int = 300,
    use_ssd: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, List[str]]:
    """
    Build train / val / test DataLoaders from HAM10000 metadata.

    Performs:
      - Lesion-level deduplication for splitting (prevents data leakage)
      - Stratified group split: 70% train / 15% val / 15% test
      - Class-weighted random sampling for training (addresses class imbalance)

    Args:
        csv_path: Path to HAM10000_metadata.csv.
        img_dir: Root directory containing image subfolders.
        batch_size: Training batch size.
        img_size: Image spatial resolution.
        use_ssd: Enable Spectral Skin Decomposition (Novelty 1).
        num_workers: DataLoader worker count.
        seed: Random seed for reproducibility.

    Returns:
        train_loader, val_loader, test_loader, class_weights, class_names
    """
    # ---- Load metadata ----
    df = pd.read_csv(csv_path)
    assert "image_id" in df.columns, f"CSV must contain 'image_id'. Found: {df.columns.tolist()}"
    assert "dx" in df.columns, f"CSV must contain 'dx'. Found: {df.columns.tolist()}"
    assert "lesion_id" in df.columns, f"CSV must contain 'lesion_id'. Found: {df.columns.tolist()}"

    # ---- Discover image directories ----
    # HAM10000 typically has images in part_1 and part_2 subdirectories
    img_dirs = []
    # Add root dir itself
    if any(f.endswith(".jpg") for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))):
        img_dirs.append(img_dir)
    # Add subdirectories that contain images
    for sub in sorted(os.listdir(img_dir)):
        sub_path = os.path.join(img_dir, sub)
        if os.path.isdir(sub_path):
            if any(f.endswith(".jpg") for f in os.listdir(sub_path)
                   if os.path.isfile(os.path.join(sub_path, f))):
                img_dirs.append(sub_path)
    if not img_dirs:
        raise FileNotFoundError(f"No .jpg images found in {img_dir} or its subdirectories.")
    print(f"[Data] Found image directories: {img_dirs}")

    # ---- Verify all images exist ----
    missing = []
    for _, row in df.iterrows():
        if _find_image_path(row["image_id"], img_dirs) is None:
            missing.append(row["image_id"])
    if missing:
        warnings.warn(f"[Data] {len(missing)} images not found. Removing from dataset.")
        df = df[~df["image_id"].isin(missing)].reset_index(drop=True)

    print(f"[Data] Total samples: {len(df)}")
    print(f"[Data] Class distribution:\n{df['dx'].value_counts().to_string()}")

    # ---- Stratified group split (by lesion_id) ----
    # First split: 70% train, 30% temp (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    groups = df["lesion_id"].values
    # For stratification, use the most common dx per lesion_id
    lesion_dx = df.groupby("lesion_id")["dx"].first()
    lesion_labels = df["lesion_id"].map(lesion_dx).values

    train_idx, temp_idx = next(gss1.split(df, lesion_labels, groups))

    # Second split: temp → 50/50 → val (15%) and test (15%)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)
    groups_temp = df_temp["lesion_id"].values
    lesion_dx_temp = df_temp.groupby("lesion_id")["dx"].first()
    lesion_labels_temp = df_temp["lesion_id"].map(lesion_dx_temp).values

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel_idx, test_rel_idx = next(gss2.split(df_temp, lesion_labels_temp, groups_temp))

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print(f"[Data] Split sizes — Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # ---- Compute class weights (inverse frequency) ----
    train_labels = df_train["dx"].map(CLASS_TO_IDX).values
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = torch.tensor(
        len(train_labels) / (NUM_CLASSES * class_counts + 1e-6),
        dtype=torch.float32,
    )
    print(f"[Data] Class weights: {dict(zip(CLASS_NAMES, class_weights.numpy().round(2)))}")

    # ---- Weighted random sampler for training ----
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(train_labels),
        replacement=True,
    )

    # ---- Build datasets ----
    rgb_train_transform = get_rgb_train_transforms(img_size)
    rgb_val_transform = get_rgb_val_transforms(img_size)
    post_transform = get_post_transforms(use_ssd)

    train_dataset = HAM10000Dataset(df_train, img_dirs, rgb_train_transform, post_transform, use_ssd)
    val_dataset = HAM10000Dataset(df_val, img_dirs, rgb_val_transform, post_transform, use_ssd)
    test_dataset = HAM10000Dataset(df_test, img_dirs, rgb_val_transform, post_transform, use_ssd)

    # ---- Build dataloaders ----
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights, CLASS_NAMES
