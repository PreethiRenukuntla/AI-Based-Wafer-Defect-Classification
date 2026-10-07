import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from sklearn.model_selection import train_test_split

from src.config import (
    WM811K_PKL_PATH, IMAGE_SIZE, CLASS_TO_IDX, DEFECT_CLASSES, SEED
)
from src.data.wm811k_loader import parse_wm811k_metadata, load_raw_df


def preprocess_wafer_matrix(wm, target_size=IMAGE_SIZE):
    """
    Standardizes a wafer map 2D numpy array:
    0: background -> 0.0
    1: normal die  -> 0.5
    2: defective die -> 1.0
    Resizes matrix using nearest-neighbor interpolation to target_size.
    Returns normalized float32 tensor of shape (1, target_size, target_size).
    """
    if not isinstance(wm, np.ndarray) or wm.ndim != 2:
        # Fallback for invalid matrices
        arr = np.zeros((target_size, target_size), dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    # Map values to [0, 128, 255] uint8 image
    img = np.zeros_like(wm, dtype=np.uint8)
    img[wm == 1] = 128
    img[wm == 2] = 255

    # Resize cleanly preserving sharp die edges
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    else:
        img_resized = img

    # Normalize to [0.0, 1.0] float32 tensor
    tensor = img_resized.astype(np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0)


class WM811KDataset(Dataset):
    """
    PyTorch Dataset for WM-811K Wafer Maps.
    Applies spatial augmentations only during training.
    """
    def __init__(self, raw_df, indices_list, metadata_df, is_train=False, augment=False):
        self.raw_df = raw_df
        self.indices = np.array(indices_list)
        self.metadata = metadata_df.iloc[self.indices].reset_index(drop=True)
        self.is_train = is_train
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        row = self.raw_df.iloc[orig_idx]
        wm = row['waferMap']
        label_str = self.metadata.iloc[idx]['failureType']
        label_idx = CLASS_TO_IDX.get(label_str, 8)  # Default to 'none' if missing

        img_tensor = preprocess_wafer_matrix(wm, target_size=IMAGE_SIZE)

        # Apply augmentation on training data
        if self.augment and self.is_train:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])
            # Random 90-degree rotations
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                img_tensor = torch.rot90(img_tensor, k=k, dims=[1, 2])

        return img_tensor, label_idx


def get_data_splits(val_ratio=0.15, seed=SEED):
    """
    Loads raw WM-811K and metadata, returning train, val, and test dataset splits.
    PRESERVES sacred official WM-811K train/test label split:
    - Official Training set (54,355 wafers) -> split into Train (~85%) & Val (~15%)
    - Official Test set (118,595 wafers) -> sacred Test set
    """
    raw_df = load_raw_df()
    meta_df = parse_wm811k_metadata()

    labeled_meta = meta_df[meta_df['is_labeled']].copy()

    official_train_mask = (labeled_meta['trianTestLabel'] == 'Training')
    official_test_mask = (labeled_meta['trianTestLabel'] == 'Test')

    official_train_df = labeled_meta[official_train_mask]
    official_test_df = labeled_meta[official_test_mask]

    # Stratified split of official training set into train and val
    train_indices, val_indices = train_test_split(
        official_train_df['original_idx'].values,
        test_size=val_ratio,
        random_state=seed,
        stratify=official_train_df['class_idx'].values
    )

    test_indices = official_test_df['original_idx'].values

    print(f"Data Splits Created:")
    print(f"  Train Set: {len(train_indices):,} samples")
    print(f"  Val Set:   {len(val_indices):,} samples")
    print(f"  Test Set:  {len(test_indices):,} samples (Official Sacred Test Set)")

    train_dataset = WM811KDataset(raw_df, train_indices, meta_df, is_train=True, augment=True)
    val_dataset = WM811KDataset(raw_df, val_indices, meta_df, is_train=False, augment=False)
    test_dataset = WM811KDataset(raw_df, test_indices, meta_df, is_train=False, augment=False)

    return raw_df, meta_df, train_dataset, val_dataset, test_dataset


def create_weighted_sampler(dataset):
    """Creates a WeightedRandomSampler to handle severe class imbalance."""
    labels = [dataset.metadata.iloc[i]['class_idx'] for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=len(DEFECT_CLASSES))
    class_weights = 1.0 / (class_counts + 1e-5)
    sample_weights = np.array([class_weights[lbl] for lbl in labels])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler, class_weights
