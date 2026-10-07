import numpy as np
import torch
import cv2
from pathlib import Path

from src.config import DATA_PROCESSED_DIR, IMAGE_SIZE, CLASS_TO_IDX, SEED
from src.data.wm811k_loader import parse_wm811k_metadata, load_raw_df


def cache_preprocessed_datasets(force_rebuild=False):
    train_cache = DATA_PROCESSED_DIR / "cache_train.npz"
    test_cache = DATA_PROCESSED_DIR / "cache_test.npz"

    if train_cache.exists() and test_cache.exists() and not force_rebuild:
        print("Loading cached preprocessed uint8 dataset tensors...")
        train_data = np.load(train_cache)
        test_data = np.load(test_cache)
        return train_data, test_data

    print("Building cached preprocessed uint8 dataset tensors for ultra-fast training & eval...")
    raw_df = load_raw_df()
    meta_df = parse_wm811k_metadata()

    labeled_meta = meta_df[meta_df['is_labeled']].copy()
    train_meta = labeled_meta[labeled_meta['trianTestLabel'] == 'Training']
    test_meta = labeled_meta[labeled_meta['trianTestLabel'] == 'Test']

    def _process_split(split_meta):
        images = []
        labels = []
        orig_indices = []

        for _, row in split_meta.iterrows():
            orig_idx = row['original_idx']
            wm = raw_df.iloc[orig_idx]['waferMap']
            cls_idx = row['class_idx']

            if isinstance(wm, np.ndarray) and wm.ndim == 2:
                img = np.zeros_like(wm, dtype=np.uint8)
                img[wm == 1] = 128
                img[wm == 2] = 255
                if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
                    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                else:
                    img_resized = img
            else:
                img_resized = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

            images.append(img_resized)
            labels.append(cls_idx)
            orig_indices.append(orig_idx)

        images_arr = np.array(images, dtype=np.uint8)
        labels_arr = np.array(labels, dtype=np.int64)
        orig_indices_arr = np.array(orig_indices, dtype=np.int32)
        return images_arr, labels_arr, orig_indices_arr

    print(f"Preprocessing {len(train_meta):,} training wafers...")
    tr_imgs, tr_lbls, tr_idxs = _process_split(train_meta)
    np.savez_compressed(train_cache, images=tr_imgs, labels=tr_lbls, orig_indices=tr_idxs)

    print(f"Preprocessing {len(test_meta):,} test wafers...")
    te_imgs, te_lbls, te_idxs = _process_split(test_meta)
    np.savez_compressed(test_cache, images=te_imgs, labels=te_lbls, orig_indices=te_idxs)

    print("Preprocessed dataset caching complete!")
    train_data = np.load(train_cache)
    test_data = np.load(test_cache)
    return train_data, test_data
