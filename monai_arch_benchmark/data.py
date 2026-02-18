import os
import glob
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from monai.data import Dataset, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd, ResizeWithPadOrCropd
)

from .config import TaskConfig


NUM_WORKERS = max(2, (os.cpu_count() or 4) // 2)
USE_CACHE_DATASET = True
CACHE_RATE = 1.0


def list_pairs(images_dir: str, labels_dir: str) -> List[Dict]:
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.nii")) +
                  glob.glob(os.path.join(images_dir, "*.nii.gz")))
    lbls = sorted(glob.glob(os.path.join(labels_dir, "*.nii")) +
                  glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    def stem(p: str) -> str:
        b = os.path.basename(p)
        if b.endswith(".nii.gz"):
            return b[:-7]
        if b.endswith(".nii"):
            return b[:-4]
        return os.path.splitext(b)[0]

    img_map = {stem(p): p for p in imgs}
    lbl_map = {stem(p): p for p in lbls}
    keys = sorted(set(img_map) & set(lbl_map))
    data = [{"image": img_map[k], "label": lbl_map[k], "case_id": k} for k in keys]
    assert len(data) > 0, (
        f"No cases found. Found {len(imgs)} images and {len(lbls)} labels; {len(keys)} matching."
    )
    return data


def make_trainval_transforms(task_cfg: TaskConfig):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=task_cfg.target_spacing,
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(
            keys=["image", "label"],
            source_key="label",
            k_divisible=16
        ),
        ResizeWithPadOrCropd(
            keys=["image", "label"],
            spatial_size=task_cfg.patch_size
        ),
    ])


def make_viz_transforms(task_cfg: TaskConfig):
    # full-volume viz pipeline (no crop/resize)
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=task_cfg.target_spacing,
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0, clip=True
        ),
    ])


def build_train_val_loaders(
    data: List[Dict],
    task_cfg: TaskConfig,
    batch_size: int,
    seed: int,
):
    tfs = make_trainval_transforms(task_cfg)

    if USE_CACHE_DATASET:
        full_ds = CacheDataset(
            data=data,
            transform=tfs,
            cache_rate=CACHE_RATE,
            num_workers=NUM_WORKERS
        )
    else:
        full_ds = Dataset(data=data, transform=tfs)

    N = len(full_ds)
    idxs = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idxs)
    n_val = max(1, int(N * task_cfg.val_fraction))
    va_idx = idxs[:n_val]
    tr_idx = idxs[n_val:]

    ds_tr = Subset(full_ds, tr_idx.tolist())
    ds_va = Subset(full_ds, va_idx.tolist())

    common_args = dict(
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    if NUM_WORKERS > 0:
        common_args.update(dict(persistent_workers=True, prefetch_factor=2))

    loader_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True, **common_args
    )
    loader_va = DataLoader(
        ds_va, batch_size=1, shuffle=False, **common_args
    )
    return loader_tr, loader_va


def build_viz_loader(
    data: List[Dict],
    task_cfg: TaskConfig,
    seed: int,
):
    tfs_viz = make_viz_transforms(task_cfg)
    if USE_CACHE_DATASET:
        full_viz = CacheDataset(
            data=data, transform=tfs_viz,
            cache_rate=CACHE_RATE, num_workers=NUM_WORKERS
        )
    else:
        full_viz = Dataset(data=data, transform=tfs_viz)

    N = len(full_viz)
    idxs = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idxs)
    n_val = max(1, int(N * task_cfg.val_fraction))
    va_idx = idxs[:n_val]
    ds_va_viz = torch.utils.data.Subset(full_viz, va_idx.tolist())

    common_args = dict(
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    if NUM_WORKERS > 0:
        common_args.update(dict(persistent_workers=True, prefetch_factor=2))

    return DataLoader(ds_va_viz, batch_size=1, shuffle=False, **common_args)
