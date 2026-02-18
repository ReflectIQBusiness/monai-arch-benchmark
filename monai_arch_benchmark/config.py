import os
from dataclasses import dataclass
from typing import Dict, Tuple

import yaml


@dataclass
class TaskConfig:
    name: str
    data_root: str
    images_dir: str
    labels_dir: str
    in_channels: int
    num_classes: int
    label_names: Dict[int, str]
    patch_size: Tuple[int, int, int]
    target_spacing: Tuple[float, float, float]
    val_fraction: float = 0.2


def load_task_config(config_path: str) -> TaskConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    images_subdir = cfg.get("images_subdir", "imagesTr")
    labels_subdir = cfg.get("labels_subdir", "labelsTr")

    images_dir = os.path.join(data_root, images_subdir)
    labels_dir = os.path.join(data_root, labels_subdir)

    label_names = {int(k): v for k, v in cfg["label_names"].items()}

    return TaskConfig(
        name=cfg["task_name"],
        data_root=data_root,
        images_dir=images_dir,
        labels_dir=labels_dir,
        in_channels=int(cfg["in_channels"]),
        num_classes=int(cfg["num_classes"]),
        label_names=label_names,
        patch_size=tuple(cfg["patch_size"]),
        target_spacing=tuple(cfg["target_spacing"]),
        val_fraction=float(cfg.get("val_fraction", 0.2)),
    )
