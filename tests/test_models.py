# tests/test_models.py

import torch

from monai_arch_benchmark.models import build_model_zoo
from monai_arch_benchmark.config import TaskConfig


def _make_dummy_task_config():
    return TaskConfig(
        name="dummy",
        data_root="/tmp",
        images_dir="/tmp/images",
        labels_dir="/tmp/labels",
        in_channels=1,
        num_classes=2,
        label_names={0: "background", 1: "class1"},
        patch_size=(32, 32, 32),
        target_spacing=(1.0, 1.0, 1.0),
        val_fraction=0.2,
    )


def test_build_model_zoo_contains_expected_keys():
    cfg = _make_dummy_task_config()
    zoo = build_model_zoo(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        patch_size=cfg.patch_size,
    )

    # We do not need to test every architecture, just a representative subset
    expected = {"BasicUNet", "UNet", "SwinUNETR"}
    assert expected.issubset(set(zoo.keys())), f"Missing models: {expected - set(zoo.keys())}"


def test_basicunet_forward_pass_3d():
    cfg = _make_dummy_task_config()
    zoo = build_model_zoo(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        patch_size=cfg.patch_size,
    )
    model = zoo["BasicUNet"]
    model.eval()

    x = torch.randn(1, cfg.in_channels, *cfg.patch_size)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, cfg.num_classes, *cfg.patch_size), (
        f"Unexpected output shape: {y.shape}"
    )
