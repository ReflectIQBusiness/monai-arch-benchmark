import copy
from typing import Dict, Tuple

import numpy as np
import torch
import monai

from monai.networks.nets import (
    UNet, UNETR, SwinUNETR, DynUNet, SegResNet,
    BasicUNet, BasicUNetPlusPlus, AttentionUnet, VNet, HighResNet
)

try:
    from thop import profile
except Exception:
    profile = None


def build_model_zoo(
    in_channels: int,
    num_classes: int,
    patch_size: Tuple[int, int, int],
) -> Dict[str, torch.nn.Module]:
    att_channels = (16, 32, 64, 128)
    att_strides = (2, 2, 2, 2)

    models = {}

    models["BasicUNetPlusPlus"] = BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        features=(16, 16, 32, 64, 128, 16),
        deep_supervision=False,
    )

    try:
        models["SwinUNETR"] = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=24,
            use_checkpoint=True,
        )
    except TypeError:
        models["SwinUNETR"] = SwinUNETR(
            img_size=patch_size,
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=24,
            use_checkpoint=True,
        )

    models["UNETR"] = UNETR(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        img_size=patch_size,
        feature_size=24,
        hidden_size=384,
        mlp_dim=1536,
        num_heads=6,
    )

    models["DynUNet"] = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        kernel_size=[[3, 3, 3]] * 5,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 4,
        upsample_kernel_size=[[2, 2, 2]] * 4,
        filters=[16, 32, 64, 128, 256],
        res_block=True,
    )

    models["UNet"] = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    models["SegResNet"] = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        init_filters=24,
        blocks_down=(1, 1, 1, 2),
        blocks_up=(1, 1, 1),
    )

    models["BasicUNet"] = BasicUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
    )

    models["AttentionUnet"] = AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=att_channels,
        strides=att_strides,
    )

    models["VNet"] = VNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
    )

    models["HighResNet"] = HighResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
    )

    return models


def params_gflops(model: torch.nn.Module, inp=(1, 1, 64, 64, 64)):
    if profile is None:
        return np.nan, sum(p.numel() for p in model.parameters())
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(*inp)
    try:
        flops, params = profile(m, inputs=(x,), verbose=False)
        return float(flops) / 1e9, float(params)
    except Exception:
        return np.nan, sum(p.numel() for p in model.parameters())
