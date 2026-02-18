import copy
import os
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

from .config import TaskConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_tensor_output(y):
    if isinstance(y, (list, tuple)):
        return y[-1]
    return y


def _pick_slice(mask_3d: np.ndarray, axis: str = "z") -> int:
    m = mask_3d > 0
    if axis == "z":
        areas = m.sum(axis=(0, 1))
    elif axis == "y":
        areas = m.sum(axis=(0, 2))
    else:
        areas = m.sum(axis=(1, 2))
    idx = int(np.argmax(areas))
    if int(areas.max()) == 0:
        idx = (
            mask_3d.shape[2] // 2 if axis == "z"
            else mask_3d.shape[1] // 2 if axis == "y"
            else mask_3d.shape[0] // 2
        )
    return idx


def _slice_2d(vol: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "z":
        return vol[..., idx]
    if axis == "y":
        return vol[:, idx, :]
    return vol[idx, :, :]


def _rgba_from_labels(mask2d: np.ndarray, alpha: float, class_colors: Dict[int, tuple]):
    h, w = mask2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for k, rgb in class_colors.items():
        m = (mask2d == k)
        if not m.any():
            continue
        rgba[m, 0] = rgb[0]
        rgba[m, 1] = rgb[1]
        rgba[m, 2] = rgb[2]
        rgba[m, 3] = alpha
    return rgba


@torch.no_grad()
def render_side_by_side_overlays_fullbrain(
    out_dir: str,
    task_cfg: TaskConfig,
    model_zoo: Dict[str, torch.nn.Module],
    viz_loader,
    axis: str = "z",
    max_cases: int = 3,
    out_prefix: str = "overlay_full",
    flair_channel: int = 0,
    overlay_alpha: float = 0.35,
):
    os.makedirs(out_dir, exist_ok=True)

    default_colors = [
        (1.00, 0.95, 0.25),
        (0.35, 0.80, 1.00),
        (1.00, 0.45, 0.65),
        (0.60, 1.00, 0.60),
        (1.00, 0.70, 0.30),
    ]
    class_colors = {
        c: default_colors[(c - 1) % len(default_colors)]
        for c in range(1, task_cfg.num_classes)
    }

    models = {}
    for name, model in model_zoo.items():
        ck = os.path.join(out_dir, "ablation_fold", f"best_{name}.pth")
        if not os.path.isfile(ck):
            print(f"[viz] skip {name}: no checkpoint at {ck}")
            continue
        m = copy.deepcopy(model).to(DEVICE).eval()
        ckpt = torch.load(ck, map_location=DEVICE)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=False)
        models[name] = m

    if not models:
        print("[viz] No models available.")
        return

    done = 0
    for batch in viz_loader:
        imgs = batch["image"].to(DEVICE)   # [1, C, H, W, D]
        gts = batch["label"].to(DEVICE)    # [1, 1, H, W, D]
        case_id = batch.get("case_id", ["case"])[0]

        gt3d = gts[0, 0].cpu().numpy()
        sl = _pick_slice(gt3d, axis=axis)

        bg3d = imgs[0, flair_channel].cpu().numpy()
        bg2d = _slice_2d(bg3d, axis, sl)
        nz = bg2d[bg2d > 0]
        if nz.size > 10:
            vmin, vmax = np.percentile(nz, [1, 99])
        else:
            vmin, vmax = float(bg2d.min()), float(bg2d.max())

        gt2d = _slice_2d(gt3d, axis, sl)
        gt_rgba = _rgba_from_labels(gt2d, overlay_alpha, class_colors)

        pred_panels = {}
        for name, m in models.items():
            predictor = lambda x, mm=m: _ensure_tensor_output(mm(x))
            try:
                logits = sliding_window_inference(
                    imgs,
                    roi_size=task_cfg.patch_size,
                    sw_batch_size=4,
                    predictor=predictor,
                    overlap=0.5,
                    mode="gaussian",
                )
            except TypeError:
                logits = sliding_window_inference(
                    imgs,
                    roi_size=task_cfg.patch_size,
                    sw_batch_size=4,
                    predictor=predictor,
                    overlap=0.5,
                    mode="constant",
                )
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
            pred2d = _slice_2d(pred, axis, sl)
            pred_panels[name] = _rgba_from_labels(pred2d, overlay_alpha, class_colors)

        cols = 1 + len(pred_panels)
        plt.figure(figsize=(4.5 * cols, 4.5))

        ax = plt.subplot(1, cols, 1)
        ax.imshow(bg2d, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.imshow(gt_rgba, interpolation="nearest")
        for cls_id, color in class_colors.items():
            ax.contour(gt2d == cls_id, levels=[0.5], linewidths=0.8, colors=[color])
        ax.set_title("Ground Truth")
        ax.axis("off")

        c = 2
        for name in sorted(pred_panels.keys()):
            ax = plt.subplot(1, cols, c)
            c += 1
            ax.imshow(bg2d, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.imshow(pred_panels[name], interpolation="nearest")
            ax.set_title(name)
            ax.axis("off")

        fname = f"{out_prefix}_{case_id}_axis-{axis}_slice-{sl:03d}.png"
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[viz] saved: {out_path}")

        done += 1
        if done >= max_cases:
            break
