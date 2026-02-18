import copy
import os
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.utils import set_determinism

from .config import TaskConfig
from .models import build_model_zoo, params_gflops
from .metrics import PerClassAggregator

# --------------------------
# Training defaults
# --------------------------
MAX_EPOCHS = 50
PATIENCE = 12
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1
SEED = 42

VAL_EVERY = 1
HEAVY_VAL_EVERY = 5
SW_OVERLAP = 0.5
SW_BATCH_SIZE = 4

SPEED_MODE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# performance / determinism knobs
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_determinism(seed=SEED)

torch.backends.cudnn.benchmark = SPEED_MODE
torch.backends.cudnn.deterministic = not SPEED_MODE


def _ensure_tensor_output(y):
    if isinstance(y, (list, tuple)):
        return y[-1]
    return y


def _make_optimizer(model: torch.nn.Module):
    try:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=BASE_LR,
            weight_decay=WEIGHT_DECAY,
            fused=True,
        )
    except TypeError:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=BASE_LR,
            weight_decay=WEIGHT_DECAY,
        )
    return opt


def train_one(
    model: torch.nn.Module,
    task_cfg: TaskConfig,
    loader_tr: DataLoader,
    loader_va: DataLoader,
    name: str,
    fold_dir: str,
):
    model = model.to(DEVICE)
    if SPEED_MODE:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = _make_optimizer(model)

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    surf_metric = SurfaceDistanceMetric(include_background=False, reduction="none")

    best_score = -1.0
    epochs_no_improve = 0
    history = defaultdict(list)

    for ep in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for batch in loader_tr:
            imgs = batch["image"].to(DEVICE)
            gts = batch["label"].long().to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(imgs)
                if isinstance(out, (list, tuple)):
                    loss = torch.stack([loss_fn(o, gts) for o in out]).mean()
                    out = out[-1]
                else:
                    loss = loss_fn(out, gts)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        train_loss = running / max(1, len(loader_tr))

        validate_now = (ep % VAL_EVERY == 0)
        mean_macro_dice = np.nan
        if validate_now:
            model.eval()
            with torch.no_grad():
                dice_metric.reset()
                agg = PerClassAggregator(task_cfg.num_classes, task_cfg.label_names)

                for batch in loader_va:
                    imgs = batch["image"].to(DEVICE)

                    logits = _ensure_tensor_output(model(imgs))

                    preds = torch.argmax(logits, dim=1, keepdim=True)
                    gts = batch["label"].to(DEVICE)

                    preds_oh = torch.nn.functional.one_hot(
                        preds.squeeze(1).long(), task_cfg.num_classes
                    ).permute(0, 4, 1, 2, 3).float()
                    gts_oh = torch.nn.functional.one_hot(
                        gts.squeeze(1).long(), task_cfg.num_classes
                    ).permute(0, 4, 1, 2, 3).float()

                    d = dice_metric(y_pred=preds_oh, y=gts_oh)

                    heavy = bool(HEAVY_VAL_EVERY) and (ep % (VAL_EVERY * HEAVY_VAL_EVERY) == 0)
                    if heavy:
                        h = hd95_metric(y_pred=preds_oh, y=gts_oh)
                        s = surf_metric(y_pred=preds_oh, y=gts_oh)
                        agg.update_batch(preds_oh, gts_oh, metrics={"dice": d, "hd95": h, "surf": s})
                    else:
                        agg.update_batch(preds_oh, gts_oh, metrics={"dice": d})

                per_class = agg.finalize()
                cls_scores = [r["Dice_macro"] for r in per_class if np.isfinite(r["Dice_macro"])]
                mean_macro_dice = float(np.mean(cls_scores)) if cls_scores else np.nan

        history["train_loss"].append(train_loss)
        history["val_macro_dice"].append(mean_macro_dice)
        history["epoch_time_sec"].append(time.time() - t0)

        if validate_now and np.isfinite(mean_macro_dice):
            if (best_score is None) or (mean_macro_dice > best_score):
                best_score = mean_macro_dice
                epochs_no_improve = 0
                to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(
                    {"model_state_dict": to_save.state_dict()},
                    os.path.join(fold_dir, f"best_{name}.pth"),
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"[{name}] Early stopping at epoch {ep}. Best Macro-Dice={best_score:.4f}")
                    break

        if validate_now:
            print(f"[{name}] ep {ep:03d} | loss={train_loss:.4f} | MacroDice={mean_macro_dice if np.isfinite(mean_macro_dice) else np.nan:.4f}")
        else:
            print(f"[{name}] ep {ep:03d} | loss={train_loss:.4f} | (validation skipped)")

    pd.DataFrame(history).to_csv(os.path.join(fold_dir, f"{name}_history.csv"), index=False)
    return best_score


@torch.no_grad()
def evaluate_per_class(
    model: torch.nn.Module,
    task_cfg: TaskConfig,
    loader_va: DataLoader,
    name: str,
    fold_dir: str,
    use_sw_for_final_eval: bool = True,
):
    ckpt_path = os.path.join(fold_dir, f"best_{name}.pth")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()

    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    surf_metric = SurfaceDistanceMetric(include_background=False, reduction="none")

    agg = PerClassAggregator(task_cfg.num_classes, task_cfg.label_names)
    for batch in loader_va:
        imgs = batch["image"].to(DEVICE)
        if use_sw_for_final_eval:
            predictor = lambda x, m=model: _ensure_tensor_output(m(x))
            try:
                logits = sliding_window_inference(
                    imgs,
                    roi_size=task_cfg.patch_size,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=predictor,
                    overlap=SW_OVERLAP,
                    mode="gaussian",
                )
            except TypeError:
                logits = sliding_window_inference(
                    imgs,
                    roi_size=task_cfg.patch_size,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=predictor,
                    overlap=SW_OVERLAP,
                    mode="constant",
                )
        else:
            logits = _ensure_tensor_output(model(imgs))

        preds = torch.argmax(logits, dim=1, keepdim=True)
        gts = batch["label"].to(DEVICE)

        preds_oh = torch.nn.functional.one_hot(
            preds.squeeze(1).long(), task_cfg.num_classes
        ).permute(0, 4, 1, 2, 3).float()
        gts_oh = torch.nn.functional.one_hot(
            gts.squeeze(1).long(), task_cfg.num_classes
        ).permute(0, 4, 1, 2, 3).float()

        d = dice_metric(y_pred=preds_oh, y=gts_oh)
        h = hd95_metric(y_pred=preds_oh, y=gts_oh)
        s = surf_metric(y_pred=preds_oh, y=gts_oh)
        agg.update_batch(preds_oh, gts_oh, metrics={"dice": d, "hd95": h, "surf": s})

    per_class = agg.finalize()
    out = pd.DataFrame(per_class)
    out.insert(0, "Model", name)
    return out


def run_per_class_ablation(
    task_cfg: TaskConfig,
    loader_tr: DataLoader,
    loader_va: DataLoader,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    fold_dir = os.path.join(out_dir, "ablation_fold")
    os.makedirs(fold_dir, exist_ok=True)

    model_zoo = build_model_zoo(
        in_channels=task_cfg.in_channels,
        num_classes=task_cfg.num_classes,
        patch_size=task_cfg.patch_size,
    )

    footprints = {}
    for mname, m in model_zoo.items():
        gf, params = params_gflops(
            m, inp=(1, task_cfg.in_channels, *task_cfg.patch_size)
        )
        footprints[mname] = {"GFLOPs@64^3": gf, "Params": params}
        size_mb = (params * 4) / 1e6
        gf_str = f"{gf:.2f}" if np.isfinite(gf) else "n/a"
        print(f"[Footprint] {mname}: params={params/1e6:.2f}M | size≈{size_mb:.2f}MB | GFLOPs@{task_cfg.patch_size[0]}^3={gf_str}")
    pd.DataFrame.from_dict(footprints, orient="index").to_csv(
        os.path.join(out_dir, "model_complexity.csv")
    )

    all_per_class_rows = []
    model_level_rows = []
    for mname, mdl in model_zoo.items():
        print(f"\n=== ABLATION (Per-Class Only) | {mname} ===")
        _ = train_one(copy.deepcopy(mdl), task_cfg, loader_tr, loader_va, mname, fold_dir)
        per_class_df = evaluate_per_class(copy.deepcopy(mdl), task_cfg, loader_va, mname, fold_dir)
        all_per_class_rows.append(per_class_df)

        mean_macro_dice = per_class_df["Dice_macro"].mean(skipna=True)
        mean_micro_dice = per_class_df["Dice_micro"].mean(skipna=True)
        mean_f1_micro = per_class_df["F1_micro"].mean(skipna=True)
        model_level_rows.append({
            "Model": mname,
            "Mean Dice_macro": float(mean_macro_dice),
            "Mean Dice_micro": float(mean_micro_dice),
            "Mean F1_micro": float(mean_f1_micro),
            "GFLOPs@64^3": float(footprints[mname]["GFLOPs@64^3"]) if np.isfinite(footprints[mname]["GFLOPs@64^3"]) else np.nan,
            "Params (M)": footprints[mname]["Params"] / 1e6,
        })

    per_class_summary = pd.concat(all_per_class_rows, ignore_index=True)
    per_class_summary.to_csv(
        os.path.join(out_dir, "per_class_summary.csv"), index=False
    )

    summary_models = pd.DataFrame(model_level_rows).sort_values(
        "Mean Dice_macro", ascending=False
    )
    summary_models.to_csv(
        os.path.join(out_dir, "summary_models.csv"), index=False
    )

    print("\n=== DONE (metrics) ===")
    print(f"Per-class summary: {os.path.abspath(os.path.join(out_dir, 'per_class_summary.csv'))}")
    print(f"Model summary:     {os.path.abspath(os.path.join(out_dir, 'summary_models.csv'))}")

    return per_class_summary, summary_models, model_zoo
