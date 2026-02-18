#!/usr/bin/env python3
import os
import argparse

from monai_arch_benchmark.config import load_task_config
from monai_arch_benchmark.data import list_pairs, build_train_val_loaders, build_viz_loader
from monai_arch_benchmark.engine import run_per_class_ablation, SEED
from monai_arch_benchmark.env_logging import log_environment
from monai_arch_benchmark.viz import render_side_by_side_overlays_fullbrain
from monai_arch_benchmark.models import build_model_zoo


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Spleen MSD ablation with MONAI architecture benchmark."
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/spleen.yaml)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="./ablation_results",
        help="Output directory for results and overlays.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    task_cfg = load_task_config(args.config)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    log_environment(
        out_dir=out_dir,
        seed=SEED,
        patch_size=task_cfg.patch_size,
        target_spacing=task_cfg.target_spacing,
        speed_mode=True,
    )

    data = list_pairs(task_cfg.images_dir, task_cfg.labels_dir)
    loader_tr, loader_va = build_train_val_loaders(
        data=data, task_cfg=task_cfg, batch_size=1, seed=SEED
    )

    per_class_df, model_df, model_zoo = run_per_class_ablation(
        task_cfg=task_cfg,
        loader_tr=loader_tr,
        loader_va=loader_va,
        out_dir=out_dir,
    )

    viz_loader = build_viz_loader(
        data=data,
        task_cfg=task_cfg,
        seed=SEED,
    )

    # For viz we re-build the zoo to have "fresh" uncompiled instances
    fresh_zoo = build_model_zoo(
        in_channels=task_cfg.in_channels,
        num_classes=task_cfg.num_classes,
        patch_size=task_cfg.patch_size,
    )

    render_side_by_side_overlays_fullbrain(
        out_dir=out_dir,
        task_cfg=task_cfg,
        model_zoo=fresh_zoo,
        viz_loader=viz_loader,
        axis="z",
        max_cases=3,
        out_prefix="overlay_full",
        flair_channel=0,
        overlay_alpha=0.35,
    )

    print("\nArtifacts written to:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
