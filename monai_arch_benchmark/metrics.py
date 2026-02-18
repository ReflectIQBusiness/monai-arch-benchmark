from typing import Dict, List

import numpy as np
import torch


class PerClassAggregator:
    """
    Keeps per-class totals ONLY (no per-case storage):
      - MICRO voxel-pooled: TP, FP, FN for precision/recall/F1 and micro Dice
      - MACRO over present-cases: running sum & count for Dice / HD95 / Surface
    """

    def __init__(self, num_classes: int, label_names: Dict[int, str], include_background: bool = False):
        self.num_classes = num_classes
        self.label_names = label_names
        self.start_idx = 1 if not include_background else 0

        C = num_classes
        self.TP = np.zeros(C, dtype=np.float64)
        self.FP = np.zeros(C, dtype=np.float64)
        self.FN = np.zeros(C, dtype=np.float64)

        self.dice_sum = np.zeros(C, dtype=np.float64)
        self.dice_cnt = np.zeros(C, dtype=np.float64)

        self.hd95_sum = np.zeros(C, dtype=np.float64)
        self.hd95_cnt = np.zeros(C, dtype=np.float64)

        self.surf_sum = np.zeros(C, dtype=np.float64)
        self.surf_cnt = np.zeros(C, dtype=np.float64)

    @torch.no_grad()
    def update_batch(self, preds_oh: torch.Tensor, gts_oh: torch.Tensor, metrics=None):
        """
        preds_oh, gts_oh: [B, C, H, W, D] float(0/1)
        metrics (optional): dict with tensors:
           'dice': [B, C-1], 'hd95': [B, C-1], 'surf': [B, C-1]
        """
        B, C, H, W, D = preds_oh.shape

        # MICRO counts
        p = preds_oh.cpu().numpy().astype(np.bool_)
        t = gts_oh.cpu().numpy().astype(np.bool_)
        for c in range(self.start_idx, C):
            pc = p[:, c]
            tc = t[:, c]
            tp = np.logical_and(pc, tc).sum()
            fp = np.logical_and(pc, np.logical_not(tc)).sum()
            fn = np.logical_and(np.logical_not(pc), tc).sum()
            self.TP[c] += tp
            self.FP[c] += fp
            self.FN[c] += fn

        # MACRO over present cases
        if metrics is not None and "dice" in metrics:
            d = metrics["dice"].cpu().numpy()
            present = (gts_oh[:, 1:, ...].sum(dim=(2, 3, 4)) > 0).cpu().numpy()
            for bi in range(d.shape[0]):
                for j in range(d.shape[1]):
                    c = j + 1
                    if present[bi, j] and np.isfinite(d[bi, j]):
                        self.dice_sum[c] += float(d[bi, j])
                        self.dice_cnt[c] += 1.0

        if metrics is not None and "hd95" in metrics:
            h = metrics["hd95"].cpu().numpy()
            present = (gts_oh[:, 1:, ...].sum(dim=(2, 3, 4)) > 0).cpu().numpy()
            for bi in range(h.shape[0]):
                for j in range(h.shape[1]):
                    c = j + 1
                    if present[bi, j] and np.isfinite(h[bi, j]):
                        self.hd95_sum[c] += float(h[bi, j])
                        self.hd95_cnt[c] += 1.0

        if metrics is not None and "surf" in metrics:
            s = metrics["surf"].cpu().numpy()
            present = (gts_oh[:, 1:, ...].sum(dim=(2, 3, 4)) > 0).cpu().numpy()
            for bi in range(s.shape[0]):
                for j in range(s.shape[1]):
                    c = j + 1
                    if present[bi, j] and np.isfinite(s[bi, j]):
                        self.surf_sum[c] += float(s[bi, j])
                        self.surf_cnt[c] += 1.0

    def finalize(self) -> List[Dict]:
        res = []
        for c in range(1, self.num_classes):
            cname = self.label_names.get(c, f"class_{c}")
            TP, FP, FN = self.TP[c], self.FP[c], self.FN[c]
            prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            rec = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            f1 = (
                2 * prec * rec / (prec + rec)
                if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0)
                else np.nan
            )
            dice_micro = (
                2 * TP / (2 * TP + FP + FN)
                if (2 * TP + FP + FN) > 0
                else np.nan
            )

            dice_macro = (
                self.dice_sum[c] / self.dice_cnt[c]
                if self.dice_cnt[c] > 0
                else np.nan
            )
            hd95_mean = (
                self.hd95_sum[c] / self.hd95_cnt[c]
                if self.hd95_cnt[c] > 0
                else np.nan
            )
            surf_mean = (
                self.surf_sum[c] / self.surf_cnt[c]
                if self.surf_cnt[c] > 0
                else np.nan
            )

            res.append({
                "Class": cname,
                "Dice_micro": float(dice_micro) if np.isfinite(dice_micro) else np.nan,
                "Dice_macro": float(dice_macro) if np.isfinite(dice_macro) else np.nan,
                "Precision_micro": float(prec) if np.isfinite(prec) else np.nan,
                "Recall_micro": float(rec) if np.isfinite(rec) else np.nan,
                "F1_micro": float(f1) if np.isfinite(f1) else np.nan,
                "HD95_mean": float(hd95_mean) if np.isfinite(hd95_mean) else np.nan,
                "SurfaceDist_mean": float(surf_mean) if np.isfinite(surf_mean) else np.nan,
                "Dice_macro_count": float(self.dice_cnt[c]),
                "HD95_count": float(self.hd95_cnt[c]),
                "Surface_count": float(self.surf_cnt[c]),
                "TP_vox": float(TP),
                "FP_vox": float(FP),
                "FN_vox": float(FN),
            })
        return res
