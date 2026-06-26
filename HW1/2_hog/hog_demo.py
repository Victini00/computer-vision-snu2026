"""
PASCAL VOC — HOG Detection Demo  (Interactive matplotlib demo)

Pages (press Next / Prev or use keyboard arrows):
  0  HOG Detection Evaluation (your detections.json vs ground_truth.json)
  1  Per-Detection IoU Visualizer

Keys:
  Right / Left arrow — navigate pages
  Q / ESC            — quit
"""

import sys
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image as PILImage

# Evaluation helpers

def calc_ap_all(pts):
    ap = 0.0
    max_p = 0.0
    env = [0.0] * len(pts)
    for i in range(len(pts) - 1, -1, -1):
        max_p = max(max_p, pts[i]["p"])
        env[i] = max_p
    for i in range(1, len(pts)):
        ap += (pts[i]["r"] - pts[i - 1]["r"]) * env[i]
    return max(0, ap)


# Page base class

class Page:
    """Base class for each demo page."""
    def __init__(self, fig, title):
        self.fig = fig
        self.title = title
        self.axes = []
        self.widgets = []

    def build(self):
        raise NotImplementedError

    def show(self):
        for ax in self.axes:
            ax.set_visible(True)
        for w in self.widgets:
            if hasattr(w, "ax"):
                w.ax.set_visible(True)

    def hide(self):
        for ax in self.axes:
            ax.set_visible(False)
        for w in self.widgets:
            if hasattr(w, "ax"):
                w.ax.set_visible(False)


# Detection helpers

def _compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _evaluate_detections(det_boxes, det_scores, gt_boxes, iou_thresh=0.5):
    """
    Evaluate detections against ground truth using VOC-style matching.

    Returns dict with: tp, fp, fn, precision, recall, f1, ap,
                       per-detection verdicts, per-gt matched flags,
                       iou_matrix, pr_points.
    """
    n_det = len(det_boxes)
    n_gt = len(gt_boxes)

    # IoU matrix
    iou_matrix = np.zeros((n_det, n_gt))
    for i in range(n_det):
        for j in range(n_gt):
            iou_matrix[i, j] = _compute_iou(det_boxes[i], gt_boxes[j])

    # Sort detections by score descending
    order = np.argsort(-np.array(det_scores))
    gt_matched = [False] * n_gt
    verdicts = [""] * n_det
    cum_tp = 0
    cum_fp = 0
    pr_points = [{"p": 1.0, "r": 0.0}]

    for idx in order:
        best_iou = 0
        best_gt = -1
        for j in range(n_gt):
            if iou_matrix[idx, j] > best_iou:
                best_iou = iou_matrix[idx, j]
                best_gt = j
        if best_iou >= iou_thresh and not gt_matched[best_gt]:
            gt_matched[best_gt] = True
            verdicts[idx] = "TP"
            cum_tp += 1
        else:
            verdicts[idx] = "FP"
            cum_fp += 1
        p = cum_tp / (cum_tp + cum_fp)
        r = cum_tp / n_gt if n_gt > 0 else 0
        pr_points.append({"p": p, "r": r})

    tp = cum_tp
    fp = cum_fp
    fn = sum(1 for m in gt_matched if not m)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / n_gt if n_gt > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0)
    ap = calc_ap_all(pr_points)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "ap": ap,
        "verdicts": verdicts, "gt_matched": gt_matched,
        "iou_matrix": iou_matrix, "pr_points": pr_points,
        "order": order,
    }


def _load_gt_and_dets(base_dir):
    """Load ground_truth.json and detections.json from base_dir."""
    gt_path = os.path.join(base_dir, "ground_truth.json")
    det_path = os.path.join(base_dir, "detections.json")
    gt_boxes, gt_labels = [], []
    det_boxes, det_scores = [], []
    img = None

    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt_data = json.load(f)
        for face in gt_data.get("faces", []):
            gt_boxes.append(face["bbox"])
            gt_labels.append(face.get("label", ""))
        img_name = gt_data.get("image", "target.png")
        img_path = os.path.join(base_dir, img_name)
        if os.path.exists(img_path):
            img = np.array(PILImage.open(img_path))

    if os.path.exists(det_path):
        with open(det_path) as f:
            det_data = json.load(f)
        for d in det_data.get("detections", []):
            det_boxes.append(d["bbox"])
            det_scores.append(d["score"])

    return gt_boxes, gt_labels, det_boxes, det_scores, img


# Page 0: HOG Detection Evaluation

class PageEvaluation(Page):
    def __init__(self, fig, base_dir):
        super().__init__(fig,
                         "HOG Detection Evaluation — Your Results vs Ground Truth")
        self.base_dir = base_dir

    def build(self):
        self.ax_img = self.fig.add_axes([0.03, 0.28, 0.42, 0.62])
        self.ax_pr = self.fig.add_axes([0.50, 0.52, 0.22, 0.38])
        self.ax_metrics = self.fig.add_axes([0.75, 0.28, 0.22, 0.62])
        self.ax_metrics.axis("off")
        self.ax_table = self.fig.add_axes([0.50, 0.28, 0.22, 0.20])
        self.ax_table.axis("off")

        self.ax_iou_sl = self.fig.add_axes([0.15, 0.16, 0.35, 0.03])
        self.sl_iou = Slider(self.ax_iou_sl, "IoU thr", 0.1, 0.9,
                             valinit=0.5, valstep=0.05)
        self.sl_iou.on_changed(self._update)

        self.axes = [self.ax_img, self.ax_pr, self.ax_metrics,
                     self.ax_table, self.ax_iou_sl]
        self.widgets = [self.sl_iou]

        self.gt_boxes, self.gt_labels, self.det_boxes, self.det_scores, \
            self.img = _load_gt_and_dets(self.base_dir)
        self._update(None)

    def _update(self, _):
        iou_t = self.sl_iou.val
        has_dets = len(self.det_boxes) > 0
        has_gt = len(self.gt_boxes) > 0

        if has_dets and has_gt:
            res = _evaluate_detections(
                self.det_boxes, self.det_scores,
                self.gt_boxes, iou_thresh=iou_t)
        else:
            res = None

        # ── Image with boxes ──
        ax = self.ax_img
        ax.clear()
        if self.img is not None:
            ax.imshow(self.img,
                      cmap="gray" if self.img.ndim == 2 else None)
        else:
            ax.text(0.5, 0.5, "target.png not found",
                    ha="center", va="center", fontsize=12, color="#999",
                    transform=ax.transAxes)

        # GT boxes (blue)
        for j, (box, label) in enumerate(
                zip(self.gt_boxes, self.gt_labels)):
            x1, y1, x2, y2 = box
            matched = res["gt_matched"][j] if res else False
            style = "-" if matched else "--"
            color = "#2563eb" if matched else "#d97706"
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor=color,
                                    facecolor="none", linestyle=style))
            tag = label + (" ✓" if matched else " FN")
            ax.text(x1, y1 - 2, tag, fontsize=7, color=color,
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none", pad=1))

        # Det boxes (green=TP, red=FP)
        if has_dets and res:
            for i, (box, score) in enumerate(
                    zip(self.det_boxes, self.det_scores)):
                x1, y1, x2, y2 = box
                is_tp = res["verdicts"][i] == "TP"
                color = "#16a34a" if is_tp else "#dc2626"
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=2, edgecolor=color,
                                        facecolor=color + "10"))
                tag = f"{'TP' if is_tp else 'FP'} {score:.2f}"
                ax.text(x1, y2 + 8, tag, fontsize=6.5, color=color,
                        fontfamily="monospace",
                        bbox=dict(facecolor="black", alpha=0.6,
                                  edgecolor="none", pad=1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Blue=GT  Green=TP  Red=FP  Dashed=FN", fontsize=8,
                     color="#666")

        # ── PR Curve ──
        ax2 = self.ax_pr
        ax2.clear()
        if res:
            pts = res["pr_points"]
            recalls = [p["r"] for p in pts]
            precisions = [p["p"] for p in pts]
            ax2.step(recalls, precisions, where="post", color="#2563eb",
                     linewidth=2)
            ax2.fill_between(recalls, precisions, step="post",
                             alpha=0.08, color="#2563eb")
            ax2.set_title(f"PR Curve  (AP={res['ap']*100:.1f}%)",
                          fontsize=9, fontweight="bold")
        else:
            ax2.set_title("PR Curve (no data)", fontsize=9)
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("Recall", fontsize=8)
        ax2.set_ylabel("Precision", fontsize=8)
        ax2.grid(True, alpha=0.15)

        # ── Metrics panel ──
        ai = self.ax_metrics
        ai.clear()
        ai.set_xlim(0, 1)
        ai.set_ylim(0, 1)
        ai.axis("off")

        if not has_gt:
            ai.text(0.05, 0.8, "ground_truth.json\nnot found",
                    fontsize=11, color="#dc2626")
        elif not has_dets:
            ai.text(0.05, 0.8, "detections.json\nnot found\n\n"
                    "Run hog.ipynb first\nto generate detections.",
                    fontsize=10, color="#d97706")
        else:
            ai.text(0.05, 0.95, "Evaluation Metrics", fontsize=12,
                    fontweight="bold")
            ai.text(0.05, 0.87, f"IoU threshold: {iou_t:.2f}",
                    fontsize=10, fontfamily="monospace", color="#666")

            y = 0.78
            metrics = [
                ("TP",        str(res["tp"]),                    "#16a34a"),
                ("FP",        str(res["fp"]),                    "#dc2626"),
                ("FN",        str(res["fn"]),                    "#d97706"),
                ("Precision", f"{res['precision']*100:.1f}%",    "#16a34a"),
                ("Recall",    f"{res['recall']*100:.1f}%",       "#d97706"),
                ("F1",        f"{res['f1']:.3f}",                "#2563eb"),
                ("AP",        f"{res['ap']*100:.1f}%",           "#7c3aed"),
            ]
            for label, val, color in metrics:
                ai.text(0.05, y, f"{label}:", fontsize=10, color="#666")
                ai.text(0.55, y, val, fontsize=13, fontweight="bold",
                        color=color, fontfamily="monospace")
                y -= 0.08

            # Summary
            ai.text(0.05, 0.15, f"GT faces: {len(self.gt_boxes)}",
                    fontsize=9, color="#666")
            ai.text(0.05, 0.08, f"Detections: {len(self.det_boxes)}",
                    fontsize=9, color="#666")

        # ── IoU detail table ──
        at = self.ax_table
        at.clear()
        at.set_xlim(0, 1)
        at.set_ylim(0, 1)
        at.axis("off")

        if res and has_dets:
            at.text(0.05, 0.92, "Det IoU details:", fontsize=8,
                    fontweight="bold", color="#333")
            y = 0.78
            order = res["order"]
            for idx in order:
                if y < 0:
                    break
                best_j = int(np.argmax(res["iou_matrix"][idx]))
                best_iou = res["iou_matrix"][idx, best_j]
                v = res["verdicts"][idx]
                vc = "#16a34a" if v == "TP" else "#dc2626"
                line = (f"s={self.det_scores[idx]:.2f}  "
                        f"IoU={best_iou:.2f}  {v}")
                at.text(0.05, y, line, fontsize=7,
                        fontfamily="monospace", color=vc)
                y -= 0.14
            if len(res["order"]) == 5:
                is_valid = True
                for idx in order:
                    if res["verdicts"][idx] != "TP":
                        is_valid = False
                if is_valid:
                    mean_IoU = 0
                    for idx in order:
                        best_j = int(np.argmax(res["iou_matrix"][idx]))
                        best_iou = res["iou_matrix"][idx, best_j]
                        mean_IoU += best_iou
                        
                    mean_IoU = mean_IoU / 5
                    at.text(0.05, y, f"Mean IoU: {mean_IoU:.4f}", fontsize=8,
                    fontweight="bold", color="#333")              
            else:
                at.text(0.05, y, f"Make sure you have exactly 5 detections,", fontsize=8,
                    fontweight="bold", color="#dc2626") 
                at.text(0.05, y - 0.14, f"all of which are true positives.", fontsize=8,
                    fontweight="bold", color="#dc2626") 
                    

        self.fig.canvas.draw_idle()


# Page 1: Per-Detection IoU Visualizer

class PageDetectionIoU(Page):
    """Show a single GT box's IoU with its best-matching detection."""

    def __init__(self, fig, base_dir, gt_index, gt_boxes, gt_labels,
                 det_boxes, det_scores, img, res):
        label = gt_labels[gt_index] or f"GT{gt_index + 1}"
        super().__init__(fig,
                         f"Per-Detection IoU — GT #{gt_index + 1}: {label}")
        self.base_dir = base_dir
        self.idx = gt_index
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.det_boxes = det_boxes
        self.det_scores = det_scores
        self.img = img
        self._res = res

    def build(self):
        # Main image panel
        self.ax_img = self.fig.add_axes([0.03, 0.28, 0.42, 0.62])
        # Overlap diagram panel
        self.ax_diag = self.fig.add_axes([0.50, 0.30, 0.24, 0.58])
        self.ax_diag.set_aspect("equal")
        # Info panel
        self.ax_info = self.fig.add_axes([0.77, 0.28, 0.20, 0.62])
        self.ax_info.axis("off")

        self.axes = [self.ax_img, self.ax_diag, self.ax_info]
        self.widgets = []

        self._render()

    def _render(self):
        n_det = len(self.det_boxes)
        n_gt = len(self.gt_boxes)

        # ── image panel ──
        ax = self.ax_img
        ax.clear()
        if self.img is not None:
            ax.imshow(self.img, cmap="gray" if self.img.ndim == 2 else None)
        else:
            ax.text(0.5, 0.5, "target.png not found",
                    ha="center", va="center", fontsize=12, color="#999",
                    transform=ax.transAxes)

        if n_det == 0 or n_gt == 0:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("No data", fontsize=8, color="#999")
            self._clear_diag()
            self._clear_info("No detections.json / ground_truth.json")
            self.fig.canvas.draw_idle()
            return

        # Draw all other GT boxes faintly
        for j, box in enumerate(self.gt_boxes):
            if j == self.idx:
                continue
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=1, edgecolor="#2563eb",
                                    facecolor="none", linestyle="--", alpha=0.3))

        # Current GT box
        gt_box = self.gt_boxes[self.idx]
        gx1, gy1, gx2, gy2 = gt_box
        gt_label = self.gt_labels[self.idx] or f"GT{self.idx + 1}"

        # Find best-matching detection for this GT (highest IoU column)
        iou_col = (self._res["iou_matrix"][:, self.idx]
                   if self._res is not None else np.array([]))
        if len(iou_col):
            best_i = int(np.argmax(iou_col))
            best_iou = float(iou_col[best_i])
        else:
            best_i = 0
            best_iou = 0.0

        det_box = self.det_boxes[best_i]
        score = self.det_scores[best_i]
        dx1, dy1, dx2, dy2 = det_box
        verdict = "TP" if best_iou >= 0.5 else "FP"
        v_color = "#16a34a" if verdict == "TP" else "#dc2626"

        # GT box (highlighted)
        ax.add_patch(Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                                linewidth=2.5, edgecolor="#2563eb",
                                facecolor="#2563eb12"))
        ax.text(gx1, gy1 - 3, gt_label,
                fontsize=7, color="#2563eb", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

        # Best-matching det box
        ax.add_patch(Rectangle((dx1, dy1), dx2 - dx1, dy2 - dy1,
                                linewidth=2.5, edgecolor=v_color,
                                facecolor=v_color + "18"))
        ax.text(dx1, dy2 + 6,
                f"{verdict}  s={score:.2f}  IoU={best_iou:.3f}",
                fontsize=7, color=v_color, fontfamily="monospace",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1))

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"GT: {gt_label}  —  Blue=GT  {'Green' if verdict == 'TP' else 'Red'}=BestDet",
                     fontsize=8, color="#666")

        # ── overlap diagram ──
        ad = self.ax_diag
        ad.clear()
        all_coords = [gx1, gx2, dx1, dx2]
        all_y = [gy1, gy2, dy1, dy2]
        x_min, x_max = min(all_coords), max(all_coords)
        y_min, y_max = min(all_y), max(all_y)
        span_x = max(x_max - x_min, 1)
        span_y = max(y_max - y_min, 1)
        pad = 0.15
        W, H = 300, 260

        def norm_x(v):
            return (v - x_min) / span_x * W * (1 - 2 * pad) + W * pad

        def norm_y(v):
            return (v - y_min) / span_y * H * (1 - 2 * pad) + H * pad

        g_x1n, g_y1n = norm_x(gx1), norm_y(gy1)
        g_x2n, g_y2n = norm_x(gx2), norm_y(gy2)
        d_x1n, d_y1n = norm_x(dx1), norm_y(dy1)
        d_x2n, d_y2n = norm_x(dx2), norm_y(dy2)

        ad.set_xlim(0, W)
        ad.set_ylim(H, 0)
        ad.set_xticks([]); ad.set_yticks([])
        ad.set_facecolor("#f5f5f3")

        # GT
        ad.add_patch(Rectangle((g_x1n, g_y1n), g_x2n - g_x1n, g_y2n - g_y1n,
                                linewidth=2.5, edgecolor="#2563eb",
                                facecolor="#2563eb15"))
        ad.text(g_x1n + 4, g_y1n + 14, "GT", fontsize=9,
                fontweight="bold", color="#2563eb", fontfamily="monospace")

        # Det
        ad.add_patch(Rectangle((d_x1n, d_y1n), d_x2n - d_x1n, d_y2n - d_y1n,
                                linewidth=2.5, edgecolor=v_color,
                                facecolor=v_color + "15"))
        ad.text(d_x1n + 4, d_y1n + 14, "Det", fontsize=9,
                fontweight="bold", color=v_color, fontfamily="monospace")

        # Intersection
        ix1n = max(g_x1n, d_x1n); iy1n = max(g_y1n, d_y1n)
        ix2n = min(g_x2n, d_x2n); iy2n = min(g_y2n, d_y2n)
        if ix2n > ix1n and iy2n > iy1n:
            ad.add_patch(Rectangle((ix1n, iy1n), ix2n - ix1n, iy2n - iy1n,
                                    linewidth=1.5, edgecolor="#16a34a",
                                    facecolor="#16a34a30", linestyle="--"))

        ad.set_title(f"IoU = {best_iou:.3f}", fontsize=10, fontweight="bold",
                     color=v_color)

        # ── info panel ──
        ai = self.ax_info
        ai.clear()
        ai.set_xlim(0, 1); ai.set_ylim(0, 1)
        ai.axis("off")

        inter_area = (max(0, min(gx2, dx2) - max(gx1, dx1)) *
                      max(0, min(gy2, dy2) - max(gy1, dy1)))
        gt_area = (gx2 - gx1) * (gy2 - gy1)
        det_area = (dx2 - dx1) * (dy2 - dy1)
        union_area = gt_area + det_area - inter_area

        ai.text(0.05, 0.95, f"GT #{self.idx + 1}: {gt_label}", fontsize=11,
                fontweight="bold", color="#2563eb")
        ai.text(0.05, 0.87, f"Best Det: #{best_i + 1}", fontsize=10,
                fontfamily="monospace", color="#666")
        ai.text(0.05, 0.80, f"Score:    {score:.3f}", fontsize=10,
                fontfamily="monospace", color="#666")
        ai.text(0.05, 0.70,
                f"IoU = {best_iou:.3f}", fontsize=28, fontweight="light",
                color=v_color, fontfamily="serif")
        ai.text(0.05, 0.58,
                ("✓  TP  (IoU ≥ 0.5)" if verdict == "TP"
                 else "✗  FP  (IoU < 0.5)"),
                fontsize=11, fontweight="bold", color=v_color,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=v_color + "12",
                          edgecolor=v_color + "33"))
        ai.text(0.05, 0.46, "Intersection:", fontsize=9, color="#16a34a")
        ai.text(0.05, 0.40, f"  {inter_area:.0f} px²", fontsize=9,
                fontfamily="monospace", color="#16a34a")
        ai.text(0.05, 0.33, "Union:", fontsize=9, color="#666")
        ai.text(0.05, 0.27, f"  {union_area:.0f} px²", fontsize=9,
                fontfamily="monospace", color="#666")
        ai.text(0.05, 0.20, "GT area:", fontsize=9, color="#2563eb")
        ai.text(0.05, 0.14, f"  {gt_area:.0f} px²", fontsize=9,
                fontfamily="monospace", color="#2563eb")
        ai.text(0.05, 0.07, "Det area:", fontsize=9, color=v_color)
        ai.text(0.05, 0.01, f"  {det_area:.0f} px²", fontsize=9,
                fontfamily="monospace", color=v_color)

        self.fig.canvas.draw_idle()

    def _clear_diag(self):
        self.ax_diag.clear()
        self.ax_diag.axis("off")

    def _clear_info(self, msg=""):
        ai = self.ax_info
        ai.clear()
        ai.set_xlim(0, 1); ai.set_ylim(0, 1)
        ai.axis("off")
        if msg:
            ai.text(0.5, 0.5, msg, ha="center", va="center",
                    fontsize=10, color="#999", transform=ai.transAxes)


# Main Application

class DemoApp:
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor("white")

        # Title axis (always visible)
        self.ax_title = self.fig.add_axes([0.0, 0.94, 1.0, 0.06])
        self.ax_title.axis("off")

        # Navigation
        self.ax_nav_prev = self.fig.add_axes([0.38, 0.01, 0.06, 0.035])
        self.ax_nav_next = self.fig.add_axes([0.56, 0.01, 0.06, 0.035])
        self.ax_nav_label = self.fig.add_axes([0.44, 0.01, 0.12, 0.035])
        self.ax_nav_label.axis("off")

        self.btn_prev = Button(self.ax_nav_prev, "< Prev")
        self.btn_next = Button(self.ax_nav_next, "Next >")
        self.btn_prev.on_clicked(lambda _: self.go(-1))
        self.btn_next.on_clicked(lambda _: self.go(1))

        # Build pages
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Pre-load data once for IoU pages
        gt_boxes, gt_labels, det_boxes, det_scores, img = \
            _load_gt_and_dets(base_dir)
        if det_boxes and gt_boxes:
            res = _evaluate_detections(
                det_boxes, det_scores, gt_boxes, iou_thresh=0.5)
        else:
            res = None

        self.pages = [PageEvaluation(self.fig, base_dir)]
        for gi in range(len(gt_boxes)):
            self.pages.append(PageDetectionIoU(
                self.fig, base_dir, gi, gt_boxes, gt_labels,
                det_boxes, det_scores, img, res))
        for p in self.pages:
            p.build()

        self.current = 0
        self._show_page(0)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _show_page(self, idx):
        for i, p in enumerate(self.pages):
            if i == idx:
                p.show()
            else:
                p.hide()

        # Update title
        self.ax_title.clear()
        self.ax_title.axis("off")
        self.ax_title.text(0.5, 0.5, self.pages[idx].title,
                           fontsize=14, fontweight="bold", ha="center",
                           va="center", color="#18181a")

        # Update nav label
        self.ax_nav_label.clear()
        self.ax_nav_label.axis("off")
        self.ax_nav_label.text(0.5, 0.5,
                               f"{idx + 1} / {len(self.pages)}",
                               fontsize=11, ha="center", va="center",
                               fontfamily="monospace", color="#999")

        self.fig.canvas.draw_idle()

    def go(self, d):
        new = self.current + d
        if 0 <= new < len(self.pages):
            self.current = new
            self._show_page(self.current)

    def _on_key(self, event):
        if event.key in ("right", "down"):
            self.go(1)
        elif event.key in ("left", "up"):
            self.go(-1)
        elif event.key in ("q", "escape"):
            plt.close("all")
            sys.exit(0)

    def run(self):
        plt.show()


if __name__ == "__main__":
    app = DemoApp()
    app.run()
