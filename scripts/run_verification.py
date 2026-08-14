"""Independent verification: run DM-Count (SH-A, QNRF) and APGCC (SH-A) on your
own test images and see the real predicted counts — before approving any of
them for production.

This does NOT trust the junior's reported MAE/bias numbers. It runs the actual
checkpoints on actual images and shows you the actual output, so you can
eyeball it against a manual head count yourself.

Usage
-----
    python scripts/run_verification.py --images_dir data/images

    # Only run specific models:
    python scripts/run_verification.py --images_dir data/images --models dmcount_sha apgcc

    # If your checkpoints live somewhere else:
    python scripts/run_verification.py --images_dir data/images \\
        --dmcount_sha_weights D:/path/dmcount_official_sh_a.pth \\
        --apgcc_weights D:/path/head_count_apgcc.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "apgcc_kit"))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------
# DM-Count: minimal VGG19 density-map model matching cvlab-stonybrook/DM-Count
# (official repo). Self-contained so this project doesn't depend on
# crowd-safety-testbed's code.
# --------------------------------------------------------------------------
class DMCountVGG19(nn.Module):
    """VGG19 backbone + regression head producing a stride-8 density map.
    Architecture matches the official DM-Count checkpoints (dmcount_official_*.pth).
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19_bn(weights=None)
        features = list(vgg.features.children())
        # Stop after the 4th max-pool block -> stride 8 feature map (matches
        # DM-Count's stated stride-8 output).
        self.features = nn.Sequential(*features[:-10])
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reg_layer(x)
        x = self.density_layer(x)
        return x


class DMCountPredictor:
    """Loads a DM-Count checkpoint and predicts a head count for a BGR frame."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = device
        self.model = DMCountVGG19().to(device).eval()
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [!] DM-Count partial load from {os.path.basename(weights_path)}: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected keys — "
                  f"architecture may not exactly match this checkpoint.")

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        h, w = rgb.shape[:2]
        # Pad to a multiple of 32 for clean pooling.
        ph, pw = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
        canvas = np.zeros((ph, pw, 3), dtype=np.float32)
        canvas[:h, :w] = rgb
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).to(self.device)
        density = self.model(tensor)[0, 0].cpu().numpy()
        count = float(density.sum())
        return density, count


# --------------------------------------------------------------------------
# APGCC: use the uploaded kit's own predictor directly.
# --------------------------------------------------------------------------
def load_apgcc_predictor(weights_path: str, device: str, conf: float, max_long_side):
    from apgcc_infer import ApgccPredictor
    return ApgccPredictor(weights=weights_path, device=device, conf=conf,
                          max_long_side=max_long_side)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def find_images(images_dir: str) -> list[str]:
    paths = []
    for name in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            paths.append(os.path.join(images_dir, name))
    return paths


def density_heatmap(dm: np.ndarray) -> np.ndarray:
    d = dm.copy()
    if d.max() > 0:
        d = d / d.max()
    d8 = (d * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


def draw_points_overlay(frame: np.ndarray, points: np.ndarray, count: float, label: str) -> np.ndarray:
    out = frame.copy()
    for x, y in points:
        cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
    text = f"{label}: {count:.1f} heads"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.rectangle(out, (10, 10), (20 + tw, 40 + th), (0, 0, 0), -1)
    cv2.putText(out, text, (15, 35 + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 255, 0), 3, cv2.LINE_AA)
    return out


def draw_count_only_overlay(frame: np.ndarray, count: float, label: str) -> np.ndarray:
    out = frame.copy()
    text = f"{label}: {count:.1f} heads (density-map model, no point locations)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (10, 10), (20 + tw, 38 + th), (0, 0, 0), -1)
    cv2.putText(out, text, (15, 32 + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default=os.path.join(ROOT, "data", "images"))
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--models", nargs="+",
                    default=["dmcount_sha", "dmcount_qnrf", "apgcc"],
                    choices=["dmcount_sha", "dmcount_qnrf", "apgcc"],
                    help="Which models to run.")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")

    ap.add_argument("--dmcount_sha_weights",
                    default=os.path.join(ROOT, "model_weights", "dmcount", "dmcount_official_sh_a.pth"))
    ap.add_argument("--dmcount_qnrf_weights",
                    default=os.path.join(ROOT, "model_weights", "dmcount", "dmcount_official_qnrf.pth"))
    ap.add_argument("--apgcc_weights",
                    default=os.path.join(ROOT, "apgcc_kit", "weights", "APGCC_SHHA_best.pth"))
    ap.add_argument("--apgcc_conf", type=float, default=0.5)
    ap.add_argument("--apgcc_max_long_side", type=int, default=1280,
                    help="Accuracy knob, not a speed knob — see apgcc_infer.py docstring. "
                         "1280 suits 1-2MP CCTV; for 4K, lower this or tile instead.")

    args = ap.parse_args()

    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"No such folder: {args.images_dir}")
    image_paths = find_images(args.images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.images_dir}")
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Found {len(image_paths)} image(s) in {args.images_dir}\n")

    predictors = {}

    if "dmcount_sha" in args.models:
        if os.path.exists(args.dmcount_sha_weights):
            predictors["dmcount_sha"] = ("DM-Count SH-A",
                                         DMCountPredictor(args.dmcount_sha_weights, args.device))
            print(f"[ok] DM-Count SH-A -> {args.dmcount_sha_weights}")
        else:
            print(f"[!] Skipping DM-Count SH-A: not found at {args.dmcount_sha_weights}")

    if "dmcount_qnrf" in args.models:
        if os.path.exists(args.dmcount_qnrf_weights):
            predictors["dmcount_qnrf"] = ("DM-Count QNRF",
                                          DMCountPredictor(args.dmcount_qnrf_weights, args.device))
            print(f"[ok] DM-Count QNRF -> {args.dmcount_qnrf_weights}")
        else:
            print(f"[!] Skipping DM-Count QNRF: not found at {args.dmcount_qnrf_weights}")

    if "apgcc" in args.models:
        if os.path.exists(args.apgcc_weights):
            try:
                predictors["apgcc"] = ("APGCC SH-A",
                                       load_apgcc_predictor(args.apgcc_weights, args.device,
                                                             args.apgcc_conf, args.apgcc_max_long_side))
                print(f"[ok] APGCC SH-A -> {args.apgcc_weights}")
            except Exception as e:
                print(f"[!] Failed to load APGCC: {e}")
                print("    Make sure apgcc_kit/apgcc/ contains the vendored upstream source "
                      "(see apgcc_kit/apgcc/PLACE_UPSTREAM_APGCC_HERE.txt)")
        else:
            print(f"[!] Skipping APGCC: not found at {args.apgcc_weights}")

    if not predictors:
        raise RuntimeError("No models loaded — check weight paths above.")

    rows = []
    print()
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[!] Could not read: {img_path}")
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f"{os.path.basename(img_path)}  ({frame.shape[1]}x{frame.shape[0]})")

        for key, (label, predictor) in predictors.items():
            if key.startswith("dmcount"):
                density, count = predictor.predict(frame)
                overlay = draw_count_only_overlay(frame, count, label)
                heat = density_heatmap(density)
                heat_path = os.path.join(args.results_dir, f"{stem}_{key}_density.png")
                cv2.imwrite(heat_path, heat)
                n_points = None
            else:  # apgcc — points, not density
                pts_scores = predictor(frame)  # (N, 3): x, y, score
                points = pts_scores[:, :2] if len(pts_scores) else np.zeros((0, 2))
                count = float(len(points))
                overlay = draw_points_overlay(frame, points, count, label)
                heat_path = ""
                n_points = int(count)

            overlay_path = os.path.join(args.results_dir, f"{stem}_{key}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)

            print(f"  {label:16s} count={count:8.2f}"
                  + (f"  points={n_points}" if n_points is not None else "")
                  + f"  -> {os.path.basename(overlay_path)}")

            rows.append({
                "image": os.path.basename(img_path),
                "model": label,
                "predicted_count": round(count, 2),
                "overlay_file": os.path.basename(overlay_path),
                "density_file": os.path.basename(heat_path) if heat_path else "",
            })
        print()

    csv_path = os.path.join(args.results_dir, "verification_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV: {csv_path}")
    print(f"Overlays and density maps: {args.results_dir}")
    print("\nNext: compare predicted_count against a manual head count on each "
          "image before trusting these models for production.")


if __name__ == "__main__":
    main()
