"""APGCC inference: BGR frame -> head points.

ONE preprocessing path, used by the annotation tool, the evaluation suite, the
HAVAN adapter and the fine-tuning validation loop. If these ever diverge, the
numbers stop being comparable and every conclusion drawn from them is void — so
they all call ``preprocess`` here.

The scale question, which is the whole game
-------------------------------------------
Measured on this project's own imagery (docs/13-apgcc-finetuning.md): a point
head trained on ShanghaiTech Part A has a NARROW head-size operating range,
roughly 5-30 px. Feed it heads outside that band, in either direction, and the
count collapses. On one hazy ghat photo P2PNet went 324 -> 3 detections as the
input long side rose 900 -> 4160 px. That is not a resolution bug; it is scale
sensitivity, and APGCC inherits it because it trains on the same data.

Consequence: ``max_long_side`` is not a performance knob, it is an ACCURACY knob,
and the right value depends on how big heads are in your camera's frame. The
1280 default is inherited from HAVAN and is a reasonable starting point for
1-2 MP CCTV; it is NOT right for 4K, where you want tiling instead (see
netra.infer.tiling). Set it deliberately per camera and record the choice.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from _common import get_logger

log = get_logger(__name__)

#: VGG16-BN needs dims divisible by 8; upstream pads to 128 and so do we, so
#: that anchor-point generation lands identically to the reference implementation.
PAD_MULTIPLE = 128

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@dataclass(frozen=True)
class PreppedFrame:
    tensor: torch.Tensor       # (1, 3, H_pad, W_pad)
    valid_hw: tuple[int, int]  # content region inside the padding
    scale_back: float          # multiply predicted xy by this to get source px


def preprocess(frame_bgr: np.ndarray, device: str,
               max_long_side: int | None = 1280) -> PreppedFrame:
    """Resize (optional), pad to a multiple of 128, normalise, to device."""
    oh, ow = frame_bgr.shape[:2]
    scale = 1.0
    if max_long_side:
        scale = min(1.0, max_long_side / float(max(oh, ow)))
    if scale < 1.0:
        frame_bgr = cv2.resize(
            frame_bgr, (int(round(ow * scale)), int(round(oh * scale))),
            interpolation=cv2.INTER_AREA,
        )
    h, w = frame_bgr.shape[:2]
    ph = ((h + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    pw = ((w + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    if (ph, pw) != (h, w):
        canvas = np.zeros((ph, pw, 3), dtype=frame_bgr.dtype)
        canvas[:h, :w] = frame_bgr
    else:
        canvas = frame_bgr

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = _NORM(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0).to(device)
    return PreppedFrame(tensor, (h, w), 1.0 / scale if scale else 1.0)


@torch.no_grad()
def predict_points(model: nn.Module, frame_bgr: np.ndarray, device: str = "cuda",
                   conf: float = 0.5, max_long_side: int | None = 1280,
                   fp16: bool = False) -> np.ndarray:
    """Return an (N, 3) float32 array of (x, y, score) in ORIGINAL frame pixels.

    Points landing in the zero-padded margin are dropped: the pad is black, the
    model can and does hallucinate on the seam, and those points have no source
    pixel behind them.
    """
    prepped = preprocess(frame_bgr, device, max_long_side)
    h, w = prepped.valid_hw

    use_amp = fp16 and str(device).startswith("cuda")
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(prepped.tensor)
    else:
        out = model(prepped.tensor)

    scores = torch.softmax(out["pred_logits"].float(), dim=-1)[0, :, 1].cpu().numpy()
    pts = out["pred_points"][0].float().cpu().numpy()
    print(f"    [debug] raw scores: min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f} n={len(scores)}")
    print(f"    [debug] raw pts x=[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] y=[{pts[:,1].min():.1f},{pts[:,1].max():.1f}]  valid_hw=({h},{w})")

    keep = (
        (scores > conf)
        & (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
        & (pts[:, 0] < w) & (pts[:, 1] < h)
    )
    pts = pts[keep] * prepped.scale_back
    return np.concatenate([pts, scores[keep, None]], axis=1).astype(np.float32)


class ApgccPredictor:
    """Reusable predictor holding one model. Build once, call many times."""

    def __init__(self, weights=None, device: str = "cuda", conf: float = 0.5,
                 max_long_side: int | None = 1280, fp16: bool = False,
                 config: str = "shha"):
        from apgcc_loader import load_apgcc
        self.model, self.info = load_apgcc(weights, device=device, config=config)
        self.device = device
        self.conf = conf
        self.max_long_side = max_long_side
        self.fp16 = fp16

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        return predict_points(self.model, frame_bgr, self.device, self.conf,
                              self.max_long_side, self.fp16)

    def count(self, frame_bgr: np.ndarray) -> int:
        return len(self(frame_bgr))
