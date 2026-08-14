"""Point-annotation dataset in APGCC's native format, with the two additions
that fine-tuning on your own cameras actually needs.

THE FORMAT (upstream, unchanged — keep it, it is refreshingly simple)
---------------------------------------------------------------------
    <root>/train.list      one line per sample:  "images/f001.jpg labels/f001.txt"
    <root>/labels/f001.txt one head per line:    "x y"   (space separated, px)

An EMPTY .txt is a valid, useful sample. See HARD NEGATIVES below.

ADDITION 1 — hard negatives
---------------------------
Measured on this project's own Nashik footage: both P2PNet and APGCC place heads
on tin roofing, empty road, rooftops, marigold garlands, litter and mud-flat
plant stubble. On one sparse market frame P2PNet reported 40 heads where ~20-25
people existed, the surplus sitting on roofs and bare tarmac.

A crop containing that texture and an empty label file teaches the model
"this is not a person" directly. It is the cheapest accuracy in the whole
pipeline, and standard random cropping under-samples it badly — an empty
rooftop is exactly the region a crowd-centric sampler skips. ``neg_ratio``
forces their share of each epoch.

ADDITION 2 — mixing public data with your own
---------------------------------------------
Fine-tuning a few hundred Nashik frames onto ShanghaiTech weights with no replay
walks the model off the public distribution — it gets better at Goda Ghat and
worse everywhere, including the next camera you install. ``MixedPointDataset``
interleaves a public set at a fixed ratio so each epoch sees both. 50/50 is the
default because it is the conservative choice, not because it is optimal; sweep
it once you have an eval set worth trusting.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

# WINDOWS DATALOADER FIX (learned the hard way on the fire kit): with
# num_workers>0, OpenCV's thread pool deadlocks against PyTorch workers on
# Windows spawn -- the loop hangs at the first batch with 0 CPU. Disabling
# OpenCV threading everywhere makes num_workers>0 usable.
cv2.setNumThreads(0)

def worker_init(worker_id: int) -> None:
    cv2.setNumThreads(0)
from torch.utils.data import Dataset

from _common import get_logger

log = get_logger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_points(path: Path) -> np.ndarray:
    """Read an APGCC .txt label. Missing or empty file -> (0, 2) — a valid negative."""
    if not path.is_file():
        return np.zeros((0, 2), dtype=np.float32)
    pts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def write_points(path: Path, points: np.ndarray) -> None:
    """Write an APGCC .txt label. An empty array writes an empty file on purpose."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{float(x):.2f} {float(y):.2f}" for x, y in np.asarray(points).reshape(-1, 2)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_list(list_file: Path) -> list[tuple[Path, Path]]:
    """Parse a .list file into (image, label) absolute path pairs."""
    root = list_file.parent
    pairs: list[tuple[Path, Path]] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        img, gt = (root / parts[0]).resolve(), (root / parts[1]).resolve()
        if img.is_file():
            pairs.append((img, gt))
        else:
            log.warning("listed image missing, skipping: %s", img)
    return pairs


@dataclass
class AugConfig:
    crop_size: int = 128        # upstream SHHA default; the model never sees a whole frame
    crops_per_image: int = 4    # upstream CROP_NUMBER
    flip_prob: float = 0.5
    scale_jitter: tuple[float, float] | None = (0.8, 1.25)
    #: Colour jitter matters here specifically because ghat scenes swing from
    #: dawn fog to hard noon sun to sodium floodlight on the same camera.
    brightness_jitter: float = 0.2


def _normalise(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1)


class PointDataset(Dataset):
    """Random crops + points, in the (samples, targets) shape APGCC's criterion wants."""

    def __init__(self, list_file: str | Path, train: bool = True,
                 aug: AugConfig | None = None):
        self.list_file = Path(list_file)
        self.pairs = read_list(self.list_file)
        if not self.pairs:
            raise ValueError(f"no usable samples in {self.list_file}")
        self.train = train
        self.aug = aug or AugConfig()
        n_neg = sum(1 for _, g in self.pairs if len(read_points(g)) == 0)
        log.info("%s: %d samples (%d hard negatives) from %s",
                 "train" if train else "val", len(self.pairs), n_neg, self.list_file.name)

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, img: np.ndarray, pts: np.ndarray):
        a = self.aug
        if a.scale_jitter:
            s = random.uniform(*a.scale_jitter)
            # Never shrink below the crop size or the crop step has nothing to take.
            min_s = max(a.crop_size / img.shape[0], a.crop_size / img.shape[1], 0.05)
            s = max(s, min_s)
            if abs(s - 1.0) > 1e-3:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                pts = pts * s
        if a.brightness_jitter > 0:
            f = 1.0 + random.uniform(-a.brightness_jitter, a.brightness_jitter)
            img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
        return img, pts

    def _crops(self, img: np.ndarray, pts: np.ndarray):
        a = self.aug
        h, w = img.shape[:2]
        cs = a.crop_size
        # Pad rather than skip: a 512x512 patch downscaled by jitter can land under
        # the crop size, and silently dropping those biases the sampler.
        if h < cs or w < cs:
            ph, pw = max(cs, h), max(cs, w)
            canvas = np.zeros((ph, pw, 3), img.dtype)
            canvas[:h, :w] = img
            img, h, w = canvas, ph, pw

        out = []
        for _ in range(a.crops_per_image):
            x0 = random.randint(0, w - cs)
            y0 = random.randint(0, h - cs)
            patch = img[y0:y0 + cs, x0:x0 + cs]
            if len(pts):
                m = ((pts[:, 0] >= x0) & (pts[:, 0] < x0 + cs)
                     & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + cs))
                p = pts[m] - np.array([x0, y0], dtype=np.float32)
            else:
                p = np.zeros((0, 2), dtype=np.float32)
            if random.random() < a.flip_prob:
                patch = patch[:, ::-1].copy()
                if len(p):
                    p[:, 0] = cs - p[:, 0]
            out.append((patch, p))
        return out

    def __getitem__(self, idx: int):
        img_path, gt_path = self.pairs[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"unreadable image: {img_path}")
        pts = read_points(gt_path)

        if not self.train:
            # Whole image, no crops — validation must measure the real frame.
            return _normalise(img), [{
                "point": torch.from_numpy(pts.astype(np.float32)),
                "labels": torch.ones(len(pts), dtype=torch.long),
                "name": img_path.name,
            }]

        img, pts = self._augment(img, pts)
        samples, targets = [], []
        for patch, p in self._crops(img, pts):
            samples.append(_normalise(patch))
            targets.append({
                "point": torch.from_numpy(p.astype(np.float32)),
                "labels": torch.ones(len(p), dtype=torch.long),
                "name": img_path.name,
            })
        return torch.stack(samples), targets


def collate(batch):
    """Flatten per-image crop groups into one batch. Targets stay a ragged list."""
    imgs, tgts = [], []
    for sample, target in batch:
        if sample.dim() == 4:          # train: (crops, 3, H, W)
            imgs.append(sample)
            tgts.extend(target)
        else:                          # val: (3, H, W)
            imgs.append(sample.unsqueeze(0))
            tgts.extend(target)
    return torch.cat(imgs, 0), tgts


class MixedPointDataset(Dataset):
    """Interleave your own data with a public set to prevent catastrophic forgetting.

    ``own_ratio=0.5`` means each epoch draws half its samples from your list and
    half from the public list, regardless of their relative sizes — so 300 Nashik
    frames are not drowned by 300k public ones, and the public set still anchors
    the features.
    """

    def __init__(self, own: PointDataset, public: PointDataset | None,
                 own_ratio: float = 0.5, length: int | None = None):
        self.own = own
        self.public = public
        self.own_ratio = 1.0 if public is None else float(own_ratio)
        self.length = length or (len(own) if public is None
                                 else int(len(own) / max(self.own_ratio, 1e-6)))
        log.info("mixed dataset: %d/epoch, own_ratio=%.2f (own=%d, public=%s)",
                 self.length, self.own_ratio, len(own),
                 len(public) if public else "none")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        use_own = self.public is None or random.random() < self.own_ratio
        ds = self.own if use_own else self.public
        return ds[random.randrange(len(ds))]
