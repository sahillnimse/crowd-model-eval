"""Cut labelling patches out of videos and stills, chosen so the label set is balanced.

WHY PATCHES AND NOT WHOLE FRAMES
---------------------------------
APGCC trains on 128x128 random crops and never sees a whole frame. So training
labels do not need whole-frame annotation. A 512x512 patch with ~40 heads takes
about 3 minutes to label; a 4K ghat frame with 800 heads takes 40+ and people
make mistakes past the first ten minutes. Whole-frame ground truth is still
needed for EVALUATION, but that is ~10 frames, not 400.

WHY MODEL-GUIDED SAMPLING
-------------------------
Uniform random patches over ghat footage return mostly water, sky and stone. You
would spend the budget labelling nothing. This script runs APGCC first and uses
its (imperfect) density to stratify:

  dense   — the crush regime the system exists to measure. Hardest, most valuable.
  mid     — ordinary operating conditions.
  sparse  — where over-counting produces nuisance alerts.
  negative— HIGH TEXTURE, NO PREDICTED HEADS: tin roofing, tarpaulin, marigold
            garlands, mud stubble, stone steps, foliage. These are the crops that
            teach the model what is NOT a person, and uniform sampling almost
            never finds them. Measured on this project's footage, false positives
            on exactly these textures are the dominant sparse-scene error.

The model's density is a SAMPLING heuristic only. It never becomes a label —
you label the patch by hand. A biased sampler costs coverage; it cannot inject
wrong ground truth.

Usage
-----
  python scripts/extract_patches.py --sources D:/rfdetr/vids D:/rfdetr/dataset \\
      --out data/nashik/patches --n 400 --patch 512
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np


from apgcc_infer import ApgccPredictor            # noqa: E402
from _common import get_logger       # noqa: E402
from _common import seed_everything      # noqa: E402

log = get_logger("extract_patches")

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Patch buckets by PREDICTED head count, and each bucket's share of the budget.
#
# "negative" is 0-3 predicted heads, not strictly 0. On real crowd footage a
# strict-zero bucket never fills: measured on this project's clips, APGCC returns
# at least one head in almost every 512 px patch, which is the over-firing the
# hard negatives exist to correct. The most valuable negative is therefore not an
# empty prediction — it is a patch of tin roofing or garlands where the model
# DID fire and a human clears it to an empty label. Combined with --min-texture,
# this bucket targets exactly those regions.
BUCKETS = [
    ("dense", 40, 10_000, 0.30),
    ("mid", 12, 40, 0.25),
    ("sparse", 3, 12, 0.20),
    ("negative", 0, 3, 0.25),
]


def iter_frames(sources: list[Path], per_video: int, seed: int):
    rng = random.Random(seed)
    files: list[Path] = []
    for s in sources:
        if s.is_file():
            files.append(s)
        elif s.is_dir():
            files += [p for p in sorted(s.rglob("*"))
                      if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT]
    log.info("scanning %d media files", len(files))
    for f in files:
        if f.suffix.lower() in IMAGE_EXT:
            img = cv2.imread(str(f))
            if img is not None:
                yield f.stem, img
            continue
        cap = cv2.VideoCapture(str(f))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            continue
        idxs = sorted(rng.sample(range(n), min(per_video, n)))
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                yield f"{f.stem}_f{i:06d}", fr
        cap.release()


def texture_score(patch_bgr: np.ndarray) -> float:
    """Laplacian variance — how much high-frequency structure a patch carries.

    Used only to keep NEGATIVE patches interesting. A negative crop of flat sky
    teaches nothing; a negative crop of corrugated tin roofing or a marigold
    garland teaches exactly the discrimination the model is getting wrong.
    """
    g = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True, help="files and/or directories")
    ap.add_argument("--out", required=True, help="output patch directory")
    ap.add_argument("--n", type=int, default=400, help="total patches to emit")
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--stride-frac", type=float, default=0.75,
                    help="patch grid stride as a fraction of patch size")
    ap.add_argument("--per-video", type=int, default=6, help="frames sampled per video")
    ap.add_argument("--min-texture", type=float, default=60.0,
                    help="minimum Laplacian variance for a NEGATIVE patch to be kept")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1229)
    args = ap.parse_args()

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    predictor = ApgccPredictor(args.weights, device=args.device, conf=args.conf)
    ps, stride = args.patch, max(1, int(args.patch * args.stride_frac))

    pool: dict[str, list] = {b[0]: [] for b in BUCKETS}
    n_frames = 0
    for name, frame in iter_frames([Path(s) for s in args.sources], args.per_video, args.seed):
        h, w = frame.shape[:2]
        if h < ps or w < ps:
            continue
        n_frames += 1
        pts = predictor(frame)[:, :2]
        for y0 in range(0, h - ps + 1, stride):
            for x0 in range(0, w - ps + 1, stride):
                if len(pts):
                    m = ((pts[:, 0] >= x0) & (pts[:, 0] < x0 + ps)
                         & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + ps))
                    c = int(m.sum())
                else:
                    c = 0
                for bname, lo, hi, _share in BUCKETS:
                    if lo <= c < hi:
                        pool[bname].append((name, x0, y0, c))
                        break
        if n_frames % 25 == 0:
            log.info("scanned %d frames; pool=%s", n_frames,
                     {k: len(v) for k, v in pool.items()})

    log.info("candidate pool: %s", {k: len(v) for k, v in pool.items()})

    # Re-open sources once more to actually cut the chosen patches.
    chosen: dict[str, list] = {}
    for bname, _lo, _hi, share in BUCKETS:
        want = int(args.n * share)
        cands = pool[bname]
        rng.shuffle(cands)
        for c in cands:
            chosen.setdefault(c[0], []).append((bname, *c[1:]))
            want -= 1
            if want <= 0:
                break

    written = {b[0]: 0 for b in BUCKETS}
    manifest = []
    for name, frame in iter_frames([Path(s) for s in args.sources], args.per_video, args.seed):
        if name not in chosen:
            continue
        for bname, x0, y0, c in chosen[name]:
            patch = frame[y0:y0 + ps, x0:x0 + ps]
            if bname == "negative" and texture_score(patch) < args.min_texture:
                continue      # a blank sky negative teaches nothing
            stem = f"{bname}_{name}_{x0}_{y0}"
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), patch,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({"stem": stem, "bucket": bname, "src": name,
                             "x0": x0, "y0": y0, "model_count": c})
            written[bname] += 1

    import json
    (out / "patches.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %d patches to %s", sum(written.values()), out / "images")
    for k, v in written.items():
        log.info("  %-9s %d", k, v)
    print(f"\nNext:\n  python scripts/annotate.py --patches {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
