"""Point-annotation tool: click heads, write APGCC-format labels.

Model-assisted: every patch opens pre-populated with APGCC's predictions, so the
job is CORRECTING (delete the dots on the tin roof, add the ones it missed)
rather than clicking from an empty image. That is roughly 4x faster and, more
importantly, keeps attention on the disagreements — which is where the training
signal lives.

    left click        add a head
    right click       delete nearest head (within 25 px)
    drag left         add a head at release point (so a slip does not misplace)
    n / SPACE         save and next
    p                 save and previous
    r                 reset to the model's prediction
    c                 clear all points  (use for hard negatives)
    u                 undo last change
    a                 re-run the model and ADD its points to what you have
    +/-               zoom in / out
    arrow keys        pan when zoomed
    h                 toggle help overlay
    q / ESC           save and quit

Labels are written on every navigation, so killing the window never loses work.
Output is exactly what apgcc_dataset and upstream APGCC expect:

    <patches>/images/<stem>.jpg
    <patches>/labels/<stem>.txt      one "x y" per line, EMPTY FILE = hard negative
    <patches>/train.list             "images/<stem>.jpg labels/<stem>.txt"
    <patches>/val.list               held-out split

An empty label file is a real, valuable sample — not a skipped one. Press `c`
on a patch of roofing or garlands and save it.

Usage
-----
  python scripts/annotate.py --patches data/nashik/patches
  python scripts/annotate.py --patches data/nashik/patches --start 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


from apgcc_dataset import read_points, write_points   # noqa: E402
from _common import get_logger                 # noqa: E402

log = get_logger("annotate")

WIN = "head-count annotate  —  left=add  right=delete  n=next  c=clear(negative)  h=help  q=quit"
DELETE_RADIUS = 25
HELP = [
    "left click / drag : add head        right click : delete nearest",
    "n or SPACE : next   p : prev        r : reset to model   c : clear (negative)",
    "u : undo            a : add model preds                  +/- : zoom  arrows : pan",
    "h : hide help       q/ESC : save and quit",
]


class Annotator:
    def __init__(self, patches: Path, predictor=None, start: int = 0,
                 val_frac: float = 0.15):
        self.root = patches
        self.img_dir = patches / "images"
        self.lbl_dir = patches / "labels"
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.files = sorted(self.img_dir.glob("*.jpg"))
        if not self.files:
            raise SystemExit(f"no images in {self.img_dir}")
        self.i = max(0, min(start, len(self.files) - 1))
        self.predictor = predictor
        self.val_frac = val_frac
        self.points: np.ndarray = np.zeros((0, 2), np.float32)
        self.undo: list[np.ndarray] = []
        self.zoom = 1.0
        self.ox = self.oy = 0
        self.show_help = True
        self.img: np.ndarray | None = None
        self.drag_from: tuple[int, int] | None = None

    # ---- persistence ----------------------------------------------------
    def label_path(self, idx: int) -> Path:
        return self.lbl_dir / (self.files[idx].stem + ".txt")

    def load(self) -> None:
        f = self.files[self.i]
        self.img = cv2.imread(str(f))
        if self.img is None:
            raise RuntimeError(f"unreadable: {f}")
        lp = self.label_path(self.i)
        if lp.exists():
            self.points = read_points(lp)          # resume prior work
        elif self.predictor is not None:
            self.points = self.predictor(self.img)[:, :2].astype(np.float32)
        else:
            self.points = np.zeros((0, 2), np.float32)
        self.undo.clear()
        self.zoom, self.ox, self.oy = 1.0, 0, 0

    def save(self) -> None:
        write_points(self.label_path(self.i), self.points)

    def write_lists(self) -> None:
        labelled = [f for j, f in enumerate(self.files) if self.label_path(j).exists()]
        if not labelled:
            return
        # Deterministic split by filename hash so re-running never reshuffles the
        # val set — a moving val set makes epoch-to-epoch numbers meaningless.
        val, train = [], []
        for f in labelled:
            (val if (hash(f.stem) % 100) < int(self.val_frac * 100) else train).append(f)
        for name, group in (("train.list", train), ("val.list", val)):
            lines = [f"images/{f.name} labels/{f.stem}.txt" for f in group]
            (self.root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("lists written: %d train, %d val", len(train), len(val))

    # ---- view maths -----------------------------------------------------
    def to_img(self, x: int, y: int) -> tuple[float, float]:
        return (x / self.zoom + self.ox, y / self.zoom + self.oy)

    def render(self) -> np.ndarray:
        assert self.img is not None
        h, w = self.img.shape[:2]
        vw, vh = int(w * self.zoom), int(h * self.zoom)
        view = cv2.resize(self.img, (vw, vh), interpolation=cv2.INTER_LINEAR)
        canvas = view[int(self.oy * self.zoom):, int(self.ox * self.zoom):].copy()
        for x, y in self.points:
            px = int((x - self.ox) * self.zoom)
            py = int((y - self.oy) * self.zoom)
            if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                cv2.circle(canvas, (px, py), 4, (60, 255, 60), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)

        done = sum(1 for j in range(len(self.files)) if self.label_path(j).exists())
        bar = np.zeros((78 if self.show_help else 34, canvas.shape[1], 3), np.uint8)
        kind = "NEGATIVE (0 heads)" if len(self.points) == 0 else f"{len(self.points)} heads"
        cv2.putText(bar, f"[{self.i+1}/{len(self.files)}]  {self.files[self.i].stem[:44]}"
                         f"   {kind}   labelled {done}/{len(self.files)}  zoom {self.zoom:.1f}x",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 255, 90), 1, cv2.LINE_AA)
        if self.show_help:
            for k, line in enumerate(HELP):
                cv2.putText(bar, line, (10, 40 + k * 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (180, 180, 180), 1, cv2.LINE_AA)
        return np.vstack([bar, canvas])

    # ---- interaction ----------------------------------------------------
    def push_undo(self) -> None:
        self.undo.append(self.points.copy())
        if len(self.undo) > 200:
            self.undo.pop(0)

    def on_mouse(self, event, x, y, flags, _param):
        bar = 78 if self.show_help else 34
        y -= bar
        if y < 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_from = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.push_undo()
            ix, iy = self.to_img(x, y)
            self.points = np.vstack([self.points, [[ix, iy]]]).astype(np.float32)
            self.drag_from = None
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points):
            ix, iy = self.to_img(x, y)
            d = np.hypot(self.points[:, 0] - ix, self.points[:, 1] - iy)
            j = int(np.argmin(d))
            if d[j] <= DELETE_RADIUS / max(self.zoom, 1e-6):
                self.push_undo()
                self.points = np.delete(self.points, j, axis=0)

    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.load()
        while True:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k in (ord('q'), 27):
                self.save(); self.write_lists(); break
            elif k in (ord('n'), 32):
                self.save(); self.i = min(self.i + 1, len(self.files) - 1); self.load()
            elif k == ord('p'):
                self.save(); self.i = max(self.i - 1, 0); self.load()
            elif k == ord('c'):
                self.push_undo(); self.points = np.zeros((0, 2), np.float32)
            elif k == ord('r'):
                self.push_undo()
                self.points = (self.predictor(self.img)[:, :2].astype(np.float32)
                               if self.predictor is not None else np.zeros((0, 2), np.float32))
            elif k == ord('a') and self.predictor is not None:
                self.push_undo()
                add = self.predictor(self.img)[:, :2].astype(np.float32)
                self.points = np.vstack([self.points, add]).astype(np.float32)
            elif k == ord('u') and self.undo:
                self.points = self.undo.pop()
            elif k == ord('h'):
                self.show_help = not self.show_help
            elif k in (ord('+'), ord('=')):
                self.zoom = min(self.zoom * 1.25, 8.0)
            elif k in (ord('-'), ord('_')):
                self.zoom = max(self.zoom / 1.25, 1.0)
                self.ox = self.oy = 0 if self.zoom == 1.0 else self.ox
            elif k == 81:   self.ox = max(0, self.ox - 40)
            elif k == 83:   self.ox += 40
            elif k == 82:   self.oy = max(0, self.oy - 40)
            elif k == 84:   self.oy += 40
        cv2.destroyAllWindows()
        done = sum(1 for j in range(len(self.files)) if self.label_path(j).exists())
        neg = sum(1 for j in range(len(self.files))
                  if self.label_path(j).exists() and len(read_points(self.label_path(j))) == 0)
        print(f"\nlabelled {done}/{len(self.files)}  ({neg} hard negatives)")
        print(f"lists: {self.root/'train.list'} , {self.root/'val.list'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", required=True, help="dir containing images/")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-model", action="store_true",
                    help="label from scratch instead of correcting model output")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    predictor = None
    if not args.no_model:
        from apgcc_infer import ApgccPredictor
        predictor = ApgccPredictor(args.weights, device=args.device, conf=args.conf,
                                   max_long_side=None)   # patches are already small
    Annotator(Path(args.patches), predictor, args.start, args.val_frac).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
