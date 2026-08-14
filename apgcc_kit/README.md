# Head-Counting (APGCC) — Training Kit

Self-contained. Copy this folder anywhere; it does **not** import from any other
project. Trains an **Apache-2.0** crowd head-counting + point-localization model
(APGCC) and lets you fine-tune it on your own camera footage.

## Why this model

- **APGCC** (ECCV 2024) — point-based crowd counting *and* head localization
  (it outputs a point per head, not just a total). MIT-licensed.
- It **replaces P2PNet**, whose weights and architecture are Tencent
  "academic-research-only" — a blocker for a paid deployment. APGCC is
  commercially usable, and on public benchmarks beats P2PNet (SHHA MAE 48.8 vs
  52.7; strict-σ localization F1 48.7 vs 40.6).
- It plugs into the surveillance console behind the same `predict(frame) ->
  [(x, y, score), ...]` interface P2PNet used, so it is a drop-in for counting.

## What's in here

| File | Purpose |
|---|---|
| `apgcc/` | vendored upstream APGCC (MIT, LICENSE kept) + two local patches (see below) |
| `weights/APGCC_SHHA_best.pth` | pretrained ShanghaiTech-A checkpoint (MIT) — the fine-tune starting point |
| `apgcc_loader.py` | build the model / load a checkpoint, offline |
| `apgcc_infer.py` | frame → head points (one preprocessing path for everything) |
| `apgcc_dataset.py` | the `.list`/`.txt` point-label format + hard negatives + public/own mixing |
| `apgcc_finetune.py` | the fine-tuning loop (AMP, EMA, cosine LR, MAE/NAE eval) |
| `extract_patches.py` | cut labelling patches from your videos, model-guided + stratified |
| `annotate.py` | click-to-label heads on those patches, writes APGCC labels |
| `train.py` | the fine-tune entry point |
| `_common.py` | tiny logging/seed helper (so the kit needs no external package) |

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```
Needs a CUDA GPU. Fine-tuning fits in ~4 GB (128px crops).

## The workflow (empty folder → fine-tuned model)

### 1. Cut labelling patches from your footage
```bash
python extract_patches.py --sources D:/path/to/your/videos --out data/patches --n 400 --patch 512
```
APGCC trains on 128px crops and never sees a whole frame, so you label **512px
patches**, not full 4K frames (~40 heads / 3 min each, versus 800 heads / 40 min).
The script runs the pretrained model first and stratifies patches by density
(dense / mid / sparse / **negative**) so you don't spend the budget labelling
empty water and sky.

### 2. Label the heads
```bash
python annotate.py --patches data/patches
```
Each patch opens **pre-populated with the model's predicted dots** — your job is
to correct (delete the false ones, add the misses), which is ~4× faster than
clicking from blank. Left-click adds, right-click deletes, `c` clears to a **hard
negative** (an empty label — a patch of "looks like a crowd but isn't"), `n`
saves and advances. Writes `train.list` + `val.list` in APGCC format.

### 3. Fine-tune
```bash
python train.py --train-list data/patches/train.list --val-list data/patches/val.list --epochs 5 --out runs/v1
```
Start from the bundled `weights/APGCC_SHHA_best.pth` (automatic). Add public-data
replay to avoid forgetting the general distribution:
```bash
python train.py ... --public-list data/shanghaitech/train.list --own-ratio 0.5
```

## Recipe — read before changing it

- **Keep the schedule SHORT (~5 epochs).** Measured on the sister fire model with
  the identical training loop: it peaks at ~5 epochs and **over-trains with
  more** — a 12-epoch run scored *worse* (recall collapsed while loss kept
  dropping). Do not default to "more epochs = better".
- **Fine-tune LRs are 10× below scratch training** (`--lr 1e-5`, backbone 1e-6)
  because you start from a converged checkpoint. Higher washes the pretrained
  features out.
- **Hard negatives are half the value.** A patch of a dense-but-not-crowd texture
  with an empty label teaches the model what *not* to count. `extract_patches`
  targets them; label plenty.

## Two upstream limitations this kit already handles

1. **APGCC's training code is not fully released.** With `AUX_EN=True` (the value
   in every shipped config), the decoder raises `NotImplemented` the moment it
   enters train mode — the auxiliary-point-guidance branch is inference-only in
   public. This kit fine-tunes with **`AUX_EN=False`**: verified to load the
   checkpoint with 0 missing keys and train cleanly. You lose APG's extra
   optimization signal but start from weights that already absorbed it.
2. **A hardcoded author path** in `apgcc/models/backbones/vgg.py`
   (`pretrained=True` → `/mnt/191/c/torch/...`) is neutralised in `apgcc_loader.py`
   — irrelevant since we load a full checkpoint over it.

Both patches live in the vendored copy / loader, documented inline.

## Windows note (important)

`apgcc_dataset.py` calls `cv2.setNumThreads(0)` and passes a `worker_init` to the
DataLoader. **Do not remove these.** Without them, `num_workers>0` deadlocks on
Windows (OpenCV threads vs PyTorch spawn) — the training loop hangs at the first
batch with 0% GPU and no error. If you *do* see a hang at the first batch,
`--num-workers 0` is the escape hatch (slower, but never hangs).

## Deploy

`apgcc_infer.ApgccPredictor(frame) -> [(x, y, score), ...]` is the console's
counting interface. After fine-tuning, point the console's counter weights at
`runs/v1/best`. Note APGCC operates at its trained input scale — a point head has
a narrow head-size band, so match `max_long_side` to your camera (see the
docstring in `apgcc_infer.py`).

## Known gaps

- Numbers here are on public ShanghaiTech-A. Real ghat/CCTV performance needs the
  fine-tune above on your own footage.
- The eval reports MAE / RMSE / NAE (counting) each epoch; localization F1 is not
  wired into this trimmed kit — add it if you need the point-accuracy metric.
