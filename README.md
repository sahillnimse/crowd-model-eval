# Crowd Model Evaluation — DM-Count, APGCC, CLIP-EBC

Standalone project for **testing/verifying head-count models before production
approval**. Separate from `crowd-safety-testbed` — this is where you sanity-check
model accuracy independently before those models get trusted anywhere.

## Folder layout

```
crowd-model-eval/
├── apgcc_kit/              <- the APGCC fine-tuning kit (uploaded code, unmodified)
│   ├── apgcc/               <- PUT upstream APGCC source here (see instructions inside)
│   ├── weights/              <- PUT your APGCC .pth checkpoint here
│   ├── apgcc_infer.py         <- ApgccPredictor: frame -> head points
│   ├── apgcc_loader.py         <- loads the model + checkpoint
│   ├── apgcc_finetune.py        <- fine-tuning loop (optional, for later)
│   ├── extract_patches.py        <- cut labelling patches from your footage
│   ├── annotate.py                <- click-to-label tool
│   ├── train.py                    <- fine-tune entry point
│   └── README.md                    <- full instructions for this kit
│
├── model_weights/
│   ├── dmcount/             <- PUT dmcount_official_sh_a.pth, dmcount_official_qnrf.pth here
│   └── clipebc/             <- PUT a CLIP-EBC checkpoint here (optional, if testing it too)
│
├── data/
│   └── images/              <- PUT your test images here
│
├── scripts/
│   └── run_verification.py  <- runs DM-Count + APGCC (+ CLIP-EBC if present) on data/images
│
└── results/                 <- output overlays, density maps, and a CSV summary land here
```

## What to copy in (manual step, per your plan)

1. **APGCC**: clone `github.com/AaronCIH/APGCC`, copy its `apgcc/` folder contents into
   `apgcc_kit/apgcc/`, and drop your `head_count_apgcc.pt` (or the official
   `APGCC_SHHA_best.pth`) into `apgcc_kit/weights/`.
2. **DM-Count**: copy `dmcount_official_sh_a.pth` and `dmcount_official_qnrf.pth`
   from your `crowd-safety-testbed/model_weights/` into `model_weights/dmcount/`.
3. **CLIP-EBC** (optional): download a checkpoint from `github.com/Yiming-M/CLIP-EBC`
   and drop it into `model_weights/clipebc/`.
4. Drop test images into `data/images/`.

## Why APGCC's own folder structure is kept as-is

`apgcc_loader.py` (in `apgcc_kit/`) hardcodes its expectations: vendored upstream
source at `apgcc_kit/apgcc/`, default checkpoint at `apgcc_kit/weights/APGCC_SHHA_best.pth`.
Rather than fight that, this project keeps the kit self-contained exactly as its
own README describes, and treats it as one component alongside DM-Count/CLIP-EBC.

## Running the verification

```bash
pip install -r apgcc_kit/requirements.txt

python scripts/run_verification.py --images_dir data/images
```

This prints predicted counts for each model on each image, saves annotated
overlays + density maps to `results/`, and writes `results/verification_summary.csv`
so you can compare the junior's reported MAE/bias claims against real output on
your own images before signing off for production.

## Note on DM-Count inference code

The uploaded files only cover **APGCC**. DM-Count doesn't have inference code in
this kit yet — `scripts/run_verification.py` includes a minimal, self-contained
DM-Count loader (VGG19 density-map architecture, matching the official
`cvlab-stonybrook/DM-Count` repo) so it can run standalone without needing the
`models/head_count/` code from crowd-safety-testbed. CLIP-EBC support is stubbed
out — it needs the official `Yiming-M/CLIP-EBC` model code, which isn't included
here; ask if you want that wired in too once you've picked a checkpoint.
