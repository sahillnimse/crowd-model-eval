"""Fine-tune APGCC on your own cameras, sized for an 8 GB laptop GPU.

WHY FINE-TUNING IS THE BIGGEST LEVER
------------------------------------
Architecture swaps move ShanghaiTech-A MAE by 3-5 points. The DOMAIN GAP costs
roughly 100: HAJJv2-CrowdCount (arXiv:2607.07322) ran APGCC on ShanghaiTech
weights zero-shot over Hajj CCTV and measured MAE 152.9 against 48.8 at home.
Nashik ghat CCTV is the same kind of transfer. No amount of model shopping
substitutes for a few hundred labelled frames from the actual cameras.

MEMORY
------
Training is on 128x128 crops (upstream CROP_SIZE), 4 per image, batch 8 images
=> 32 crops of 128px per step. That is small: expect ~2-3 GB. The 8 GB card is
not the constraint here; annotation throughput is.

FINE-TUNE, DO NOT RETRAIN
-------------------------
Defaults below are 10x lower LR than upstream pretraining and ~10x fewer epochs,
because we start from a converged checkpoint. Upstream's 1e-4 / 3500 epochs will
wash the pretrained features out and you will end up worse than where you began.
The backbone gets a further 10x reduction: VGG features generalise fine, it is
the decoder and the classification head that need to learn "saffron cloth is not
a head" and "tin roofing is not a crowd".
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from apgcc_dataset import PointDataset, MixedPointDataset, collate, worker_init
from apgcc_loader import _vendored_on_path, _disable_hardcoded_imagenet_init
from _common import get_logger

log = get_logger(__name__)


@dataclass
class FinetuneConfig:
    # data
    train_list: str = ""
    val_list: str = ""
    public_list: str = ""            # optional replay set; empty disables mixing
    own_ratio: float = 0.5

    # optimisation — fine-tuning values, NOT upstream pretraining values
    epochs: int = 300
    batch_size: int = 8              # images; x crops_per_image = actual crops
    lr: float = 1e-5                 # upstream pretrain: 1e-4
    lr_backbone: float = 1e-6        # upstream pretrain: 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    clip_grad: float = 0.1
    ema_decay: float = 0.999

    # eval / io
    eval_every: int = 5
    conf: float = 0.5
    val_max_long_side: int = 1280
    out_dir: str = "runs/apgcc_finetune"
    weights: str = ""                # start point; empty -> weights/APGCC_SHHA_best.pth
    config: str = "shha"
    device: str = "cuda"
    amp: bool = True
    num_workers: int = 2   # >0 needs the cv2 fix in apgcc_dataset (bundled)
    seed: int = 1229
    resume: str = ""
    aux_en: bool = False              # upstream training path for APG is unreleased

    extra: dict = field(default_factory=dict)


def build_model_and_criterion(config: str, device: str, aux_en: bool = False):
    """APGCC's model + its own Hungarian-matcher criterion.

    ``aux_en`` DEFAULTS TO FALSE, AND THAT IS NOT AN ARBITRARY CHOICE
    ----------------------------------------------------------------
    The upstream release does not ship the auxiliary-point-guidance training
    path — the APG that the paper is named after. ``models/Decoder.py`` ends:

        if not self.aux_en or not self.training:
            return out
        else:
            raise NotImplemented        # still refinement, will be announced ASAP

    So with ``AUX_EN=True`` (the value in every shipped config) the model raises
    the instant it is put in train mode. The published repo is inference-only for
    the full method. Verified against the checkpoint: building with AUX_EN=False
    loads SHHA_best.pth with 0 missing and 0 unexpected keys, because the
    auxiliary anchors are loss-side only and carry no parameters, and
    ``build_model`` correctly drops ``loss_aux`` from the weight dict, leaving
    ``{'loss_ce': 1, 'loss_points': 0.0002}``.

    WHAT THIS COSTS YOU. Fine-tuning without APG is a weaker optimisation signal
    than the authors used to produce SHHA_best.pth. It is still sound — you start
    from weights that already absorbed APG's benefit during pretraining, and the
    point-matching + classification losses are exactly P2PNet's proven recipe.
    But do not expect fine-tuning to reproduce the paper's training dynamics, and
    if upstream later publishes the APG path, revisit this.

    Set ``aux_en=True`` only if you have implemented the aux branch yourself.
    """
    from apgcc_loader import APGCC_CONFIGS
    cfg_path = APGCC_CONFIGS.get(config, Path(config))
    with _vendored_on_path():
        _disable_hardcoded_imagenet_init()
        from config import cfg as _cfg, merge_from_file  # type: ignore
        from models import build_model  # type: ignore
        merged = merge_from_file(_cfg, str(cfg_path))
        merged.MODEL.AUX_EN = bool(aux_en)
        model, criterion = build_model(merged, training=True)
    if not aux_en:
        log.info("APGCC aux branch disabled (upstream training path unreleased); "
                 "losses = %s", getattr(criterion, "weight_dict", {}))
    return model.to(device), criterion.to(device)


class ModelEma:
    """Exponential moving average of weights. Cheap, and reliably worth ~1-2 MAE."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                e.copy_(m)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str,
             conf: float = 0.5) -> dict[str, float]:
    """MAE / RMSE / NAE on whole validation frames.

    NAE (normalised absolute error) is reported because MAE alone is dominated by
    the densest frames — a model can post a respectable MAE while being 300% wrong
    on every sparse frame, which is precisely the false-alarm behaviour that makes
    operators stop trusting a control-room display.
    """
    model.eval()
    errs, gts = [], []
    for samples, targets in loader:
        samples = samples.to(device)
        out = model(samples)
        scores = torch.softmax(out["pred_logits"].float(), -1)[0, :, 1]
        pred = int((scores > conf).sum().item())
        gt = int(targets[0]["point"].shape[0])
        errs.append(pred - gt)
        gts.append(gt)
    e = np.asarray(errs, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "nae": float((np.abs(e) / np.maximum(g, 1)).mean()),
        "bias": float(e.mean()),          # signed: negative = undercounting = dangerous
        "n": int(len(e)),
    }


def finetune(cfg: FinetuneConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device = cfg.device
    model, criterion = build_model_and_criterion(cfg.config, device, cfg.aux_en)

    start = Path(cfg.weights) if cfg.weights else \
        Path(__file__).resolve().parents[3] / "weights" / "APGCC_SHHA_best.pth"
    if start.is_file():
        blob = torch.load(start, map_location="cpu", weights_only=False)
        state = blob.get("model", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info("init from %s (missing=%d unexpected=%d)", start.name,
                 len(missing), len(unexpected))
        if len(missing) > 20:
            raise RuntimeError(
                f"{len(missing)} missing keys initialising from {start}. That is not a "
                "fine-tune, it is a mostly-random model. Check --config matches the "
                "checkpoint's architecture."
            )
    else:
        log.warning("no start weights at %s — training from scratch is NOT recommended", start)

    # data
    own = PointDataset(cfg.train_list, train=True)
    public = PointDataset(cfg.public_list, train=True) if cfg.public_list else None
    train_ds = MixedPointDataset(own, public, cfg.own_ratio)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate,
                          persistent_workers=cfg.num_workers > 0, drop_last=True,
                          worker_init_fn=(worker_init if cfg.num_workers > 0 else None))
    val_ld = None
    if cfg.val_list:
        val_ds = PointDataset(cfg.val_list, train=False)
        val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=collate)

    # param groups — backbone learns 10x slower
    backbone, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if n.startswith("encoder") else head).append(p)
    opt = torch.optim.AdamW(
        [{"params": head, "lr": cfg.lr},
         {"params": backbone, "lr": cfg.lr_backbone}],
        weight_decay=cfg.weight_decay,
    )
    log.info("params: head=%d tensors, backbone=%d tensors", len(head), len(backbone))

    steps_per_epoch = max(1, len(train_ld))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = ModelEma(model, cfg.ema_decay)
    use_amp = cfg.amp and device.startswith("cuda")

    best = {"mae": float("inf"), "epoch": -1}
    history = []
    step = 0
    log.info("fine-tuning %d epochs, %d steps/epoch, batch %d images (=%d crops)",
             cfg.epochs, steps_per_epoch, cfg.batch_size,
             cfg.batch_size * own.aug.crops_per_image)

    for epoch in range(cfg.epochs):
        model.train()
        criterion.train()
        t0 = time.perf_counter()
        running = 0.0
        for samples, targets in train_ld:
            samples = samples.to(device, non_blocking=True)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in t.items()} for t in targets]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(samples)
            # Loss in fp32: the Hungarian matcher's cost matrix and the point
            # regression are numerically fragile in reduced precision.
            outputs = {k: (v.float() if torch.is_tensor(v) else v)
                       for k, v in outputs.items()}
            loss_dict = criterion(outputs, targets)
            weights = criterion.weight_dict
            loss = sum(loss_dict[k] * weights[k] for k in loss_dict if k in weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()
            sched.step()
            ema.update(model)
            running += float(loss.detach())
            step += 1

        dt = time.perf_counter() - t0
        rec = {"epoch": epoch, "loss": running / steps_per_epoch,
               "lr": sched.get_last_lr()[0], "sec": round(dt, 1)}

        if val_ld is not None and ((epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1):
            raw = evaluate(model, val_ld, device, cfg.conf)
            ema_m = evaluate(ema.ema, val_ld, device, cfg.conf)
            rec["val"] = raw
            rec["val_ema"] = ema_m
            pick, tag = (ema_m, "ema") if ema_m["mae"] <= raw["mae"] else (raw, "raw")
            if pick["mae"] < best["mae"]:
                best = {"mae": pick["mae"], "epoch": epoch, "which": tag, **pick}
                torch.save({"model": (ema.ema if tag == "ema" else model).state_dict(),
                            "epoch": epoch, "metrics": pick, "config": asdict(cfg)},
                           out_dir / "best.pth")
                log.info("epoch %d: NEW BEST mae=%.2f (%s) bias=%+.1f -> best.pth",
                         epoch, pick["mae"], tag, pick["bias"])
            log.info("epoch %d loss=%.4f | raw mae=%.2f nae=%.3f | ema mae=%.2f | %.0fs",
                     epoch, rec["loss"], raw["mae"], raw["nae"], ema_m["mae"], dt)
        else:
            log.info("epoch %d loss=%.4f lr=%.2e %.0fs", epoch, rec["loss"], rec["lr"], dt)

        history.append(rec)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "ema": ema.ema.state_dict(),
                    "epoch": epoch, "optimizer": opt.state_dict()}, out_dir / "last.pth")

    log.info("done. best mae=%.2f at epoch %d -> %s",
             best["mae"], best["epoch"], out_dir / "best.pth")
    return {"best": best, "history": history, "out_dir": str(out_dir)}
