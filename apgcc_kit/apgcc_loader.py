"""Build an APGCC model and load weights, without fighting the upstream repo.

Two upstream facts this module exists to absorb:

1. APGCC is not a package. It expects to be run from inside ``apgcc/`` with that
   directory on ``sys.path`` (``from config import cfg``, ``from models import
   build_model``). We add the vendored path once, import, and restore — so the
   rest of netra can just ``from netra.apgcc import load_apgcc``.

2. ``models/Encoder.py`` calls ``models.vgg16_bn(pretrained=True)`` and the
   vendored ``backbones/vgg.py`` resolves that to a hardcoded path on the
   author's machine (``/mnt/191/c/torch/vgg16/vgg16_bn-6c64b313.pth``), which
   raises FileNotFoundError on every other computer on earth. The ImageNet init
   is irrelevant whenever we immediately load a trained checkpoint over it, so we
   patch ``pretrained`` to False at import time rather than editing the vendored
   source. If you ever train APGCC FROM SCRATCH you *do* want a real ImageNet
   init — pass ``imagenet_init=True`` and supply the weights yourself.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from _common import get_logger

log = get_logger(__name__)

# Repo root is four parents up from src/netra/apgcc/loader.py
_HERE = Path(__file__).resolve().parent
VENDOR_DIR = _HERE / "apgcc"
DEFAULT_WEIGHTS = _HERE / "weights" / "APGCC_SHHA_best.pth"

#: Upstream configs. SHHA_test is the IFI decoder that matches SHHA_best.pth.
APGCC_CONFIGS = {
    "shha": VENDOR_DIR / "configs" / "SHHA_test.yml",
    "shha_ifi": VENDOR_DIR / "configs" / "SHHA_IFI.yml",
    "shha_basic": VENDOR_DIR / "configs" / "SHHA_basic.yml",
}


@contextlib.contextmanager
def _vendored_on_path():
    """Put the vendored apgcc dir on sys.path for the duration of an import."""
    if not VENDOR_DIR.is_dir():
        raise FileNotFoundError(
            f"APGCC sources not found at {VENDOR_DIR}.\n"
            "Vendor them with:\n"
            "  git clone --depth 1 https://github.com/AaronCIH/APGCC.git tmp_apgcc\n"
            f"  cp -r tmp_apgcc/apgcc {VENDOR_DIR}\n"
            f"  cp tmp_apgcc/LICENSE {VENDOR_DIR}/LICENSE"
        )
    p = str(VENDOR_DIR)
    added = p not in sys.path
    if added:
        sys.path.insert(0, p)
    try:
        yield
    finally:
        if added and p in sys.path:
            sys.path.remove(p)


def _disable_hardcoded_imagenet_init() -> None:
    """Stop backbones/vgg.py reaching for the author's local ImageNet weights."""
    try:
        from models.backbones import vgg as _vgg_mod  # type: ignore
    except Exception:  # pragma: no cover - upstream layout changed
        return
    if getattr(_vgg_mod, "_netra_patched", False):
        return
    original = _vgg_mod._vgg

    def _vgg_no_pretrained(arch, cfg_, batch_norm, pretrained, progress, **kwargs):
        # pretrained is forced False; a trained checkpoint is loaded straight after.
        return original(arch, cfg_, batch_norm, False, progress, **kwargs)

    _vgg_mod._vgg = _vgg_no_pretrained
    _vgg_mod._netra_patched = True


def build_apgcc(config: str | Path = "shha", imagenet_init: bool = False) -> nn.Module:
    """Construct an untrained APGCC model from an upstream YAML config."""
    cfg_path = APGCC_CONFIGS.get(str(config), None) if not isinstance(config, Path) else config
    if cfg_path is None:
        cfg_path = Path(config)
    cfg_path = Path(cfg_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"APGCC config not found: {cfg_path}")

    with _vendored_on_path():
        if not imagenet_init:
            _disable_hardcoded_imagenet_init()
        from config import cfg as _cfg, merge_from_file  # type: ignore
        from models import build_model  # type: ignore

        merged = merge_from_file(_cfg, str(cfg_path))
        model = build_model(merged, training=False)
    return model


def load_apgcc(
    weights: str | Path | None = None,
    device: str = "cuda",
    config: str | Path = "shha",
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build APGCC and load a checkpoint.

    Returns ``(model, info)`` where ``info`` carries the load report. We surface
    missing/unexpected keys rather than swallowing them: a silently partial load
    produces a model that runs, returns plausible-looking points, and is wrong —
    which is the single worst failure mode in this whole system. HAVAN's own
    ``load_model`` has a partial-load fallback that only prints a warning; this
    one raises unless you explicitly opt out with ``strict=False``.
    """
    weights_path = Path(weights) if weights is not None else DEFAULT_WEIGHTS
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"APGCC checkpoint not found: {weights_path}\n"
            "Download the SHHA checkpoint (MIT) with:\n"
            "  pip install gdown && python -c \"import gdown; gdown.download("
            "id='1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu', "
            f"output=r'{weights_path}')\""
        )

    model = build_apgcc(config)
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    info = {
        "weights": str(weights_path),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "n_tensors": len(state),
        "epoch": blob.get("epoch") if isinstance(blob, dict) else None,
    }
    if (missing or unexpected) and strict:
        raise RuntimeError(
            f"APGCC checkpoint does not match the '{config}' architecture: "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys.\n"
            f"  first missing:    {list(missing)[:5]}\n"
            f"  first unexpected: {list(unexpected)[:5]}\n"
            "A partial load yields confident nonsense. Fix the config or pass "
            "strict=False if you genuinely intend a partial load."
        )
    if missing or unexpected:
        log.warning("APGCC partial load: %d missing, %d unexpected",
                    len(missing), len(unexpected))

    model.to(device).eval()
    log.info("APGCC loaded from %s (%d tensors, device=%s)",
             weights_path.name, info["n_tensors"], device)
    return model, info
