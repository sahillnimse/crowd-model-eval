"""Fine-tune APGCC on your own labelled patches.

  python scripts/finetune_apgcc.py \\
      --train-list data/nashik/patches/train.list \\
      --val-list   data/nashik/patches/val.list \\
      --epochs 300 --out runs/apgcc_nashik_v1

Add public replay once you have ShanghaiTech prepared, to stop the model
forgetting everything that is not Goda Ghat:

      --public-list data/processed/shanghaitech_a/train.list --own-ratio 0.5

Defaults are FINE-TUNING values (lr 1e-5, 300 epochs), not upstream pretraining
values (lr 1e-4, 3500 epochs). Using the latter on a few hundred images will
destroy the pretrained features. See the recipe notes in README.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


from apgcc_finetune import FinetuneConfig, finetune   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--val-list", default="")
    ap.add_argument("--public-list", default="", help="replay set to prevent forgetting")
    ap.add_argument("--own-ratio", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8, help="images; x4 crops each")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-backbone", type=float, default=1e-6)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--weights", default="", help="start point (default: SHHA_best)")
    ap.add_argument("--config", default="shha")
    ap.add_argument("--out", default="runs/apgcc_finetune")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1229)
    a = ap.parse_args()

    cfg = FinetuneConfig(
        train_list=a.train_list, val_list=a.val_list, public_list=a.public_list,
        own_ratio=a.own_ratio, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, lr_backbone=a.lr_backbone, warmup_epochs=a.warmup_epochs,
        eval_every=a.eval_every, ema_decay=a.ema_decay, weights=a.weights,
        config=a.config, out_dir=a.out, device=a.device,
        num_workers=a.num_workers, amp=not a.no_amp, seed=a.seed,
    )
    result = finetune(cfg)
    b = result["best"]
    print(f"\nbest MAE {b['mae']:.2f} (epoch {b['epoch']}, {b.get('which','?')} weights)")
    print(f"checkpoint: {Path(result['out_dir'])/'best.pth'}")
    print("\nWire it into HAVAN:")
    print(f"  NETRA_APGCC_WEIGHTS={Path(result['out_dir'])/'best.pth'}")
    print("  see README.md (Deploy section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
