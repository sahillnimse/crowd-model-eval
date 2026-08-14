"""Tiny standalone helpers so this kit needs no external package."""
import logging, random
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def seed_everything(seed: int = 1229) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except Exception:
        pass
