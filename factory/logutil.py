"""Shared logging helpers — prefer warning over silent ``except: pass``."""
from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; ensure a basic stderr handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        root = logging.getLogger("ea_factory")
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
            root.addHandler(handler)
            root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)
