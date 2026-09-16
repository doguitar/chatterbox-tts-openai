"""Inference device selection. Pure helpers so they can be unit-tested without a GPU."""

from __future__ import annotations


def select_device(requested: str, cuda: bool, mps: bool) -> str:
    """Pick a torch device string.

    Explicit TTS_DEVICE wins. Otherwise CUDA, then MPS, then CPU.
    """
    requested = (requested or "").strip()
    if requested:
        return requested
    if cuda:
        return "cuda:0"
    if mps:
        return "mps"
    return "cpu"


def device_kind(device: str) -> str:
    return (device or "cpu").split(":", 1)[0].lower()
