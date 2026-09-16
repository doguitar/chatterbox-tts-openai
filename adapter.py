"""Chatterbox load/generate adapter. Isolated so tests can fake the engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models import parse_variant


def tensor_to_float32_1d(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach()
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    if hasattr(wav, "numpy"):
        wav = wav.numpy()
    audio = np.asarray(wav, dtype=np.float32)
    return np.ravel(audio)


def generation_kwargs(
    variant: str,
    *,
    audio_prompt_path: str | None,
    language: str | None,
    exaggeration: float | None,
    cfg_weight: float | None,
) -> dict:
    variant = parse_variant(variant)
    kwargs: dict = {}
    if audio_prompt_path:
        kwargs["audio_prompt_path"] = audio_prompt_path
    if variant == "multilingual":
        kwargs["language_id"] = (language or "en").strip().lower() or "en"
    if variant in {"english", "multilingual"}:
        if exaggeration is not None:
            kwargs["exaggeration"] = float(exaggeration)
        if cfg_weight is not None:
            kwargs["cfg_weight"] = float(cfg_weight)
    return kwargs


def call_generate(
    engine,
    variant: str,
    text: str,
    *,
    audio_prompt_path: str | None = None,
    language: str | None = None,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
) -> tuple[np.ndarray, int]:
    kwargs = generation_kwargs(
        variant,
        audio_prompt_path=audio_prompt_path,
        language=language,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    wav = engine.generate(text, **kwargs)
    sr = int(getattr(engine, "sr", 24000))
    return tensor_to_float32_1d(wav), sr


def load_engine(variant: str, device: str, model_path: str = "", t3_model: str = "v3"):
    variant = parse_variant(variant)
    local = Path(model_path) if (model_path or "").strip() else None
    use_local = bool(local and local.is_dir())
    if variant in {"turbo", "nano"}:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        nano = variant == "nano"
        if use_local:
            return ChatterboxTurboTTS.from_local(str(local), device, nano=nano)
        return ChatterboxTurboTTS.from_pretrained(device=device, nano=nano)
    if variant == "english":
        from chatterbox.tts import ChatterboxTTS

        if use_local:
            return ChatterboxTTS.from_local(str(local), device)
        return ChatterboxTTS.from_pretrained(device=device)
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    t3 = (t3_model or "v3").strip() or "v3"
    if use_local:
        return ChatterboxMultilingualTTS.from_local(str(local), device, t3_model=t3)
    return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=t3)
