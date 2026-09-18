"""Chatterbox load/generate adapter. Isolated so tests can fake the engine."""

from __future__ import annotations

import inspect
import json
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
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    max_gen_len: int | None = None,
    norm_loudness: bool | None = None,
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
    if variant in {"turbo", "nano"}:
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = float(repetition_penalty)
        if max_gen_len is not None:
            kwargs["max_gen_len"] = int(max_gen_len)
        if norm_loudness is not None:
            kwargs["norm_loudness"] = bool(norm_loudness)
    return kwargs


def supported_generate_kwargs(engine, kwargs: dict) -> dict:
    generate = getattr(engine, "generate", None)
    if generate is None:
        return kwargs
    try:
        params = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def call_generate(
    engine,
    variant: str,
    text: str,
    *,
    audio_prompt_path: str | None = None,
    language: str | None = None,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    max_gen_len: int | None = None,
    norm_loudness: bool | None = None,
) -> tuple[np.ndarray, int]:
    kwargs = generation_kwargs(
        variant,
        audio_prompt_path=audio_prompt_path,
        language=language,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_gen_len=max_gen_len,
        norm_loudness=norm_loudness,
    )
    wav = engine.generate(text, **supported_generate_kwargs(engine, kwargs))
    sr = int(getattr(engine, "sr", 24000))
    return tensor_to_float32_1d(wav), sr


def is_turbo_pack(path: Path) -> bool:
    return path.is_dir() and (path / "pack.json").is_file()


def is_official_local_ckpt(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        (path / name).is_file()
        for name in (
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "t3_turbo_v1.safetensors",
            "s3gen.pt",
            "ve.pt",
        )
    )


def load_turbo_pack(pack_dir: Path, device: str):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    pack_dir = Path(pack_dir)
    pack = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    engine = ChatterboxTurboTTS.from_pretrained(device=device, nano=False)
    merged_name = str(pack.get("merged") or "t3_turbo_finetuned_merged.safetensors")
    merged = pack_dir / merged_name
    if not merged.is_file():
        raise FileNotFoundError(f"pack merged weights missing: {merged}")
    state = load_file(str(merged))
    if "model" in state:
        inner = state["model"]
        state = inner[0] if isinstance(inner, (list, tuple)) else inner
    engine.t3.load_state_dict(state, strict=False)
    tfmr = getattr(engine.t3, "tfmr", None)
    if tfmr is not None and hasattr(tfmr, "wte"):
        del tfmr.wte
    tok = pack_dir / str(pack.get("tokenizer") or "tokenizer")
    if tok.is_dir():
        engine.tokenizer = AutoTokenizer.from_pretrained(str(tok))
        if engine.tokenizer.pad_token is None:
            engine.tokenizer.pad_token = engine.tokenizer.eos_token
    return engine


def load_engine(variant: str, device: str, model_path: str = "", t3_model: str = "v3"):
    variant = parse_variant(variant)
    local = Path(model_path) if (model_path or "").strip() else None
    if local and is_turbo_pack(local):
        return load_turbo_pack(local, device)
    use_local = bool(local and is_official_local_ckpt(local))
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
