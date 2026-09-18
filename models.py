"""Engine-variant helpers and validated voice-document parsing. Pure — no torch."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import NamedTuple

PUBLIC_MODEL_ALIASES = frozenset({"tts-1", "chatterbox-tts"})
ENGINE_VARIANTS = frozenset({"turbo", "nano", "english", "multilingual"})
BUILTIN_DEFAULT_VOICE = "default"
ENGINE_SAMPLE_RATE = 24000

GENERATION_KEYS = (
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "max_gen_len",
    "norm_loudness",
    "exaggeration",
    "cfg_weight",
)
_GENERATION_INT_KEYS = frozenset({"top_k", "max_gen_len"})
_GENERATION_BOOL_KEYS = frozenset({"norm_loudness"})
GenerationValue = int | float | bool
GenerationConfig = dict[str, GenerationValue]


def parse_generation_config(raw: object | None, *, prefix: str = "generation") -> GenerationConfig | None:
    """Strict generation-config parser. None means missing; {} is an explicit empty object."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix} must be an object")
    out: GenerationConfig = {}
    for key, value in raw.items():
        loc = f"{prefix}.{key}"
        if key not in GENERATION_KEYS:
            raise ValueError(f"{loc} unknown key")
        if key in _GENERATION_BOOL_KEYS:
            if type(value) is not bool:
                raise ValueError(f"{loc} must be a boolean")
            out[key] = value
            continue
        if key in _GENERATION_INT_KEYS:
            if type(value) is bool or type(value) is not int:
                raise ValueError(f"{loc} must be a positive integer")
            if value < 1:
                raise ValueError(f"{loc} must be >= 1")
            out[key] = value
            continue
        if type(value) is bool or not isinstance(value, (int, float)):
            raise ValueError(f"{loc} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{loc} must be finite")
        out[key] = number
    return out


def merge_generation(*layers: GenerationConfig | None) -> GenerationConfig:
    merged: GenerationConfig = {}
    for layer in layers:
        if layer:
            merged.update(layer)
    return merged


def try_parse_generation(raw: object | None, *, prefix: str = "generation") -> GenerationConfig | None:
    try:
        return parse_generation_config(raw, prefix=prefix)
    except ValueError:
        return None

MULTILINGUAL_LANGUAGES = (
    "ar",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "sw",
    "tr",
    "zh",
)

_VARIANT_META = {
    "turbo": {
        "public_id": "chatterbox-turbo",
        "voice_mode": "reference",
        "sample_rate": ENGINE_SAMPLE_RATE,
        "builtin_default": False,
        "paralinguistic_tags": True,
        "exaggeration": False,
        "languages": ("en",),
    },
    "nano": {
        "public_id": "chatterbox-nano",
        "voice_mode": "reference",
        "sample_rate": ENGINE_SAMPLE_RATE,
        "builtin_default": False,
        "paralinguistic_tags": True,
        "exaggeration": False,
        "languages": ("en",),
    },
    "english": {
        "public_id": "chatterbox",
        "voice_mode": "builtin_or_reference",
        "sample_rate": ENGINE_SAMPLE_RATE,
        "builtin_default": True,
        "paralinguistic_tags": False,
        "exaggeration": True,
        "languages": ("en",),
    },
    "multilingual": {
        "public_id": "chatterbox-multilingual",
        "voice_mode": "builtin_or_reference",
        "sample_rate": ENGINE_SAMPLE_RATE,
        "builtin_default": True,
        "paralinguistic_tags": False,
        "exaggeration": True,
        "languages": MULTILINGUAL_LANGUAGES,
    },
}


def load_pack_document(path: Path) -> dict | None:
    pack_path = path / "pack.json"
    if not pack_path.is_file():
        return None
    try:
        data = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def discover_packs(root: Path) -> list[tuple[str, Path, dict]]:
    """Find Chatterbox train packs (pack.json) under root or at root itself."""
    if not root.is_dir():
        return []
    found: list[tuple[str, Path, dict]] = []
    own = load_pack_document(root)
    if own is not None:
        alias = str(own.get("name") or root.name).strip() or root.name
        return [(alias, root, own)]
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        data = load_pack_document(child)
        if data is None:
            continue
        alias = str(data.get("name") or child.name).strip() or child.name
        found.append((alias, child, data))
    return found


def pack_voice_overlays(packs: list[tuple[str, Path, dict]]) -> list[VoiceOverlay]:
    out: list[VoiceOverlay] = []
    for alias, path, data in packs:
        ref = str(data.get("reference") or "reference.wav").strip() or "reference.wav"
        rel = f"{path.name}/{ref}".replace("\\", "/")
        gen = try_parse_generation(data.get("generation"), prefix=f"{alias}: generation")
        out.append(VoiceOverlay(alias, "", alias, "", "voice_clone", rel, "", gen))
    return out


PRETRAINED_ENGINE_KEY = "__pretrained__"


def resolve_pack_for_voice(
    name: str,
    overlays: list[VoiceOverlay],
    catalog: list[tuple[str, Path, dict]],
) -> tuple[str, Path, dict] | None:
    """Map a public voice to a train pack, or None for the stock pretrained engine."""
    by_alias = {alias.lower(): (alias, path, data) for alias, path, data in catalog}
    by_folder = {path.name.lower(): (alias, path, data) for alias, path, data in catalog}
    key = (name or "").strip().lower()
    if key in by_alias:
        return by_alias[key]
    if key in by_folder:
        return by_folder[key]
    for item in overlays:
        if item.alias.strip().lower() != key:
            continue
        hint = (item.model or "").strip().lower()
        if hint in by_alias:
            return by_alias[hint]
        if hint in by_folder:
            return by_folder[hint]
    return None


def engine_cache_key(pack: tuple[str, Path, dict] | None) -> str:
    return pack[0] if pack is not None else PRETRAINED_ENGINE_KEY


def lru_touch(order: list[str], key: str) -> list[str]:
    return [item for item in order if item != key] + [key]


def lru_victim(order: list[str], keep: str) -> str | None:
    for item in order:
        if item != keep:
            return item
    return None


def is_memory_full_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "outofmemory" in name:
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def parse_variant(raw: str) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return "turbo"
    if value in ENGINE_VARIANTS:
        return value
    raise ValueError(f"TTS_VARIANT must be one of {sorted(ENGINE_VARIANTS)}; got {raw!r}")


def parse_load_policy(raw: str) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return "lazy"
    if value in {"lazy", "one", "all"}:
        return value
    raise ValueError(f"TTS_LOAD_POLICY must be one of {{'lazy', 'one', 'all'}}; got {raw!r}")


def engine_metadata(variant: str) -> dict:
    key = parse_variant(variant)
    return dict(_VARIANT_META[key])


def supports_builtin_default(variant: str) -> bool:
    return bool(engine_metadata(variant)["builtin_default"])


def is_public_model_request(requested: str, public_name: str) -> bool:
    requested = (requested or "").strip()
    if not requested:
        return True
    if requested in PUBLIC_MODEL_ALIASES:
        return True
    return requested == (public_name or "").strip()


class VoiceOverlay(NamedTuple):
    alias: str
    speaker: str
    model: str | None
    instructions: str
    kind: str
    ref_audio: str
    ref_text: str
    generation: GenerationConfig | None = None


def parse_voice_overlays(data: object | None, env_speakers: str) -> list[VoiceOverlay]:
    out: list[VoiceOverlay] = []
    if isinstance(data, dict):
        voices = data.get("voices", data)
        if isinstance(voices, dict):
            for name, spec in voices.items():
                if isinstance(spec, str):
                    out.append(VoiceOverlay(str(name), spec, None, "", "", "", ""))
                elif isinstance(spec, dict):
                    speaker = str(spec.get("speaker") or spec.get("name") or name)
                    model = spec.get("model")
                    model_id = str(model).strip() if model else None
                    raw_preset = spec.get("instructions")
                    preset = raw_preset.strip() if isinstance(raw_preset, str) else ""
                    kind_raw = spec.get("kind")
                    kind = kind_raw.strip().lower() if isinstance(kind_raw, str) else ""
                    ref_audio = spec.get("ref_audio").strip() if isinstance(spec.get("ref_audio"), str) else ""
                    ref_text = spec.get("ref_text").strip() if isinstance(spec.get("ref_text"), str) else ""
                    if kind == "voice_clone" or ref_audio or ref_text:
                        kind = "voice_clone"
                        speaker = str(spec.get("speaker") or "")
                    gen = try_parse_generation(spec.get("generation"), prefix=f"{name}: generation")
                    out.append(
                        VoiceOverlay(
                            str(name), speaker, model_id or None, preset, kind, ref_audio, ref_text, gen
                        )
                    )
        elif isinstance(voices, list):
            for name in voices:
                out.append(VoiceOverlay(str(name), str(name), None, "", "", "", ""))
    for name in (env_speakers or "").split(","):
        name = name.strip()
        if name:
            out.append(VoiceOverlay(name, name, None, "", "", "", ""))
    return out


def voices_file_writable(path: Path) -> bool:
    try:
        if path.is_file():
            return os.access(path, os.W_OK)
        return path.parent.is_dir() and os.access(path.parent, os.W_OK)
    except OSError:
        return False


def load_voices_document(path: Path) -> tuple[dict, str | None]:
    if not path.is_file():
        return {"voices": {}}, None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"voices": {}}, str(exc)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"voices": {}}, f"invalid JSON: {exc}"
    if isinstance(decoded, dict) and "voices" in decoded and isinstance(decoded["voices"], dict):
        return decoded, None
    if isinstance(decoded, dict) and "voices" not in decoded:
        return {"voices": decoded}, None
    return {"voices": {}}, "voices.json must be a JSON object"


def validate_voices_document(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("voices.json must be a JSON object")
    voices = data["voices"] if "voices" in data else data
    if not isinstance(voices, dict):
        raise ValueError("voices must be a JSON object")
    normalized: dict[str, str | dict[str, str]] = {}
    for raw_key, spec in voices.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("empty alias")
        if isinstance(spec, str):
            if not spec.strip():
                raise ValueError(f"{key}: alias target must be a non-empty string")
            normalized[key] = spec
            continue
        if isinstance(spec, dict):
            kind_raw = spec.get("kind")
            kind = kind_raw.strip().lower() if isinstance(kind_raw, str) else ""
            if kind and kind != "voice_clone":
                raise ValueError(f"{key}: unknown kind")
            is_clone = kind == "voice_clone" or "ref_audio" in spec or "ref_text" in spec
            if is_clone:
                audio_raw = spec.get("ref_audio")
                text_raw = spec.get("ref_text")
                if not isinstance(audio_raw, str) or not audio_raw.strip():
                    raise ValueError(f"{key}: clone requires ref_audio")
                if text_raw is not None and not isinstance(text_raw, str):
                    raise ValueError(f"{key}: ref_text must be a string")
                entry: dict[str, str] = {
                    "kind": "voice_clone",
                    "ref_audio": audio_raw.strip(),
                }
                if isinstance(text_raw, str) and text_raw.strip():
                    entry["ref_text"] = text_raw.strip()
                speaker = spec.get("speaker")
                if isinstance(speaker, str) and speaker.strip():
                    entry["speaker"] = speaker
                model = spec.get("model")
                if model is not None:
                    if not isinstance(model, str):
                        raise ValueError(f"{key}: model must be a string")
                    stripped_model = model.strip()
                    if stripped_model:
                        if stripped_model not in ENGINE_VARIANTS and stripped_model not in PUBLIC_MODEL_ALIASES:
                            raise ValueError(f"{key}: model must be a Chatterbox variant or tts-1")
                        entry["model"] = stripped_model
                if "instructions" in spec:
                    instructions = spec.get("instructions")
                    if not isinstance(instructions, str):
                        raise ValueError(f"{key}: instructions must be a string")
                    stripped = instructions.strip()
                    if stripped:
                        entry["instructions"] = stripped
                if "generation" in spec:
                    parsed = parse_generation_config(spec.get("generation"), prefix=f"{key}: generation")
                    if parsed is not None:
                        entry["generation"] = parsed
                normalized[key] = entry
                continue
            speaker = spec.get("speaker")
            if not isinstance(speaker, str) or not speaker.strip():
                raise ValueError(f"{key}: speaker must be a non-empty string")
            entry = {"speaker": speaker}
            model = spec.get("model")
            if model is not None:
                if not isinstance(model, str):
                    raise ValueError(f"{key}: model must be a string")
                if model.strip():
                    entry["model"] = model
            if "instructions" in spec:
                instructions = spec.get("instructions")
                if not isinstance(instructions, str):
                    raise ValueError(f"{key}: instructions must be a string")
                stripped = instructions.strip()
                if stripped:
                    entry["instructions"] = stripped
            if "generation" in spec:
                parsed = parse_generation_config(spec.get("generation"), prefix=f"{key}: generation")
                if parsed is not None:
                    entry["generation"] = parsed
            normalized[key] = entry
            continue
        raise ValueError(f"{key}: value must be a string or object")
    out: dict = {"voices": normalized}
    if "generation" in data:
        parsed = parse_generation_config(data.get("generation"), prefix="generation")
        if parsed is not None:
            out["generation"] = parsed
    return out


def write_voices_document(path: Path, document: dict) -> None:
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise


def overlay_instructions(name: str, overlays: list[VoiceOverlay]) -> str:
    key = (name or "").strip().lower()
    preset = ""
    for item in overlays:
        if item.alias.strip().lower() == key:
            preset = item.instructions
    return preset


def overlay_kind(name: str, overlays: list[VoiceOverlay]) -> str:
    key = (name or "").strip().lower()
    kind = ""
    for item in overlays:
        if item.alias.strip().lower() == key:
            kind = item.kind
    return kind


def overlay_clone_ref(name: str, overlays: list[VoiceOverlay]) -> tuple[str, str]:
    key = (name or "").strip().lower()
    ref = ("", "")
    for item in overlays:
        if item.alias.strip().lower() == key:
            ref = (item.ref_audio, item.ref_text)
    return ref


def overlay_generation(name: str, overlays: list[VoiceOverlay]) -> GenerationConfig | None:
    key = (name or "").strip().lower()
    generation = None
    for item in overlays:
        if item.alias.strip().lower() == key:
            generation = item.generation
    return generation


def path_under(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except AttributeError:
        try:
            return os.path.commonpath([str(path), str(root)]) == str(root)
        except ValueError:
            return False


def resolve_reference_wav(rel: str, *roots: Path) -> Path | None:
    rel = (rel or "").strip()
    if not rel:
        return None
    candidate = Path(rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    for root in roots:
        if not root:
            continue
        base = root.resolve()
        path = (root / rel).resolve()
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        if path_under(path, base):
            return path
    return None


def listed_voice_names(overlays: list[VoiceOverlay], variant: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    if supports_builtin_default(variant):
        names.append(BUILTIN_DEFAULT_VOICE)
        seen.add(BUILTIN_DEFAULT_VOICE)
    for item in overlays:
        alias = item.alias.strip()
        if not alias:
            continue
        key = alias.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(alias)
    return names


def public_default_voice(overlays: list[VoiceOverlay], requested: str, variant: str) -> str:
    listed = listed_voice_names(overlays, variant)
    if not listed:
        return ""
    requested = (requested or "").strip()
    by_key = {name.lower(): name for name in listed}
    if requested and requested.lower() in by_key:
        return by_key[requested.lower()]
    return listed[0]


def resolve_voice_route(
    name: str | None,
    overlays: list[VoiceOverlay],
    default_voice: str,
    stock_voices: set[str] | frozenset[str],
    variant: str,
) -> tuple[str, bool, str]:
    listed = {n.lower(): n for n in listed_voice_names(overlays, variant)}
    default_key = (default_voice or "").strip().lower()
    default_name = listed.get(default_key, default_voice)
    key = (name or "").strip().lower()
    if not key or key in stock_voices:
        reason = "empty voice" if not key else f"openai stock voice {key}"
        return default_name, bool(key), reason
    if key in listed:
        return listed[key], False, ""
    return default_name, True, f"unknown voice {name!r}; using {default_voice}"
