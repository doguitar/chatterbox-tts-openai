"""Engine-variant helpers and validated voice-document parsing. Pure — no torch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

PUBLIC_MODEL_ALIASES = frozenset({"tts-1", "chatterbox-tts"})
ENGINE_VARIANTS = frozenset({"turbo", "nano", "english", "multilingual"})
BUILTIN_DEFAULT_VOICE = "default"
ENGINE_SAMPLE_RATE = 24000
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
    if value == "all":
        raise ValueError(
            "TTS_LOAD_POLICY=all is not supported: Chatterbox loads one resident engine per process"
        )
    if value in {"lazy", "one"}:
        return value
    raise ValueError(f"TTS_LOAD_POLICY must be one of {{'lazy', 'one'}}; got {raw!r}")


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
                    out.append(
                        VoiceOverlay(str(name), speaker, model_id or None, preset, kind, ref_audio, ref_text)
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
            normalized[key] = entry
            continue
        raise ValueError(f"{key}: value must be a string or object")
    return {"voices": normalized}


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


def path_under(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except AttributeError:
        try:
            return os.path.commonpath([str(path), str(root)]) == str(root)
        except ValueError:
            return False


def resolve_reference_wav(rel: str, config_dir: Path) -> Path | None:
    rel = (rel or "").strip()
    if not rel:
        return None
    candidate = Path(rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    root = config_dir.resolve()
    path = (config_dir / rel).resolve()
    if not path.is_file() or path.suffix.lower() != ".wav":
        return None
    if not path_under(path, root):
        return None
    return path


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
