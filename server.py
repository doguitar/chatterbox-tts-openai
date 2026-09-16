#!/usr/bin/env python3
"""OpenAI-compatible TTS API for Chatterbox.

  GET  /
  GET  /config
  GET  /health
  GET  /v1/models
  GET  /v1/voices
  POST /v1/audio/speech
"""
from __future__ import annotations

import asyncio
import gc
import io
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, field_validator

from adapter import call_generate, load_engine
from device import select_device
from models import (
    VoiceOverlay,
    engine_metadata,
    is_public_model_request,
    listed_voice_names,
    load_voices_document,
    overlay_clone_ref,
    overlay_instructions,
    overlay_kind,
    parse_load_policy,
    parse_variant,
    parse_voice_overlays,
    public_default_voice,
    resolve_reference_wav,
    resolve_voice_route,
    supports_builtin_default,
    validate_voices_document,
    voices_file_writable,
    write_voices_document,
)

OPENAI_STOCK_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}

AUDIO_FORMATS = {
    "wav": ("audio/wav", ["-f", "wav", "-acodec", "pcm_s16le"]),
    "mp3": ("audio/mpeg", ["-f", "mp3", "-acodec", "libmp3lame", "-q:a", "2"]),
    "opus": ("audio/opus", ["-f", "opus"]),
    "aac": ("audio/aac", ["-f", "adts", "-acodec", "aac"]),
    "flac": ("audio/flac", ["-f", "flac"]),
    "pcm": ("application/octet-stream", None),
}

HOST = os.environ.get("TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TTS_PORT", "8080"))
MODEL_PATH = os.environ.get("TTS_MODEL", os.environ.get("MODEL_PATH", "")).strip()
VOICES_PATH = Path(os.environ.get("TTS_VOICES", "/config/voices.json"))
CLONES_DIR = VOICES_PATH.parent / "clones"
DEFAULT_LANGUAGE = os.environ.get("TTS_LANGUAGE", "en")
MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "tts-1")
LOAD_POLICY = parse_load_policy(os.environ.get("TTS_LOAD_POLICY", ""))
VARIANT = parse_variant(os.environ.get("TTS_VARIANT", ""))
T3_MODEL = os.environ.get("TTS_T3_MODEL", "v3").strip() or "v3"
STATIC_DIR = Path(__file__).resolve().parent / "static"
VARIANT_META = engine_metadata(VARIANT)


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return float(raw)


TTS_EXAGGERATION = _env_float("TTS_EXAGGERATION")
TTS_CFG_WEIGHT = _env_float("TTS_CFG_WEIGHT")


def mps_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def pick_device() -> str:
    return select_device(os.environ.get("TTS_DEVICE", ""), torch.cuda.is_available(), mps_available())


DEVICE = pick_device()

app = FastAPI(title="Chatterbox TTS OpenAI")
lock = threading.Lock()
engine: Any = None
voice_overlays: list[VoiceOverlay] = []
default_voice = ""
ready_error: str | None = None
BODY_LOG_LIMIT = int(os.environ.get("TTS_LOG_BODY_LIMIT", "8000"))


def preview_body(raw: bytes | None) -> str:
    text = (raw or b"").decode("utf-8", "replace")
    if len(text) > BODY_LOG_LIMIT:
        return text[:BODY_LOG_LIMIT] + "...(truncated)"
    return text


def log_api_error(request: Request, status: int, detail: Any) -> None:
    raw = getattr(request.state, "raw_body", b"")
    print(
        f"API error {request.method} {request.url.path} status={status} "
        f"detail={detail!r} body={preview_body(raw)!r}",
        flush=True,
    )


@app.middleware("http")
async def capture_request_body(request: Request, call_next):
    raw = await request.body()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    request = Request(request.scope, receive)
    request.state.raw_body = raw
    try:
        response = await call_next(request)
    except Exception as exc:
        log_api_error(request, 500, repr(exc))
        raise
    if response.status_code >= 400:
        log_api_error(request, response.status_code, f"http {response.status_code}")
    return response


class OpenAISpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: str = ""
    text: str = ""
    voice: Any = ""
    model: str = ""
    instructions: str | None = None
    language: str | None = None
    response_format: str = "mp3"
    speed: float | None = None
    stream_format: str | None = None

    @field_validator("voice", mode="before")
    @classmethod
    def coerce_voice(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return str(value.get("id") or value.get("voice") or value.get("name") or "")
        return str(value)


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _read_overlays() -> list[VoiceOverlay]:
    data = None
    if VOICES_PATH.is_file():
        data = json.loads(VOICES_PATH.read_text(encoding="utf-8"))
    return parse_voice_overlays(data, os.environ.get("TTS_SPEAKERS", ""))


def encode_audio(audio: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "mp3").lower().strip()
    if fmt not in AUDIO_FORMATS:
        fmt = "mp3"
    media, ffmpeg_args = AUDIO_FORMATS[fmt]
    wav = wav_bytes(audio, sr)
    if fmt == "wav":
        return wav, media
    if fmt == "pcm":
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return pcm, media
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *ffmpeg_args, "pipe:1"],
        input=wav,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        print(f"ffmpeg {fmt} failed: {proc.stderr.decode('utf-8', 'replace')}", flush=True)
        return wav, "audio/wav"
    return proc.stdout, media


def _rebuild_voices() -> None:
    global voice_overlays, default_voice
    voice_overlays = _read_overlays()
    default_voice = public_default_voice(
        voice_overlays,
        os.environ.get("TTS_DEFAULT_VOICE", ""),
        VARIANT,
    )


def _voices_ui_payload() -> dict:
    document, error = load_voices_document(VOICES_PATH)
    return {
        "path": str(VOICES_PATH),
        "exists": VOICES_PATH.is_file(),
        "writable": voices_file_writable(VOICES_PATH),
        "document": document,
        "speakers_env": os.environ.get("TTS_SPEAKERS", ""),
        "error": error,
        "variant": VARIANT,
    }


def _load_one() -> Any:
    global engine
    if engine is not None:
        return engine
    engine = load_engine(VARIANT, DEVICE, MODEL_PATH, T3_MODEL)
    print(
        f"loaded variant={VARIANT} device={DEVICE} model_path={MODEL_PATH or '(pretrained)'} "
        f"t3={T3_MODEL}",
        flush=True,
    )
    return engine


def _unload_one() -> None:
    global engine
    if engine is None:
        return
    del engine
    engine = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("unloaded chatterbox engine", flush=True)


def _ensure_engine() -> Any:
    return _load_one()


@app.on_event("startup")
def startup() -> None:
    global ready_error
    _rebuild_voices()
    try:
        if LOAD_POLICY == "one":
            _load_one()
    except Exception as exc:
        ready_error = str(exc)
        raise
    print(
        f"policy={LOAD_POLICY} public={MODEL_NAME} variant={VARIANT} "
        f"voices={listed_voice_names(voice_overlays, VARIANT)} default_voice={default_voice} "
        f"device={DEVICE}",
        flush=True,
    )


@app.get("/")
def ui_index():
    path = STATIC_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui missing")
    return FileResponse(path)


@app.get("/design")
def ui_design():
    raise HTTPException(status_code=404, detail="voice design is not supported by Chatterbox")


@app.get("/clone")
def ui_clone():
    path = STATIC_DIR / "clone.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui missing")
    return FileResponse(path)


@app.get("/config")
def ui_config():
    path = STATIC_DIR / "config.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui missing")
    return FileResponse(path)


@app.get("/ui/voices")
def ui_voices_get():
    return _voices_ui_payload()


@app.put("/ui/voices")
async def ui_voices_put(request: Request):
    if not voices_file_writable(VOICES_PATH):
        raise HTTPException(status_code=403, detail="voices.json is not writable")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="voices.json must be a JSON object")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="voices.json must be a JSON object")
    try:
        document = validate_voices_document(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        write_voices_document(VOICES_PATH, document)
    except OSError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    def _reload():
        with lock:
            _rebuild_voices()

    await asyncio.to_thread(_reload)
    payload = _voices_ui_payload()
    payload["voices"] = listed_voice_names(voice_overlays, VARIANT)
    payload["default"] = default_voice
    return payload


def _reload_voices_locked():
    with lock:
        _rebuild_voices()


@app.post("/ui/clone-preset")
async def ui_clone_preset(request: Request):
    if not voices_file_writable(VOICES_PATH):
        raise HTTPException(status_code=403, detail="voices.json is not writable")
    form = await request.form()
    alias = _safe_alias(str(form.get("alias") or ""))
    if not alias:
        raise HTTPException(status_code=400, detail="clone requires a public name")
    ref_text = str(form.get("ref_text") or "").strip()
    upload = form.get("ref_audio")
    raw = await _read_clone_wav(upload)
    CLONES_DIR.mkdir(parents=True, exist_ok=True)
    dest = CLONES_DIR / f"{alias}.wav"
    dest.write_bytes(raw)
    document, _error = load_voices_document(VOICES_PATH)
    voices = dict(document.get("voices") or {})
    entry = {"kind": "voice_clone", "ref_audio": f"clones/{alias}.wav"}
    if ref_text:
        entry["ref_text"] = ref_text
    voices[alias] = entry
    try:
        document = validate_voices_document({"voices": voices})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        write_voices_document(VOICES_PATH, document)
    except OSError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    await asyncio.to_thread(_reload_voices_locked)
    payload = _voices_ui_payload()
    payload["voices"] = listed_voice_names(voice_overlays, VARIANT)
    payload["default"] = default_voice
    return payload


@app.delete("/ui/clone-preset/{alias}")
async def ui_clone_preset_delete(alias: str):
    if not voices_file_writable(VOICES_PATH):
        raise HTTPException(status_code=403, detail="voices.json is not writable")
    safe = _safe_alias(alias)
    if not safe:
        raise HTTPException(status_code=400, detail="clone requires a public name")
    wav = CLONES_DIR / f"{safe}.wav"
    if wav.is_file():
        try:
            wav.unlink()
        except OSError:
            pass
    document, _error = load_voices_document(VOICES_PATH)
    voices = dict(document.get("voices") or {})
    voices.pop(safe, None)
    try:
        document = validate_voices_document({"voices": voices})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        write_voices_document(VOICES_PATH, document)
    except OSError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    await asyncio.to_thread(_reload_voices_locked)
    payload = _voices_ui_payload()
    payload["voices"] = listed_voice_names(voice_overlays, VARIANT)
    payload["default"] = default_voice
    return payload


@app.get("/health")
def health():
    if ready_error:
        return {"ok": False, "error": ready_error}
    return {
        "ok": True,
        "voices": listed_voice_names(voice_overlays, VARIANT),
        "default": default_voice,
        "device": DEVICE,
        "model": MODEL_NAME,
        "variant": VARIANT,
        "loaded": engine is not None,
        "policy": LOAD_POLICY,
        "voice_mode": VARIANT_META["voice_mode"],
        "sample_rate": VARIANT_META["sample_rate"],
    }


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
def openai_voices(model: str | None = None):
    if model and not is_public_model_request(model, MODEL_NAME):
        raise HTTPException(status_code=400, detail=f"unknown model {model!r}")
    names = listed_voice_names(voice_overlays, VARIANT)
    return {
        "object": "list",
        "data": [
            {
                "voice_id": n,
                "name": n,
                "kind": overlay_kind(n, voice_overlays)
                or ("builtin" if n.lower() == "default" else "alias"),
                "instructions": overlay_instructions(n, voice_overlays) or None,
            }
            for n in names
        ],
        "default": default_voice,
    }


@app.get("/v1/models")
def openai_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    log_api_error(request, 422, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    log_api_error(request, exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log_api_error(request, 500, repr(exc))
    return JSONResponse(status_code=500, content={"detail": str(exc)})


CLONE_WAV_MAX = 50 * 1024 * 1024
_WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


def _safe_alias(name: str) -> str:
    cleaned = "".join(ch for ch in (name or "") if ch.isalnum() or ch in "._-")
    return cleaned[:64]


def _is_wav_upload(filename: str, content_type: str, raw: bytes) -> bool:
    name = (filename or "").lower()
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if name.endswith(".wav") or name.endswith(".wave") or ctype in _WAV_TYPES:
        return True
    return raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"


async def _read_clone_wav(upload) -> bytes:
    filename = getattr(upload, "filename", "") or ""
    content_type = getattr(upload, "content_type", "") or ""
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="clone requires a wav file")
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="clone requires a wav file")
    if len(raw) > CLONE_WAV_MAX:
        raise HTTPException(status_code=400, detail="clone wav must be 50MB or smaller")
    if not _is_wav_upload(filename, content_type, raw):
        raise HTTPException(status_code=400, detail="clone requires a wav file")
    return raw


def _run_speech(
    voice: str,
    text: str,
    language: str,
    fmt: str,
    ref_audio: str | None,
    ref_text: str | None,
    speed: float | None,
):
    with lock:
        used, fell_back, reason = resolve_voice_route(
            voice,
            voice_overlays,
            default_voice,
            OPENAI_STOCK_VOICES,
            VARIANT,
        )
        shot_a = (ref_audio or "").strip()
        shot_t = (ref_text or "").strip()
        if shot_a and not shot_t:
            raise HTTPException(status_code=400, detail="clone requires ref_audio and ref_text")
        if shot_t and not shot_a:
            raise HTTPException(status_code=400, detail="clone requires ref_audio and ref_text")
        okind = overlay_kind(voice, voice_overlays)
        prompt_path: str | None = None
        if shot_a and shot_t:
            prompt_path = shot_a
            used = voice.strip() or "clone"
            fell_back = False
            reason = ""
        elif okind == "voice_clone":
            rel, _rtext = overlay_clone_ref(voice, voice_overlays)
            wav = resolve_reference_wav(rel, VOICES_PATH.parent)
            if wav is None:
                raise HTTPException(status_code=400, detail="clone preset missing wav")
            prompt_path = str(wav)
            used = voice.strip()
        elif used.lower() == "default" and supports_builtin_default(VARIANT):
            prompt_path = None
        elif not prompt_path:
            rel, _rtext = overlay_clone_ref(used, voice_overlays)
            wav = resolve_reference_wav(rel, VOICES_PATH.parent)
            if wav is None:
                if not supports_builtin_default(VARIANT):
                    raise HTTPException(
                        status_code=400,
                        detail="voice requires a reference WAV preset (ref_audio)",
                    )
            else:
                prompt_path = str(wav)

        if speed is not None:
            print(f"speed={speed} ignored; Chatterbox has no speed parameter", flush=True)
        eng = _ensure_engine()
        print(
            f"speech public={MODEL_NAME} variant={VARIANT} voice={voice!r} -> {used} "
            f"format={fmt} chars={len(text)} fallback={fell_back} ref={prompt_path!r}",
            flush=True,
        )
        audio, sr = call_generate(
            eng,
            VARIANT,
            text,
            audio_prompt_path=prompt_path,
            language=language,
            exaggeration=TTS_EXAGGERATION,
            cfg_weight=TTS_CFG_WEIGHT,
        )
    body, media = encode_audio(audio, sr, fmt)
    return VARIANT, used, fell_back, reason, body, media, speed is not None


def _speech_response(mid, used, fell_back, reason, body, media, speed_ignored: bool):
    headers = {"X-TTS-Voice-Used": used, "X-TTS-Model": mid, "X-TTS-Variant": VARIANT}
    if fell_back:
        headers["X-TTS-Fell-Back"] = "1"
        headers["X-TTS-Fell-Back-Reason"] = reason
    if speed_ignored:
        headers["X-TTS-Speed-Ignored"] = "1"
    return Response(content=body, media_type=media, headers=headers)


async def _openai_speech_multipart(request: Request):
    form = await request.form()
    text = str(form.get("input") or form.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")
    voice = str(form.get("voice") or "")
    language = str(form.get("language") or DEFAULT_LANGUAGE)
    fmt = str(form.get("response_format") or "mp3").lower().strip()
    speed_raw = form.get("speed")
    speed = float(speed_raw) if speed_raw not in (None, "") else None
    ref_text = str(form.get("ref_text") or "").strip()
    upload = form.get("ref_audio")
    if upload is None or not hasattr(upload, "read") or not ref_text:
        raise HTTPException(status_code=400, detail="clone requires ref_audio and ref_text")
    raw = await _read_clone_wav(upload)
    tmp_dir = CLONES_DIR if CLONES_DIR.parent.is_dir() else Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(tmp_dir))
    try:
        tmp.write(raw)
        tmp.close()
        mid, used, fell_back, reason, body, media, speed_ignored = await asyncio.to_thread(
            _run_speech,
            voice,
            text,
            language,
            fmt,
            tmp.name,
            ref_text,
            speed,
        )
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass
    return _speech_response(mid, used, fell_back, reason, body, media, speed_ignored)


@app.post("/v1/audio/speech")
async def openai_speech(request: Request):
    if ready_error:
        raise HTTPException(status_code=503, detail=ready_error)
    ct = (request.headers.get("content-type") or "").lower()
    if ct.startswith("multipart/form-data"):
        return await _openai_speech_multipart(request)
    try:
        req = OpenAISpeechRequest.model_validate(await request.json())
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")
    if req.model and not is_public_model_request(req.model, MODEL_NAME):
        raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
    text = (req.input or req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")
    fmt = (req.response_format or "mp3").lower().strip()
    language = req.language or DEFAULT_LANGUAGE
    try:
        mid, used, fell_back, reason, body, media, speed_ignored = await asyncio.to_thread(
            _run_speech,
            str(req.voice or ""),
            text,
            language,
            fmt,
            None,
            None,
            req.speed,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return _speech_response(mid, used, fell_back, reason, body, media, speed_ignored)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
