# chatterbox-tts-openai

OpenAI-compatible HTTP API for [Chatterbox TTS](https://github.com/resemble-ai/chatterbox).

Default engine is **Chatterbox-Turbo** (`TTS_VARIANT=turbo`). Voice identity is a **reference WAV preset**, not a checkpoint speaker id. `instructions` is not Qwen voice design; Chatterbox has no voice-design API. Request `speed` is accepted and ignored (`X-TTS-Speed-Ignored: 1`).

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | `ok`, `variant`, configured voice aliases |
| GET | `/v1/models` | public id `tts-1` (override with `TTS_MODEL_NAME`) |
| GET | `/v1/voices` | aliases from `voices.json` plus built-in `default` for `english` and `multilingual` |
| POST | `/v1/audio/speech` | JSON or multipart clone |
| GET | `/` `/clone` `/config` | web UI |
| GET | `/design` | **404** — no Chatterbox equivalent |

JSON fields: `input`/`text`, `voice`, `model`, `language`, `response_format`, `speed` (ignored). Formats: `wav`, `pcm`, `mp3`, `opus`, `aac`, `flac`.

## Variants (`TTS_VARIANT`)

| Value | Class | Reference audio | Extra |
| --- | --- | --- | --- |
| `turbo` (default) | `ChatterboxTurboTTS.from_pretrained(device=...)` | **required** (`audio_prompt_path`) | paralinguistic tags `[laugh]` `[chuckle]` `[cough]` |
| `nano` | `ChatterboxTurboTTS.from_pretrained(device=..., nano=True)` | **required** | same tags; intended for CPU |
| `english` | `ChatterboxTTS.from_pretrained(device=...)` | optional; built-in `default` uses checkpoint conditionals | `TTS_EXAGGERATION`, `TTS_CFG_WEIGHT` |
| `multilingual` | `ChatterboxMultilingualTTS.from_pretrained(device=..., t3_model=v3)` | optional | `language` → `language_id`; default T3 is v3 via `TTS_T3_MODEL` |

`TTS_MODEL` is an optional **local checkpoint directory** passed to `from_local`. Official `from_pretrained` loaders hard-code Hugging Face repo ids, so a Hub id in `TTS_MODEL` is unused unless it is an existing local directory.

The Docker image installs Chatterbox from the GitHub repository so `nano=True` and multilingual `t3_model="v3"` match the current README. PyPI `chatterbox-tts==0.1.7` only exposes `from_pretrained(device)` and cannot load Nano or V3.

Turbo/Nano downloads: `ResembleAI/chatterbox-turbo` or `ResembleAI/chatterbox-nano`. English/multilingual: `ResembleAI/chatterbox`. Cache is Hugging Face's default (`HF_HOME` / `~/.cache/huggingface`).

### Multilingual language ids

`ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh`

## voices.json

```json
{
  "voices": {
    "jane": {
      "kind": "voice_clone",
      "ref_audio": "clones/jane.wav",
      "ref_text": "Transcript of the reference clip."
    }
  }
}
```

`ref_audio` must be a `.wav` under the config directory. Missing/non-WAV/out-of-root paths return HTTP 400. `ref_text` and `instructions` are metadata only; generation uses the WAV.

One-shot clone: `multipart/form-data` with `input`, `voice`, `ref_text`, and `ref_audio`. Temporary upload is deleted after the request. Missing `ref_text` → HTTP 400.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `TTS_VARIANT` | `turbo` | `turbo` `nano` `english` `multilingual` |
| `TTS_DEVICE` | auto `cuda:0` → `mps` → `cpu` | explicit torch device |
| `TTS_MODEL` | `/models` | pack root (bind-mount this); `from_local` or Turbo LoRA `pack.json` |
| `TTS_T3_MODEL` | `v3` | multilingual T3 (`v2` or `v3`) |
| `TTS_VOICES` | `/config/voices.json` | presets |
| `TTS_MODEL_NAME` | `tts-1` | public `/v1/models` id |
| `TTS_LANGUAGE` | `en` | default `language_id` |
| `TTS_DEFAULT_VOICE` | first listed | default alias |
| `TTS_LOAD_POLICY` | `lazy` | `lazy` keeps every pack loaded on first use; `one` unloads the previous pack on switch; `all` preloads every pack. On CUDA/CPU OOM while loading, the least-recent pack is unloaded and the load is retried |
| `TTS_EXAGGERATION` | unset | english/multilingual `exaggeration` |
| `TTS_CFG_WEIGHT` | unset | english/multilingual `cfg_weight` |
| `TTS_HOST` / `TTS_PORT` | `0.0.0.0` / `8080` | bind |

## Docker

```bash
docker build --build-arg TORCH_BACKEND=cpu -t chatterbox-tts-openai:cpu .
docker build --build-arg TORCH_BACKEND=cuda -t chatterbox-tts-openai:cuda .

docker run --rm -p 8080:8080 \
  -e TTS_VARIANT=turbo \
  -v "$PWD/config:/config" \
  -v /path/to/chatterbox-packs:/models \
  chatterbox-tts-openai:cpu
```

Drop a pack directory with `pack.json` under the host models folder (for example `GOOD-en-serling-turbo-v2-e50`), then `POST /ui/rescan` or use **Rescan models** in the UI. Do not copy packs into a running container; `/models` is the bind mount.

CUDA:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -e TTS_DEVICE=cuda:0 \
  -e TTS_VARIANT=turbo \
  -v "$PWD/config:/config" \
  -v /path/to/chatterbox-packs:/models \
  ghcr.io/doguitar/chatterbox-tts-openai:cuda
```

Published images: `ghcr.io/doguitar/chatterbox-tts-openai:cpu` (`:latest`) and `:cuda`. First start downloads weights into the Hugging Face cache (add a writable `HF_HOME` volume if you want them persisted).

## curl

```bash
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/v1/voices

curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"jane","input":"Hello from Chatterbox.","response_format":"wav"}' \
  --output out.wav

curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -F input='Hello from Chatterbox.' \
  -F voice=clone \
  -F ref_text='This is the transcript of the reference wav.' \
  -F ref_audio=@clones/jane.wav \
  -F response_format=wav \
  --output clone.wav
```

Header `X-TTS-Voice-Used` is the alias actually used.

## Tests

```bash
python -m unittest discover -p 'test_*.py'
```

Helper/adapter tests do not download weights.

## License

MIT. Chatterbox weights and package are from Resemble AI; see their repository for model terms.
