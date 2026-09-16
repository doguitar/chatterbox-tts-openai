# chatterbox-tts-openai

OpenAI-compatible HTTP API for [Chatterbox TTS](https://github.com/resemble-ai/chatterbox). Fork of [qwen3-tts-openai](https://github.com/doguitar/qwen3-tts-openai) with the Qwen3 inference layer replaced by Chatterbox.

Default engine is **Chatterbox-Turbo** (`TTS_VARIANT=turbo`). A public `voice` is a **reference WAV** (config preset or train pack), not a Qwen checkpoint speaker id. `instructions` is not Qwen voice design; Chatterbox has no voice-design API (`GET /design` is 404). Request `speed` is accepted and ignored (`X-TTS-Speed-Ignored: 1`).

## What lives where

| Path | Role |
| --- | --- |
| `/config` (`TTS_VOICES`) | `voices.json` plus `clones/*.wav` presets |
| `/models` (`TTS_MODEL`) | **Train packs** (`pack.json` folders). Bind-mount this; do not `docker cp` into a running container |
| `HF_HOME` (or `~/.cache/huggingface`) | Resemble Hub backbone (`ResembleAI/chatterbox-turbo`, etc.). Not the packs folder |

A **pack** is a train export (`pack.json`, merged T3 weights, tokenizer, `reference.wav`). A **voice** is the name clients send (`pack.json` `"name"` or a `voices.json` alias). The Hub **backbone** (S3Gen, voice encoder, stock Turbo T3) is still loaded with `from_pretrained`; the pack then replaces T3 with the merged file. The PEFT `adapter/` directory in a pack is kept on disk and is not applied at load time.

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | `ok`, `variant`, voice list, `loaded` (resident engine keys), `models` (discovered packs) |
| GET | `/v1/models` | public id `tts-1` (override with `TTS_MODEL_NAME`) |
| GET | `/v1/voices` | pack names plus `voices.json` aliases; built-in `default` for `english` / `multilingual` |
| POST | `/v1/audio/speech` | JSON or multipart clone |
| POST | `/ui/rescan` | re-read `/models`; Speak UI **Rescan models** |
| GET | `/` `/clone` `/config` | web UI |
| GET | `/design` | **404** |

JSON fields: `input`/`text`, `voice`, `model`, `language`, `response_format`, `speed` (ignored). Formats: `wav`, `pcm`, `mp3`, `opus`, `aac`, `flac`.

Response headers: `X-TTS-Voice-Used` (alias), `X-TTS-Model` (engine cache key: pack name or `__pretrained__`), `X-TTS-Variant`.

## Variants (`TTS_VARIANT`)

| Value | Class | Reference audio | Extra |
| --- | --- | --- | --- |
| `turbo` (default) | `ChatterboxTurboTTS.from_pretrained(device=...)` | **required** (`audio_prompt_path`) | paralinguistic tags `[laugh]` `[chuckle]` `[cough]` |
| `nano` | `ChatterboxTurboTTS.from_pretrained(device=..., nano=True)` | **required** | same tags; intended for CPU |
| `english` | `ChatterboxTTS.from_pretrained(device=...)` | optional; built-in `default` uses checkpoint conditionals | `TTS_EXAGGERATION`, `TTS_CFG_WEIGHT` |
| `multilingual` | `ChatterboxMultilingualTTS.from_pretrained(device=..., t3_model=v3)` | optional | `language` → `language_id`; `TTS_T3_MODEL` |

The Docker image installs Chatterbox from GitHub so `nano=True` and multilingual `t3_model="v3"` match the current Resemble README. PyPI `chatterbox-tts==0.1.7` cannot load Nano or V3.

Hub ids: Turbo `ResembleAI/chatterbox-turbo`, Nano `ResembleAI/chatterbox-nano`, English/multilingual `ResembleAI/chatterbox`.

### Multilingual language ids

`ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh`

## Train packs

Drop one directory per pack under the `/models` bind. Example `pack.json`:

```json
{
  "name": "alice",
  "model": "turbo-lora",
  "merged": "t3_turbo_finetuned_merged.safetensors",
  "reference": "reference.wav",
  "tokenizer": "tokenizer",
  "engine": "chatterbox-turbo"
}
```

- `name` is the public voice id (`alice`).
- `merged` is **T3 only** (base Turbo T3 with LoRA baked in), not the full Chatterbox stack.
- Speech with that voice loads that pack’s merged T3 on the Hub Turbo backbone and uses the pack’s `reference.wav`.
- `voices.json` aliases that are not pack names (for example `jane`) use the stock Hub engine (`__pretrained__`) plus their clone WAV.
- A second pack is another resident engine, not another speaker on the same T3. Speak each pack voice to load it.

`TTS_MODEL` as a **single official checkpoint directory** (`ve.safetensors` / `t3_turbo_v1.safetensors` / …) still uses `from_local`. An empty `/models` directory is only a pack catalog root; it is not treated as a checkpoint.

## Engine cache (`TTS_LOAD_POLICY`)

Same idea as the Qwen image’s checkpoint cache, keyed by pack:

| Policy | Behavior |
| --- | --- |
| `lazy` (default) | load on first speech for that pack; **keep** every pack that has been used |
| `one` | unload the previous pack before loading the next |
| `all` | preload every discovered pack at startup (and Hub pretrained if there are no packs) |

If load hits CUDA/CPU **out of memory**, the least-recent other pack is unloaded (`gc` + `empty_cache`) and the load is retried. Logs look like `memory full loading 'orwell'; unloading 'alice'`. `/health` `loaded` is the resident list.

**Hub downloads** happen when an engine actually loads (`from_pretrained`), not on a lazy boot with no speech. Persist them with a writable `HF_HOME` volume. Packs stay on `/models`.

`TTS_DEFAULT_MODEL` selects which pack `one`/`all` load first (pack `name` or folder name). Rescan re-reads the catalog; engines for packs that disappeared are unloaded.

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

`ref_audio` must be a `.wav` under the config directory (or a pack-relative path resolved under `/models`). Missing / non-WAV / out-of-root paths return HTTP 400. `ref_text` and `instructions` are metadata; generation uses the WAV.

One-shot clone: `multipart/form-data` with `input`, `voice`, `ref_text`, and `ref_audio`. The temporary upload is deleted after the request. Missing `ref_text` → HTTP 400.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `TTS_VARIANT` | `turbo` | `turbo` `nano` `english` `multilingual` |
| `TTS_DEVICE` | auto `cuda:0` → `mps` → `cpu` | explicit torch device (no XPU image) |
| `TTS_MODEL` | `/models` | pack catalog root and/or local official checkpoint |
| `TTS_DEFAULT_MODEL` | first pack | pack `name` or folder to prefer on `one`/`all` |
| `TTS_T3_MODEL` | `v3` | multilingual T3 (`v2` or `v3`) |
| `TTS_VOICES` | `/config/voices.json` | presets |
| `TTS_MODEL_NAME` | `tts-1` | public `/v1/models` id |
| `TTS_LANGUAGE` | `en` | default `language_id` |
| `TTS_DEFAULT_VOICE` | first listed | default alias |
| `TTS_LOAD_POLICY` | `lazy` | `lazy` / `one` / `all` (see above); OOM evicts LRU |
| `TTS_EXAGGERATION` | unset | english/multilingual `exaggeration` |
| `TTS_CFG_WEIGHT` | unset | english/multilingual `cfg_weight` |
| `HF_HOME` | Hugging Face default | persist Hub backbone downloads |
| `TTS_HOST` / `TTS_PORT` | `0.0.0.0` / `8080` | bind |

## Docker

CPU and CUDA images only (`TORCH_BACKEND=cpu` or `cuda`). There is no XPU tag.

```bash
docker build --build-arg TORCH_BACKEND=cpu -t chatterbox-tts-openai:cpu .
docker build --build-arg TORCH_BACKEND=cuda -t chatterbox-tts-openai:cuda .

docker run --rm -p 8080:8080 \
  -e TTS_VARIANT=turbo \
  -e TTS_LOAD_POLICY=lazy \
  -e HF_HOME=/hf \
  -v "$PWD/config:/config" \
  -v /path/to/chatterbox-packs:/models \
  -v /path/to/hf-cache:/hf \
  chatterbox-tts-openai:cpu
```

Drop a pack directory with `pack.json` under the host packs folder, then `POST /ui/rescan` or **Rescan models**. `/models` is the bind mount.

CUDA:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -e TTS_DEVICE=cuda:0 \
  -e TTS_VARIANT=turbo \
  -e HF_HOME=/hf \
  -v "$PWD/config:/config" \
  -v /path/to/chatterbox-packs:/models \
  -v /path/to/hf-cache:/hf \
  ghcr.io/doguitar/chatterbox-tts-openai:cuda
```

Published images: `ghcr.io/doguitar/chatterbox-tts-openai:cpu` (`:latest`) and `:cuda`.

## Unraid

Templates live in `unraid/`:

- `chatterbox-tts-openai.xml` — CUDA (`:cuda`, `--gpus=all`)
- `chatterbox-tts-openai-cpu.xml` — CPU (`:cpu`)

On the Unraid host, copy them into Docker’s user-template directory (`my-` prefix is required for Add Container to list them):

```bash
mkdir -p /boot/config/plugins/dockerMan/templates-user
curl -fsSL -o /boot/config/plugins/dockerMan/templates-user/my-chatterbox-tts-openai.xml \
  https://raw.githubusercontent.com/doguitar/chatterbox-tts-openai/main/unraid/chatterbox-tts-openai.xml
curl -fsSL -o /boot/config/plugins/dockerMan/templates-user/my-chatterbox-tts-openai-cpu.xml \
  https://raw.githubusercontent.com/doguitar/chatterbox-tts-openai/main/unraid/chatterbox-tts-openai-cpu.xml
```

Then **Docker → Add Container**, open the **Template** dropdown, and pick `chatterbox-tts-openai` or `chatterbox-tts-openai-cpu`. Paths default to `/mnt/user/appdata/chatterbox-tts-openai/{config,models,hf}`. GHCR packages start private: `docker login ghcr.io` on the Unraid host, or make the package public.

## curl

```bash
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/v1/voices
curl -sS -X POST http://127.0.0.1:8080/ui/rescan

curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"alice","input":"Hello from Chatterbox.","response_format":"wav"}' \
  --output alice.wav

curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"jane","input":"Hello from Chatterbox.","response_format":"wav"}' \
  --output jane.wav

curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -F input='Hello from Chatterbox.' \
  -F voice=clone \
  -F ref_text='This is the transcript of the reference wav.' \
  -F ref_audio=@clones/jane.wav \
  -F response_format=wav \
  --output clone.wav
```

## Tests

```bash
python -m unittest discover -p 'test_*.py'
```

Helper/adapter/HTTP tests do not download weights.

## License

MIT. Chatterbox weights and package are from Resemble AI; see their repository for model terms.
