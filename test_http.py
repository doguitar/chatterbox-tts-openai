#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class FakeEngine:
    sr = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.zeros((1, 480), dtype=np.float32)


class SpeechHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        clones = self.root / "clones"
        clones.mkdir()
        self.wav = clones / "jane.wav"
        self.wav.write_bytes(b"RIFF....WAVE")
        (self.root / "voices.json").write_text(
            json.dumps(
                {
                    "voices": {
                        "jane": {
                            "kind": "voice_clone",
                            "ref_audio": "clones/jane.wav",
                            "ref_text": "Hello from Jane.",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        self.env = {
            "TTS_VOICES": str(self.root / "voices.json"),
            "TTS_VARIANT": "turbo",
            "TTS_DEVICE": "cpu",
            "TTS_LOAD_POLICY": "lazy",
            "TTS_MODEL": "",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _client(self, variant="turbo"):
        self.env["TTS_VARIANT"] = variant
        fake = FakeEngine()
        with patch.dict("os.environ", self.env, clear=False):
            import importlib
            import server as server_mod

            importlib.reload(server_mod)
            server_mod.VOICES_PATH = self.root / "voices.json"
            server_mod.CLONES_DIR = self.root / "clones"
            server_mod.VARIANT = variant
            server_mod._rebuild_voices()
            server_mod.loaded = {server_mod.PRETRAINED_ENGINE_KEY: fake}
            server_mod.lru_order = [server_mod.PRETRAINED_ENGINE_KEY]
            server_mod.engine_variant_by = {server_mod.PRETRAINED_ENGINE_KEY: variant}
            from fastapi.testclient import TestClient

            return TestClient(server_mod.app), fake, server_mod

    def test_json_speech_wav(self):
        client, fake, _mod = self._client("turbo")
        res = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "voice": "jane",
                "input": "Hello from Chatterbox.",
                "response_format": "wav",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.headers.get("content-type"), "audio/wav")
        self.assertEqual(res.headers.get("x-tts-voice-used"), "jane")
        self.assertEqual(res.headers.get("x-tts-model"), "__pretrained__")
        self.assertEqual(res.headers.get("x-tts-load-state"), "hot")
        self.assertEqual(res.headers.get("x-tts-input-tokens"), "unavailable")
        self.assertRegex(res.headers.get("x-tts-processing-time-ms") or "", r"^\d+$")
        self.assertGreater(len(res.content), 44)
        self.assertEqual(fake.calls[0][0], "Hello from Chatterbox.")
        self.assertEqual(fake.calls[0][1]["audio_prompt_path"], str(self.wav.resolve()))

    def test_multilingual_forwards_language(self):
        client, fake, _mod = self._client("multilingual")
        res = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "voice": "jane",
                "input": "Bonjour.",
                "language": "fr",
                "response_format": "wav",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(fake.calls[0][1]["language_id"], "fr")

    def test_english_clone_generation_exaggeration(self):
        client, fake, server_mod = self._client("english")
        (self.root / "voices.json").write_text(
            json.dumps(
                {
                    "generation": {"exaggeration": 0.4, "cfg_weight": 0.2},
                    "voices": {
                        "jane": {
                            "kind": "voice_clone",
                            "ref_audio": "clones/jane.wav",
                            "ref_text": "Hello from Jane.",
                            "generation": {"exaggeration": 0.9, "cfg_weight": 0.1},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        server_mod._rebuild_voices()
        res = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "voice": "jane", "input": "Hello.", "response_format": "wav"},
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(fake.calls[0][1]["exaggeration"], 0.9)
        self.assertEqual(fake.calls[0][1]["cfg_weight"], 0.1)


    def test_health_and_voices(self):
        client, _fake, _mod = self._client("turbo")
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        body = health.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["variant"], "turbo")
        self.assertIn("jane", body["voices"])
        voices = client.get("/v1/voices").json()
        ids = [row["voice_id"] for row in voices["data"]]
        self.assertEqual(ids, ["jane"])

    def test_design_404(self):
        client, _fake, _mod = self._client("turbo")
        res = client.get("/design")
        self.assertEqual(res.status_code, 404)

    def test_missing_clone_wav_400(self):
        client, _fake, _mod = self._client("turbo")
        (self.root / "voices.json").write_text(
            json.dumps(
                {
                    "voices": {
                        "ghost": {
                            "kind": "voice_clone",
                            "ref_audio": "clones/missing.wav",
                            "ref_text": "x",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        _mod = client.app
        import server as server_mod

        server_mod._rebuild_voices()
        res = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "voice": "ghost", "input": "Hi", "response_format": "wav"},
        )
        self.assertEqual(res.status_code, 400)

    def test_multipart_clone_and_tmp_cleanup(self):
        client, fake, server_mod = self._client("turbo")
        before = set(self.root.glob("**/*"))
        res = client.post(
            "/v1/audio/speech",
            data={
                "input": "Hello clone.",
                "voice": "oneshot",
                "ref_text": "This is the transcript.",
                "response_format": "wav",
            },
            files={"ref_audio": ("ref.wav", self.wav.read_bytes(), "audio/wav")},
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertTrue(fake.calls)
        after = set(self.root.glob("**/*"))
        extras = {p for p in after - before if p.is_file() and p.suffix == ".wav" and p != self.wav}
        self.assertEqual(extras, set())

    def test_multipart_missing_ref_text_400(self):
        client, _fake, _mod = self._client("turbo")
        res = client.post(
            "/v1/audio/speech",
            data={"input": "Hello clone.", "voice": "oneshot", "response_format": "wav"},
            files={"ref_audio": ("ref.wav", self.wav.read_bytes(), "audio/wav")},
        )
        self.assertEqual(res.status_code, 400)


    def test_json_speech_cold_load_headers(self):
        client, _server_mod, engines = self._client_unloaded()
        res = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "voice": "jane",
                "input": "Hello from Chatterbox.",
                "response_format": "wav",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.headers.get("content-type"), "audio/wav")
        self.assertEqual(res.headers.get("x-tts-model"), "__pretrained__")
        self.assertEqual(res.headers.get("x-tts-load-state"), "cold")
        self.assertEqual(res.headers.get("x-tts-input-tokens"), "unavailable")
        self.assertRegex(res.headers.get("x-tts-processing-time-ms") or "", r"^\d+$")
        self.assertEqual(len(engines), 1)
        hot = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "voice": "jane",
                "input": "Second line.",
                "response_format": "wav",
            },
        )
        self.assertEqual(hot.status_code, 200, hot.text)
        self.assertEqual(hot.headers.get("x-tts-load-state"), "hot")




    def _client_unloaded(self, variant="turbo"):
        self.env["TTS_VARIANT"] = variant
        self.env["TTS_MODEL"] = str(self.root / "empty-models")
        (self.root / "empty-models").mkdir(exist_ok=True)
        engines = []

        def fake_load(v, device, model_path="", t3_model="v3"):
            eng = FakeEngine()
            engines.append(eng)
            return eng

        with patch.dict("os.environ", self.env, clear=False):
            import importlib
            import server as server_mod

            importlib.reload(server_mod)
            server_mod.VOICES_PATH = self.root / "voices.json"
            server_mod.CLONES_DIR = self.root / "clones"
            server_mod.MODEL_PATH = str(self.root / "empty-models")
            server_mod.VARIANT = variant
            server_mod.load_engine = fake_load
            server_mod._rescan_catalog(unload=False)
            from fastapi.testclient import TestClient

            return TestClient(server_mod.app), server_mod, engines

    def test_pretrained_catalog_before_and_after_speech(self):
        (self.root / "voices.json").write_text(
            json.dumps(
                {
                    "generation": {"temperature": 0.7, "max_gen_len": 1200},
                    "voices": {
                        "jane": {
                            "kind": "voice_clone",
                            "ref_audio": "clones/jane.wav",
                            "ref_text": "Hello from Jane.",
                            "generation": {"temperature": 0.6},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        client, server_mod, engines = self._client_unloaded()
        health = client.get("/health").json()
        models = health["models"]
        pretrained = [m for m in models if m["id"] == "__pretrained__"]
        self.assertEqual(len(pretrained), 1)
        self.assertEqual(pretrained[0]["path"], "(pretrained)")
        self.assertEqual(pretrained[0]["kind"], "turbo")
        self.assertFalse(pretrained[0]["loaded"])
        rescan = client.post("/ui/rescan").json()
        self.assertEqual(sum(1 for m in rescan["models"] if m["id"] == "__pretrained__"), 1)
        self.assertFalse(next(m["loaded"] for m in rescan["models"] if m["id"] == "__pretrained__"))
        res = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "voice": "jane",
                "input": "This is sentence one. This is sentence two.",
                "response_format": "wav",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.headers.get("x-tts-model"), "__pretrained__")
        self.assertEqual(engines[0].calls[0][1]["temperature"], 0.6)
        self.assertEqual(engines[0].calls[0][1]["max_gen_len"], 1200)
        after = client.get("/health").json()["models"]
        self.assertEqual(sum(1 for m in after if m["id"] == "__pretrained__"), 1)
        self.assertTrue(next(m["loaded"] for m in after if m["id"] == "__pretrained__"))

    def test_clone_and_pack_generation_headers(self):
        (self.root / "voices.json").write_text(
            json.dumps(
                {
                    "generation": {"temperature": 0.7, "max_gen_len": 1200},
                    "voices": {
                        "jane": {
                            "kind": "voice_clone",
                            "ref_audio": "clones/jane.wav",
                            "ref_text": "Hello from Jane.",
                            "generation": {"temperature": 0.6},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        models = self.root / "models"
        models.mkdir()
        pack = models / "GOOD-en-alice"
        pack.mkdir()
        (pack / "pack.json").write_text(
            json.dumps(
                {
                    "name": "alice",
                    "model": "turbo-lora",
                    "reference": "reference.wav",
                    "generation": {"max_gen_len": 900},
                }
            ),
            encoding="utf-8",
        )
        (pack / "reference.wav").write_bytes(b"RIFF....WAVE")
        self.env["TTS_MODEL"] = str(models)
        engines = {}

        def fake_load(variant, device, model_path="", t3_model="v3"):
            eng = FakeEngine()
            engines[model_path or "__pretrained__"] = eng
            return eng

        with patch.dict("os.environ", self.env, clear=False):
            import importlib
            import server as server_mod

            importlib.reload(server_mod)
            server_mod.VOICES_PATH = self.root / "voices.json"
            server_mod.CLONES_DIR = self.root / "clones"
            server_mod.MODEL_PATH = str(models)
            server_mod.load_engine = fake_load
            server_mod._rescan_catalog(unload=False)
            from fastapi.testclient import TestClient

            client = TestClient(server_mod.app)

        sentence = "This is sentence one. This is sentence two."
        clone = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "voice": "jane", "input": sentence, "response_format": "wav"},
        )
        self.assertEqual(clone.status_code, 200, clone.text)
        self.assertEqual(clone.headers.get("x-tts-model"), "__pretrained__")
        pre = engines["__pretrained__"]
        self.assertEqual(pre.calls[0][1]["temperature"], 0.6)
        self.assertEqual(pre.calls[0][1]["max_gen_len"], 1200)
        pack_res = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "voice": "alice", "input": sentence, "response_format": "wav"},
        )
        self.assertEqual(pack_res.status_code, 200, pack_res.text)
        self.assertEqual(pack_res.headers.get("x-tts-model"), "alice")
        pack_eng = next(eng for path, eng in engines.items() if path != "__pretrained__")
        self.assertEqual(pack_eng.calls[0][1]["temperature"], 0.7)
        self.assertEqual(pack_eng.calls[0][1]["max_gen_len"], 900)

    def test_ui_voices_generation_round_trip_and_errors(self):
        client, _fake, _mod = self._client("turbo")
        payload = {
            "generation": {
                "temperature": 0.0,
                "norm_loudness": False,
                "max_gen_len": 12,
                "exaggeration": 0.5,
                "cfg_weight": 0.25,
            },
            "voices": {
                "jane": {
                    "kind": "voice_clone",
                    "ref_audio": "clones/jane.wav",
                    "generation": {"temperature": 0.6, "exaggeration": 0.8},
                }
            },
        }
        res = client.put("/ui/voices", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()["document"]
        self.assertEqual(body["generation"]["temperature"], 0.0)
        self.assertIs(body["generation"]["norm_loudness"], False)
        self.assertEqual(body["generation"]["exaggeration"], 0.5)
        self.assertEqual(body["generation"]["cfg_weight"], 0.25)
        self.assertEqual(body["voices"]["jane"]["generation"]["exaggeration"], 0.8)
        again = client.get("/ui/voices").json()["document"]
        self.assertEqual(again["generation"]["temperature"], 0.0)
        self.assertIs(again["generation"]["norm_loudness"], False)
        self.assertEqual(again["generation"]["exaggeration"], 0.5)
        bad = client.put(
            "/ui/voices",
            json={"generation": {"nope": 1}, "voices": payload["voices"]},
        )
        self.assertEqual(bad.status_code, 400)
        self.assertIn("generation.nope", bad.json()["detail"])
        bad = client.put(
            "/ui/voices",
            json={"generation": {"top_k": 0}, "voices": payload["voices"]},
        )
        self.assertEqual(bad.status_code, 400)
        self.assertIn("top_k", bad.json()["detail"])
        bad = client.put(
            "/ui/voices",
            json={"generation": {"max_gen_len": 0}, "voices": payload["voices"]},
        )
        self.assertEqual(bad.status_code, 400)
        bad = client.put(
            "/ui/voices",
            json={"generation": {"temperature": "x"}, "voices": payload["voices"]},
        )
        self.assertEqual(bad.status_code, 400)
        bad = client.put(
            "/ui/voices",
            content=b'{"generation":{"top_p":Infinity},"voices":{"jane":{"kind":"voice_clone","ref_audio":"clones/jane.wav"}}}',
            headers={"content-type": "application/json"},
        )
        self.assertEqual(bad.status_code, 400)
        self.assertIn("finite", bad.json()["detail"])


    def _pack(self, root, name, folder):
        pack = root / folder
        pack.mkdir()
        (pack / "pack.json").write_text(
            json.dumps({"name": name, "model": "turbo-lora", "reference": "reference.wav"}),
            encoding="utf-8",
        )
        (pack / "reference.wav").write_bytes(b"RIFF....WAVE")
        return pack

    def _client_with_packs(self, policy="lazy"):
        models = self.root / "models"
        models.mkdir()
        self._pack(models, "alice", "GOOD-en-alice")
        self._pack(models, "orwell", "GOOD-en-orwell")
        self.env["TTS_MODEL"] = str(models)
        self.env["TTS_LOAD_POLICY"] = policy
        engines = []

        def fake_load(variant, device, model_path="", t3_model="v3"):
            eng = FakeEngine()
            eng.model_path = model_path
            engines.append(eng)
            if getattr(self, "_oom_once", False):
                self._oom_once = False
                raise RuntimeError("CUDA out of memory")
            return eng

        with patch.dict("os.environ", self.env, clear=False):
            import importlib
            import server as server_mod

            importlib.reload(server_mod)
            server_mod.VOICES_PATH = self.root / "voices.json"
            server_mod.CLONES_DIR = self.root / "clones"
            server_mod.MODEL_PATH = str(models)
            server_mod.LOAD_POLICY = policy
            server_mod._rescan_catalog(unload=False)
            server_mod.load_engine = fake_load
            from fastapi.testclient import TestClient

            return TestClient(server_mod.app), server_mod, engines

    def test_lazy_keeps_two_packs(self):
        client, server_mod, engines = self._client_with_packs("lazy")
        for voice in ("alice", "orwell"):
            res = client.post(
                "/v1/audio/speech",
                json={"model": "tts-1", "voice": voice, "input": "Hi", "response_format": "wav"},
            )
            self.assertEqual(res.status_code, 200, res.text)
            self.assertEqual(res.headers.get("x-tts-model"), voice)
        self.assertEqual(set(server_mod.loaded), {"alice", "orwell"})
        self.assertEqual(len(engines), 2)

    def test_one_unloads_previous_pack(self):
        client, server_mod, _engines = self._client_with_packs("one")
        client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "voice": "alice", "input": "Hi", "response_format": "wav"},
        )
        client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "voice": "orwell", "input": "Hi", "response_format": "wav"},
        )
        self.assertEqual(list(server_mod.loaded), ["orwell"])

    def test_memory_full_unloads_lru(self):
        client, server_mod, _engines = self._client_with_packs("lazy")
        client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "voice": "alice", "input": "Hi", "response_format": "wav"},
        )
        self._oom_once = True
        res = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "voice": "orwell", "input": "Hi", "response_format": "wav"},
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(list(server_mod.loaded), ["orwell"])



if __name__ == "__main__":
    unittest.main()
