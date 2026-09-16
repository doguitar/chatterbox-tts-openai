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
            server_mod.engine = fake
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


if __name__ == "__main__":
    unittest.main()
