#!/usr/bin/env python3
import unittest
from unittest.mock import MagicMock

import numpy as np

from adapter import call_generate, generation_kwargs, tensor_to_float32_1d


class TensorConvertTests(unittest.TestCase):
    def test_flattens_2d_float32(self):
        wav = np.zeros((1, 8), dtype=np.float32)
        out = tensor_to_float32_1d(wav)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (8,))


class GenerationKwargsTests(unittest.TestCase):
    def test_turbo_forwards_ref_omits_language(self):
        kwargs = generation_kwargs(
            "turbo",
            audio_prompt_path="/tmp/a.wav",
            language="fr",
            exaggeration=0.7,
            cfg_weight=0.3,
        )
        self.assertEqual(kwargs, {"audio_prompt_path": "/tmp/a.wav"})

    def test_english_exaggeration(self):
        kwargs = generation_kwargs(
            "english",
            audio_prompt_path=None,
            language="en",
            exaggeration=0.7,
            cfg_weight=0.3,
        )
        self.assertEqual(kwargs, {"exaggeration": 0.7, "cfg_weight": 0.3})

    def test_multilingual_language_and_ref(self):
        kwargs = generation_kwargs(
            "multilingual",
            audio_prompt_path="/tmp/a.wav",
            language="FR",
            exaggeration=0.5,
            cfg_weight=0.5,
        )
        self.assertEqual(kwargs["language_id"], "fr")
        self.assertEqual(kwargs["audio_prompt_path"], "/tmp/a.wav")
        self.assertEqual(kwargs["exaggeration"], 0.5)


class CallGenerateTests(unittest.TestCase):
    def test_turbo_call(self):
        engine = MagicMock()
        engine.sr = 24000
        engine.generate.return_value = np.ones((1, 4), dtype=np.float32)
        audio, sr = call_generate(
            engine, "turbo", "hello", audio_prompt_path="/tmp/ref.wav", language="en"
        )
        engine.generate.assert_called_once_with("hello", audio_prompt_path="/tmp/ref.wav")
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.shape, (4,))

    def test_multilingual_call(self):
        engine = MagicMock()
        engine.sr = 24000
        engine.generate.return_value = np.ones((1, 4), dtype=np.float32)
        call_generate(
            engine,
            "multilingual",
            "bonjour",
            audio_prompt_path="/tmp/ref.wav",
            language="fr",
            exaggeration=0.5,
            cfg_weight=0.5,
        )
        engine.generate.assert_called_once_with(
            "bonjour",
            audio_prompt_path="/tmp/ref.wav",
            language_id="fr",
            exaggeration=0.5,
            cfg_weight=0.5,
        )


if __name__ == "__main__":
    unittest.main()
