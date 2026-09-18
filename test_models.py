#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from models import (
    ENGINE_VARIANTS,
    PRETRAINED_ENGINE_KEY,
    VoiceOverlay,
    discover_packs,
    engine_cache_key,
    engine_metadata,
    is_memory_full_error,
    is_public_model_request,
    listed_voice_names,
    lru_touch,
    lru_victim,
    merge_generation,
    overlay_clone_ref,
    overlay_generation,
    overlay_instructions,
    overlay_kind,
    pack_voice_overlays,
    parse_generation_config,
    parse_load_policy,
    parse_variant,
    parse_voice_overlays,
    public_default_voice,
    resolve_pack_for_voice,
    resolve_reference_wav,
    resolve_voice_route,
    supports_builtin_default,
    validate_voices_document,
)



class ParseVariantTests(unittest.TestCase):
    def test_default_turbo(self):
        self.assertEqual(parse_variant(""), "turbo")
        self.assertEqual(parse_variant(" Turbo "), "turbo")

    def test_accepted(self):
        for name in ENGINE_VARIANTS:
            self.assertEqual(parse_variant(name), name)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_variant("qwen")
        self.assertIn("turbo", str(ctx.exception))


class ParseLoadPolicyTests(unittest.TestCase):
    def test_empty_defaults_lazy(self):
        self.assertEqual(parse_load_policy(""), "lazy")
        self.assertEqual(parse_load_policy(" One "), "one")

    def test_all_accepted(self):
        self.assertEqual(parse_load_policy("all"), "all")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_load_policy("resident")
        self.assertIn("lazy", str(ctx.exception))


class EngineMetadataTests(unittest.TestCase):
    def test_turbo_reference_only(self):
        meta = engine_metadata("turbo")
        self.assertEqual(meta["public_id"], "chatterbox-turbo")
        self.assertEqual(meta["voice_mode"], "reference")
        self.assertFalse(supports_builtin_default("turbo"))
        self.assertFalse(supports_builtin_default("nano"))

    def test_english_builtin(self):
        meta = engine_metadata("english")
        self.assertEqual(meta["public_id"], "chatterbox")
        self.assertTrue(supports_builtin_default("english"))
        self.assertTrue(supports_builtin_default("multilingual"))
        self.assertEqual(meta["sample_rate"], 24000)


class PublicModelTests(unittest.TestCase):
    def test_aliases(self):
        self.assertTrue(is_public_model_request("", "tts-1"))
        self.assertTrue(is_public_model_request("tts-1", "tts-1"))
        self.assertTrue(is_public_model_request("chatterbox-tts", "tts-1"))
        self.assertFalse(is_public_model_request("alpha", "tts-1"))


class VoiceListingTests(unittest.TestCase):
    def test_turbo_lists_aliases_only(self):
        overlays = parse_voice_overlays(
            {
                "voices": {
                    "jane": {
                        "kind": "voice_clone",
                        "ref_audio": "clones/jane.wav",
                        "ref_text": "Hello.",
                    }
                }
            },
            "",
        )
        self.assertEqual(listed_voice_names(overlays, "turbo"), ["jane"])
        self.assertEqual(public_default_voice(overlays, "", "turbo"), "jane")

    def test_english_includes_default(self):
        overlays = parse_voice_overlays({"voices": {}}, "")
        self.assertEqual(listed_voice_names(overlays, "english"), ["default"])
        self.assertEqual(public_default_voice(overlays, "", "english"), "default")
        overlays = parse_voice_overlays(
            {"voices": {"jane": {"kind": "voice_clone", "ref_audio": "a.wav", "ref_text": "x"}}},
            "",
        )
        self.assertEqual(listed_voice_names(overlays, "english"), ["default", "jane"])

    def test_requested_default(self):
        overlays = [
            VoiceOverlay("jane", "", None, "", "voice_clone", "clones/jane.wav", "Hi"),
            VoiceOverlay("bob", "", None, "", "voice_clone", "clones/bob.wav", "Hi"),
        ]
        self.assertEqual(public_default_voice(overlays, "bob", "turbo"), "bob")


class OverlayHelpersTests(unittest.TestCase):
    def test_overlay_fields(self):
        overlays = parse_voice_overlays(
            {
                "voices": {
                    "jane": {
                        "kind": "voice_clone",
                        "ref_audio": "clones/jane.wav",
                        "ref_text": "Hello.",
                        "instructions": "warm",
                    }
                }
            },
            "",
        )
        self.assertEqual(overlay_kind("jane", overlays), "voice_clone")
        self.assertEqual(overlay_instructions("jane", overlays), "warm")
        self.assertEqual(overlay_clone_ref("jane", overlays), ("clones/jane.wav", "Hello."))
        self.assertIsNone(overlay_generation("jane", overlays))


class ResolveVoiceTests(unittest.TestCase):
    def test_alias_and_stock(self):
        overlays = parse_voice_overlays(
            {"voices": {"jane": {"kind": "voice_clone", "ref_audio": "a.wav", "ref_text": "x"}}},
            "",
        )
        used, fell, reason = resolve_voice_route(
            "jane", overlays, "jane", frozenset({"alloy"}), "turbo"
        )
        self.assertEqual((used, fell), ("jane", False))
        used, fell, reason = resolve_voice_route(
            "alloy", overlays, "jane", frozenset({"alloy"}), "turbo"
        )
        self.assertEqual((used, fell), ("jane", True))
        self.assertIn("openai stock voice", reason)
        used, fell, reason = resolve_voice_route(
            "missing", overlays, "jane", frozenset({"alloy"}), "turbo"
        )
        self.assertTrue(fell)
        self.assertEqual(used, "jane")


class ReferencePathTests(unittest.TestCase):
    def test_resolves_wav_under_config(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            clones = root / "clones"
            clones.mkdir()
            wav = clones / "jane.wav"
            wav.write_bytes(b"RIFF")
            self.assertEqual(resolve_reference_wav("clones/jane.wav", root), wav.resolve())
            self.assertIsNone(resolve_reference_wav("clones/missing.wav", root))
            self.assertIsNone(resolve_reference_wav("../jane.wav", root))
            self.assertIsNone(resolve_reference_wav("clones/jane.mp3", root))
            (clones / "jane.mp3").write_bytes(b"x")
            self.assertIsNone(resolve_reference_wav("clones/jane.mp3", root))

    def test_rejects_absolute(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wav = root / "a.wav"
            wav.write_bytes(b"RIFF")
            self.assertIsNone(resolve_reference_wav(str(wav), root))


class ValidateCloneTests(unittest.TestCase):
    def test_clone_requires_ref_audio(self):
        with self.assertRaises(ValueError) as ctx:
            validate_voices_document(
                {"voices": {"jane": {"kind": "voice_clone", "ref_text": "Hello."}}}
            )
        self.assertIn("clone requires ref_audio", str(ctx.exception))

    def test_clone_ref_text_optional(self):
        out = validate_voices_document(
            {"voices": {"jane": {"kind": "voice_clone", "ref_audio": "clones/jane.wav"}}}
        )
        self.assertEqual(out["voices"]["jane"]["kind"], "voice_clone")
        self.assertNotIn("ref_text", out["voices"]["jane"])


class DiscoverPacksTests(unittest.TestCase):
    def test_nested_pack_and_voice_overlay(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pack = root / "GOOD-en-alice-turbo-v2-e50"
            pack.mkdir()
            (pack / "pack.json").write_text(
                json.dumps(
                    {
                        "name": "alice",
                        "model": "turbo-lora",
                        "reference": "reference.wav",
                        "engine": "chatterbox-turbo",
                        "generation": {"max_gen_len": 900, "norm_loudness": False},
                    }
                ),
                encoding="utf-8",
            )
            (pack / "reference.wav").write_bytes(b"RIFF....WAVE")
            found = discover_packs(root)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0][0], "alice")
            overlays = pack_voice_overlays(found)
            self.assertEqual(overlays[0].alias, "alice")
            self.assertEqual(overlays[0].model, "alice")
            self.assertEqual(overlays[0].kind, "voice_clone")
            self.assertEqual(overlays[0].generation, {"max_gen_len": 900, "norm_loudness": False})
            wav = resolve_reference_wav(overlays[0].ref_audio, root)
            self.assertEqual(wav, (pack / "reference.wav").resolve())
            jane = VoiceOverlay("jane", "", None, "", "voice_clone", "clones/jane.wav", "")
            self.assertEqual(resolve_pack_for_voice("alice", overlays, found)[0], "alice")
            self.assertIsNone(resolve_pack_for_voice("jane", [jane], found))
            self.assertEqual(engine_cache_key(None), PRETRAINED_ENGINE_KEY)

    def test_missing_root(self):
        self.assertEqual(discover_packs(Path("/no/such/models-dir")), [])


class EngineCacheHelperTests(unittest.TestCase):
    def test_lru_evicts_oldest_other(self):
        order = lru_touch([], "a")
        order = lru_touch(order, "b")
        order = lru_touch(order, "a")
        self.assertEqual(order, ["b", "a"])
        self.assertEqual(lru_victim(order, "a"), "b")
        self.assertIsNone(lru_victim(["a"], "a"))

    def test_memory_full_error(self):
        self.assertTrue(is_memory_full_error(RuntimeError("CUDA out of memory")))
        self.assertFalse(is_memory_full_error(RuntimeError("missing weights")))


class GenerationConfigTests(unittest.TestCase):
    def test_parse_preserves_zero_and_false(self):
        parsed = parse_generation_config(
            {
                "temperature": 0.0,
                "norm_loudness": False,
                "top_k": 1,
                "max_gen_len": 1,
                "exaggeration": 0.5,
                "cfg_weight": 0.3,
            }
        )
        self.assertEqual(parsed["temperature"], 0.0)
        self.assertIs(parsed["norm_loudness"], False)
        self.assertEqual(parsed["top_k"], 1)
        self.assertEqual(parsed["exaggeration"], 0.5)
        self.assertEqual(parsed["cfg_weight"], 0.3)

    def test_missing_is_none(self):
        self.assertIsNone(parse_generation_config(None))

    def test_merge_precedence(self):
        merged = merge_generation(
            {"temperature": 0.8, "max_gen_len": 1200},
            {"temperature": 0.6},
        )
        self.assertEqual(merged, {"temperature": 0.6, "max_gen_len": 1200})

    def test_rejects_unknown_type_nonfinite_and_zero_ints(self):
        with self.assertRaises(ValueError) as ctx:
            parse_generation_config({"nope": 1}, prefix="generation")
        self.assertIn("generation.nope", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            parse_generation_config({"temperature": "hot"}, prefix="jane: generation")
        self.assertIn("jane: generation.temperature", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            parse_generation_config({"top_k": 0})
        self.assertIn("top_k", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            parse_generation_config({"max_gen_len": 0})
        self.assertIn("max_gen_len", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            parse_generation_config({"top_p": float("nan")})
        self.assertIn("finite", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            parse_generation_config({"norm_loudness": 1})
        self.assertIn("boolean", str(ctx.exception))

    def test_validate_global_and_clone(self):
        out = validate_voices_document(
            {
                "generation": {"temperature": 0.0, "norm_loudness": False},
                "voices": {
                    "jane": {
                        "kind": "voice_clone",
                        "ref_audio": "clones/jane.wav",
                        "generation": {"temperature": 0.6},
                    }
                },
            }
        )
        self.assertEqual(out["generation"]["temperature"], 0.0)
        self.assertIs(out["generation"]["norm_loudness"], False)
        self.assertEqual(out["voices"]["jane"]["generation"], {"temperature": 0.6})

    def test_parse_voice_overlay_generation(self):
        overlays = parse_voice_overlays(
            {
                "voices": {
                    "jane": {
                        "kind": "voice_clone",
                        "ref_audio": "clones/jane.wav",
                        "generation": {"temperature": 0.7},
                    }
                }
            },
            "",
        )
        self.assertEqual(overlay_generation("jane", overlays), {"temperature": 0.7})



if __name__ == "__main__":
    unittest.main()
