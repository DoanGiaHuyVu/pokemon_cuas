"""Step 1 checks use temporary artifacts and never load models or send input."""

import contextlib
import builtins
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.cli import main, readiness_report
from harness.settings import ConfigurationError, load_settings


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "game.nds").touch()
        for name in (
            "source/pyproject.toml", "source/laya/__init__.py",
            "checkpoint/model.safetensors", "checkpoint/rl_agent_config.json",
            "checkpoint/questions.json", "checkpoint/encoder/config.json",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (self.root / "checkpoint/tokenizer").mkdir()
        self.env_file = self.root / ".env"
        self.base = {
            "GEMINI_API_KEY": "test-secret-never-print",
            "ROM_PATH": "game.nds",
            "LAYA_MODEL_PATH": "checkpoint",
            "LAYA_SOURCE_PATH": "source",
        }

    def load(self, **overrides):
        return load_settings(
            self.env_file, environ={**self.base, **overrides}, project_root=self.root
        )

    def invoke(self, arguments):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = main(arguments)
        return status, output.getvalue(), error.getvalue()

    def test_valid_settings_anchor_relative_paths_and_hide_key(self):
        settings = self.load()
        self.assertEqual(settings.rom_path, self.root / "game.nds")
        self.assertFalse(settings.geometry_configured)
        self.assertEqual(settings.mode.value, "preflight")
        self.assertNotIn(self.base["GEMINI_API_KEY"], repr(settings))
        self.assertNotIn(self.base["GEMINI_API_KEY"], settings.model_dump_json())

    def test_missing_or_empty_key_fails_without_echoing_values(self):
        for key in (None, "", "   ", "your_google_ai_studio_key_here"):
            values = dict(self.base)
            if key is None:
                values.pop("GEMINI_API_KEY")
            else:
                values["GEMINI_API_KEY"] = key
            with self.subTest(key=key), self.assertRaises(ConfigurationError) as caught:
                load_settings(self.env_file, environ=values, project_root=self.root)
            self.assertIn("GEMINI_API_KEY", str(caught.exception))
            self.assertNotIn("your_google_ai_studio_key_here", str(caught.exception))

    def test_file_settings_do_not_interpolate_or_modify_environment(self):
        self.env_file.write_text(
            'GEMINI_API_KEY="literal-${OTHER_SECRET}"\n'
            'ROM_PATH=game.nds\nLAYA_MODEL_PATH=checkpoint\nLAYA_SOURCE_PATH=source\n'
        )
        before = dict(os.environ)
        settings = load_settings(self.env_file, environ={}, project_root=self.root)
        self.assertEqual(settings.gemini_api_key.get_secret_value(), "literal-${OTHER_SECRET}")
        self.assertEqual(dict(os.environ), before)
        overridden = load_settings(
            self.env_file, environ={"GEMINI_API_KEY": "override-key"}, project_root=self.root
        )
        self.assertEqual(overridden.gemini_api_key.get_secret_value(), "override-key")

    def test_invalid_geometry_reports_field_without_leaking_secret(self):
        with self.assertRaises(ConfigurationError) as caught:
            self.load(EMU_WIDTH=self.base["GEMINI_API_KEY"])
        self.assertIn("EMU_WIDTH", str(caught.exception))
        self.assertNotIn(self.base["GEMINI_API_KEY"], str(caught.exception))
        with self.assertRaises(ConfigurationError):
            self.load(EMU_WIDTH="0", EMU_HEIGHT="100", EMU_LEFT="0", EMU_TOP="0")

    def test_partial_geometry_fails_and_full_geometry_is_not_calibrated(self):
        with self.assertRaises(ConfigurationError):
            self.load(EMU_LEFT="0")
        settings = self.load(EMU_LEFT="-100", EMU_TOP="0", EMU_WIDTH="512", EMU_HEIGHT="768")
        self.assertTrue(settings.geometry_configured)
        self.assertIn("not used by window capture", readiness_report(settings, None))

    def test_missing_checkpoint_file_and_wrong_rom_fail(self):
        (self.root / "checkpoint/questions.json").unlink()
        with self.assertRaises(ConfigurationError) as caught:
            self.load()
        self.assertIn("questions.json", str(caught.exception))
        (self.root / "checkpoint/questions.json").touch()
        with self.assertRaises(ConfigurationError):
            self.load(ROM_PATH="missing.nds")
        with self.assertRaises(ConfigurationError):
            self.load(ROM_PATH="source/pyproject.toml")

    def test_bare_and_empty_path_settings_fail_clearly(self):
        for path in (None, ""):
            with self.subTest(path=path), self.assertRaises(ConfigurationError) as caught:
                self.load(LAYA_MODEL_PATH=path)
            self.assertIn("LAYA_MODEL_PATH", str(caught.exception))

    def test_request_is_one_preflight_and_redacts_secret(self):
        settings = self.load()
        with patch("harness.cli.load_settings", return_value=settings), patch(
            "harness.cli.dependency_availability", return_value={"laya": False}
        ), patch("harness.cli.find_spec", side_effect=AssertionError("unexpected import lookup")):
            status, output, error = self.invoke(
                ["--prompt", f"Please start the game. {self.base['GEMINI_API_KEY']}"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(error, "")
        self.assertIn("one request; execution disabled", output)
        self.assertIn("Request accepted for preflight only", output)
        self.assertIn("laya: not installed", output)
        self.assertNotIn(self.base["GEMINI_API_KEY"], output)

    def test_check_does_not_request_interactive_input(self):
        with patch("harness.cli.load_settings", return_value=self.load()), patch(
            "builtins.input", side_effect=AssertionError("unexpected prompt")
        ):
            status, output, _ = self.invoke(["--check"])
        self.assertEqual(status, 0)
        self.assertNotIn("Your request:", output)

    def test_preflight_never_imports_model_or_desktop_runtimes(self):
        original_import = builtins.__import__
        forbidden = {"pyautogui", "torch", "transformers", "accelerate", "laya"}

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in forbidden or name == "google.genai":
                raise AssertionError(f"Unexpected runtime import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("harness.cli.load_settings", return_value=self.load()), patch(
            "builtins.__import__", side_effect=guarded_import
        ):
            status, _, error = self.invoke(["--prompt", "Press Start"])
        self.assertEqual(status, 0)
        self.assertEqual(error, "")

    def test_noninteractive_start_requires_explicit_request_or_check(self):
        with patch("harness.cli.load_settings", return_value=self.load()), patch(
            "sys.stdin.isatty", return_value=False
        ):
            status, output, error = self.invoke([])
        self.assertEqual(status, 2)
        self.assertEqual(output, "")
        self.assertIn("Supply --prompt or --check", error)

    def test_blank_and_excessive_prompts_fail(self):
        with patch("harness.cli.load_settings", return_value=self.load()):
            for prompt in ("  ", "x" * 4001):
                with self.subTest(length=len(prompt)):
                    status, output, error = self.invoke(["--prompt", prompt])
                    self.assertEqual(status, 2)
                    self.assertEqual(output, "")
                    self.assertIn("between 1 and 4000", error)

    def test_interactive_request_and_cancellation(self):
        with patch("harness.cli.load_settings", return_value=self.load()), patch(
            "sys.stdin.isatty", return_value=True
        ), patch("builtins.input", return_value="Inspect the screen"):
            status, output, _ = self.invoke([])
        self.assertEqual(status, 0)
        self.assertIn("Your request: Inspect the screen", output)
        with patch("harness.cli.load_settings", return_value=self.load()), patch(
            "sys.stdin.isatty", return_value=True
        ), patch("builtins.input", side_effect=KeyboardInterrupt):
            status, output, _ = self.invoke([])
        self.assertEqual(status, 130)
        self.assertIn("no actions executed", output)

    def test_configuration_error_has_no_traceback(self):
        with patch("harness.cli.load_settings", side_effect=ConfigurationError("Missing GEMINI_API_KEY")):
            status, output, error = self.invoke(["--check"])
        self.assertEqual(status, 2)
        self.assertEqual(output, "")
        self.assertIn("Missing GEMINI_API_KEY", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
