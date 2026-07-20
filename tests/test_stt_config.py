from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config


ROOT = Path(__file__).resolve().parents[1]


class SttConfigTests(unittest.TestCase):
    def test_gigaam_is_default_and_each_engine_has_models(self):
        self.assertEqual(config.DEFAULTS["stt_engine"], "gigaam")
        self.assertEqual(config.DEFAULTS["gigaam_model"], "v3_e2e_rnnt")
        self.assertEqual(config.models_for_engine("gigaam"), ["v3_e2e_rnnt"])
        self.assertIn("medium", config.models_for_engine("whisper"))

    def test_unknown_engine_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown STT engine"):
            config.models_for_engine("unknown")

    def test_old_config_without_engine_inherits_gigaam_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"model_size": "medium"}', encoding="utf-8")
            with patch("config._config_path", return_value=str(path)):
                loaded = config.load_config()
        self.assertEqual(loaded["stt_engine"], "gigaam")
        self.assertEqual(loaded["model_size"], "medium")

    def test_setup_dialog_contains_engine_selector_and_saves_selection(self):
        source = (ROOT / "setup_dialog.py").read_text(encoding="utf-8")
        self.assertIn('text="Recognition engine:"', source)
        self.assertIn('result["stt_engine"] = engine_var.get()', source)
        self.assertIn("models_for_engine", source)

    def test_settings_launcher_reopens_and_saves_config(self):
        configure_path = ROOT / "configure.py"
        launcher_path = ROOT / "settings.bat"
        self.assertTrue(configure_path.is_file(), "configure.py must exist")
        self.assertTrue(launcher_path.is_file(), "settings.bat must exist")
        configure = configure_path.read_text(encoding="utf-8")
        launcher = launcher_path.read_text(encoding="utf-8")
        self.assertIn("run_setup(load_config())", configure)
        self.assertIn("save_config(cfg)", configure)
        self.assertIn("configure.py", launcher)


if __name__ == "__main__":
    unittest.main()
