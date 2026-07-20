from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FastFinalModeTests(unittest.TestCase):
    def test_realtime_preview_is_enabled_by_default(self):
        config_source = (ROOT / "config.py").read_text(encoding="utf-8")

        self.assertIn('"realtime_preview": True', config_source)

    def test_transcription_worker_skips_interim_transcribe_when_preview_disabled(self):
        app_source = (ROOT / "dictate_realtime.py").read_text(encoding="utf-8")

        self.assertIn("REALTIME_PREVIEW = cfg", app_source)
        self.assertIn("if not REALTIME_PREVIEW:", app_source)


if __name__ == "__main__":
    unittest.main()
