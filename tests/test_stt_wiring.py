from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SttWiringTests(unittest.TestCase):
    def test_realtime_entrypoint_uses_backend_facade(self):
        source = (ROOT / "dictate_realtime.py").read_text(encoding="utf-8")
        self.assertIn("from stt_backend import create_stt_backend", source)
        self.assertIn("stt_backend = create_stt_backend(cfg)", source)
        self.assertNotIn("from faster_whisper import WhisperModel", source)
        self.assertNotIn("model.transcribe(", source)
        self.assertNotIn("detect_compute", source)
        self.assertGreaterEqual(source.count("stt_backend.transcribe("), 5)

    def test_ready_message_reports_selected_backend_device(self):
        source = (ROOT / "dictate_realtime.py").read_text(encoding="utf-8")
        self.assertIn("stt_backend.engine", source)
        self.assertIn("stt_backend.device", source)


if __name__ == "__main__":
    unittest.main()
