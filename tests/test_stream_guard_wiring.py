from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StreamGuardWiringTests(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "dictate_realtime.py").read_text(encoding="utf-8")

    def test_toggle_on_checks_stream_health_and_reopens(self):
        self.assertIn("from audio_stream_guard import", self.src)
        self.assertIn("recover_stream(", self.src)
        self.assertIn("def _ensure_stream_alive", self.src)

    def test_widget_drag_is_not_clamped_to_screen(self):
        # Пользователь ставит виджет на панель задач — ограничивать нельзя.
        self.assertNotIn("clamp_to_screen", self.src)


if __name__ == "__main__":
    unittest.main()


class StartBatRestartTests(unittest.TestCase):
    def test_start_bat_kills_old_instance_before_writing_log(self):
        src = (ROOT / "start.bat").read_text(encoding="utf-8")
        kill_pos = src.find("taskkill")
        log_pos = src.find("> \"%~dp0stl_log.txt\"")
        self.assertGreater(kill_pos, -1, "start.bat must taskkill the old PID")
        self.assertGreater(log_pos, kill_pos, "kill must happen before log redirect")
