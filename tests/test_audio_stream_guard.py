import unittest
from unittest.mock import Mock

from audio_stream_guard import clamp_to_screen, recover_stream, stream_stalled


class StreamStalledTests(unittest.TestCase):
    def test_alive_when_callbacks_arrived(self):
        self.assertFalse(stream_stalled(10, 13))

    def test_stalled_when_counter_did_not_move(self):
        self.assertTrue(stream_stalled(10, 10))


class RecoverStreamTests(unittest.TestCase):
    def test_keeps_stream_that_delivers_callbacks(self):
        stream = Mock()
        reopen = Mock()

        result = recover_stream(stream, 5, 9, reopen, log=lambda _: None)

        self.assertIs(result, stream)
        reopen.assert_not_called()
        stream.close.assert_not_called()

    def test_reopens_stalled_stream_and_closes_old_one(self):
        stream = Mock()
        fresh = object()
        reopen = Mock(return_value=fresh)

        result = recover_stream(stream, 5, 5, reopen, log=lambda _: None)

        self.assertIs(result, fresh)
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_survives_dead_stream_raising_on_close(self):
        stream = Mock()
        stream.stop.side_effect = RuntimeError("PortAudio error")
        stream.close.side_effect = RuntimeError("PortAudio error")
        fresh = object()

        result = recover_stream(stream, 5, 5, lambda: fresh, log=lambda _: None)

        self.assertIs(result, fresh)

    def test_returns_none_and_logs_when_reopen_fails(self):
        logs = []

        result = recover_stream(Mock(), 5, 5, lambda: None, log=logs.append)

        self.assertIsNone(result)
        self.assertTrue(any("reopen failed" in m for m in logs))


class ClampToScreenTests(unittest.TestCase):
    def test_window_dragged_below_taskbar_is_pulled_back(self):
        # Реальный случай: виджет 320x38 утащили на Y=1396 при экране 3440x1440,
        # и его накрыла панель задач.
        x, y = clamp_to_screen(221, 1396, 320, 38, 3440, 1440, margin=60)

        self.assertEqual(x, 221)
        self.assertEqual(y, 1440 - 38 - 60)

    def test_window_inside_screen_is_untouched(self):
        self.assertEqual(clamp_to_screen(100, 18, 320, 38, 3440, 1440), (100, 18))

    def test_negative_coordinates_are_clamped_to_zero(self):
        self.assertEqual(clamp_to_screen(-50, -10, 320, 38, 3440, 1440), (0, 0))


if __name__ == "__main__":
    unittest.main()
