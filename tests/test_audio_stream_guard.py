import unittest
from unittest.mock import Mock

from audio_stream_guard import recover_stream, stream_stalled


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


if __name__ == "__main__":
    unittest.main()
