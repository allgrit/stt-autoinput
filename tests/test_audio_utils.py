import unittest

import numpy as np

from audio_utils import trim_silence


class AudioUtilsTests(unittest.TestCase):
    def test_trim_silence_removes_quiet_edges_and_keeps_padding(self):
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        speech = np.full(sr * 2, 0.02, dtype=np.float32)
        audio = np.concatenate([silence, speech, silence])

        trimmed = trim_silence(
            audio,
            sample_rate=sr,
            threshold=0.005,
            window_sec=0.1,
            padding_sec=0.05,
        )

        self.assertGreaterEqual(len(trimmed), int(sr * 2.0))
        self.assertLessEqual(len(trimmed), int(sr * 2.2))

    def test_trim_silence_returns_empty_for_all_silence(self):
        audio = np.zeros(16000, dtype=np.float32)

        trimmed = trim_silence(audio, sample_rate=16000, threshold=0.005)

        self.assertEqual(len(trimmed), 0)


if __name__ == "__main__":
    unittest.main()
