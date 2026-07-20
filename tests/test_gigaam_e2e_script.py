from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]


class GigaAME2EScriptTests(unittest.TestCase):
    def test_requirements_and_e2e_script_use_selected_model(self):
        script_path = ROOT / "scripts" / "e2e_gigaam.py"
        self.assertTrue(script_path.is_file(), "scripts/e2e_gigaam.py must exist")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("gigaam[torch] @", requirements)
        self.assertIn('"gigaam_model": "v3_e2e_rnnt"', script)
        self.assertIn("GigaAMBackend", script)

    def test_load_window_returns_mono_float32_at_16khz(self):
        script_path = ROOT / "scripts" / "e2e_gigaam.py"
        self.assertTrue(script_path.is_file(), "scripts/e2e_gigaam.py must exist")
        from scripts.e2e_gigaam import load_window

        source = np.column_stack(
            [np.full(8000, 12000, dtype=np.int16), np.zeros(8000, dtype=np.int16)]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo-8khz.wav"
            wavfile.write(path, 8000, source)
            audio = load_window(path)

        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        self.assertLessEqual(abs(len(audio) - 16000), 1)
        self.assertGreater(float(np.mean(np.abs(audio))), 0.1)


if __name__ == "__main__":
    unittest.main()
