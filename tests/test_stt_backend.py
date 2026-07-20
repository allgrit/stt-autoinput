import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

import stt_backend


class Result:
    text = "Проверка распознавания."


class FakeGigaAM:
    def __init__(self, error=None):
        self.error = error
        self.path = None
        self.wav = None
        self.frames = None

    def transcribe(self, path):
        self.path = path
        with wave.open(path, "rb") as wav:
            self.wav = (wav.getnchannels(), wav.getframerate(), wav.getsampwidth())
            self.frames = wav.readframes(wav.getnframes())
        if self.error:
            raise self.error
        return Result()


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        segment = type("Segment", (), {"text": " резерв "})()
        return [segment], object()


class SttBackendTests(unittest.TestCase):
    def test_gigaam_writes_pcm16_16khz_and_removes_temp_file(self):
        model = FakeGigaAM()
        backend = stt_backend.GigaAMBackend({}, model=model)

        text = backend.transcribe(
            np.array([-2.0, 0.0, 2.0], dtype=np.float32), "final"
        )

        self.assertEqual(text, "Проверка распознавания.")
        self.assertEqual(model.wav, (1, 16000, 2))
        samples = np.frombuffer(model.frames, dtype="<i2")
        self.assertEqual(samples.tolist(), [-32767, 0, 32767])
        self.assertFalse(os.path.exists(model.path))

    def test_gigaam_removes_temp_file_after_error(self):
        model = FakeGigaAM(RuntimeError("boom"))
        backend = stt_backend.GigaAMBackend({}, model=model)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            backend.transcribe(np.zeros(1600, dtype=np.float32), "final")

        self.assertFalse(os.path.exists(model.path))

    def test_gigaam_retries_model_load_on_cpu_after_cuda_error(self):
        marker = object()
        load_model = Mock(side_effect=[RuntimeError("cuda failed"), marker])
        fake_gigaam = SimpleNamespace(load_model=load_model)
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True)
        )

        with patch.dict(sys.modules, {"gigaam": fake_gigaam, "torch": fake_torch}):
            backend = stt_backend.GigaAMBackend(
                {"gigaam_model": "v3_e2e_rnnt", "compute_device": "auto"}
            )

        self.assertIs(backend.model, marker)
        self.assertEqual(backend.device, "cpu")
        self.assertEqual(load_model.call_args_list[0].kwargs["device"], "cuda")
        self.assertEqual(load_model.call_args_list[1].kwargs["device"], "cpu")

    def test_whisper_maps_interim_and_final_settings(self):
        model = FakeWhisper()
        cfg = {
            "language": "ru",
            "beam_interim": 1,
            "beam_final": 5,
            "vad_filter": False,
            "initial_prompt": "Подсказка",
        }
        backend = stt_backend.WhisperBackend(cfg, model=model)

        self.assertEqual(
            backend.transcribe(np.zeros(10), "interim"), "резерв"
        )
        backend.transcribe(np.zeros(10), "final", force_vad=True)

        self.assertEqual(model.calls[0]["beam_size"], 1)
        self.assertEqual(model.calls[1]["beam_size"], 5)
        self.assertFalse(model.calls[0]["vad_filter"])
        self.assertTrue(model.calls[1]["vad_filter"])
        self.assertEqual(model.calls[1]["initial_prompt"], "Подсказка")

    def test_factory_creates_only_selected_engine(self):
        marker = object()
        with patch(
            "stt_backend.GigaAMBackend", return_value=marker
        ) as gigaam_cls, patch("stt_backend.WhisperBackend") as whisper_cls:
            result = stt_backend.create_stt_backend({"stt_engine": "gigaam"})

        self.assertIs(result, marker)
        gigaam_cls.assert_called_once()
        whisper_cls.assert_not_called()

    def test_factory_explains_missing_dependency(self):
        with patch(
            "stt_backend.GigaAMBackend", side_effect=ImportError("missing")
        ):
            with self.assertRaisesRegex(
                RuntimeError, "pip install -r requirements.txt"
            ):
                stt_backend.create_stt_backend({"stt_engine": "gigaam"})

    def test_factory_rejects_unknown_engine(self):
        with self.assertRaisesRegex(ValueError, "Unknown STT engine"):
            stt_backend.create_stt_backend({"stt_engine": "unknown"})


if __name__ == "__main__":
    unittest.main()
