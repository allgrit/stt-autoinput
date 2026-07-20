"""Единый интерфейс для GigaAM и faster-whisper."""

from __future__ import annotations

import os
import tempfile
import wave

import numpy as np


SAMPLE_RATE = 16000


def _write_pcm16_wav(path, audio):
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


class GigaAMBackend:
    engine = "gigaam"

    def __init__(self, cfg, model=None):
        self.model_name = cfg.get("gigaam_model", "v3_e2e_rnnt")
        self.device = "injected"
        self.model = model if model is not None else self._load(cfg)

    def _load(self, cfg):
        import gigaam
        import torch

        requested = cfg.get("compute_device", "auto")
        if requested == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested
        print(f"[init] GigaAM '{self.model_name}' on {self.device}...")
        try:
            return gigaam.load_model(
                self.model_name,
                device=self.device,
                fp16_encoder=self.device != "cpu",
            )
        except Exception as error:
            if self.device == "cpu":
                raise
            print(f"[init] GigaAM GPU failed ({error}), falling back to CPU...")
            self.device = "cpu"
            return gigaam.load_model(
                self.model_name, device="cpu", fp16_encoder=False
            )

    def transcribe(self, audio, quality="final", force_vad=False):
        del quality, force_vad
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = handle.name
        handle.close()
        try:
            _write_pcm16_wav(path, audio)
            result = self.model.transcribe(path)
            if isinstance(result, str):
                text = result
            else:
                text = (
                    getattr(result, "text", None)
                    or getattr(result, "transcription", "")
                )
            return (text or "").strip()
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class WhisperBackend:
    engine = "whisper"

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.device = "injected"
        self.model = model if model is not None else self._load(cfg)

    def _load(self, cfg):
        from faster_whisper import WhisperModel

        requested = cfg.get("compute_device", "auto")
        self.device = "cuda" if requested == "auto" else requested
        compute_type = cfg.get("compute_type", "float16")
        if self.device == "cpu":
            compute_type = "float32"
        print(
            f"[init] Whisper '{cfg['model_size']}' on "
            f"{self.device} ({compute_type})..."
        )
        try:
            return WhisperModel(
                cfg["model_size"],
                device=self.device,
                compute_type=compute_type,
            )
        except Exception as error:
            if self.device == "cpu":
                raise
            print(f"[init] Whisper GPU failed ({error}), falling back to CPU...")
            self.device = "cpu"
            return WhisperModel(
                cfg["model_size"], device="cpu", compute_type="float32"
            )

    def transcribe(self, audio, quality="final", force_vad=False):
        if quality not in {"interim", "final"}:
            raise ValueError(f"Unknown transcription quality: {quality}")
        beam = (
            self.cfg["beam_final"]
            if quality == "final"
            else self.cfg["beam_interim"]
        )
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg["language"],
            beam_size=beam,
            vad_filter=force_vad or self.cfg["vad_filter"],
            initial_prompt=self.cfg.get("initial_prompt", ""),
        )
        return " ".join(segment.text for segment in segments).strip()


def create_stt_backend(cfg):
    engine = cfg.get("stt_engine", "gigaam")
    try:
        if engine == "gigaam":
            return GigaAMBackend(cfg)
        if engine == "whisper":
            return WhisperBackend(cfg)
    except ImportError as error:
        raise RuntimeError(
            f"Движок {engine} недоступен: установите зависимости через "
            "'python -m pip install -r requirements.txt' или выберите другой "
            "Recognition engine в настройках."
        ) from error
    raise ValueError(f"Unknown STT engine: {engine}")
