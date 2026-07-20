from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stt_backend import GigaAMBackend  # noqa: E402


SAMPLE_RATE = 16000
WINDOW_SECONDS = 20


def load_window(path):
    sample_rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        divisor = gcd(SAMPLE_RATE, sample_rate)
        audio = resample_poly(
            audio, SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype(np.float32)

    window = WINDOW_SECONDS * SAMPLE_RATE
    if len(audio) <= window:
        return audio

    starts = list(range(0, len(audio) - window + 1, window // 2))
    last_start = len(audio) - window
    if starts[-1] != last_start:
        starts.append(last_start)
    start = max(
        starts,
        key=lambda offset: float(np.mean(audio[offset:offset + window] ** 2)),
    )
    return audio[start:start + window]


def main():
    parser = argparse.ArgumentParser(
        description="Сквозная проверка GigaAM-v3 на коротком окне WAV."
    )
    parser.add_argument("wav", type=Path)
    args = parser.parse_args()

    audio = load_window(args.wav)
    cfg = {
        "gigaam_model": "v3_e2e_rnnt",
        "compute_device": "auto",
    }
    started = time.perf_counter()
    backend = GigaAMBackend(cfg)
    text = backend.transcribe(audio, quality="final")
    elapsed = time.perf_counter() - started
    if not text:
        raise SystemExit("GigaAM вернула пустой результат")
    print(
        f"model={backend.model_name} device={backend.device} seconds={elapsed:.1f}"
    )
    print(text)


if __name__ == "__main__":
    main()
