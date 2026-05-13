from __future__ import annotations

import numpy as np


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float,
    window_sec: float = 0.1,
    padding_sec: float = 0.15,
) -> np.ndarray:
    if audio.size == 0:
        return audio

    window = max(1, int(sample_rate * window_sec))
    padding = max(0, int(sample_rate * padding_sec))

    active_windows = []
    for start in range(0, len(audio), window):
        chunk = audio[start:start + window]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms >= threshold:
            active_windows.append((start, min(start + window, len(audio))))

    if not active_windows:
        return audio[:0]

    first = max(0, active_windows[0][0] - padding)
    last = min(len(audio), active_windows[-1][1] + padding)
    return audio[first:last]
