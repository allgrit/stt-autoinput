"""Защита аудиопотока от «тихой смерти».

PortAudio-поток, открытый один раз при старте, может перестать доставлять
callback'и (переподключение USB-микрофона, сброс драйвера) без единого
исключения. Снаружи это выглядит как запись с chunks=0. Здесь чистая логика,
без sounddevice, чтобы её можно было тестировать.
"""
from __future__ import annotations

from typing import Callable, Optional


def stream_stalled(cb_count_before: int, cb_count_now: int) -> bool:
    """Поток жив, только если счётчик callback'ов вырос."""
    return cb_count_now <= cb_count_before


def recover_stream(
    stream,
    cb_count_before: int,
    cb_count_now: int,
    reopen: Callable[[], Optional[object]],
    log: Callable[[str], None] = print,
):
    """Вернуть живой поток: старый, если он доставлял данные, иначе переоткрытый.

    ``reopen`` возвращает новый поток либо None, если открыть не удалось.
    Старый поток закрывается перед переоткрытием, ошибки закрытия игнорируются:
    у мёртвого потока stop()/close() могут бросать.
    """
    if not stream_stalled(cb_count_before, cb_count_now):
        return stream
    log("[warn] audio stream stalled (no callbacks) — reopening")
    for closer in ("stop", "close"):
        try:
            getattr(stream, closer)()
        except Exception:
            pass
    new_stream = reopen()
    if new_stream is None:
        log("[error] audio stream reopen failed")
    else:
        log("[ok] audio stream reopened")
    return new_stream
