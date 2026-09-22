"""Защита аудиопотока от «тихой смерти» и удержание виджета в экране.

PortAudio-поток, открытый один раз при старте, может перестать доставлять
callback'и (переподключение USB-микрофона, сброс драйвера) без единого
исключения. Снаружи это выглядит как запись с chunks=0. Здесь чистая логика,
без sounddevice и tkinter, чтобы её можно было тестировать.
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


def clamp_to_screen(x: int, y: int, w: int, h: int, screen_w: int, screen_h: int,
                    margin: int = 0) -> tuple[int, int]:
    """Координаты окна, при которых оно целиком остаётся в пределах экрана."""
    max_x = max(0, screen_w - w - margin)
    max_y = max(0, screen_h - h - margin)
    return min(max(x, 0), max_x), min(max(y, 0), max_y)
