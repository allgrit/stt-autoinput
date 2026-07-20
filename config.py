from __future__ import annotations
import json
import os
import sys

DEFAULTS = {
    "stt_engine": "gigaam",
    "gigaam_model": "v3_e2e_rnnt",
    "model_size": "small",
    "language": "ru",
    "toggle_key": "f9",
    "device_index": None,
    "device_name": None,
    "compute_device": "auto",
    "compute_type": "float16",
    "realtime_preview": True,
    "vad_filter": False,
    "trim_silence": True,
    "trim_silence_rms": 0.003,
    "stream_interval": 0.8,
    "beam_interim": 1,
    "beam_final": 1,
    "silence_rms": 0.003,
    "commit_pause": 1.5,
    "max_buffer_s": 7.0,
    "hard_max_buffer_s": 14.0,
    "vad_min_silence_ms": 350,
    "vad_threshold": 0.5,
    "beep_on": [1000, 100],
    "beep_off": [600, 100],
    "beep_cmd": [800, 50],
    "initial_prompt": "Здравствуйте. Запятые, точки и заглавные буквы расставлены правильно.",
}

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
STT_ENGINES = ["gigaam", "whisper"]
GIGAAM_MODELS = ["v3_e2e_rnnt"]


def models_for_engine(engine):
    if engine == "gigaam":
        return list(GIGAAM_MODELS)
    if engine == "whisper":
        return list(MODEL_SIZES)
    raise ValueError(f"Unknown STT engine: {engine}")


def _config_path():
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "config.json")


def load_config():
    path = _config_path()
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                user = json.load(f)
            cfg.update(user)
        except Exception:
            pass
    return cfg


def save_config(cfg):
    path = _config_path()
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


def needs_setup(cfg):
    return cfg.get("device_index") is None


def detect_compute():
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel("tiny", device="cuda", compute_type="float16")
        del m
        return "cuda", "float16"
    except Exception:
        return "cpu", "float32"
