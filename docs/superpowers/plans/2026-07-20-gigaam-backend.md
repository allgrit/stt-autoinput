# Переключаемое распознавание GigaAM / Whisper — план реализации

> **Для агентных исполнителей:** ОБЯЗАТЕЛЬНЫЙ НАВЫК: использовать `superpowers:subagent-driven-development` (рекомендуется) или `superpowers:executing-plans` и выполнять план по задачам. Шаги отмечены чекбоксами `- [ ]`.

**Цель:** сделать GigaAM-v3 `v3_e2e_rnnt` основным движком диктовки, сохранив Whisper как вручную выбираемый резерв и подтвердив работу реальной модели сквозным тестом.

**Архитектура:** отдельный `stt_backend.py` предоставляет единый строковый результат для GigaAM и Whisper. `dictate_realtime.py` продолжает управлять аудиобуфером, VAD и вставкой текста, но больше не знает API конкретной модели. GigaAM получает короткий PCM16 WAV во временном файле; Whisper продолжает получать массив `numpy` напрямую.

**Технологии:** Python 3.11, `numpy`, `faster-whisper`, `gigaam`, `torch`, `torchaudio`, `unittest`/`pytest`, Tkinter.

---

## Структура файлов

- Создать `stt_backend.py`: загрузка движков, преобразование WAV и единый интерфейс распознавания.
- Создать `tests/test_stt_backend.py`: поведенческие тесты обоих адаптеров и очистки временных файлов.
- Создать `tests/test_stt_config.py`: значения по умолчанию, список моделей и сохранение выбора в UI на уровне доступных чистых функций/исходного кода.
- Создать `configure.py` и `settings.bat`: повторное открытие настроек без запуска диктовки.
- Создать `tests/test_stt_wiring.py`: статическая проверка, что рабочий entrypoint использует адаптер, а не загружает Whisper напрямую.
- Изменить `config.py`: настройки движка, модели GigaAM и чистая функция выбора моделей.
- Изменить `setup_dialog.py`: переключатель движка и динамический список моделей.
- Изменить `dictate_realtime.py`: создание выбранного backend и замена прямых вызовов `model.transcribe`.
- Изменить `requirements.txt`: официальная зависимость GigaAM с `torch` extra.
- Создать `scripts/e2e_gigaam.py`: воспроизводимая проверка настоящей модели на коротком WAV.

### Задача 1: Конфигурация и переключатель UI

**Файлы:**

- Изменить: `config.py`
- Изменить: `setup_dialog.py`
- Создать: `configure.py`
- Создать: `settings.bat`
- Создать: `tests/test_stt_config.py`

- [ ] **Шаг 1: написать падающий тест конфигурации**

```python
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config


ROOT = Path(__file__).resolve().parents[1]


class SttConfigTests(unittest.TestCase):
    def test_gigaam_is_default_and_each_engine_has_models(self):
        self.assertEqual(config.DEFAULTS["stt_engine"], "gigaam")
        self.assertEqual(config.DEFAULTS["gigaam_model"], "v3_e2e_rnnt")
        self.assertEqual(config.models_for_engine("gigaam"), ["v3_e2e_rnnt"])
        self.assertIn("medium", config.models_for_engine("whisper"))

    def test_old_config_without_engine_inherits_gigaam_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"model_size": "medium"}', encoding="utf-8")
            with patch("config._config_path", return_value=str(path)):
                loaded = config.load_config()
        self.assertEqual(loaded["stt_engine"], "gigaam")
        self.assertEqual(loaded["model_size"], "medium")

    def test_setup_dialog_contains_engine_selector_and_saves_selection(self):
        source = (ROOT / "setup_dialog.py").read_text(encoding="utf-8")
        self.assertIn('text="Recognition engine:"', source)
        self.assertIn('result["stt_engine"] = engine_var.get()', source)
        self.assertIn("models_for_engine", source)

    def test_settings_launcher_reopens_and_saves_config(self):
        configure = (ROOT / "configure.py").read_text(encoding="utf-8")
        launcher = (ROOT / "settings.bat").read_text(encoding="utf-8")
        self.assertIn("run_setup(load_config())", configure)
        self.assertIn("save_config(cfg)", configure)
        self.assertIn("configure.py", launcher)
```

- [ ] **Шаг 2: запустить тест и подтвердить правильное падение**

Команда: `python -m pytest tests/test_stt_config.py -q`

Ожидается: FAIL из-за отсутствующих `stt_engine`, `gigaam_model` и `models_for_engine`.

- [ ] **Шаг 3: добавить минимальную конфигурацию**

В `config.py` добавить:

```python
DEFAULTS = {
    "stt_engine": "gigaam",
    "gigaam_model": "v3_e2e_rnnt",
    # остальные существующие поля без изменений
}

STT_ENGINES = ["gigaam", "whisper"]
GIGAAM_MODELS = ["v3_e2e_rnnt"]


def models_for_engine(engine):
    if engine == "gigaam":
        return list(GIGAAM_MODELS)
    if engine == "whisper":
        return list(MODEL_SIZES)
    raise ValueError(f"Unknown STT engine: {engine}")
```

В `setup_dialog.py` заменить локальный импорт и блок модели на следующий код:

```python
from config import STT_ENGINES, models_for_engine

ttk.Label(main, text="Recognition engine:", style="Dark.TLabel").pack(anchor="w")
engine_var = tk.StringVar(value=cfg.get("stt_engine", "gigaam"))
engine_combo = ttk.Combobox(
    main, textvariable=engine_var, values=STT_ENGINES,
    state="readonly", width=20, font=("Segoe UI", 9),
)
engine_combo.pack(anchor="w", pady=(2, 10))

ttk.Label(main, text="Model:", style="Dark.TLabel").pack(anchor="w")


def configured_model(engine):
    if engine == "gigaam":
        return cfg.get("gigaam_model", "v3_e2e_rnnt")
    return cfg.get("model_size", "medium")


model_var = tk.StringVar(value=configured_model(engine_var.get()))
model_combo = ttk.Combobox(
    main, textvariable=model_var, values=models_for_engine(engine_var.get()),
    state="readonly", width=20, font=("Segoe UI", 9),
)
model_combo.pack(anchor="w", pady=(2, 10))

vram_hint = {
    "v3_e2e_rnnt": "GigaAM-v3, GPU/CPU",
    "tiny": "~1 GB VRAM", "base": "~1 GB", "small": "~2 GB",
    "medium": "~5 GB", "large-v2": "~10 GB", "large-v3": "~10 GB",
}
model_hint = ttk.Label(
    main, text=vram_hint.get(model_var.get(), ""), style="Dark.TLabel",
)
model_hint.pack(anchor="w", pady=(0, 10))


def on_model_change(event=None):
    model_hint.config(text=vram_hint.get(model_var.get(), ""))


def on_engine_change(event=None):
    values = models_for_engine(engine_var.get())
    model_combo.config(values=values)
    model_var.set(configured_model(engine_var.get()))
    on_model_change()


engine_combo.bind("<<ComboboxSelected>>", on_engine_change)
model_combo.bind("<<ComboboxSelected>>", on_model_change)
```

В `on_save()` после сохранения микрофона добавить:

```python
result["stt_engine"] = engine_var.get()
if engine_var.get() == "gigaam":
    result["gigaam_model"] = model_var.get()
else:
result["model_size"] = model_var.get()
```

Создать `configure.py`:

```python
from config import load_config, save_config
from setup_dialog import run_setup


def main():
    cfg = run_setup(load_config())
    save_config(cfg)


if __name__ == "__main__":
    main()
```

Создать `settings.bat`:

```bat
@echo off
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%~dp0configure.py"
```

- [ ] **Шаг 4: запустить тест конфигурации**

Команда: `python -m pytest tests/test_stt_config.py -q`

Ожидается: `4 passed`.

- [ ] **Шаг 5: зафиксировать изменение**

```powershell
git add -- config.py setup_dialog.py configure.py settings.bat tests/test_stt_config.py
git commit -m "Добавить выбор движка распознавания"
```

### Задача 2: Адаптеры GigaAM и Whisper

**Файлы:**

- Создать: `stt_backend.py`
- Создать: `tests/test_stt_backend.py`

- [ ] **Шаг 1: написать падающие тесты адаптеров**

```python
import os
import wave
import unittest
from unittest.mock import patch

import numpy as np

from stt_backend import GigaAMBackend, WhisperBackend, create_stt_backend


class Result:
    text = "Проверка распознавания."


class FakeGigaAM:
    def __init__(self, error=None):
        self.error = error
        self.path = None
        self.wav = None

    def transcribe(self, path):
        self.path = path
        with wave.open(path, "rb") as wav:
            self.wav = (wav.getnchannels(), wav.getframerate(), wav.getsampwidth())
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
        backend = GigaAMBackend({}, model=model)
        text = backend.transcribe(np.array([-2.0, 0.0, 2.0], dtype=np.float32), "final")
        self.assertEqual(text, "Проверка распознавания.")
        self.assertEqual(model.wav, (1, 16000, 2))
        self.assertFalse(os.path.exists(model.path))

    def test_gigaam_removes_temp_file_after_error(self):
        model = FakeGigaAM(RuntimeError("boom"))
        backend = GigaAMBackend({}, model=model)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            backend.transcribe(np.zeros(1600, dtype=np.float32), "final")
        self.assertFalse(os.path.exists(model.path))

    def test_whisper_maps_interim_and_final_settings(self):
        model = FakeWhisper()
        cfg = {"language": "ru", "beam_interim": 1, "beam_final": 5,
               "vad_filter": False, "initial_prompt": "Подсказка"}
        backend = WhisperBackend(cfg, model=model)
        self.assertEqual(backend.transcribe(np.zeros(10), "interim"), "резерв")
        backend.transcribe(np.zeros(10), "final", force_vad=True)
        self.assertEqual(model.calls[0]["beam_size"], 1)
        self.assertEqual(model.calls[1]["beam_size"], 5)
        self.assertFalse(model.calls[0]["vad_filter"])
        self.assertTrue(model.calls[1]["vad_filter"])

    def test_factory_creates_only_selected_engine(self):
        marker = object()
        with patch("stt_backend.GigaAMBackend", return_value=marker) as gigaam_cls, \
             patch("stt_backend.WhisperBackend") as whisper_cls:
            result = create_stt_backend({"stt_engine": "gigaam"})
        self.assertIs(result, marker)
        gigaam_cls.assert_called_once()
        whisper_cls.assert_not_called()
```

- [ ] **Шаг 2: запустить тест и подтвердить правильное падение**

Команда: `python -m pytest tests/test_stt_backend.py -q`

Ожидается: ошибка импорта `stt_backend`.

- [ ] **Шаг 3: реализовать минимальные адаптеры**

`stt_backend.py` должен:

```python
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
    def __init__(self, cfg, model=None):
        self.model_name = cfg.get("gigaam_model", "v3_e2e_rnnt")
        self.model = model if model is not None else self._load(cfg)

    def _load(self, cfg):
        import gigaam
        import torch
        requested = cfg.get("compute_device", "auto")
        device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        try:
            return gigaam.load_model(self.model_name, device=device, fp16_encoder=device != "cpu")
        except Exception:
            if device == "cpu":
                raise
            return gigaam.load_model(self.model_name, device="cpu", fp16_encoder=False)

    def transcribe(self, audio, quality="final", force_vad=False):
        del quality, force_vad
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = handle.name
        handle.close()
        try:
            _write_pcm16_wav(path, audio)
            result = self.model.transcribe(path)
            text = result if isinstance(result, str) else getattr(result, "text", "")
            return (text or "").strip()
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class WhisperBackend:
    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.model = model if model is not None else self._load(cfg)

    @staticmethod
    def _load(cfg):
        from faster_whisper import WhisperModel
        device = cfg.get("compute_device", "auto")
        if device == "auto":
            device = "cuda"
        compute_type = cfg.get("compute_type", "float16")
        try:
            return WhisperModel(cfg["model_size"], device=device, compute_type=compute_type)
        except Exception:
            if device == "cpu":
                raise
            return WhisperModel(cfg["model_size"], device="cpu", compute_type="float32")

    def transcribe(self, audio, quality="final", force_vad=False):
        beam = self.cfg["beam_final"] if quality == "final" else self.cfg["beam_interim"]
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
    except ImportError as exc:
        raise RuntimeError(
            f"Движок {engine} недоступен: установите зависимости через "
            "'python -m pip install -r requirements.txt' или выберите другой "
            "Recognition engine в настройках."
        ) from exc
    raise ValueError(f"Unknown STT engine: {engine}")
```

- [ ] **Шаг 4: запустить тесты адаптеров**

Команда: `python -m pytest tests/test_stt_backend.py -q`

Ожидается: `4 passed`.

- [ ] **Шаг 5: зафиксировать изменение**

```powershell
git add -- stt_backend.py tests/test_stt_backend.py
git commit -m "Добавить адаптеры GigaAM и Whisper"
```

### Задача 3: Подключение адаптера к рабочему entrypoint

**Файлы:**

- Изменить: `dictate_realtime.py`
- Создать: `tests/test_stt_wiring.py`

- [ ] **Шаг 1: написать падающий тест подключения**

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SttWiringTests(unittest.TestCase):
    def test_realtime_entrypoint_uses_backend_facade(self):
        source = (ROOT / "dictate_realtime.py").read_text(encoding="utf-8")
        self.assertIn("from stt_backend import create_stt_backend", source)
        self.assertIn("stt_backend = create_stt_backend(cfg)", source)
        self.assertNotIn("from faster_whisper import WhisperModel", source)
        self.assertNotIn("model.transcribe(", source)
        self.assertGreaterEqual(source.count("stt_backend.transcribe("), 5)
```

- [ ] **Шаг 2: запустить тест и подтвердить правильное падение**

Команда: `python -m pytest tests/test_stt_wiring.py -q`

Ожидается: FAIL, потому что entrypoint ещё импортирует и вызывает Whisper напрямую.

- [ ] **Шаг 3: заменить загрузку и вызовы модели**

В `dictate_realtime.py`:

```python
from faster_whisper.vad import get_speech_timestamps, VadOptions
from stt_backend import create_stt_backend

# после определения аудиоустройства
print(f"[init] STT engine: {cfg['stt_engine']}")
stt_backend = create_stt_backend(cfg)

# каждый путь инференса под существующим model_lock
text = stt_backend.transcribe(audio, quality="interim")
final = stt_backend.transcribe(audio_snap, quality="final")
head_text = stt_backend.transcribe(head, quality="final", force_vad=True)
raw_text = stt_backend.transcribe(audio, quality="final")
```

Сохранить существующие `model_lock`, фильтр галлюцинаций, очистку буфера и обработку исключений. В warm-up вызвать backend на секунде нулевого аудио и отдельно прогреть Silero VAD.

- [ ] **Шаг 4: запустить тест подключения и текущие быстрые тесты**

Команды:

```powershell
python -m pytest tests/test_stt_wiring.py tests/test_fast_final_mode.py tests/test_audio_utils.py -q
python test_dictate.py
```

Ожидается: все тесты проходят, `test_dictate.py` завершает таблицу проверок без `FAIL`.

- [ ] **Шаг 5: зафиксировать изменение**

```powershell
git add -- dictate_realtime.py tests/test_stt_wiring.py
git commit -m "Подключить переключаемый STT backend"
```

### Задача 4: Зависимость и воспроизводимый E2E

**Файлы:**

- Изменить: `requirements.txt`
- Создать: `scripts/e2e_gigaam.py`
- Создать: `tests/test_gigaam_e2e_script.py`

- [ ] **Шаг 1: написать падающий тест контракта E2E**

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GigaAME2EScriptTests(unittest.TestCase):
    def test_requirements_and_e2e_script_use_selected_model(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        script = (ROOT / "scripts" / "e2e_gigaam.py").read_text(encoding="utf-8")
        self.assertIn("gigaam[torch] @", requirements)
        self.assertIn('"gigaam_model": "v3_e2e_rnnt"', script)
        self.assertIn("GigaAMBackend", script)
```

- [ ] **Шаг 2: запустить тест и подтвердить правильное падение**

Команда: `python -m pytest tests/test_gigaam_e2e_script.py -q`

Ожидается: FAIL из-за отсутствующего E2E-скрипта и зависимости.

- [ ] **Шаг 3: добавить зависимость и E2E-скрипт**

В `requirements.txt` добавить:

```text
gigaam[torch] @ https://github.com/salute-developers/GigaAM/archive/refs/heads/main.tar.gz
```

Создать `scripts/e2e_gigaam.py`:

```python
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
    if audio.ndim > 1:
        audio = audio.astype(np.float32).mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    if sample_rate != SAMPLE_RATE:
        divisor = gcd(SAMPLE_RATE, sample_rate)
        audio = resample_poly(
            audio, SAMPLE_RATE // divisor, sample_rate // divisor,
        ).astype(np.float32)
    window = WINDOW_SECONDS * SAMPLE_RATE
    if len(audio) <= window:
        return audio
    starts = range(0, len(audio) - window + 1, window // 2)
    start = max(starts, key=lambda offset: float(np.mean(audio[offset:offset + window] ** 2)))
    return audio[start:start + window]


def main():
    parser = argparse.ArgumentParser()
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
    print(f"model={backend.model_name} seconds={elapsed:.1f}")
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Шаг 4: запустить тест контракта**

Команда: `python -m pytest tests/test_gigaam_e2e_script.py -q`

Ожидается: `1 passed`.

- [ ] **Шаг 5: зафиксировать изменение**

```powershell
git add -- requirements.txt scripts/e2e_gigaam.py tests/test_gigaam_e2e_script.py
git commit -m "Добавить зависимость и E2E GigaAM"
```

### Задача 5: Реальная установка и сквозная проверка

**Файлы:**

- Проверить: весь репозиторий
- Не изменять тестовые WAV из `jitsi-meet-recordings`

- [ ] **Шаг 1: проверить системные предпосылки**

Команды:

```powershell
python -V
ffmpeg -version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Ожидается: Python не ниже 3.10, доступный `ffmpeg`, `torch` импортируется.

- [ ] **Шаг 2: установить зависимости**

Команда: `python -m pip install -r requirements.txt`

Ожидается: успешно установлены `gigaam` и совместимый `torchaudio`; существующий CUDA `torch` не заменён несовместимой сборкой.

- [ ] **Шаг 3: выполнить полный автоматический набор**

Команды:

```powershell
python -m pytest -q
python test_dictate.py
```

Ожидается: все тесты проходят без ошибок.

- [ ] **Шаг 4: выполнить настоящий GigaAM E2E**

Команда: `python scripts/e2e_gigaam.py C:\Users\allgrit\Documents\codex\jitsi-meet-recordings\test-meeting3.wav`

Если файл длиннее 25 секунд, скрипт сам использует первые 20 секунд; если там нет речи, выбрать внутри скрипта первое 20-секундное окно с RMS выше порога. Ожидается: загрузка `v3_e2e_rnnt`, непустой русский текст и код выхода `0`.

- [ ] **Шаг 5: проверить запуск приложения до готовности**

Запустить `dictate_realtime.py` с временной копией конфигурации, указывающей `stt_engine=gigaam`, дождаться строки `[init] Ready`, затем штатно завершить процесс. Повторить загрузочный smoke с `stt_engine=whisper`, не выполняя распознавание микрофона.

- [ ] **Шаг 6: проверить diff и зафиксировать итоговые поправки**

Команды:

```powershell
git diff --check
git status --short
```

Ожидается: нет ошибок whitespace; пользовательский `project_architecture.html` остаётся untracked и не входит ни в один коммит.
