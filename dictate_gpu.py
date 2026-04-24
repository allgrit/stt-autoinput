import sys
import os
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import keyboard
import pyperclip

MODEL_SIZE = "medium"
LANG = "ru"
PUSH_KEY = "f9"
SAMPLERATE = 16000
BEAM_SIZE = 5

print(f"Загрузка модели '{MODEL_SIZE}' на GPU...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print(f"Модель загружена. Удерживай {PUSH_KEY.upper()} и говори.")

audio_buffer = []


def audio_callback(indata, frames, time_info, status):
    audio_buffer.append(indata.copy())


with sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=audio_callback):
    while True:
        if keyboard.is_pressed(PUSH_KEY):
            audio_buffer.clear()
            print("[REC] Запись...")

            while keyboard.is_pressed(PUSH_KEY):
                time.sleep(0.05)

            print("[...] Распознавание...")

            if not audio_buffer:
                print("[!] Пустая запись, пропуск.")
                continue

            audio = np.concatenate(audio_buffer, axis=0).flatten()

            if len(audio) < SAMPLERATE * 0.3:
                print("[!] Слишком короткая запись, пропуск.")
                continue

            segments, info = model.transcribe(
                audio,
                language=LANG,
                beam_size=BEAM_SIZE,
            )

            text = " ".join(seg.text for seg in segments).strip()

            if text:
                pyperclip.copy(text)
                keyboard.press_and_release("ctrl+v")
                print(f"[OK] {text}")
            else:
                print("[!] Ничего не распознано.")

        time.sleep(0.05)
