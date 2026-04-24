import sys
import os
import re
import time
import threading
import winsound
import tkinter as tk

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly
from math import gcd
from faster_whisper import WhisperModel
import keyboard
import pyperclip

# ===== CONFIG =====
MODEL_SIZE = "medium"
LANG = "ru"
TOGGLE_KEY = "f9"
WHISPER_SR = 16000
STREAM_INTERVAL = 0.8
BEAM_INTERIM = 1
BEAM_FINAL = 5
DEVICE_INDEX = 1
SILENCE_RMS = 0.01

BEEP_ON = (1000, 100)
BEEP_OFF = (600, 100)
BEEP_CMD = (800, 50)

# ===== HALLUCINATION FILTER =====
HALLUCINATIONS = [
    "спасибо за внимание", "спасибо за просмотр",
    "подписывайтесь на канал", "продолжение следует",
    "до новых встреч", "до свидания", "с вами был",
    "редактор субтитров", "корректор", "добро пожаловать",
    "music", "you", "thank you", "the end",
]


def is_hallucination(text: str) -> bool:
    t = text.lower().strip().rstrip(".")
    if len(t) < 3:
        return True
    return any(h in t for h in HALLUCINATIONS)


# ===== VOICE COMMANDS =====
COMMANDS: list[tuple[re.Pattern, callable]] = []


def cmd(pattern: str):
    def decorator(fn):
        COMMANDS.append((re.compile(pattern, re.IGNORECASE), fn))
        return fn
    return decorator


@cmd(r"^удали(ть)?\s+слово\.?$")
def cmd_delete_word():
    keyboard.send("ctrl+backspace")
    return "удалить слово"


@cmd(r"^удали(ть)?\s+(последнее\s+)?предложение\.?$")
def cmd_delete_sentence():
    keyboard.send("home")
    time.sleep(0.02)
    keyboard.send("shift+end")
    time.sleep(0.02)
    keyboard.send("backspace")
    return "удалить предложение"


@cmd(r"^удали(ть)?\s+строк[уа]\.?$")
def cmd_delete_line():
    keyboard.send("home")
    time.sleep(0.02)
    keyboard.send("shift+end")
    time.sleep(0.02)
    keyboard.send("backspace")
    return "удалить строку"


@cmd(r"^удали(ть)?\s+вс[её]\.?$")
def cmd_delete_all():
    keyboard.send("ctrl+a")
    time.sleep(0.02)
    keyboard.send("backspace")
    return "удалить всё"


@cmd(r"^отмен(ить|а|и)\.?$")
def cmd_undo():
    keyboard.send("ctrl+z")
    return "отмена"


@cmd(r"^нов(ая|ую)\s+строк[уа]\.?$")
def cmd_newline():
    keyboard.send("enter")
    return "новая строка"


@cmd(r"^нов(ый|ому)\s+(абзац|параграф)\.?$")
def cmd_paragraph():
    keyboard.send("enter")
    time.sleep(0.02)
    keyboard.send("enter")
    return "новый абзац"


@cmd(r"^таб(уляция)?\.?$")
def cmd_tab():
    keyboard.send("tab")
    return "таб"


def try_command(text: str) -> str | None:
    clean = text.strip().rstrip(".,!?;:")
    for pattern, fn in COMMANDS:
        if pattern.match(clean):
            return fn()
    return None


# ===== AUDIO DEVICE =====
dev_info = sd.query_devices(DEVICE_INDEX)
DEVICE_SR = int(dev_info["default_samplerate"])
_g = gcd(WHISPER_SR, DEVICE_SR)
_UP, _DOWN = WHISPER_SR // _g, DEVICE_SR // _g


def resample_to_whisper(audio: np.ndarray) -> np.ndarray:
    if DEVICE_SR == WHISPER_SR:
        return audio
    return resample_poly(audio, _UP, _DOWN).astype(np.float32)


# ===== MODEL =====
print(f"[init] {dev_info['name']} ({DEVICE_SR} Hz)")
print(f"[init] Whisper '{MODEL_SIZE}'...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print("[init] Готово.\n")

# ===== STATE =====
audio_lock = threading.Lock()
text_lock = threading.Lock()
toggle_lock = threading.Lock()
audio_buffer: list[np.ndarray] = []
is_recording = False
stream_active = False
current_text = ""
rec_start_time = 0.0
last_status = ""


def get_audio_snapshot() -> np.ndarray | None:
    with audio_lock:
        if not audio_buffer:
            return None
        raw = np.concatenate(audio_buffer, axis=0).flatten()
    return resample_to_whisper(raw)


def audio_is_silent(audio: np.ndarray) -> bool:
    return np.sqrt(np.mean(audio ** 2)) < SILENCE_RMS


def send_backspaces(n: int):
    if n <= 0:
        return
    for _ in range(n):
        keyboard.send("backspace")
    time.sleep(0.01)


def paste_text(text: str):
    if not text:
        return
    pyperclip.copy(text)
    time.sleep(0.02)
    keyboard.press_and_release("ctrl+v")
    time.sleep(0.02)


def apply_diff(old_text: str, new_text: str):
    common = 0
    for a, b in zip(old_text, new_text):
        if a == b:
            common += 1
        else:
            break
    send_backspaces(len(old_text) - common)
    paste_text(new_text[common:])


# ===== STREAMING WORKER =====
def transcription_worker():
    global current_text

    while True:
        if not stream_active:
            time.sleep(0.1)
            continue

        time.sleep(STREAM_INTERVAL)
        if not stream_active:
            continue

        audio = get_audio_snapshot()
        if audio is None or len(audio) < WHISPER_SR * 0.3 or audio_is_silent(audio):
            continue

        try:
            segments, _ = model.transcribe(
                audio, language=LANG, beam_size=BEAM_INTERIM, vad_filter=True,
            )
            new_text = " ".join(seg.text for seg in segments).strip()
        except Exception:
            continue

        if not new_text or not stream_active or is_hallucination(new_text):
            continue

        with text_lock:
            if new_text != current_text:
                apply_diff(current_text, new_text)
                current_text = new_text


# ===== TOGGLE =====
def do_finalize():
    global current_text, last_status

    audio = get_audio_snapshot()

    if audio is None or len(audio) < WHISPER_SR * 0.3 or audio_is_silent(audio):
        with text_lock:
            if current_text:
                send_backspaces(len(current_text))
            current_text = ""
        last_status = ""
        return

    try:
        segments, _ = model.transcribe(
            audio, language=LANG, beam_size=BEAM_FINAL, vad_filter=True,
        )
        raw_text = " ".join(seg.text for seg in segments).strip()
    except Exception:
        raw_text = ""

    if not raw_text or is_hallucination(raw_text):
        with text_lock:
            if current_text:
                send_backspaces(len(current_text))
            current_text = ""
        last_status = ""
        return

    cmd_result = try_command(raw_text)
    if cmd_result is not None:
        with text_lock:
            if current_text:
                send_backspaces(len(current_text))
            current_text = ""
        threading.Thread(target=lambda: winsound.Beep(*BEEP_CMD), daemon=True).start()
        last_status = f"cmd: {cmd_result}"
        print(f"[cmd] {cmd_result}")
        return

    with text_lock:
        if current_text:
            send_backspaces(len(current_text))
        paste_text(raw_text)
        current_text = ""
    last_status = raw_text
    print(f"[ok] {raw_text}")


def on_toggle():
    global is_recording, stream_active, current_text, rec_start_time, last_status

    if not toggle_lock.acquire(blocking=False):
        return
    try:
        if not is_recording:
            is_recording = True
            stream_active = True
            rec_start_time = time.time()
            with audio_lock:
                audio_buffer.clear()
            with text_lock:
                current_text = ""
            pyperclip.copy("")
            last_status = ""
            threading.Thread(target=lambda: winsound.Beep(*BEEP_ON), daemon=True).start()
            print("[rec] ON")
        else:
            stream_active = False
            is_recording = False
            threading.Thread(target=lambda: winsound.Beep(*BEEP_OFF), daemon=True).start()
            do_finalize()
    finally:
        toggle_lock.release()


# ===== AUDIO CALLBACK =====
def audio_callback(indata, frames, time_info, status):
    if is_recording:
        with audio_lock:
            audio_buffer.append(indata.copy())


# ===== FLOATING WIDGET =====
class Overlay:
    BG = "#16161e"
    RED = "#ff3b3b"
    GRAY = "#555555"
    WHITE = "#e0e0e0"
    DIM = "#777777"
    CYAN = "#66cccc"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("STL")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self.root.configure(bg=self.BG)

        screen_w = self.root.winfo_screenwidth()
        self.root.geometry(f"320x38+{screen_w - 340}+18")

        frame = tk.Frame(self.root, bg=self.BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.canvas = tk.Canvas(frame, width=16, height=16, bg=self.BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=(0, 8))
        self.dot = self.canvas.create_oval(2, 2, 14, 14, fill=self.GRAY, outline="")

        self.label = tk.Label(frame, text="Готово  [F9]", fg=self.DIM, bg=self.BG,
                              font=("Segoe UI", 10), anchor="w")
        self.label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._pulse_on = True
        self.root.bind("<Button-1>", self._drag_start)
        self.root.bind("<B1-Motion>", self._drag_move)
        self._dx = 0
        self._dy = 0

        self.root.after(120, self._tick)

    def _drag_start(self, e):
        self._dx = e.x
        self._dy = e.y

    def _drag_move(self, e):
        x = self.root.winfo_x() + e.x - self._dx
        y = self.root.winfo_y() + e.y - self._dy
        self.root.geometry(f"+{x}+{y}")

    def _tick(self):
        if is_recording:
            self._pulse_on = not self._pulse_on
            self.canvas.itemconfig(self.dot, fill=self.RED if self._pulse_on else "#aa2222")

            elapsed = int(time.time() - rec_start_time)
            mins, secs = divmod(elapsed, 60)
            timer = f"{mins:02d}:{secs:02d}"

            with text_lock:
                txt = current_text
            if txt:
                show = txt if len(txt) <= 26 else "..." + txt[-23:]
                self.label.config(text=f"{timer}  {show}", fg=self.WHITE)
            else:
                self.label.config(text=f"{timer}  ...", fg="#ff8888")
        else:
            self.canvas.itemconfig(self.dot, fill=self.GRAY)
            if last_status and last_status.startswith("cmd:"):
                self.label.config(text=last_status, fg=self.CYAN)
            else:
                self.label.config(text="Готово  [F9]", fg=self.DIM)

        self.root.after(120, self._tick)

    def run(self):
        self.root.mainloop()


# ===== MAIN =====
threading.Thread(target=transcription_worker, daemon=True).start()
keyboard.on_press_key(TOGGLE_KEY, lambda _: threading.Thread(target=on_toggle, daemon=True).start())

audio_stream = sd.InputStream(
    device=DEVICE_INDEX, samplerate=DEVICE_SR, channels=1,
    blocksize=1024, latency="high", callback=audio_callback,
)
audio_stream.start()

print("[ready] F9 = toggle")
print("[cmds]  удалить слово / предложение / строку / всё")
print("[cmds]  отменить, новая строка, новый абзац, таб")
print("[filter] VAD + hallucination filter active\n")

Overlay().run()

audio_stream.stop()
audio_stream.close()
