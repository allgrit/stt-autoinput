from __future__ import annotations
import sys
import os
import re
import time
import threading
import signal
import atexit
import winsound
import tkinter as tk

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

_PID_PATH = os.path.join(os.getcwd(), "stl.pid")
_killed_old = False
try:
    old_pid = int(open(_PID_PATH).read().strip())
    os.kill(old_pid, signal.SIGTERM)
    _killed_old = True
    print(f"[init] Killed old instance (PID {old_pid})")
except (FileNotFoundError, ValueError, OSError):
    pass
with open(_PID_PATH, "w") as _f:
    _f.write(str(os.getpid()))
atexit.register(lambda: os.path.exists(_PID_PATH) and os.remove(_PID_PATH))

import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly
from math import gcd
from faster_whisper import WhisperModel
from faster_whisper.vad import get_speech_timestamps, VadOptions
import keyboard
import pyperclip

from config import load_config, save_config, needs_setup, detect_compute
from setup_dialog import run_setup
from audio_utils import trim_silence

# ===== CONFIG =====
cfg = load_config()

if needs_setup(cfg):
    cfg = run_setup(cfg)
    save_config(cfg)

MODEL_SIZE = cfg["model_size"]
LANG = cfg["language"]
TOGGLE_KEY = cfg["toggle_key"]

# Auto-resolve device by name if index doesn't match
DEVICE_INDEX = cfg["device_index"]
_device_name = cfg.get("device_name", "")


def _device_name_matches(target, candidate):
    """Bidirectional substring: handles PortAudio name truncation on MME."""
    return target in candidate or candidate in target


if DEVICE_INDEX is not None and _device_name:
    try:
        info = sd.query_devices(DEVICE_INDEX)
        if not _device_name_matches(_device_name, info.get("name", "")):
            for i, d in enumerate(sd.query_devices()):
                if _device_name_matches(_device_name, d["name"]) and d["max_input_channels"] > 0:
                    print(f"[init] Device index changed: {DEVICE_INDEX} -> {i} ({d['name']})")
                    DEVICE_INDEX = i
                    cfg["device_index"] = i
                    save_config(cfg)
                    break
    except Exception:
        for i, d in enumerate(sd.query_devices()):
            if _device_name_matches(_device_name, d["name"]) and d["max_input_channels"] > 0:
                DEVICE_INDEX = i
                cfg["device_index"] = i
                save_config(cfg)
                break
WHISPER_SR = 16000
STREAM_INTERVAL = cfg["stream_interval"]
BEAM_INTERIM = cfg["beam_interim"]
BEAM_FINAL = cfg["beam_final"]
REALTIME_PREVIEW = cfg["realtime_preview"]
VAD_FILTER = cfg["vad_filter"]
TRIM_SILENCE = cfg["trim_silence"]
TRIM_SILENCE_RMS = cfg["trim_silence_rms"]
SILENCE_RMS = cfg["silence_rms"]
COMMIT_PAUSE = cfg["commit_pause"]
MAX_BUFFER_S = cfg.get("max_buffer_s", 7.0)
HARD_MAX_BUFFER_S = cfg.get("hard_max_buffer_s", 14.0)
VAD_MIN_SILENCE_MS = cfg.get("vad_min_silence_ms", 350)
VAD_THRESHOLD = cfg.get("vad_threshold", 0.5)
INITIAL_PROMPT = cfg.get("initial_prompt", "")

BEEP_ON = tuple(cfg["beep_on"])
BEEP_OFF = tuple(cfg["beep_off"])
BEEP_CMD = tuple(cfg["beep_cmd"])

# ===== COMPUTE DEVICE =====
if cfg["compute_device"] == "auto":
    COMPUTE_DEVICE, COMPUTE_TYPE = detect_compute()
else:
    COMPUTE_DEVICE = cfg["compute_device"]
    COMPUTE_TYPE = cfg["compute_type"]

# ===== HALLUCINATION FILTER =====
HALLUCINATIONS = [
    "спасибо за внимание", "спасибо за просмотр",
    "подписывайтесь на канал", "продолжение следует",
    "до новых встреч", "до свидания", "с вами был",
    "редактор субтитров", "корректор", "добро пожаловать",
    "music", "you", "thank you", "the end",
]


def is_hallucination(text):
    t = text.lower().strip().rstrip(".")
    if len(t) < 3:
        return True
    return any(h in t for h in HALLUCINATIONS)


# ===== VOICE COMMANDS =====
COMMANDS = []


def cmd(pattern):
    def decorator(fn):
        COMMANDS.append((re.compile(pattern, re.IGNORECASE), fn))
        return fn
    return decorator


@cmd(r"^удали(ть|те)?\s+слово$")
def cmd_delete_word():
    keyboard.send("ctrl+backspace")
    return "удалить слово"


@cmd(r"^удали(ть|те)?\s+(последнее\s+)?предложение$")
def cmd_delete_sentence():
    keyboard.send("home")
    time.sleep(0.02)
    keyboard.send("shift+end")
    time.sleep(0.02)
    keyboard.send("backspace")
    return "удалить предложение"


@cmd(r"^удали(ть|те)?\s+строк[уа]$")
def cmd_delete_line():
    keyboard.send("home")
    time.sleep(0.02)
    keyboard.send("shift+end")
    time.sleep(0.02)
    keyboard.send("backspace")
    return "удалить строку"


@cmd(r"^удали(ть|те)?\s+вс[её]$")
def cmd_delete_all():
    keyboard.send("ctrl+a")
    time.sleep(0.02)
    keyboard.send("backspace")
    return "удалить всё"


@cmd(r"^отмен(ить|а|и|ите)$")
def cmd_undo():
    keyboard.send("ctrl+z")
    return "отмена"


@cmd(r"^нов(ая|ую)\s+строк[уа]$")
def cmd_newline():
    keyboard.send("enter")
    return "новая строка"


@cmd(r"^(enter|энтер|ввод)$")
def cmd_enter():
    keyboard.send("enter")
    return "enter"


@cmd(r"^нов(ый|ому)\s+(абзац|параграф)$")
def cmd_paragraph():
    keyboard.send("enter")
    time.sleep(0.02)
    keyboard.send("enter")
    return "новый абзац"


@cmd(r"^таб(уляция)?$")
def cmd_tab():
    keyboard.send("tab")
    return "таб"


def _clean(text):
    return text.strip().rstrip(".,!?;:").strip()


def match_command_full(text):
    clean = _clean(text)
    for pattern, fn in COMMANDS:
        if pattern.match(clean):
            return fn
    return None


def match_command_trailing(text):
    words = text.split()
    for n in range(1, min(6, len(words) + 1)):
        tail = _clean(" ".join(words[-n:]))
        for pattern, fn in COMMANDS:
            if pattern.match(tail):
                remaining = " ".join(words[:-n]).strip()
                return remaining, fn
    return text, None


# ===== AUDIO DEVICE =====
dev_info = sd.query_devices(DEVICE_INDEX)
DEVICE_SR = int(dev_info["default_samplerate"])
_g = gcd(WHISPER_SR, DEVICE_SR)
_UP, _DOWN = WHISPER_SR // _g, DEVICE_SR // _g


def resample_to_whisper(audio):
    if DEVICE_SR == WHISPER_SR:
        return audio
    return resample_poly(audio, _UP, _DOWN).astype(np.float32)


# ===== MODEL =====
print(f"[init] {dev_info['name']} ({DEVICE_SR} Hz)")
print(f"[init] Whisper '{MODEL_SIZE}' on {COMPUTE_DEVICE} ({COMPUTE_TYPE})...")
try:
    model = WhisperModel(MODEL_SIZE, device=COMPUTE_DEVICE, compute_type=COMPUTE_TYPE)
except Exception as e:
    print(f"[init] GPU failed ({e}), falling back to CPU...")
    COMPUTE_DEVICE, COMPUTE_TYPE = "cpu", "float32"
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="float32")

# Warm up both models so the first real utterance doesn't eat a multi-second
# lazy-load stall (Silero VAD + CTranslate2 graph allocation).
try:
    _warm = np.zeros(WHISPER_SR, dtype=np.float32)
    get_speech_timestamps(_warm, VadOptions(), sampling_rate=WHISPER_SR)
    list(model.transcribe(_warm, language=LANG, beam_size=BEAM_FINAL)[0])
except Exception as e:
    print(f"[init] warmup skipped: {e}")
print("[init] Ready.\n")

# ===== STATE =====
_T0 = time.monotonic()


def ts():
    return f"{time.monotonic() - _T0:7.2f}s"


audio_lock = threading.Lock()
text_lock = threading.Lock()
toggle_lock = threading.Lock()
# Serializes GPU inference: the worker (interim + vad-commit), do_finalize and
# background _bg_refine threads all share one CTranslate2 model, which is not
# safe for concurrent transcribe() calls — overlap caused multi-second stalls.
model_lock = threading.Lock()
audio_buffer = []
is_recording = False
stream_active = False
current_text = ""
rec_start_time = 0.0
last_status = ""


def get_audio_snapshot():
    with audio_lock:
        if not audio_buffer:
            return None
        raw = np.concatenate(audio_buffer, axis=0).flatten()
    return resample_to_whisper(raw)


def audio_is_silent(audio):
    return np.sqrt(np.mean(audio ** 2)) < SILENCE_RMS


def _vad_has_speech(audio16):
    """VAD-based speech presence. Robust to mic level, unlike an absolute RMS
    gate — the C922 mic records speech at RMS ~0.006, below any safe RMS floor,
    so RMS silence detection drops whole phrases. Silero VAD keys on spectral
    speech features instead. Fail-open on error: assume speech, let Whisper decide."""
    try:
        segs = get_speech_timestamps(
            audio16,
            VadOptions(
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=150,
                min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                speech_pad_ms=100,
            ),
            sampling_rate=WHISPER_SR,
        )
        return len(segs) > 0
    except Exception:
        return True


def find_vad_split(audio16):
    """Find a commit boundary (seconds) inside a silence gap, keeping the last
    (possibly ongoing) speech region as tail. Returns None if no safe split —
    e.g. fewer than two speech regions (one continuous phrase)."""
    try:
        segs = get_speech_timestamps(
            audio16,
            VadOptions(
                threshold=VAD_THRESHOLD,
                min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                min_speech_duration_ms=200,
                speech_pad_ms=200,
            ),
            sampling_rate=WHISPER_SR,
        )
    except Exception:
        return None
    if len(segs) < 2:
        return None
    # Cut in the middle of the silence gap before the last speech region, so the
    # head holds all completed phrases and the tail holds the ongoing one.
    gap_start = segs[-2]["end"]
    gap_end = segs[-1]["start"]
    if gap_end <= gap_start:  # padding swallowed the gap — no clean cut
        return None
    return ((gap_start + gap_end) // 2) / WHISPER_SR


def recent_audio_is_silent():
    with audio_lock:
        if not audio_buffer:
            return True
        n_chunks = int(COMMIT_PAUSE * DEVICE_SR / 1024)
        recent = audio_buffer[-n_chunks:]
        if not recent:
            return True
        raw = np.concatenate(recent, axis=0).flatten()
    return not _vad_has_speech(resample_to_whisper(raw))


def send_backspaces(n):
    if n <= 0:
        return
    for _ in range(n):
        keyboard.send("backspace")
    time.sleep(0.01)


def paste_text(text):
    if not text:
        return
    pyperclip.copy(text)
    time.sleep(0.02)
    keyboard.press_and_release("ctrl+v")
    time.sleep(0.02)


def apply_diff(old_text, new_text):
    common = 0
    for a, b in zip(old_text, new_text):
        if a == b:
            common += 1
        else:
            break
    send_backspaces(len(old_text) - common)
    paste_text(new_text[common:])


# ===== BACKGROUND REFINE =====
def _bg_refine(audio_snap, old_text):
    try:
        with model_lock:
            segments, _ = model.transcribe(
                audio_snap, language=LANG, beam_size=BEAM_FINAL, vad_filter=VAD_FILTER,
                initial_prompt=INITIAL_PROMPT,
            )
            final = " ".join(seg.text for seg in segments).strip()
    except Exception:
        return
    if not final or is_hallucination(final) or final == old_text:
        return
    with text_lock:
        # stream_active guards against finishing after F9-off: do_finalize has
        # already typed the final text, so refining here would inject stray
        # backspaces/text into whatever window now has focus.
        if stream_active and not current_text:
            send_backspaces(len(old_text))
            paste_text(final)
            print(f"[refine] {final}")


# ===== STREAMING WORKER =====
def transcription_worker():
    global current_text, last_status
    last_speech_time = 0.0

    while True:
        if not stream_active:
            last_speech_time = 0.0
            time.sleep(0.1)
            continue

        if not REALTIME_PREVIEW:
            time.sleep(0.1)
            continue

        time.sleep(STREAM_INTERVAL)
        if not stream_active:
            continue

        silence = recent_audio_is_silent()

        if current_text and last_speech_time and silence:
            if time.time() - last_speech_time > COMMIT_PAUSE:
                commit_audio = get_audio_snapshot()
                commit_old = current_text
                with audio_lock:
                    audio_buffer.clear()
                with text_lock:
                    current_text = ""
                last_speech_time = 0.0
                print(f"[commit {ts()}] {commit_old}")

                if commit_audio is not None and len(commit_audio) >= WHISPER_SR * 0.3:
                    threading.Thread(
                        target=_bg_refine, args=(commit_audio, commit_old), daemon=True,
                    ).start()
                continue

        if silence:
            continue

        audio = get_audio_snapshot()
        if audio is None or len(audio) < WHISPER_SR * 0.3:
            continue

        # ----- VAD segmentation: bound the buffer during continuous speech -----
        # Without this the whole buffer is re-transcribed every interim pass, so
        # cost grows O(n) per pass (and O(n^2) overall): the longer you speak the
        # slower and worse it gets, and chunks get dropped. We lock completed
        # phrases at a natural pause with beam=final and keep only the tail.
        dur = len(audio) / WHISPER_SR
        if dur >= MAX_BUFFER_S:
            split_t = find_vad_split(audio)
            head = None
            whole = False
            if split_t is not None:
                cand = audio[: int(split_t * WHISPER_SR)]
                if len(cand) >= WHISPER_SR * 0.3:
                    head = cand
            if head is None and dur >= HARD_MAX_BUFFER_S:
                head, whole, split_t = audio, True, dur
            if head is not None:
                try:
                    with model_lock:
                        h_segs, _ = model.transcribe(
                            head, language=LANG, beam_size=BEAM_FINAL, vad_filter=True,
                            initial_prompt=INITIAL_PROMPT,
                        )
                        head_text = " ".join(s.text for s in h_segs).strip()
                except Exception:
                    head_text = ""
                if not stream_active:
                    # F9-off landed during transcription — do_finalize owns the
                    # text now; don't paste a stale head or trim a cleared buffer.
                    continue
                ok = bool(head_text) and not is_hallucination(head_text)
                with text_lock:
                    if current_text:
                        send_backspaces(len(current_text))
                    if ok:
                        paste_text(head_text)
                    current_text = ""
                with audio_lock:
                    if whole:
                        audio_buffer.clear()
                    else:
                        del audio_buffer[: int(split_t * DEVICE_SR / 1024)]
                last_speech_time = 0.0
                if ok:
                    print(f"[vad-commit] {head_text}")
                continue

        try:
            with model_lock:
                segments, _ = model.transcribe(
                    audio, language=LANG, beam_size=BEAM_INTERIM, vad_filter=VAD_FILTER,
                    initial_prompt=INITIAL_PROMPT,
                )
                new_text = " ".join(seg.text for seg in segments).strip()
        except Exception:
            continue

        if not new_text or not stream_active or is_hallucination(new_text):
            continue

        if len(new_text) > len(current_text):
            last_speech_time = time.time()

        remaining, cmd_fn = match_command_trailing(new_text)

        if cmd_fn is not None:
            with text_lock:
                if current_text:
                    send_backspaces(len(current_text))
                if remaining:
                    paste_text(remaining)
                current_text = ""
            time.sleep(0.05)
            result = cmd_fn()
            with audio_lock:
                audio_buffer.clear()
            last_speech_time = 0.0
            last_status = f"cmd: {result}"
            print(f"[cmd] {result}")
            winsound.Beep(*BEEP_CMD)
            continue

        with text_lock:
            if new_text != current_text:
                common = 0
                for a, b in zip(current_text, new_text):
                    if a == b:
                        common += 1
                    else:
                        break
                to_delete = len(current_text) - common
                if to_delete <= 5:
                    send_backspaces(to_delete)
                    paste_text(new_text[common:])
                    current_text = new_text


# ===== TOGGLE =====
def do_finalize():
    global current_text, last_status

    audio = get_audio_snapshot()

    if audio is None or len(audio) < WHISPER_SR * 0.3:
        with text_lock:
            if current_text:
                send_backspaces(len(current_text))
            current_text = ""
        last_status = ""
        print("[final] no audio")
        return

    raw_seconds = len(audio) / WHISPER_SR
    if TRIM_SILENCE:
        audio = trim_silence(audio, WHISPER_SR, TRIM_SILENCE_RMS)

    if len(audio) < WHISPER_SR * 0.3 or not _vad_has_speech(audio):
        with text_lock:
            if current_text:
                send_backspaces(len(current_text))
            current_text = ""
        last_status = ""
        print(f"[final] no speech after trim ({raw_seconds:.1f}s raw)")
        return

    audio_seconds = len(audio) / WHISPER_SR
    last_status = "Transcribing..."
    print(f"[final] audio={audio_seconds:.1f}s raw={raw_seconds:.1f}s beam={BEAM_FINAL} vad={VAD_FILTER}")
    started = time.perf_counter()
    try:
        with model_lock:
            segments, _ = model.transcribe(
                audio, language=LANG, beam_size=BEAM_FINAL, vad_filter=VAD_FILTER,
                initial_prompt=INITIAL_PROMPT,
            )
            raw_text = " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        print(f"[final:error] {type(e).__name__}: {e}")
        raw_text = ""
    elapsed = time.perf_counter() - started
    print(f"[final] done in {elapsed:.1f}s")

    if not raw_text or is_hallucination(raw_text):
        print(f"[final] rejected: '{raw_text}'")
        with text_lock:
            if current_text:
                send_backspaces(len(current_text))
            current_text = ""
        last_status = ""
        return

    cmd_fn = match_command_full(raw_text)
    if cmd_fn is not None:
        with text_lock:
            if current_text:
                send_backspaces(len(current_text))
            current_text = ""
        time.sleep(0.05)
        result = cmd_fn()
        winsound.Beep(*BEEP_CMD)
        last_status = f"cmd: {result}"
        print(f"[cmd] {result}")
        return

    remaining, cmd_fn = match_command_trailing(raw_text)
    if cmd_fn is not None:
        with text_lock:
            if current_text:
                send_backspaces(len(current_text))
            if remaining:
                paste_text(remaining)
            current_text = ""
        time.sleep(0.05)
        result = cmd_fn()
        winsound.Beep(*BEEP_CMD)
        last_status = f"cmd: {result}"
        print(f"[ok+cmd] {remaining} | {result}")
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
            print(f"[rec {ts()}] ON")
        else:
            stream_active = False
            is_recording = False
            with audio_lock:
                _n_chunks = len(audio_buffer)
            print(f"[rec {ts()}] OFF  chunks={_n_chunks} ({_n_chunks*1024/DEVICE_SR:.1f}s)")
            threading.Thread(target=lambda: winsound.Beep(*BEEP_OFF), daemon=True).start()
            print("[rec] OFF")
            do_finalize()
    finally:
        toggle_lock.release()


# ===== AUDIO CALLBACK =====
_cb_count = 0

def audio_callback(indata, frames, time_info, status):
    global _cb_count
    _cb_count += 1
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

        self.label = tk.Label(frame, text=f"Ready  [{TOGGLE_KEY.upper()}]", fg=self.DIM,
                              bg=self.BG, font=("Segoe UI", 10), anchor="w")
        self.label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._pulse_on = True
        self.root.bind("<Button-1>", self._drag_start)
        self.root.bind("<B1-Motion>", self._drag_move)
        self.root.bind("<Button-3>", self._context_menu)
        self._dx = 0
        self._dy = 0

        self._menu = tk.Menu(self.root, tearoff=0, bg="#2a2a3e", fg="#e0e0e0",
                              activebackground="#3a3a5e", font=("Segoe UI", 9))
        self._menu.add_command(label="Settings...", command=self._open_settings)
        self._menu.add_separator()
        self._menu.add_command(label="Exit", command=self._quit)

        self.root.after(120, self._tick)

    def _drag_start(self, e):
        self._dx = e.x
        self._dy = e.y

    def _drag_move(self, e):
        x = self.root.winfo_x() + e.x - self._dx
        y = self.root.winfo_y() + e.y - self._dy
        self.root.geometry(f"+{x}+{y}")

    def _context_menu(self, e):
        self._menu.post(e.x_root, e.y_root)

    def _open_settings(self):
        new_cfg = run_setup(cfg)
        save_config(new_cfg)
        self.label.config(text="Restart to apply", fg="#ffcc00")

    def _quit(self):
        self.root.destroy()

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
            elif last_status:
                self.label.config(text=last_status, fg=self.WHITE)
            else:
                self.label.config(text=f"Ready  [{TOGGLE_KEY.upper()}]", fg=self.DIM)

        self.root.after(120, self._tick)

    def run(self):
        self.root.mainloop()


# ===== ENTER HOOK =====
def on_enter_pressed():
    global current_text
    if is_recording and current_text:
        with text_lock:
            current_text = ""
        with audio_lock:
            audio_buffer.clear()
        print("[commit] enter detected")


# ===== MAIN =====
threading.Thread(target=transcription_worker, daemon=True).start()
keyboard.on_press_key(TOGGLE_KEY, lambda _: threading.Thread(target=on_toggle, daemon=True).start())
keyboard.on_press_key("enter", lambda _: on_enter_pressed())

audio_stream = None

_API_PRIORITY = {"MME": 0, "Windows DirectSound": 1, "Windows WASAPI": 2, "Windows WDM-KS": 3}


def _try_open_stream(dev_idx, sr):
    global _cb_count
    try:
        _cb_count = 0
        s = sd.InputStream(device=dev_idx, samplerate=sr, channels=1,
                           blocksize=1024, latency="high", callback=audio_callback)
        s.start()
        time.sleep(0.3)
        if _cb_count == 0:
            s.stop()
            s.close()
            print(f"[warn] device {dev_idx} opened but no audio callbacks")
            return None
        return s
    except Exception as e:
        print(f"[warn] device {dev_idx} open failed: {e}")
        return None


def _apply_device(idx, sr):
    global DEVICE_SR, _UP, _DOWN
    DEVICE_SR = sr
    _g = gcd(WHISPER_SR, DEVICE_SR)
    _UP = WHISPER_SR // _g
    _DOWN = DEVICE_SR // _g


def _open_with_retry(dev_idx, sr, retries=3):
    for attempt in range(retries):
        if attempt > 0:
            wait = 0.5 * (2 ** attempt)
            print(f"[retry] device {dev_idx}, attempt {attempt + 1}/{retries} (wait {wait:.1f}s)")
            time.sleep(wait)
        stream = _try_open_stream(dev_idx, sr)
        if stream is not None:
            return stream
    return None


if _killed_old:
    time.sleep(0.5)
    audio_stream = _open_with_retry(DEVICE_INDEX, DEVICE_SR)
else:
    audio_stream = _try_open_stream(DEVICE_INDEX, DEVICE_SR)

if audio_stream is None:
    _target_name = cfg.get("device_name", "")
    _same_mic = []
    _other = []
    for idx in range(len(sd.query_devices())):
        info = sd.query_devices(idx)
        if info["max_input_channels"] == 0 or idx == DEVICE_INDEX:
            continue
        if _target_name and _device_name_matches(_target_name, info["name"]):
            _same_mic.append(idx)
        else:
            _other.append(idx)
    _same_mic.sort(key=lambda i: _API_PRIORITY.get(
        sd.query_hostapis(sd.query_devices(i)["hostapi"])["name"], 9))
    print(f"[fallback] same mic: {_same_mic}, other: {len(_other)}")
    for idx in _same_mic + _other:
        info = sd.query_devices(idx)
        sr = int(info["default_samplerate"])
        audio_stream = _try_open_stream(idx, sr)
        if audio_stream is not None:
            _apply_device(idx, sr)
            api_name = sd.query_hostapis(info["hostapi"])["name"]
            print(f"[ok] using [{idx}] {info['name']} ({sr} Hz, {api_name})")
            cfg["device_index"] = idx
            save_config(cfg)
            break

if audio_stream is None:
    print("[error] No working audio device found!")
    sys.exit(1)

print(f"[ready] {TOGGLE_KEY.upper()} = toggle | {COMPUTE_DEVICE}")
print("[cmds]  удалить слово / предложение / строку / всё")
print("[cmds]  отменить, новая строка, новый абзац, таб, энтер")
print(
    f"[mode] realtime_preview={REALTIME_PREVIEW} beam_final={BEAM_FINAL} "
    f"vad={VAD_FILTER} trim={TRIM_SILENCE}"
)
print(f"[filter] hallucination filter + auto-commit ({COMMIT_PAUSE}s)\n")

Overlay().run()

audio_stream.stop()
audio_stream.close()
