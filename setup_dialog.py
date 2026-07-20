from __future__ import annotations
import threading
import tkinter as tk
from tkinter import ttk

import sounddevice as sd
import numpy as np


def get_input_devices():
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append({
                "index": i,
                "name": d["name"],
                "sr": int(d["default_samplerate"]),
                "channels": d["max_input_channels"],
            })
    return devices


def run_setup(cfg):
    result = dict(cfg)
    confirmed = [False]

    root = tk.Tk()
    root.title("STL Autoinput — Setup")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")
    root.attributes("-topmost", True)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Dark.TFrame", background="#1e1e2e")
    style.configure("Dark.TLabel", background="#1e1e2e", foreground="#e0e0e0",
                     font=("Segoe UI", 10))
    style.configure("Dark.TButton", font=("Segoe UI", 10))
    style.configure("Header.TLabel", background="#1e1e2e", foreground="#ffffff",
                     font=("Segoe UI", 14, "bold"))

    main = ttk.Frame(root, style="Dark.TFrame", padding=20)
    main.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main, text="STL Autoinput", style="Header.TLabel").pack(pady=(0, 15))

    # --- Microphone ---
    ttk.Label(main, text="Microphone:", style="Dark.TLabel").pack(anchor="w")

    devices = get_input_devices()
    dev_names = [f"[{d['index']}] {d['name']} ({d['sr']} Hz)" for d in devices]

    mic_var = tk.StringVar()
    mic_combo = ttk.Combobox(main, textvariable=mic_var, values=dev_names,
                              state="readonly", width=50, font=("Segoe UI", 9))
    mic_combo.pack(fill=tk.X, pady=(2, 5))

    preselect = 0
    if cfg.get("device_index") is not None:
        for i, d in enumerate(devices):
            if d["index"] == cfg["device_index"]:
                preselect = i
                break
    mic_combo.current(preselect)

    # VU meter
    vu_canvas = tk.Canvas(main, height=14, bg="#2a2a3e", highlightthickness=0)
    vu_canvas.pack(fill=tk.X, pady=(0, 5))
    vu_bar = vu_canvas.create_rectangle(0, 0, 0, 14, fill="#44cc66", outline="")

    vu_active = [False]
    vu_stream = [None]

    def update_vu():
        if not vu_active[0]:
            return
        root.after(80, update_vu)

    def start_vu_test():
        sel = mic_combo.current()
        if sel < 0:
            return
        dev = devices[sel]
        vu_active[0] = True

        def audio_cb(indata, frames, time_info, status):
            if not vu_active[0]:
                return
            rms = float(np.sqrt(np.mean(indata ** 2)))
            level = min(rms * 15, 1.0)
            w = int(vu_canvas.winfo_width() * level)
            try:
                vu_canvas.coords(vu_bar, 0, 0, w, 14)
                color = "#44cc66" if level < 0.6 else "#ffcc00" if level < 0.85 else "#ff4444"
                vu_canvas.itemconfig(vu_bar, fill=color)
            except tk.TclError:
                pass

        try:
            if vu_stream[0]:
                vu_stream[0].stop()
                vu_stream[0].close()
            s = sd.InputStream(device=dev["index"], samplerate=dev["sr"],
                               channels=1, blocksize=1024, callback=audio_cb)
            s.start()
            vu_stream[0] = s
            root.after(80, update_vu)
        except Exception as e:
            vu_canvas.coords(vu_bar, 0, 0, 0, 14)

    def stop_vu():
        vu_active[0] = False
        if vu_stream[0]:
            try:
                vu_stream[0].stop()
                vu_stream[0].close()
            except Exception:
                pass
            vu_stream[0] = None
        vu_canvas.coords(vu_bar, 0, 0, 0, 14)

    def on_mic_change(event=None):
        stop_vu()
        start_vu_test()

    mic_combo.bind("<<ComboboxSelected>>", on_mic_change)

    btn_frame_mic = ttk.Frame(main, style="Dark.TFrame")
    btn_frame_mic.pack(fill=tk.X, pady=(0, 10))
    ttk.Button(btn_frame_mic, text="Test mic", command=start_vu_test).pack(side=tk.LEFT)
    ttk.Button(btn_frame_mic, text="Stop", command=stop_vu).pack(side=tk.LEFT, padx=5)

    # --- Recognition engine and model ---
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

    # --- Language ---
    ttk.Label(main, text="Language:", style="Dark.TLabel").pack(anchor="w")
    lang_var = tk.StringVar(value=cfg.get("language", "ru"))
    lang_entry = ttk.Entry(main, textvariable=lang_var, width=10, font=("Segoe UI", 9))
    lang_entry.pack(anchor="w", pady=(2, 10))

    # --- Hotkey ---
    ttk.Label(main, text="Toggle key:", style="Dark.TLabel").pack(anchor="w")
    hotkey_var = tk.StringVar(value=cfg.get("toggle_key", "f9"))
    hotkey_entry = ttk.Entry(main, textvariable=hotkey_var, width=15, font=("Segoe UI", 9))
    hotkey_entry.pack(anchor="w", pady=(2, 10))

    # --- GPU ---
    gpu_label = ttk.Label(main, text="GPU: detecting...", style="Dark.TLabel")
    gpu_label.pack(anchor="w", pady=(0, 15))

    def detect_gpu():
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                gpu_label.config(text=f"GPU: {name} (CUDA)")
                return
        except Exception:
            pass
        gpu_label.config(text="GPU: not found (CPU mode)")

    threading.Thread(target=detect_gpu, daemon=True).start()

    # --- Save ---
    def on_save():
        stop_vu()
        sel = mic_combo.current()
        if sel >= 0:
            dev = devices[sel]
            result["device_index"] = dev["index"]
            result["device_name"] = dev["name"]
        result["stt_engine"] = engine_var.get()
        if engine_var.get() == "gigaam":
            result["gigaam_model"] = model_var.get()
        else:
            result["model_size"] = model_var.get()
        result["language"] = lang_var.get()
        result["toggle_key"] = hotkey_var.get()
        confirmed[0] = True
        root.destroy()

    ttk.Button(main, text="Save", command=on_save).pack(pady=(5, 0))

    root.protocol("WM_DELETE_WINDOW", lambda: (stop_vu(), root.destroy()))

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    start_vu_test()
    root.mainloop()

    if not confirmed[0]:
        raise SystemExit("Setup cancelled")

    return result
