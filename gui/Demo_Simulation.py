"""
ARIS UV Nitrate Analyser  -  PPT Demo Simulation
=================================================
Replays real .spz sample spectra through the full 12-step autosampler sequence
automatically, with live PLS predictions.  No hardware required.

Usage:  python "Demo_Simulation.py"
"""

import threading, time, logging
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.signal import savgol_filter
import joblib

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ArisDemo")

# ---- CONFIG ----------------------------------------------------------------
PIXELS      = 2048
WL_START    = 182.98
WL_END      = 988.74
WAVELENGTHS = np.linspace(WL_START, WL_END, PIXELS)

_HERE      = Path(__file__).resolve().parent
_MODELS    = (_HERE / "Pure_PW" / "models") if (_HERE / "Pure_PW" / "models").exists() \
             else (_HERE / "models")
MODEL_PATH = _MODELS / "pls_model.pkl"
WV_PATH    = _MODELS / "wavelengths.npy"
PP_PATH    = _MODELS / "preprocess.pkl"

_DATA_DIRS = [
    _HERE / "Pure_PW" / "N-Pure",
    _HERE / "Pure_PW" / "Re-PW-N",
    _HERE / "Pure_PW" / "Most Recent",
]
_DI_PATH = _HERE / "Pure_PW" / "Re-PW-N" / "Di wasser.spz"

# Autosampler demo sequence: (arduino_tag, delay_s, display_label, hex_colour)
_SEQ = [
    ("SEQUENCE_STARTED",          0.6,  "Sequence started  --  running...",                "#00ff9f"),
    ("STEP 1",                    1.8,  "Step 1  --  DI In  (8s)  [GPA2]",                "#ffcc00"),
    ("STEP 2",                    1.5,  "Step 2  --  Drain  (10s)  [GPA0]",               "#ffaa00"),
    ("STEP 3",                    1.8,  "Step 3  --  DI Reference  (8s)  [GPA2]",         "#ffcc00"),
    ("STATUS: REFERENCE_TRIGGER", 0.3,  "REFERENCE window  --  capturing DI baseline...", "#00ffe8"),
    ("STEP 4",                    1.5,  "Step 4  --  DI Drain  (10s)  [GPA0]",            "#ffaa00"),
    ("STEP 5",                    1.8,  "Step 5  --  Sample flush 1  (10s)  [GPA1]",      "#ffcc00"),
    ("STEP 6",                    1.5,  "Step 6  --  Drain  (8s)  [GPA0]",                "#ffaa00"),
    ("STEP 7",                    0.3,  "Step 7  --  skipped",                            "#7888b8"),
    ("STEP 8",                    1.5,  "Step 8  --  Drain  (8s)  [GPA0]",                "#ffaa00"),
    ("STEP 9",                    1.8,  "Step 9  --  Sample In  (7s)  [GPA1]  << MEASURE","#00ffe8"),
    ("STATUS: MEASURE_NOW",       0.6,  "MEASURING  --  running PLS prediction...",        "#00ffe8"),
    ("STEP 10",                   1.5,  "Step 10  --  Drain  (10s)  [GPA0]",              "#ffaa00"),
    ("STEP 11",                   1.5,  "Step 11  --  DI In  (9s)  [GPA2]",               "#ffcc00"),
    ("STEP 12",                   1.5,  "Step 12  --  Final Drain  (11s)  [GPA0]",        "#ffaa00"),
    ("SEQUENCE_COMPLETE",         0.5,  "Sequence complete  --  ready for next sample",    "#00ff9f"),
]

# ---- COLOURS (same as real app) --------------------------------------------
C_BG0  = "#06080d";  C_BG1  = "#0a0d18";  C_BG2  = "#0c1020";  C_BG3  = "#121626"
C_EDGE = "#253558"
C_TEAL = "#00ffe8";  C_BLUE  = "#38cfff";  C_VIO   = "#d580ff";  C_AMBER = "#ffcc00"
C_RED  = "#ff3b3b";  C_GREEN = "#00ff9f";  C_WARN  = "#ffaa00"
C_REF_LINE  = "#00ff9f";  C_SAMP_LINE = "#ff7a30"
C_TEXT = "#eef2ff";  C_DIM   = "#7888b8";  C_MUTED = "#2a3550"

PLOT_RC = {
    "figure.facecolor": C_BG0, "axes.facecolor":  C_BG2,
    "axes.edgecolor":   "#3a5080", "axes.labelcolor": "#ffffff",
    "axes.labelsize":   9.5,
    "xtick.color": "#ffffff", "ytick.color": "#ffffff",
    "xtick.labelsize": 8.5,   "ytick.labelsize": 8.5,
    "text.color":  "#ffffff", "grid.color": "#1e2d50",
    "grid.linestyle": "-",    "grid.alpha": 1.0,
    "lines.linewidth": 1.8,   "font.family": "monospace",
}


# ---- SPZ FILE READER -------------------------------------------------------
def read_spz(path):
    """
    Parse an Avenir Photonics .spz text file.
    Returns (wavelengths_nm, absorbance) as float64 numpy arrays.
    """
    with open(path, 'r', errors='replace') as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Wavelength"):
            start = i + 1
            break
    if start is None:
        raise ValueError(f"No wavelength header found in {path}")
    wl, ab = [], []
    for line in lines[start:]:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            try:
                wl.append(float(parts[0]))
                ab.append(float(parts[1]))
            except ValueError:
                pass
    return np.array(wl, dtype=np.float64), np.array(ab, dtype=np.float64)


# ---- DEMO LIBRARY ----------------------------------------------------------
class DemoLibrary:
    """Loads all .spz sample files from data directories for playback."""

    _SKIP = {"total", "a", "di wasser"}

    def __init__(self):
        self._samples = []   # list of (name_str, abs_on_WAVELENGTHS_grid float32)
        self._di_abs  = np.zeros(PIXELS, dtype=np.float32)
        self._idx     = 0
        self._load()

    def _load(self):
        if _DI_PATH.exists():
            try:
                wl, ab = read_spz(_DI_PATH)
                self._di_abs = np.interp(WAVELENGTHS, wl, ab).astype(np.float32)
                log.info("DI water reference loaded: %s", _DI_PATH.name)
            except Exception as e:
                log.warning("DI water load failed: %s", e)

        seen = set()
        for d in _DATA_DIRS:
            if not d.exists():
                continue
            for spz in sorted(d.glob("*.spz")):
                if spz.name in seen:
                    continue
                if spz.stem.lower() in self._SKIP:
                    continue
                try:
                    wl, ab = read_spz(spz)
                    arr = np.interp(WAVELENGTHS, wl, ab).astype(np.float32)
                    arr = np.where(np.isfinite(arr), arr, 0.0)
                    self._samples.append((spz.stem, arr))
                    seen.add(spz.name)
                except Exception:
                    pass

        log.info("DemoLibrary: %d samples loaded", len(self._samples))

        if not self._samples:
            log.warning("No .spz samples found; using synthetic fallback")

    def di_spectrum(self):
        return self._di_abs.copy()

    def next(self):
        """Return (name, abs_array), cycling infinitely through all samples."""
        if not self._samples:
            t = self._idx * 0.4
            wl = WAVELENGTHS
            conc = 5.0 + 9.0 * abs(np.sin(t * 0.5))
            ab = (conc * 0.11 * np.exp(-((wl - 220) ** 2) / (2 * 12 ** 2)) + 0.03)
            self._idx += 1
            return f"SYN_{self._idx}", ab.astype(np.float32)
        name, arr = self._samples[self._idx % len(self._samples)]
        self._idx += 1
        return name, arr

    @property
    def total(self):
        return max(len(self._samples), 1)

    @property
    def current_index(self):
        return self._idx


# ---- FILE DRIVER -----------------------------------------------------------
class FileDriver:
    """
    Simulates the ARIS 546 sensor by streaming real .spz absorbance spectra
    with small Gaussian noise to give a live-acquisition feel.
    """
    _NOISE = 0.003

    def __init__(self, library: DemoLibrary):
        self._rng  = np.random.default_rng(7)
        self._base = library.di_spectrum()
        self._lib  = library

    def set_di(self):
        self._base = self._lib.di_spectrum()

    def set_sample(self, abs_arr: np.ndarray):
        self._base = abs_arr.copy()

    def get_spectrum(self, exp_ms=2.0):
        noise = self._rng.normal(0, self._NOISE, PIXELS).astype(np.float32)
        return np.clip(self._base + noise, 0.0, None)

    def close(self):
        pass


# ---- PREDICTOR (unchanged from real app) -----------------------------------
class Predictor:
    def __init__(self):
        self.ready     = False
        self._model    = None
        self._model_wv = None
        self._pp       = {"window": 11, "polyorder": 2, "deriv": 1, "snv": False}
        try:
            self._model    = joblib.load(MODEL_PATH)
            self._model_wv = np.load(WV_PATH)
            if PP_PATH.exists():
                self._pp = joblib.load(PP_PATH)
            overlap = (min(WAVELENGTHS.max(), self._model_wv.max())
                       - max(WAVELENGTHS.min(), self._model_wv.min()))
            if overlap < 10:
                log.error("Model wavelength overlap < 10 nm")
                return
            self.ready = True
            log.info("PLS model loaded  %.1f-%.1f nm  LV=%s  SG1(w=%s)",
                     self._model_wv.min(), self._model_wv.max(),
                     getattr(self._model, 'n_components', '?'),
                     self._pp.get('window', '?'))
        except Exception as e:
            log.warning("Model not loaded: %s", e)

    def predict(self, absorbance: np.ndarray):
        if not self.ready or np.all(absorbance == 0):
            return None
        x = np.interp(self._model_wv, WAVELENGTHS, absorbance)
        x = np.where(np.isfinite(x), x, 0.0)
        pp = self._pp
        if pp.get("snv", False):
            x = (x - x.mean()) / (x.std() + 1e-10)
        x = savgol_filter(x, window_length=pp["window"],
                          polyorder=pp["polyorder"], deriv=pp["deriv"])
        try:
            return max(0.0, float(self._model.predict(x.reshape(1, -1))[0]))
        except Exception:
            return None


# ---- MAIN APP --------------------------------------------------------------
class App:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ARIS UV  -  Nitrate Analyser  [PPT DEMO]")
        self.root.configure(bg=C_BG0)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        # ---- state ----
        self.current_raw    = np.zeros(PIXELS, dtype=np.float32)
        self.dark           = np.zeros(PIXELS, dtype=np.float32)
        self.ref            = None
        self.last_processed = np.zeros(PIXELS, dtype=np.float32)
        self.last_sample    = np.zeros(PIXELS, dtype=np.float32)
        self.is_running     = True
        self._calib_status  = {"ref": False, "samp": False}

        # ---- demo state ----
        self._lib          = DemoLibrary()
        self._pending_abs  = None    # loaded at STEP 9
        self._current_name = "---"

        # ---- tk vars ----
        self.averaging = tk.IntVar(value=5)
        self.smooth_w  = tk.IntVar(value=7)
        self.auto_pred = tk.BooleanVar(value=True)

        # ---- prediction history ----
        self.HIST      = 200
        self._ppm_hist = np.full(self.HIST, np.nan)
        self._t_hist   = np.arange(-self.HIST, 0) * 0.1

        # ---- hardware + model ----
        self._pred = Predictor()
        self.hw    = FileDriver(self._lib)

        # ---- build UI ----
        self._build_window()
        self._build_plots()

        # ---- start sensing thread ----
        threading.Thread(target=self._sense_loop, daemon=True).start()

        # ---- auto-start demo ----
        self.root.after(1500, self._start_sequence)

    # ========================================================================
    # SENSING LOOP
    # ========================================================================

    def _sense_loop(self):
        while self.is_running:
            n = max(1, self.averaging.get())
            frames = [self.hw.get_spectrum() for _ in range(n)]
            frames = [f for f in frames if f is not None]
            if frames:
                self.current_raw = np.mean(frames, axis=0).astype(np.float32)
                self.root.after(0, self._update_ui)
            time.sleep(0.03)

    def _update_ui(self):
        raw = self.current_raw
        w   = self.smooth_w.get()
        if w % 2 == 0:
            w += 1
        abs_v = savgol_filter(raw.copy(), w, 3) if w > 1 else raw.copy()
        self.last_processed = abs_v

        mx = float(raw.max())
        self._lbl_cts.config(text=f"{mx:.3g} AU")
        if mx > 1.9:
            self._lbl_status.config(text="HIGH",     fg=C_WARN)
        elif mx < 1e-5:
            self._lbl_status.config(text="NO SIGNAL",fg=C_RED)
        else:
            self._lbl_status.config(text="LIVE",     fg=C_GREEN)

        self._ln_live.set_ydata(abs_v)
        self._fill_live.remove()
        self._fill_live = self._ax_live.fill_between(
            WAVELENGTHS, 0, np.clip(abs_v, 0, None), color=C_WARN, alpha=0.10)

        idx220 = int(np.argmin(np.abs(WAVELENGTHS - 220)))
        self._lbl_peak.config(text=f"{abs_v[idx220]:.4f} AU")

        if hasattr(self, '_ln_uv'):
            self._ln_uv.set_ydata(abs_v)

        if self.auto_pred.get() and self._pred.ready:
            self._run_pls(abs_v)

        self._canvas.draw_idle()

    def _run_pls(self, abs_v):
        ppm = self._pred.predict(abs_v)
        if ppm is None:
            return
        self._ppm_hist = np.roll(self._ppm_hist, -1)
        self._ppm_hist[-1] = ppm
        self._ln_trend.set_ydata(self._ppm_hist)
        self._fill_trend.remove()
        filled = np.where(np.isfinite(self._ppm_hist), self._ppm_hist, 0.0)
        self._fill_trend = self._ax_trend.fill_between(
            self._t_hist, 0, filled, color=C_AMBER, alpha=0.12)
        valid = self._ppm_hist[np.isfinite(self._ppm_hist)]
        if len(valid):
            self._ax_trend.set_ylim(0, max(float(valid.max()) * 1.3, 5.0))
        self._lbl_ppm.config(text=f"{ppm:.2f}")
        self._bar[0].set_height(ppm)
        self._bar_tx.set_text(f"{ppm:.2f}")
        self._bar_tx.set_y(ppm)

    # ========================================================================
    # DEMO SEQUENCE RUNNER
    # ========================================================================

    def _start_sequence(self):
        """Load next sample and begin the 12-step autosampler sequence."""
        name, abs_arr      = self._lib.next()
        self._current_name = name
        self._pending_abs  = abs_arr
        self.hw.set_di()   # start flowing DI water until Step 9

        n = self._lib.current_index
        t = self._lib.total
        self._lbl_sample.config(text=f"SAMPLE:  {name}")
        self._lbl_counter.config(text=f"{n}  /  {t}")

        self._log("-" * 36)
        self._log(f"AUTO-SEQUENCE  >>  {name}")
        self._log("-" * 36)

        self._run_step(0)

    def _run_step(self, idx: int):
        """Execute step idx of _SEQ, then schedule the next one."""
        if idx >= len(_SEQ):
            return

        tag, delay, label, color = _SEQ[idx]
        self._lbl_step.config(text=label, fg=color)
        self._log(f"[ARD] {tag}")

        # ---- handle events ----
        if tag == "STATUS: REFERENCE_TRIGGER":
            self._capture_ref()

        elif tag == "STEP 9":
            if self._pending_abs is not None:
                self.hw.set_sample(self._pending_abs)   # sample starts flowing

        elif tag == "STATUS: MEASURE_NOW":
            self._do_measure()

        elif tag == "SEQUENCE_COMPLETE":
            self._log(f"[*] Complete  --  {self._current_name}")
            # pause 3 s, then auto-start next sample
            self.root.after(3000, self._start_sequence)
            return

        self.root.after(int(delay * 1000), self._run_step, idx + 1)

    def _capture_ref(self):
        """Freeze current DI spectrum as the reference baseline."""
        self.ref = self.current_raw.copy()
        self._ln_ref.set_ydata(self.ref)
        self._calib_status["ref"] = True
        self._refresh_calib_plot()
        self._log(f"[AUTO] DI reference captured  mean={self.ref.mean():.4g} AU")

    def _do_measure(self):
        """Freeze sample spectrum, run PLS, update all displays."""
        self.last_sample = self.last_processed.copy()
        ppm = self._pred.predict(self.last_sample) if self._pred.ready else None

        if ppm is not None:
            self._lbl_ppm.config(text=f"{ppm:.2f}")
            self._bar[0].set_height(ppm)
            self._bar_tx.set_text(f"{ppm:.2f}")
            self._bar_tx.set_y(ppm)

        self._ln_samp.set_ydata(self.last_sample)
        self._calib_status["samp"] = True
        self._refresh_calib_plot()

        if ppm is not None:
            self._log(f"[RESULT] >>>  {self._current_name}:  NO3 = {ppm:.2f} ppm  <<<")
        else:
            self._log("[AUTO] spectrum frozen  (model not ready)")

    def _refresh_calib_plot(self):
        """Auto-scale calibration overlay axes."""
        segs = []
        if self._calib_status["ref"] and self.ref is not None:
            segs.append(self.ref)
        if self._calib_status["samp"]:
            segs.append(self.last_sample)
        if segs:
            combined = np.concatenate(segs)
            v = combined[np.isfinite(combined)]
            if len(v):
                ymax = float(v.max()) * 1.18
                ymin = float(v.min()) - abs(ymax) * 0.04
                self._ax_calib.set_ylim(min(ymin, -0.02), max(ymax, 0.1))
        badges = (["Ref [v]"] if self._calib_status["ref"]  else []) + \
                 (["Sample [v]"] if self._calib_status["samp"] else [])
        badge_str = "  |  ".join(badges)
        title = f"CALIBRATION SPECTRA   {badge_str}" if badge_str else "CALIBRATION SPECTRA"
        self._ax_calib.set_title(title, color=C_TEXT, fontsize=8.5,
                                  fontweight="bold", loc="left", pad=5)
        if badges:
            self._calib_hint.set_visible(False)
        self._canvas.draw_idle()

    # ========================================================================
    # LOG
    # ========================================================================

    def _log(self, msg: str):
        self._log_box.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}\n")
        self._log_box.see(tk.END)
        log.info(msg)

    # ========================================================================
    # QUIT
    # ========================================================================

    def _quit(self):
        self.is_running = False
        try:
            self.hw.close()
        except Exception:
            pass
        self.root.destroy()

    # ========================================================================
    # UI CONSTRUCTION
    # ========================================================================

    def _build_window(self):
        # ---- top toolbar ----
        tb = tk.Frame(self.root, bg=C_BG1, pady=12, padx=16)
        tb.pack(fill=tk.X)

        left = tk.Frame(tb, bg=C_BG1)
        left.pack(side=tk.LEFT)
        tk.Label(left, text="ARIS UV SPECTROSCOPY",
                 fg=C_TEAL, bg=C_BG1,
                 font=("Courier New", 15, "bold")).pack(anchor="w")
        tk.Label(left, text="[DEMO MODE]  FILE PLAYBACK  --  PPT SIMULATION",
                 fg=C_AMBER, bg=C_BG1,
                 font=("Courier New", 10)).pack(anchor="w")

        mid = tk.Frame(tb, bg=C_BG1)
        mid.pack(side=tk.LEFT, padx=50)
        self._lbl_status = tk.Label(mid, text="LIVE", fg=C_GREEN, bg=C_BG1,
                                    font=("Courier New", 12, "bold"))
        self._lbl_status.grid(row=0, column=0, padx=14)
        self._lbl_cts = tk.Label(mid, text="0 AU", fg=C_TEXT, bg=C_BG1,
                                  font=("Courier New", 11))
        self._lbl_cts.grid(row=1, column=0, padx=14)

        # ---- body ----
        body = tk.Frame(self.root, bg=C_BG0)
        body.pack(fill=tk.BOTH, expand=True)

        plot_area = tk.Frame(body, bg=C_BG0)
        plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._fig    = Figure(figsize=(17.0, 9.0), facecolor=C_BG0)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_area)
        self._canvas.get_tk_widget().configure(bg=C_BG0)

        tb2 = tk.Frame(plot_area, bg=C_BG1)
        tb2.pack(fill=tk.X, side=tk.TOP)
        self._toolbar = NavigationToolbar2Tk(self._canvas, tb2)
        self._toolbar.update()
        self._toolbar.configure(bg=C_BG1)
        for w in self._toolbar.winfo_children():
            try:
                w.configure(bg=C_BG1, fg=C_TEXT, activebackground=C_BG3,
                             activeforeground=C_TEAL, highlightbackground=C_BG1,
                             relief=tk.FLAT)
            except Exception:
                pass
        if hasattr(self._toolbar, "_message_label"):
            self._toolbar._message_label.configure(bg=C_BG1, fg=C_DIM,
                                                    font=("Courier New", 8))

        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

        # ---- right panel ----
        right = tk.Frame(body, bg=C_BG1, width=275)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)
        right.pack_propagate(False)
        self._build_right(right)

    def _build_right(self, p):
        def sep():
            ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ---- NO3 concentration card ----
        card = tk.Frame(p, bg=C_BG3, pady=6)
        card.pack(fill=tk.X, padx=8, pady=(8, 3))
        tk.Label(card, text="NO3-  CONCENTRATION",
                 fg=C_VIO, bg=C_BG3,
                 font=("Courier New", 8, "bold"),
                 anchor="w", padx=10).pack(fill=tk.X)
        self._lbl_ppm = tk.Label(card, text="--",
                                  fg=C_AMBER, bg=C_BG3,
                                  font=("Courier New", 34, "bold"))
        self._lbl_ppm.pack()
        tk.Label(card, text="ppm", fg=C_DIM, bg=C_BG3,
                 font=("Courier New", 9)).pack(pady=(0, 4))

        # ---- absorbance @ 220 nm ----
        c2 = tk.Frame(p, bg=C_BG3, pady=4)
        c2.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(c2, text="ABSORBANCE @ 220 nm",
                 fg=C_VIO, bg=C_BG3,
                 font=("Courier New", 8, "bold"),
                 anchor="w", padx=10).pack(fill=tk.X)
        self._lbl_peak = tk.Label(c2, text="--",
                                   fg=C_TEAL, bg=C_BG3,
                                   font=("Courier New", 16, "bold"))
        self._lbl_peak.pack(pady=(0, 4))

        sep()

        # ---- model status ----
        ok  = "MODEL  LOADED  [R2=0.897]" if self._pred.ready else "MODEL  NOT FOUND"
        col = C_GREEN                      if self._pred.ready else C_RED
        mc = tk.Frame(p, bg=C_BG3, pady=4)
        mc.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(mc, text=ok, fg=col, bg=C_BG3,
                 font=("Courier New", 9, "bold")).pack(padx=10, pady=4)

        sep()

        # ---- current sample card ----
        sc = tk.Frame(p, bg=C_BG3, pady=4)
        sc.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(sc, text="CURRENT SAMPLE",
                 fg=C_BLUE, bg=C_BG3,
                 font=("Courier New", 8, "bold"),
                 anchor="w", padx=10).pack(fill=tk.X)
        self._lbl_sample = tk.Label(sc, text="SAMPLE:  ---",
                                     fg=C_TEXT, bg=C_BG3,
                                     font=("Courier New", 9, "bold"),
                                     anchor="w", padx=10)
        self._lbl_sample.pack(fill=tk.X, pady=(2, 0))
        self._lbl_counter = tk.Label(sc, text="-- / --",
                                      fg=C_DIM, bg=C_BG3,
                                      font=("Courier New", 9),
                                      anchor="w", padx=10)
        self._lbl_counter.pack(fill=tk.X, pady=(0, 4))

        sep()

        # ---- autosampler step ----
        stc = tk.Frame(p, bg=C_BG3, pady=4)
        stc.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(stc, text="AUTOSAMPLER STEP",
                 fg=C_GREEN, bg=C_BG3,
                 font=("Courier New", 8, "bold"),
                 anchor="w", padx=10).pack(fill=tk.X)
        self._lbl_step = tk.Label(stc,
                                   text="STANDBY  --  initialising demo...",
                                   fg=C_DIM, bg=C_BG3,
                                   font=("Courier New", 8),
                                   anchor="w", padx=10,
                                   wraplength=245, justify="left")
        self._lbl_step.pack(fill=tk.X, pady=(2, 6))

        sep()

        # ---- event log ----
        lh = tk.Frame(p, bg=C_BG1)
        lh.pack(fill=tk.X, padx=8)
        tk.Label(lh, text="EVENT LOG",
                 fg=C_TEAL, bg=C_BG1,
                 font=("Courier New", 8, "bold")).pack(side=tk.LEFT, pady=4)
        tk.Button(lh, text="CLEAR",
                  bg="#1a2540", fg=C_TEAL,
                  activebackground=C_BG2, activeforeground="white",
                  font=("Courier New", 7, "bold"), relief=tk.FLAT,
                  cursor="hand2", padx=8, pady=2,
                  command=lambda: self._log_box.delete("1.0", tk.END)
                  ).pack(side=tk.RIGHT, pady=2)

        self._log_box = tk.Text(p, bg="#080b12", fg="#00ffc8",
                                 font=("Courier New", 8), relief=tk.FLAT,
                                 padx=10, pady=6,
                                 insertbackground=C_TEXT)
        self._log_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 6))

    # ========================================================================
    # PLOTS
    # ========================================================================

    def _build_plots(self):
        import matplotlib.pyplot as plt
        plt.rcParams.update(PLOT_RC)
        self._fig.patch.set_facecolor(C_BG0)

        gs = GridSpec(2, 4, figure=self._fig,
                      width_ratios=[1.65, 0.75, 1.3, 1.55],
                      height_ratios=[1.05, 1.0],
                      hspace=0.22, wspace=0.18,
                      left=0.044, right=0.999,
                      top=0.968, bottom=0.058)

        self._ax_live  = self._fig.add_subplot(gs[0, :3])
        self._ax_uv    = self._fig.add_subplot(gs[0, 3])
        self._ax_calib = self._fig.add_subplot(gs[1, :2])
        self._ax_trend = self._fig.add_subplot(gs[1, 2])
        self._ax_bar   = self._fig.add_subplot(gs[1, 3])

        def _style(ax):
            ax.grid(True, color="#1e2d50", linewidth=0.8, alpha=1.0)
            ax.tick_params(colors="#ffffff", labelcolor="#ffffff",
                           labelsize=9, length=5, width=1.2, pad=4)
            ax.xaxis.label.set_color("#ffffff")
            ax.yaxis.label.set_color("#ffffff")
            for sp in ax.spines.values():
                sp.set_edgecolor("#3a5080")
                sp.set_linewidth(1.3)
            ax.set_facecolor(C_BG2)
        for ax in self._fig.get_axes():
            _style(ax)

        # ---- live spectrum (full range) ----
        ax = self._ax_live
        ax.set_title("LIVE ABSORBANCE  [Direct  -  device output]",
                     color=C_WARN, fontsize=9.5, fontweight="bold", loc="left", pad=3)
        ax.set_xlabel("Wavelength (nm)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_ylabel("Absorbance (AU)",  fontsize=9, color="#ffffff", labelpad=4)
        ax.set_xlim(WL_START, WL_END)
        ax.set_ylim(-0.05, 2.0)
        self._ln_live, = ax.plot(WAVELENGTHS, np.zeros(PIXELS), color=C_TEAL, lw=2.0)
        self._fill_live = ax.fill_between(WAVELENGTHS, 0, 0, color=C_WARN, alpha=0.10)
        ax.axvspan(200, 240, color=C_VIO, alpha=0.06, zorder=0)
        ax.axvline(220, color=C_VIO,   lw=1.0, ls="--", alpha=0.7, label="NO3- 220 nm")
        ax.axvline(210, color="#ffcc00", lw=1.0, ls="--", alpha=0.7, label="NO2- 210 nm")
        ax.legend(fontsize=8, facecolor=C_BG1, labelcolor="#ffffff",
                  framealpha=0.95, edgecolor="#3a5080", loc="upper right")

        # ---- UV zoom (200-350 nm) ----
        ax = self._ax_uv
        ax.set_title("UV  200-350 nm", color=C_VIO, fontsize=9, fontweight="bold",
                     loc="left", pad=3)
        ax.set_xlabel("Wavelength (nm)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_ylabel("AU",              fontsize=9, color="#ffffff", labelpad=4)
        ax.set_xlim(200, 350)
        ax.set_ylim(-0.05, 2.0)
        self._ln_uv, = ax.plot(WAVELENGTHS, np.zeros(PIXELS), color=C_VIO, lw=2.0)
        ax.axvline(220, color=C_TEAL,    lw=1.0, ls="--", alpha=0.7)
        ax.axvline(210, color="#ffcc00", lw=1.0, ls="--", alpha=0.6)
        ax.axvspan(200, 240, color=C_VIO, alpha=0.08, zorder=0)

        # ---- calibration overlay (ref + sample) ----
        ax = self._ax_calib
        ax.set_title("CALIBRATION SPECTRA", color="#ffffff", fontsize=9.5,
                     fontweight="bold", loc="left", pad=3)
        ax.set_xlabel("Wavelength (nm)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_ylabel("Absorbance (AU)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_xlim(WL_START, WL_END)
        ax.set_ylim(0, 1.0)
        self._ln_ref,  = ax.plot(WAVELENGTHS, np.zeros(PIXELS),
                                  color=C_REF_LINE,  lw=2.0, label="DI Reference  (Step 3)")
        self._ln_samp, = ax.plot(WAVELENGTHS, np.zeros(PIXELS),
                                  color=C_SAMP_LINE, lw=2.0, label="Sample  (Step 9)")
        ax.axvline(220, color=C_VIO,     lw=0.8, ls="--", alpha=0.5)
        ax.axvline(210, color="#ffcc00", lw=0.8, ls="--", alpha=0.4)
        ax.axvspan(200, 240, color=C_VIO, alpha=0.04, zorder=0)
        ax.legend(fontsize=7.5, facecolor=C_BG1, labelcolor=C_TEXT,
                  framealpha=0.95, edgecolor=C_EDGE, loc="upper right")
        self._calib_hint = ax.text(
            0.5, 0.48,
            "Waiting for autosampler sequence...",
            transform=ax.transAxes, ha="center", va="center",
            color=C_DIM, fontsize=8, style="italic")

        # ---- nitrate trend ----
        ax = self._ax_trend
        ax.set_title("NITRATE TREND", color=C_AMBER, fontsize=9.5,
                     fontweight="bold", loc="left", pad=3)
        ax.set_xlabel("Time (s)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_ylabel("ppm",      fontsize=9, color="#ffffff", labelpad=4)
        ax.set_xlim(self._t_hist[0], 0)
        ax.set_ylim(0, 30)
        self._ln_trend,  = ax.plot(self._t_hist, self._ppm_hist,
                                    color=C_AMBER, lw=1.4)
        self._fill_trend  = ax.fill_between(
            self._t_hist, 0,
            np.where(np.isfinite(self._ppm_hist), self._ppm_hist, 0.0),
            color=C_AMBER, alpha=0.15)

        # ---- concentration bar ----
        ax = self._ax_bar
        ax.set_title("NITRATE NOW", color=C_AMBER, fontsize=9.5,
                     fontweight="bold", loc="left", pad=3)
        ax.set_ylabel("ppm", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_xticks([])
        ax.set_ylim(0, 30)
        self._bar    = ax.bar(["NO3-"], [0], color=C_VIO, width=0.5,
                               zorder=3, edgecolor=C_EDGE)
        self._bar_tx = ax.text(0, 0.5, "--",
                                ha="center", va="bottom",
                                color=C_TEXT, fontsize=15,
                                fontweight="bold", fontfamily="monospace")

        self._canvas.draw()

    # ========================================================================
    # SCROLL ZOOM
    # ========================================================================

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        f  = 0.85 if event.button == "up" else 1.15
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xc = event.xdata if event.xdata else (xl[0] + xl[1]) / 2
        yc = event.ydata if event.ydata else (yl[0] + yl[1]) / 2
        ax.set_xlim(xc - (xc - xl[0]) * f, xc + (xl[1] - xc) * f)
        ax.set_ylim(yc - (yc - yl[0]) * f, yc + (yl[1] - yc) * f)
        self._canvas.draw_idle()


# ---- ENTRY POINT -----------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1600x950")
    root.state("zoomed")
    App(root)
    root.mainloop()
