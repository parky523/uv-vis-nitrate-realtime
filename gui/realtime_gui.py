"""
ARIS UV Nitrate Analyser
==========================
Based on the working InspectiveUI driver and acquisition logic.
Premium dark instrument GUI with live absorbance, sample analysis,
scroll-to-zoom, PLS prediction, averaging, and CSV export.
"""

import threading, time, struct, csv, logging
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.signal import savgol_filter
import joblib

import usb.core, usb.util

import queue
try:
    import serial
    _SERIAL_OK = True
except ImportError:
    serial = None
    _SERIAL_OK = False

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Aris")

import sys as _sys
ARDUINO_PORT = "COM10" if _sys.platform == "win32" else "/dev/ttyACM0"
ARDUINO_BAUD = 9600
TRIGGER_REF  = "STATUS: REFERENCE_TRIGGER"
TRIGGER_MSG  = "STATUS: MEASURE_NOW"

# -
# CONFIG
# -
PIXELS      = 2048
ADC_MAX     = 65535
WL_START    = 182.98
WL_END      = 988.74
WAVELENGTHS = np.linspace(WL_START, WL_END, PIXELS)

# Paths  -  auto-finds models/ whether run on Windows (Pure_PW/models/) or Pi (models/)
_HERE   = Path(__file__).resolve().parent
_MODELS = (_HERE / "Pure_PW" / "models") if (_HERE / "Pure_PW" / "models").exists() \
          else (_HERE / "models")
MODEL_PATH = _MODELS / "pls_model.pkl"
WV_PATH    = _MODELS / "wavelengths.npy"
PP_PATH    = _MODELS / "preprocess.pkl"

# -
# 1.  DRIVER   -  original working ArisDriver, unchanged
# -

class ArisDriver:
    def __init__(self):
        self.dev = usb.core.find(idVendor=0x354F)
        if self.dev is None:
            raise ConnectionError("Aris 546 Not Found. Check USB connection.")
        self.dev.set_configuration()
        self.pixels = PIXELS
        log.info("Aris 546 online")

    def get_spectrum(self, exp_ms=2.0):
        try:
            self.dev.write(0x01, struct.pack('<II', 0x1100, int(exp_ms * 1000)))
            self.dev.write(0x01, struct.pack('<II', 0x0004, 1))
            ready = False
            for _ in range(100):
                self.dev.write(0x01, struct.pack('<I', 0x3000) + b'\x00' * 4)
                reply = self.dev.read(0x81, 16, timeout=500)
                if len(reply) >= 8:
                    if (struct.unpack('<I', reply[4:8])[0] >> 8) & 0xFFFF > 0:
                        ready = True
                        break
                time.sleep(0.005)
            if not ready:
                return None
            self.dev.write(0x01, struct.pack('<I', 0x4000) + b'\x00' * 4)
            data_reply = self.dev.read(0x81, 16492, timeout=2000)
            if len(data_reply) >= 8260:
                return np.frombuffer(
                    data_reply[68:68 + self.pixels * 4], dtype=np.float32
                ).copy()
            return None
        except Exception:
            return None

    def close(self):
        try:
            usb.util.dispose_resources(self.dev)
        except Exception:
            pass


class SyntheticDriver:
    """Realistic demo driver when hardware is unavailable."""
    def __init__(self):
        self._rng   = np.random.default_rng(42)
        self._frame = 0
        log.info("Synthetic driver active (demo mode)")

    def get_spectrum(self, exp_ms=2.0):
        self._frame += 1
        wl = WAVELENGTHS
        t  = self._frame * 0.05
        ref = (
            30000 * np.exp(-((wl - 500) ** 2) / (2 * 150 ** 2))
            + 8000 * np.exp(-((wl - 300) ** 2) / (2 * 80  ** 2))
            + 2000
        )
        conc  = 5.0 + 4.0 * np.sin(t * 0.3)
        a_no3 = 0.8 * conc * np.exp(-((wl - 220) ** 2) / (2 * 12 ** 2))
        a_no2 = 0.20       * np.exp(-((wl - 210) ** 2) / (2 * 8  ** 2))
        sample = ref * 10 ** (-(a_no3 + a_no2 + 0.05))
        return np.clip(
            sample + self._rng.normal(0, 20, PIXELS), 0, ADC_MAX
        ).astype(np.float32)

    def close(self):
        pass


# -
# AUTOSAMPLER SERIAL THREAD
# -

STEP_LABELS = {
    "SEQUENCE_STARTED":          "Sequence started  --  running...",
    "CLEANUP":                   "CLEANUP  --  Post-abort drain  (8s)  [GPA0]",
    "STEP 1":                    "Step 1   --  DI flush  (7s)  [GPA2]",
    "STEP 2":                    "Step 2   --  Drain  (10s)  [GPA0]",
    "STEP 3":                    "Step 3   --  DI Reference fill  (7s)  [GPA2]",
    "STATUS: REFERENCE_TRIGGER": "REFERENCE window  (15s)  --  capturing DI baseline...",
    "STEP 4":                    "Step 4   --  DI Drain  (10s)  [GPA0]",
    "STEP 5":                    "Step 5   --  Sample flush  (8s)  [GPA1]",
    "STEP 6":                    "Step 6   --  Drain  (10s)  [GPA0]",
    "STEP 7":                    "Step 7   --  skipped",
    "STEP 8":                    "Step 8   --  Drain  (10s)  [GPA0]",
    "STEP 9":                    "Step 9   --  Sample fill  (7s)  [GPA1]  << MEASURE",
    "STATUS: MEASURE_NOW":       "MEASURE window  (15s)  --  running PLS prediction...",
    "STEP 10":                   "Step 10  --  Drain  (10s)  [GPA0]",
    "STEP 11":                   "Step 11  --  DI rinse  (7s)  [GPA2]",
    "STEP 12":                   "Step 12  --  Final Drain  (15s)  [GPA0]",
    "SEQUENCE_COMPLETE":         "Sequence complete  --  ready for next sample",
    "CLEANUP_DRAIN":             "Post-abort cleanup drain  (8s)  [GPA0]",
    "STATUS: ABORTED":           "ABORTED  --  cleanup drain will run next trigger",
    "STATUS: STANDBY":           "STANDBY  --  flip LEFT or press START",
}


class ArduinoThread(threading.Thread):
    def __init__(self, port, baud, msg_queue, trigger_cb, ref_trigger_cb=None):
        super().__init__(daemon=True)
        self.port            = port
        self.baud            = baud
        self.q               = msg_queue
        self.trigger_cb      = trigger_cb
        self.ref_trigger_cb  = ref_trigger_cb
        self._stop_evt       = threading.Event()
        self._cmd_q          = queue.Queue()
        self.ser             = None

    def send_cmd(self, cmd):
        self._cmd_q.put(cmd)

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(0.5)
            self.ser.reset_input_buffer()
            self.q.put(("open", f"Connected  {self.port}  {self.baud} baud"))
            ref_triggered = False
            triggered     = False
            while not self._stop_evt.is_set():
                # flush pending commands to Arduino
                try:
                    while True:
                        cmd = self._cmd_q.get_nowait()
                        self.ser.write((cmd + "\n").encode())
                except queue.Empty:
                    pass

                if self.ser.in_waiting:
                    try:
                        line = self.ser.readline().decode("utf-8", errors="replace").strip()
                    except Exception:
                        continue
                    if line:
                        self.q.put(("line", line))
                        if not ref_triggered and TRIGGER_REF in line:
                            ref_triggered = True
                            if self.ref_trigger_cb:
                                threading.Thread(target=self.ref_trigger_cb,
                                                 daemon=True).start()
                        if not triggered and TRIGGER_MSG in line:
                            triggered = True
                            threading.Thread(target=self.trigger_cb,
                                             daemon=True).start()
                        if "SEQUENCE_COMPLETE" in line:
                            ref_triggered = False   # reset for next run
                            triggered     = False
                            self.q.put(("done", None))
                        if "STATUS: ABORTED" in line or "STATUS: EMERGENCY_STOP" in line:
                            ref_triggered = False
                            triggered     = False
                            self.q.put(("aborted", None))
                        if "STATUS: SEQUENCE_STARTED" in line:
                            ref_triggered = False
                            triggered     = False
                            self.q.put(("seq_started", None))
                else:
                    time.sleep(0.02)
        except Exception as exc:
            self.q.put(("err", str(exc)))
        finally:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass

    def stop(self):
        self._stop_evt.set()
        try:
            if self.ser and self.ser.is_open:
                self.ser.write(b"off\n")
        except Exception:
            pass


# -
# 2.  PLS PREDICTOR
# -

class Predictor:
    def __init__(self):
        self.ready     = False
        self._model    = None
        self._model_wv = None
        self._pp       = {"window": 11, "polyorder": 2, "deriv": 1}  # SG1 defaults
        try:
            self._model    = joblib.load(MODEL_PATH)
            self._model_wv = np.load(WV_PATH)
            if PP_PATH.exists():
                self._pp = joblib.load(PP_PATH)
            overlap = (
                min(WAVELENGTHS.max(), self._model_wv.max())
                - max(WAVELENGTHS.min(), self._model_wv.min())
            )
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

    def predict(self, absorbance):
        if not self.ready:
            return None
        if np.all(absorbance == 0):
            return None
        # 1. interpolate to model wavelength grid
        x = np.interp(self._model_wv, WAVELENGTHS, absorbance)
        x = np.where(np.isfinite(x), x, 0.0)
        # 2. apply same preprocessing as training (SNV if used, then SG1)
        from scipy.signal import savgol_filter as _sg
        pp = self._pp
        if pp.get("snv", False):
            x = (x - x.mean()) / (x.std() + 1e-10)
        x = _sg(x, window_length=pp["window"],
                polyorder=pp["polyorder"], deriv=pp["deriv"])
        try:
            return max(0.0, float(self._model.predict(x.reshape(1, -1))[0]))
        except Exception:
            return None


# -
# 3.  COLOUR PALETTE
# -

# Background layers
C_BG0   = "#06080d"   # deepest  -  window bg
C_BG1   = "#0a0d18"   # toolbar / panels
C_BG2   = "#0c1020"   # plot facecolor  (slightly lighter for contrast)
C_BG3   = "#121626"   # card bg

# Borders
C_EDGE  = "#253558"   # brighter edge for grid visibility

# Accents   -  vivid neon-on-dark palette
C_TEAL  = "#00ffe8"   # live spectrum   -  electric cyan (brighter)
C_BLUE  = "#38cfff"   # sample spectrum  -  sky blue
C_VIO   = "#d580ff"   # UV / NO3  -  bright violet (brighter)
C_AMBER = "#ffcc00"   # trend  -  bright gold (purer yellow)
C_RED   = "#ff3b3b"   # errors  -  vivid red
C_GREEN = "#00ff9f"   # success  -  neon mint
C_WARN  = "#ffaa00"   # warnings  -  amber

# Calibration overlay colours
C_DARK_LINE = "#b0becc"   # dark capture  -  bright silver
C_REF_LINE  = "#00ff9f"   # DI reference  -  neon mint
C_SAMP_LINE = "#ff7a30"   # sample  -  vivid orange

# Text
C_TEXT  = "#eef2ff"   # near-white
C_DIM   = "#7888b8"   # muted labels (brighter)
C_MUTED = "#2a3550"

PLOT_RC = {
    "figure.facecolor": C_BG0,
    "axes.facecolor":   C_BG2,
    "axes.edgecolor":   "#3a5080",
    "axes.labelcolor":  "#ffffff",   # pure white axis labels
    "axes.labelsize":   9.5,
    "xtick.color":      "#ffffff",   # pure white tick labels
    "ytick.color":      "#ffffff",
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8.5,
    "text.color":       "#ffffff",
    "grid.color":       "#1e2d50",
    "grid.linestyle":   "-",
    "grid.alpha":       1.0,
    "lines.linewidth":  1.8,
    "font.family":      "monospace",
}


# -
# 4.  MAIN APPLICATION
# -

class App:

    def __init__(self, root):
        self.root = root
        self.root.title("ARIS UV   -   Nitrate Analyser")
        self.root.configure(bg=C_BG0)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        # - state -
        self.current_raw    = np.zeros(PIXELS, dtype=np.float32)
        self.dark           = None
        self.ref            = None
        self.last_processed = np.zeros(PIXELS)   # last absorbance shown
        self.last_sample    = np.zeros(PIXELS)   # frozen sample snapshot
        self.is_running     = True

        # - autosampler -
        self._ard_thread    = None
        self._ard_q         = queue.Queue()
        self._ard_running   = False
        self._current_mode  = 1   # 1=auto-sample  2=gui-trigger  3=manual
        self._mode2_warned  = False

        # - user-adjustable settings -
        self.exp_time     = tk.DoubleVar(value=10.0)
        self.averaging    = tk.IntVar(value=50)
        self.smooth_w     = tk.IntVar(value=7)
        self.sub_dark_var = tk.BooleanVar(value=True)
        self.auto_pred    = tk.BooleanVar(value=True)
        self.direct_abs   = tk.BooleanVar(value=True)    # device streams absorbance directly

        # - prediction history -
        self.HIST = 200
        self._ppm_hist = np.full(self.HIST, np.nan)
        self._t_hist   = np.arange(-self.HIST, 0) * 0.1

        # - validation data -
        self._val_results   = []   # (known_ppm, measured_ppm)
        self._blank_results = []   # measured_ppm on DI blanks
        self._rep_results   = []   # measured_ppm from repeatability run
        self._ref_results   = []   # (our_ppm, reference_ppm)

        # - hardware -
        self._pred = Predictor()
        try:
            self.hw   = ArisDriver()
            self._hw_label = "[*] HARDWARE ONLINE"
            self._hw_color = C_GREEN
        except Exception as e:
            self.hw   = SyntheticDriver()
            self._hw_label = "[*] DEMO MODE"
            self._hw_color = C_WARN
            log.warning("Demo mode: %s", e)

        # - build UI -
        self._build_window()
        self._build_plots()

        # - start sensing thread -
        threading.Thread(target=self._sense_loop, daemon=True).start()
        self._auto_connect_arduino()
        self.root.after(100, self._poll_arduino)
        self.root.after(200, self._launch_dialogs)

    # -
    # SENSING THREAD   -  identical logic to original sensing_thread
    # -

    def _sense_loop(self):
        while self.is_running:
            n       = max(1, self.averaging.get())
            samples = []
            for _ in range(n):
                d = self.hw.get_spectrum(self.exp_time.get())
                if d is not None:
                    samples.append(d)
            if samples:
                self.current_raw = np.mean(samples, axis=0).astype(np.float32)
                self.root.after(0, self._update_ui)
            time.sleep(0.02)

    # -
    # UI UPDATE   -  runs in main thread via root.after()
    # -

    def _update_ui(self):
        raw   = self.current_raw
        maxc  = float(raw.max())
        meanc = float(raw.mean())

        # Status  -  threshold adapts to float vs integer count range
        self._lbl_cts.config(text=f"{maxc:.3g} cts")
        self._lbl_mean.config(text=f"{meanc:.3g} mean")

        if maxc > 64000:
            self._lbl_status.config(text="SATURATED", fg=C_RED)
        elif maxc < 1e-6:
            self._lbl_status.config(text="NO SIGNAL", fg=C_WARN)
        else:
            self._lbl_status.config(text="LIVE", fg=C_GREEN)

        w = self.smooth_w.get()
        if w % 2 == 0:
            w += 1

        # - Direct-ABS mode: device already returns absorbance -
        if self.direct_abs.get():
            abs_v = raw.copy()
            if w > 1:
                abs_v = savgol_filter(abs_v, w, 3)
            self.last_processed = abs_v
            self._ax_live.set_title("LIVE ABSORBANCE  [Direct  -  device output]",
                                    color=C_WARN, fontsize=8, fontweight="bold",
                                    loc="left", pad=3)
            self._ax_live.set_ylim(-0.05, 2.0)
            self._ln_live.set_ydata(abs_v)
            try:
                self._fill_live.remove()
            except Exception:
                pass
            self._fill_live = self._ax_live.fill_between(
                WAVELENGTHS, 0, np.clip(abs_v, 0, None),
                color=C_WARN, alpha=0.10
            )
            idx = int(np.argmin(np.abs(WAVELENGTHS - 220)))
            self._lbl_peak.config(text=f"{abs_v[idx]:.4f} AU")
            if self.auto_pred.get() and self._pred.ready:
                self._run_pls(abs_v)
            self._canvas.draw_idle()
            return

        # - Beer-Lambert mode -
        if self.ref is not None:
            dark_sub = self.dark if (self.sub_dark_var.get() and self.dark is not None) \
                       else np.zeros(PIXELS)
            denom = self.ref - dark_sub
            denom = np.where(np.abs(denom) < 1e-7, 1e-7, denom)
            ratio = (raw - dark_sub) / denom
            abs_v = -np.log10(np.clip(ratio, 0.001, 10.0))
            if w > 1:
                abs_v = savgol_filter(abs_v, w, 3)
            self.last_processed = abs_v

            self._ln_live.set_ydata(abs_v)
            try:
                self._fill_live.remove()
            except Exception:
                pass
            self._fill_live = self._ax_live.fill_between(
                WAVELENGTHS, 0, np.clip(abs_v, 0, None),
                color=C_TEAL, alpha=0.08
            )

            idx = int(np.argmin(np.abs(WAVELENGTHS - 220)))
            self._lbl_peak.config(text=f"{abs_v[idx]:.4f} AU")

            if self.auto_pred.get() and self._pred.ready:
                self._run_pls(abs_v)
        else:
            self._ln_live.set_ydata(raw)
            yhi = max(float(raw.max()) * 1.1, 0.01)
            self._ax_live.set_ylim(0, yhi)

        self._canvas.draw_idle()

    def _run_pls(self, abs_v):
        ppm = self._pred.predict(abs_v)
        if ppm is not None:
            self._ppm_hist = np.roll(self._ppm_hist, -1)
            self._ppm_hist[-1] = ppm
            self._ln_trend.set_ydata(self._ppm_hist)
            try:
                self._fill_trend.remove()
            except Exception:
                pass
            filled = np.where(np.isfinite(self._ppm_hist), self._ppm_hist, 0.0)
            self._fill_trend = self._ax_trend.fill_between(
                self._t_hist, 0, filled, color=C_AMBER, alpha=0.12
            )
            valid = self._ppm_hist[np.isfinite(self._ppm_hist)]
            if len(valid):
                ymax = max(float(valid.max()) * 1.3, 5.0)
                self._ax_trend.set_ylim(0, ymax)
            self._lbl_ppm.config(text=f"{ppm:.2f}")
            self._bar[0].set_height(ppm)
            self._bar_tx.set_text(f"{ppm:.2f}")
            self._bar_tx.set_y(ppm)

    # -
    # CALIBRATION ACTIONS
    # -

    def _refresh_calib_plot(self):
        """Auto-scale calibration overlay to UV region and update title status badges."""
        # scale y-axis to the UV region (200-400nm) where nitrate absorbs
        uv_mask = (WAVELENGTHS >= 200) & (WAVELENGTHS <= 400)
        all_segs = []
        if self._calib_status["dark"] and self.dark is not None:
            all_segs.append(self.dark[uv_mask])
        if self._calib_status["ref"] and self.ref is not None:
            all_segs.append(self.ref[uv_mask])
        if self._calib_status["samp"]:
            all_segs.append(self.last_sample[uv_mask])
            diff = self.last_sample - (self.ref if self.ref is not None
                                       else np.zeros(PIXELS))
            all_segs.append(diff[uv_mask])
        if all_segs:
            combined = np.concatenate(all_segs)
            valid = combined[np.isfinite(combined)]
            if len(valid):
                ymax = float(valid.max()) * 1.25
                ymin = float(valid.min()) - abs(float(valid.max())) * 0.10
                self._ax_calib.set_ylim(min(ymin, -0.05), max(ymax, 0.1))
        badges = []
        if self._calib_status["dark"]: badges.append("Dark [v]")
        if self._calib_status["ref"]:  badges.append("Ref [v]")
        if self._calib_status["samp"]: badges.append("Sample [v]")
        status = "  |  ".join(badges) if badges else ""
        title = f"CALIBRATION SPECTRA   {status}" if status else "CALIBRATION SPECTRA"
        self._ax_calib.set_title(title, color=C_TEXT, fontsize=8.5,
                                  fontweight="bold", loc="left", pad=5)
        if badges:
            self._calib_hint.set_visible(False)
        self._canvas.draw_idle()

    def _cap_dark(self):
        self.dark = self.current_raw.copy()
        mean = float(self.dark.mean())
        idx220 = int(np.argmin(np.abs(WAVELENGTHS - 220)))
        self._btn_dark.config(text=f"DARK [v]  {mean:.3f}", bg="#1a3530")
        self._ln_dark.set_ydata(self.dark)
        self._calib_status["dark"] = True
        self._refresh_calib_plot()
        self._log(f"Dark captured  mean={mean:.4f} AU  @220nm={self.dark[idx220]:.4f} AU")

    def _cap_ref(self):
        self.ref = self.current_raw.copy()
        mean = float(self.ref.mean())
        idx220 = int(np.argmin(np.abs(WAVELENGTHS - 220)))
        au220  = float(self.ref[idx220])
        self._btn_ref.config(text=f"REF [v] {au220:.3f}AU@220", bg="#1a3a3a")
        self._ax_live.set_ylim(-0.05, 2.0)
        self._ax_live.set_title("LIVE ABSORBANCE", color=C_TEAL,
                                 fontsize=9, fontweight="bold", loc="left")
        self._update_ref_plot(mean)
        self._log(f"DI Reference captured  mean={mean:.4f} AU  @220nm={au220:.4f} AU")

    def _cap_sample(self):
        if self.ref is None and not self.direct_abs.get():
            messagebox.showwarning("Reference Missing",
                                   "Capture reference (DI water) first.")
            return
        self.last_sample = self.last_processed.copy()
        idx220 = int(np.argmin(np.abs(WAVELENGTHS - 220)))
        au220  = float(self.last_sample[idx220])
        if self._pred.ready:
            ppm = self._pred.predict(self.last_sample)
            if ppm is not None:
                self._bar[0].set_height(ppm)
                self._bar_tx.set_text(f"{ppm:.2f}")
                self._bar_tx.set_y(ppm)
                self._lbl_ppm.config(text=f"{ppm:.2f}")
                self._freeze_sample_plot(ppm)
                self._log(f"Sample analyzed  NO3={ppm:.2f} ppm  @220nm={au220:.4f} AU")
                return
        self._freeze_sample_plot()
        self._log(f"Sample captured  @220nm={au220:.4f} AU  (model not ready)")

    def _save_csv(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"spectrum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        if not p:
            return
        use_processed = self.ref is not None or self.direct_abs.get()
        data = self.last_processed if use_processed else self.current_raw
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wavelength_nm", "value", "raw_counts"])
            for wl, v, r in zip(WAVELENGTHS, data, self.current_raw):
                w.writerow([f"{wl:.4f}", f"{v:.6f}", f"{r:.1f}"])
        self._log(f"Saved -> {Path(p).name}")

    def _run_diag(self):
        """Full calibration validation diagnostic."""
        raw    = self.current_raw
        probes = [210, 215, 220, 250, 300, 380]
        self._log("=" * 36)
        self._log("VALIDATION DIAGNOSTIC")
        self._log("=" * 36)
        self._log(f"  Mode: {'Direct-ABS' if self.direct_abs.get() else 'Beer-Lambert'}")
        self._log(f"  PLS model: {'READY' if self._pred.ready else 'NOT LOADED'}")
        self._log(f"  Dark:  {'captured' if self.dark is not None else 'NOT captured'}")
        self._log(f"  Ref:   {'captured' if self.ref  is not None else 'NOT captured'}")
        self._log(f"  Sample:{'captured' if self._calib_status['samp'] else 'NOT captured'}")
        self._log("-- LIVE absorbance (UV range) --")
        for wl in probes:
            idx = int(np.argmin(np.abs(WAVELENGTHS - wl)))
            r_v = f"{raw[idx]:.4f}"
            d_v = f"{self.dark[idx]:.4f}" if self.dark is not None else "  --  "
            f_v = f"{self.ref[idx]:.4f}"  if self.ref  is not None else "  --  "
            s_v = f"{self.last_sample[idx]:.4f}" if self._calib_status["samp"] else "  --  "
            self._log(f"  {wl}nm  live={r_v}  dark={d_v}  ref={f_v}  samp={s_v}")
        if self.ref is not None and self._calib_status["samp"]:
            idx220 = int(np.argmin(np.abs(WAVELENGTHS - 220)))
            diff   = self.last_sample[idx220] - self.ref[idx220]
            self._log(f"-- Differential @220nm = {diff:.4f} AU (samp-ref)")
            if self._pred.ready:
                ppm = self._pred.predict(self.last_sample)
                self._log(f"-- PLS prediction = {ppm:.2f} ppm" if ppm is not None
                          else "-- PLS prediction = FAILED")
        self._log("=" * 36)

    # -------------------------------------------------------
    # VALIDATION METHODS  (Mode 4 / ISO 17025)
    # -------------------------------------------------------

    def _run_val_standard(self):
        if not self._pred.ready:
            self._log("[VAL] PLS not ready"); return
        if self.last_sample is None or not self._calib_status.get("samp"):
            self._log("[VAL] No sample captured — run ANALYZE first"); return
        known   = float(self._val_known_ppm.get())
        measured = self._pred.predict(self.last_processed)
        if measured is None:
            self._log("[VAL] Prediction failed"); return
        error_pct = (measured - known) / known * 100 if known != 0 else float("nan")
        self._val_results.append((known, measured))
        n = len(self._val_results)
        self._lbl_val_result.config(
            text=f"n={n}  meas={measured:.2f}  err={error_pct:+.1f}%")
        self._log(f"[VAL] known={known:.2f}  meas={measured:.2f}  err={error_pct:+.1f}%")
        if len(self._val_results) >= 3:
            kn = np.array([x[0] for x in self._val_results])
            ms = np.array([x[1] for x in self._val_results])
            m_, b_ = np.polyfit(kn, ms, 1)
            ss_res = np.sum((ms - (m_ * kn + b_)) ** 2)
            ss_tot = np.sum((ms - ms.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            self._log(f"[VAL] Linearity  n={len(kn)}  R²={r2:.4f}  slope={m_:.4f}  int={b_:+.3f}")

    def _run_repeatability(self, n: int = 5):
        if not self._pred.ready:
            self._log("[VAL] PLS not ready"); return
        if not self._calib_status.get("samp"):
            self._log("[VAL] No sample captured — run ANALYZE first"); return
        vals = [self._pred.predict(self.last_processed) for _ in range(n)]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            self._log("[VAL] Not enough predictions"); return
        m_, s_ = float(np.mean(vals)), float(np.std(vals, ddof=1))
        rsd = s_ / m_ * 100 if m_ != 0 else float("nan")
        self._rep_results.extend(vals)
        self._log(f"[VAL] Repeatability  n={len(vals)}  mean={m_:.2f}  "
                  f"SD={s_:.3f}  RSD={rsd:.1f}%  "
                  f"{'PASS' if rsd < 5 else 'FAIL(>5%)'}")
        self._lbl_val_result.config(
            text=f"RepeatRSD={rsd:.1f}% ({'PASS' if rsd < 5 else 'FAIL'})")

    def _run_blank_lod(self, n: int = 10):
        if not self._pred.ready:
            self._log("[VAL] PLS not ready"); return
        if not self._calib_status.get("samp"):
            self._log("[VAL] Capture a DI blank as the sample first"); return
        vals = [self._pred.predict(self.last_processed) for _ in range(n)]
        vals = [v for v in vals if v is not None]
        if len(vals) < 3:
            self._log("[VAL] Not enough blank predictions"); return
        s_  = float(np.std(vals, ddof=1))
        lod = 3.0  * s_
        loq = 10.0 * s_
        self._blank_results.extend(vals)
        self._log(f"[VAL] Blank LOD/LOQ  n={len(vals)}  SD={s_:.4f}  "
                  f"LOD={lod:.3f}ppm  LOQ={loq:.3f}ppm")
        self._lbl_val_result.config(
            text=f"LOD={lod:.3f}  LOQ={loq:.3f}ppm")

    def _add_ref_point(self):
        if not self._pred.ready:
            self._log("[VAL] PLS not ready"); return
        if not self._calib_status.get("samp"):
            self._log("[VAL] No sample captured — run ANALYZE first"); return
        ref_val  = float(self._val_ref_var.get())
        our_val  = self._pred.predict(self.last_processed)
        if our_val is None:
            self._log("[VAL] Prediction failed"); return
        bias = our_val - ref_val
        self._ref_results.append((our_val, ref_val))
        n = len(self._ref_results)
        self._log(f"[VAL] RefCmp n={n}  ours={our_val:.2f}  ref={ref_val:.2f}  "
                  f"bias={bias:+.2f}ppm ({bias/ref_val*100:+.1f}%)" if ref_val != 0 else
                  f"[VAL] RefCmp n={n}  ours={our_val:.2f}  ref={ref_val:.2f}  bias={bias:+.2f}")
        if len(self._ref_results) >= 2:
            ours = np.array([x[0] for x in self._ref_results])
            refs = np.array([x[1] for x in self._ref_results])
            r    = np.corrcoef(ours, refs)[0, 1]
            mean_bias = float(np.mean(ours - refs))
            self._log(f"[VAL] RefCmp summary  n={len(ours)}  r={r:.4f}  "
                      f"mean_bias={mean_bias:+.2f}ppm")

    def _export_val_report(self):
        import datetime, csv as _csv
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(r"C:\Report Writing\Figures") / f"validation_report_{ts}.csv"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        with open(out, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["# UV-Vis PLS Nitrate — ISO 17025 Validation Report"])
            w.writerow(["# Generated", datetime.datetime.now().isoformat()])
            w.writerow([])

            # --- Linearity ---
            w.writerow(["SECTION", "LINEARITY"])
            w.writerow(["known_ppm", "measured_ppm", "error_pct"])
            for kn, ms in self._val_results:
                err = (ms - kn) / kn * 100 if kn else float("nan")
                w.writerow([f"{kn:.3f}", f"{ms:.3f}", f"{err:+.2f}"])
            if len(self._val_results) >= 2:
                kn_ = np.array([x[0] for x in self._val_results])
                ms_ = np.array([x[1] for x in self._val_results])
                m_, b_ = np.polyfit(kn_, ms_, 1)
                ss_res = np.sum((ms_ - (m_ * kn_ + b_)) ** 2)
                ss_tot = np.sum((ms_ - ms_.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                w.writerow(["R2", f"{r2:.5f}", "slope", f"{m_:.5f}", "intercept", f"{b_:+.5f}"])
            w.writerow([])

            # --- Repeatability ---
            w.writerow(["SECTION", "REPEATABILITY"])
            w.writerow(["measured_ppm"])
            for v in self._rep_results:
                w.writerow([f"{v:.4f}"])
            if len(self._rep_results) >= 2:
                m_ = float(np.mean(self._rep_results))
                s_ = float(np.std(self._rep_results, ddof=1))
                rsd = s_ / m_ * 100 if m_ else float("nan")
                w.writerow(["mean", f"{m_:.4f}", "SD", f"{s_:.4f}", "RSD%", f"{rsd:.2f}"])
            w.writerow([])

            # --- LOD/LOQ ---
            w.writerow(["SECTION", "LOD_LOQ_BLANKS"])
            w.writerow(["blank_ppm"])
            for v in self._blank_results:
                w.writerow([f"{v:.4f}"])
            if len(self._blank_results) >= 2:
                s_ = float(np.std(self._blank_results, ddof=1))
                w.writerow(["LOD(3*SD)", f"{3*s_:.4f}ppm",
                             "LOQ(10*SD)", f"{10*s_:.4f}ppm"])
            w.writerow([])

            # --- Reference comparison ---
            w.writerow(["SECTION", "REFERENCE_COMPARISON"])
            w.writerow(["our_ppm", "reference_ppm", "bias_ppm"])
            for ou, rf in self._ref_results:
                w.writerow([f"{ou:.4f}", f"{rf:.4f}", f"{ou-rf:+.4f}"])
            if len(self._ref_results) >= 2:
                ours = np.array([x[0] for x in self._ref_results])
                refs = np.array([x[1] for x in self._ref_results])
                r    = np.corrcoef(ours, refs)[0, 1]
                mb   = float(np.mean(ours - refs))
                w.writerow(["Pearson_r", f"{r:.5f}", "mean_bias_ppm", f"{mb:+.4f}"])
        self._log(f"[VAL] Report saved -> {out.name}")
        self._lbl_val_result.config(text=f"Saved: {out.name}")

    def _reset_session(self):
        """Reset all calibration, predictions, and plots back to startup state."""
        if not messagebox.askyesno("Reset Session",
                                   "Reset dark, reference, sample and all predictions?\n\n"
                                   "The autosampler connection and mode will be kept.\n"
                                   "You will be prompted to recapture dark."):
            return

        # clear calibration data
        self.dark           = None
        self.ref            = None
        self.last_processed = np.zeros(PIXELS)
        self.last_sample    = np.zeros(PIXELS)

        # clear prediction history
        self._ppm_hist = np.full(self.HIST, np.nan)

        # reset plot data
        self._ln_live.set_ydata(np.zeros(PIXELS))
        self._fill_live.remove()
        self._fill_live = self._ax_live.fill_between(
            WAVELENGTHS, 0, 0, color=C_TEAL, alpha=0.13)
        self._ax_live.set_ylim(0, ADC_MAX * 1.05)
        self._ax_live.set_title("LIVE SPECTRUM", color=C_TEAL,
                                 fontsize=9.5, fontweight="bold", loc="left", pad=3)

        self._ln_uv.set_ydata(np.zeros(PIXELS))

        self._ln_dark.set_ydata(np.zeros(PIXELS))
        self._ln_ref.set_ydata(np.zeros(PIXELS))
        self._ln_samp.set_ydata(np.zeros(PIXELS))
        self._ln_diff.set_ydata(np.zeros(PIXELS))
        self._calib_status = {"dark": False, "ref": False, "samp": False}
        self._calib_hint.set_visible(True)
        self._ax_calib.set_xlim(WL_START, WL_END)
        self._ax_calib.set_title("CALIBRATION SPECTRA", color=C_TEXT,
                                  fontsize=8.5, fontweight="bold", loc="left", pad=5)

        self._ln_trend.set_ydata(self._ppm_hist)
        self._fill_trend.remove()
        self._fill_trend = self._ax_trend.fill_between(
            self._t_hist, 0, 0, color=C_AMBER, alpha=0.15)
        self._ax_trend.set_ylim(0, 30)

        self._bar[0].set_height(0)
        self._bar_tx.set_text("--")
        self._bar_tx.set_y(0.5)
        self._ax_bar.set_ylim(0, 30)

        self._canvas.draw_idle()

        # reset toolbar button labels
        self._btn_dark.config(text="SET DARK", bg="#2a3a4a")
        self._btn_ref.config(text="SET REF",   bg="#1a3a2a")

        # reset readout labels
        self._lbl_ppm.config(text="--")
        self._lbl_peak.config(text="--")

        self._log("=" * 38)
        self._log("[RESET]  session cleared  --  recapture dark")
        self._log("=" * 38)

        # re-run dark calibration dialog
        self.root.after(200, self._show_dark_dialog)

    # -
    # ZOOM   -  scroll wheel on any axes (same as original)
    # -

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        f  = 0.85 if event.button == "up" else 1.15
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        xc = event.xdata if event.xdata else (xl[0] + xl[1]) / 2
        yc = event.ydata if event.ydata else (yl[0] + yl[1]) / 2
        ax.set_xlim(xc - (xc - xl[0]) * f, xc + (xl[1] - xc) * f)
        ax.set_ylim(yc - (yc - yl[0]) * f, yc + (yl[1] - yc) * f)
        self._canvas.draw_idle()

    def _on_right_click(self, event):
        """Right-click resets zoom on the clicked axes."""
        if event.button == 3 and event.inaxes is not None:
            ax = event.inaxes
            if ax is self._ax_live:
                ax.set_xlim(WL_START, WL_END)
                ax.set_ylim(-0.05, 1.5) if self.ref is not None \
                    else ax.set_ylim(0, ADC_MAX * 1.05)
            elif ax is self._ax_calib:
                ax.set_xlim(WL_START, WL_END)
                ax.autoscale(axis="y")
            elif ax is self._ax_trend:
                ax.set_xlim(self._t_hist[0], 0)
                ax.set_ylim(0, 30)
            self._canvas.draw_idle()

    # -
    # LOG
    # -

    def _log(self, msg):
        self._log_box.insert(
            tk.END,
            f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}\n"
        )
        self._log_box.see(tk.END)
        log.info(msg)

    # -
    # QUIT
    # -

    def _quit(self):
        self.is_running = False
        if self._ard_thread:
            self._ard_thread.stop()
        try:
            self.hw.close()
        except Exception:
            pass
        self.root.destroy()

    # -
    # AUTOSAMPLER
    # -


    def _auto_connect_arduino(self):
        if not _SERIAL_OK:
            return
        self._ard_thread = ArduinoThread(
            ARDUINO_PORT, ARDUINO_BAUD, self._ard_q,
            self._on_ard_trigger, self._on_ard_ref_trigger)
        self._ard_thread.start()

    def _start_autosampler(self):
        if not _SERIAL_OK:
            self._log("[AUTOSAMPLER] pyserial not installed -- pip install pyserial")
            return
        # Mode 2: warn user once to set switch to RIGHT (standby) to avoid auto-trigger
        if self._current_mode == 2 and not self._mode2_warned:
            messagebox.showinfo(
                "Mode 2  --  GUI Trigger",
                "Ensure the toggle switch is in the RIGHT position (standby).\n\n"
                "If the switch is LEFT (LOW), it will auto-trigger the sequence "
                "independently of this button.\n\nPress OK when switch is RIGHT.")
            self._mode2_warned = True
        self._btn_ard_start.config(state=tk.DISABLED)
        self._btn_ard_stop.config(state=tk.NORMAL)
        self._lbl_ard_step.config(text="Step: starting...", fg=C_WARN)
        self._log("-" * 38)
        self._log(f"[AUTOSAMPLER] START  {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
        self._log("-" * 38)
        self._ard_running = True
        if self._ard_thread and self._ard_thread.is_alive():
            self._ard_thread.send_cmd("start")
        else:
            import threading as _t, time as _ti
            self._ard_thread = ArduinoThread(
                ARDUINO_PORT, ARDUINO_BAUD, self._ard_q,
                self._on_ard_trigger, self._on_ard_ref_trigger)
            self._ard_thread.start()
            def _delayed():
                _ti.sleep(1.0)
                self._ard_thread.send_cmd("start")
            _t.Thread(target=_delayed, daemon=True).start()

    def _stop_autosampler(self):
        """Legacy stop (no drain). Use _emergency_stop for drain-after-stop."""
        if self._ard_thread:
            self._ard_thread.send_cmd("off")
        self._ard_running = False
        if hasattr(self, "_btn_ard_start"):
            self._btn_ard_start.config(state=tk.NORMAL)
        if hasattr(self, "_btn_ard_stop"):
            self._btn_ard_stop.config(state=tk.DISABLED)
        self._lbl_ard_step.config(text="stopped  --  flip LEFT or press START", fg=C_RED)
        self._log("[AUTOSAMPLER] stopped  --  all pumps OFF")

    def _on_ard_ref_trigger(self):
        """Called in worker thread when REFERENCE LOADED is seen."""
        self.root.after(0, self._auto_set_ref)

    def _update_ref_plot(self, mean_val):
        """Freeze DI reference spectrum into the calibration overlay."""
        self._ln_ref.set_ydata(self.ref)
        self._calib_status["ref"] = True
        self._refresh_calib_plot()

    def _auto_set_ref(self):
        """Capture DI water as reference during automated reference window."""
        self.ref = self.current_raw.copy()
        mean_val = float(self.ref.mean())
        self._btn_ref.config(text=f"REF  [v]  {mean_val:.0f}cts", bg="#1a3a3a")
        self._ax_live.set_ylim(-0.05, 2.0)
        self._ax_live.set_title("LIVE ABSORBANCE", color=C_TEAL,
                                 fontsize=9, fontweight="bold", loc="left")
        self._update_ref_plot(mean_val)
        self._log(f"[AUTOSAMPLER] DI reference captured  mean={mean_val:.4g}")

    def _on_ard_trigger(self):
        """Called in worker thread when SAMPLE LOADED is seen."""
        self.root.after(0, self._auto_measure)

    def _freeze_sample_plot(self, ppm=None):
        """Freeze sample spectrum and differential into calibration overlay."""
        self._ln_samp.set_ydata(self.last_sample)
        self._calib_status["samp"] = True
        # differential: sample - ref (shows nitrate peak at 220nm)
        if self.ref is not None:
            diff = self.last_sample - self.ref
            self._ln_diff.set_ydata(diff)
            idx220 = int(np.argmin(np.abs(WAVELENGTHS - 220)))
            self._log(f"  Differential @220nm = {diff[idx220]:.4f} AU  "
                      f"(ref={self.ref[idx220]:.4f}  samp={self.last_sample[idx220]:.4f})")
        else:
            self._ln_diff.set_ydata(self.last_sample)
        # zoom calib plot to UV region to show nitrate peak clearly
        self._ax_calib.set_xlim(200, 400)
        self._refresh_calib_plot()

    def _auto_measure(self):
        """Freeze current spectrum and run PLS  -  called from main thread."""
        self.last_sample = self.last_processed.copy()
        idx220 = int(np.argmin(np.abs(WAVELENGTHS - 220)))
        au220  = float(self.last_sample[idx220])
        if self._pred.ready:
            ppm = self._pred.predict(self.last_sample)
            if ppm is not None:
                self._bar[0].set_height(ppm)
                self._bar_tx.set_text(f"{ppm:.2f}")
                self._bar_tx.set_y(ppm)
                self._lbl_ppm.config(text=f"{ppm:.2f}")
                self._freeze_sample_plot(ppm)
                self._log(f"[AUTOSAMPLER] >>>  NO3 = {ppm:.2f} ppm  <<<  @220nm={au220:.4f} AU")
                return
        self._freeze_sample_plot()
        self._log(f"[AUTOSAMPLER] spectrum frozen @220nm={au220:.4f} AU  (model not ready)")

    def _poll_arduino(self):
        """Drain autosampler message queue  -  runs in main thread every 50 ms."""
        try:
            while True:
                kind, data = self._ard_q.get_nowait()

                if kind == "open":
                    self._lbl_ard_status.config(
                        text=f"[*]  {data}", fg=C_GREEN)
                    self._log(f"[AUTOSAMPLER] {data}")

                elif kind == "line":
                    self._log(f"[ARD] {data}")
                    # update step label from well-known markers
                    for marker, label in STEP_LABELS.items():
                        if marker in data:
                            if "MEASURE_NOW" in marker or "REFERENCE_TRIGGER" in marker:
                                color = C_TEAL
                            elif "ABORTED" in marker or "EMERGENCY" in marker:
                                color = C_RED
                            elif "COMPLETE" in marker or "STANDBY" in marker:
                                color = C_GREEN
                            else:
                                color = C_WARN
                            self._lbl_ard_step.config(text=label, fg=color)
                            break

                elif kind == "done":
                    self._lbl_ard_step.config(
                        text="Sequence complete  --  ready for next sample", fg=C_TEAL)
                    self._ard_running = False
                    if hasattr(self, "_btn_ard_start"):
                        self._btn_ard_start.config(state=tk.NORMAL)
                    if hasattr(self, "_btn_ard_stop"):
                        self._btn_ard_stop.config(state=tk.DISABLED)
                    self._log("[AUTOSAMPLER] [*] sequence complete")

                elif kind == "seq_started":
                    self._ard_running = True
                    if hasattr(self, "_btn_ard_start"):
                        self._btn_ard_start.config(state=tk.DISABLED)
                    if hasattr(self, "_btn_ard_stop"):
                        self._btn_ard_stop.config(state=tk.NORMAL)

                elif kind == "aborted":
                    self._lbl_ard_step.config(
                        text="ABORTED  --  cleanup drain runs on next trigger",
                        fg=C_RED)
                    self._ard_running = False
                    if hasattr(self, "_btn_ard_start"):
                        self._btn_ard_start.config(state=tk.NORMAL)
                    if hasattr(self, "_btn_ard_stop"):
                        self._btn_ard_stop.config(state=tk.DISABLED)
                    self._log("[AUTOSAMPLER] [x] ABORTED  --  10s cleanup drain on next trigger")

                elif kind == "err":
                    self._lbl_ard_status.config(
                        text=f"[x]  {data}", fg=C_RED)
                    self._lbl_ard_step.config(text="connection error", fg=C_RED)
                    self._ard_running = False
                    if hasattr(self, "_btn_ard_start"):
                        self._btn_ard_start.config(state=tk.NORMAL)
                    if hasattr(self, "_btn_ard_stop"):
                        self._btn_ard_stop.config(state=tk.DISABLED)
                    self._log(f"[AUTOSAMPLER] ERROR: {data}")

        except queue.Empty:
            pass
        finally:
            self.root.after(50, self._poll_arduino)

    # -
    # LAUNCH DIALOGS  (mode selection + dark calibration on every start)
    # -

    def _launch_dialogs(self):
        self._show_mode_dialog()
        self._show_dark_dialog()
        self._rebuild_mode_controls()

    def _show_mode_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Select Operating Mode")
        dlg.configure(bg=C_BG0)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.geometry("450x300")
        dlg.update_idletasks()
        px = self.root.winfo_x() + max(0, (self.root.winfo_width()  - 450) // 2)
        py = self.root.winfo_y() + max(0, (self.root.winfo_height() - 300) // 2)
        dlg.geometry(f"+{px}+{py}")

        tk.Label(dlg, text="SELECT OPERATING MODE",
                 fg=C_TEAL, bg=C_BG0,
                 font=("Courier New", 14, "bold")).pack(pady=(18, 12))

        def _choose(n):
            self._current_mode = n
            dlg.destroy()

        for num, name, desc, bg, fg in [
            (1, "AUTO-SAMPLE",  "Toggle switch triggers 12-step sequence",    "#005533", "#00ffb0"),
            (2, "GUI TRIGGER",  "Press START button to run one sequence",      "#003355", "#38cfff"),
            (3, "MANUAL LAB",   "Manual calibration  --  use toolbar buttons", "#332200", "#ffcc00"),
            (4, "VALIDATE",     "ISO 17025: linearity, spike recovery, LOD",   "#1a1a55", "#9999ff"),
        ]:
            row = tk.Frame(dlg, bg=C_BG3, pady=3)
            row.pack(fill=tk.X, padx=20, pady=3)
            tk.Button(row, text=f"MODE {num}:  {name}",
                      bg=bg, fg=fg,
                      activebackground=bg, activeforeground="white",
                      font=("Courier New", 10, "bold"), relief=tk.FLAT,
                      cursor="hand2", padx=10, pady=7,
                      command=lambda n=num: _choose(n)).pack(side=tk.LEFT, padx=8)
            tk.Label(row, text=desc, fg=C_DIM, bg=C_BG3,
                     font=("Courier New", 9)).pack(side=tk.LEFT, padx=8)

        dlg.protocol("WM_DELETE_WINDOW", lambda: _choose(1))
        self.root.wait_window(dlg)

    def _show_dark_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Dark Calibration")
        dlg.configure(bg=C_BG0)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.geometry("380x210")
        dlg.update_idletasks()
        px = self.root.winfo_x() + max(0, (self.root.winfo_width()  - 380) // 2)
        py = self.root.winfo_y() + max(0, (self.root.winfo_height() - 210) // 2)
        dlg.geometry(f"+{px}+{py}")

        tk.Label(dlg, text="DARK CALIBRATION",
                 fg=C_TEAL, bg=C_BG0,
                 font=("Courier New", 12, "bold")).pack(pady=(18, 6))
        tk.Label(dlg, text="1.  Turn OFF the light source",
                 fg=C_WARN, bg=C_BG0,
                 font=("Courier New", 10)).pack(pady=2)
        tk.Label(dlg, text="2.  Click CAPTURE DARK below",
                 fg=C_TEXT, bg=C_BG0,
                 font=("Courier New", 10)).pack(pady=(0, 14))

        def _capture():
            self._cap_dark()
            dlg.destroy()

        bf = tk.Frame(dlg, bg=C_BG0)
        bf.pack()
        tk.Button(bf, text="CAPTURE DARK",
                  bg="#2a3a4a", fg=C_TEXT,
                  activebackground="#3a4a5a", activeforeground="white",
                  font=("Courier New", 10, "bold"), relief=tk.FLAT,
                  cursor="hand2", padx=14, pady=8,
                  command=_capture).pack(side=tk.LEFT, padx=8)
        tk.Button(bf, text="SKIP",
                  bg=C_BG3, fg=C_DIM,
                  activebackground=C_BG2, activeforeground=C_TEXT,
                  font=("Courier New", 10), relief=tk.FLAT,
                  cursor="hand2", padx=14, pady=8,
                  command=dlg.destroy).pack(side=tk.LEFT, padx=8)

        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        self.root.wait_window(dlg)

    def _select_mode(self, n):
        if n == self._current_mode:
            return
        names = {1: "AUTO-SAMPLE", 2: "GUI TRIGGER", 3: "MANUAL LAB", 4: "VALIDATE"}
        if not messagebox.askyesno("Switch Mode",
                                   f"Switch to Mode {n}: {names[n]}?"):
            return
        self._current_mode = n
        self._mode2_warned = False
        self._rebuild_mode_controls()
        self._log(f"[MODE] Mode {n}: {names[n]}")

    def _rebuild_mode_controls(self):
        """Show the correct mode sub-frame, update button highlight."""
        if not hasattr(self, "_mode_ctrl_frame"):
            return
        m = self._current_mode
        for frm in [self._m1_frame, self._m2_frame, self._m3_frame, self._m4_frame]:
            frm.pack_forget()
        if m == 1:
            self._m1_frame.pack(fill=tk.X)
        elif m == 2:
            self._m2_frame.pack(fill=tk.X)
        elif m == 4:
            self._m4_frame.pack(fill=tk.X)
        else:
            self._m3_frame.pack(fill=tk.X)
        for n, btn in self._mode_btns.items():
            btn.config(relief=tk.SUNKEN if n == m else tk.FLAT)

    def _emergency_stop(self):
        """Mode 1 / Mode 2 emergency stop: stop pumps, run drain, show countdown."""
        if self._ard_thread:
            self._ard_thread.send_cmd("off")
            self._ard_thread.send_cmd("drain")   # firmware runs 15s drain
        self._ard_running = False
        self._lbl_ard_step.config(
            text="[!] EMERGENCY STOP  --  drain running 15s...", fg=C_RED)
        self._log("[EMERGENCY STOP]  pumps off  --  drain 15s")
        self._btn_emg_stop.config(state=tk.DISABLED)
        if hasattr(self, "_btn_ard_stop"):
            self._btn_ard_stop.config(state=tk.DISABLED)
        self.root.after(15000, self._estop_done)

    def _estop_done(self):
        self._btn_emg_stop.config(state=tk.NORMAL)
        if hasattr(self, "_btn_ard_start"):
            self._btn_ard_start.config(state=tk.NORMAL)
        self._lbl_ard_step.config(
            text="Ready  --  flip LEFT or press START", fg=C_GREEN)
        self._log("[EMERGENCY STOP]  drain complete  --  ready for next sample")

    # -
    # UI CONSTRUCTION
    # -

    def _build_window(self):
        # - top toolbar -
        tb = tk.Frame(self.root, bg=C_BG1, pady=5, padx=8)
        tb.pack(fill=tk.X)

        # left: title + hw status
        left = tk.Frame(tb, bg=C_BG1)
        left.pack(side=tk.LEFT)
        tk.Label(left, text="ARIS UV SPECTROSCOPY",
                 fg=C_TEAL, bg=C_BG1,
                 font=("Courier New", 11, "bold")).pack(anchor="w")
        tk.Label(left, text=self._hw_label,
                 fg=self._hw_color, bg=C_BG1,
                 font=("Courier New", 8)).pack(anchor="w")

        # centre: signal indicators
        mid = tk.Frame(tb, bg=C_BG1)
        mid.pack(side=tk.LEFT, padx=20)
        self._lbl_status = tk.Label(mid, text="LIVE", fg=C_GREEN, bg=C_BG1,
                                    font=("Courier New", 10, "bold"))
        self._lbl_status.grid(row=0, column=0, padx=8)
        self._lbl_cts = tk.Label(mid, text="0 cts", fg=C_TEXT, bg=C_BG1,
                                 font=("Courier New", 9))
        self._lbl_cts.grid(row=1, column=0, padx=8)
        self._lbl_mean = tk.Label(mid, text="0 mean", fg="#aac4e8", bg=C_BG1,
                                  font=("Courier New", 8))
        self._lbl_mean.grid(row=2, column=0, padx=8)

        # centre-right: settings
        sett = tk.Frame(tb, bg=C_BG1)
        sett.pack(side=tk.LEFT, padx=10)

        def param(parent, label, var, col, width=5):
            tk.Label(parent, text=label, fg="#aac4e8", bg=C_BG1,
                     font=("Courier New", 7)).grid(row=0, column=col, padx=5)
            tk.Entry(parent, textvariable=var, width=width,
                     bg=C_BG3, fg=C_TEXT, bd=0,
                     font=("Courier New", 9),
                     insertbackground=C_TEXT).grid(row=1, column=col, padx=5)

        param(sett, "Exposure (ms)", self.exp_time,  0)
        param(sett, "Averaging (N)", self.averaging,  1)
        param(sett, "Smooth Win",    self.smooth_w,   2)

        tk.Checkbutton(sett, text="Dark Sub",
                       variable=self.sub_dark_var,
                       fg=C_TEXT, bg=C_BG1, selectcolor=C_BG0,
                       activebackground=C_BG1,
                       font=("Courier New", 8)).grid(row=1, column=3, padx=6)

        tk.Checkbutton(sett, text="Auto predict",
                       variable=self.auto_pred,
                       fg=C_TEXT, bg=C_BG1, selectcolor=C_BG0,
                       activebackground=C_BG1,
                       font=("Courier New", 8)).grid(row=1, column=4, padx=6)

        tk.Label(sett, text="Direct ABS", fg=C_WARN, bg=C_BG1,
                 font=("Courier New", 7)).grid(row=0, column=5, padx=5)
        tk.Checkbutton(sett, text="(bypass BL)",
                       variable=self.direct_abs,
                       fg=C_WARN, bg=C_BG1, selectcolor=C_BG0,
                       activebackground=C_BG1,
                       font=("Courier New", 8)).grid(row=1, column=5, padx=5)

        # right: action buttons
        right = tk.Frame(tb, bg=C_BG1)
        right.pack(side=tk.RIGHT)

        def abtn(text, color, cmd, **kw):
            return tk.Button(right, text=text, bg=color, fg="white",
                             activebackground=color, relief=tk.FLAT,
                             cursor="hand2", font=("Courier New", 8, "bold"),
                             padx=8, pady=5, command=cmd, **kw)

        self._btn_dark  = abtn("SET DARK",   "#2a3a4a", self._cap_dark)
        self._btn_ref   = abtn("SET REF",    "#1a3a2a", self._cap_ref)
        self._btn_samp  = abtn("ANALYZE",    "#4a1a2a", self._cap_sample)
        self._btn_save  = abtn("SAVE CSV",   "#1a2a4a", self._save_csv)
        self._btn_diag  = abtn("DIAG",       "#3a2a0a", self._run_diag)
        self._btn_reset = abtn("[R] RESET",  "#5a1a1a", self._reset_session)

        for b in [self._btn_dark, self._btn_ref, self._btn_samp,
                  self._btn_save, self._btn_diag, self._btn_reset]:
            b.pack(side=tk.LEFT, padx=4)

        # - main body -
        body = tk.Frame(self.root, bg=C_BG0)
        body.pack(fill=tk.BOTH, expand=True)

        # plot area
        plot_area = tk.Frame(body, bg=C_BG0)
        plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._fig    = Figure(figsize=(17.0, 9.0), facecolor=C_BG0)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_area)
        self._canvas.get_tk_widget().configure(bg=C_BG0)

        # toolbar  -  force dark theme on every child widget
        tb2 = tk.Frame(plot_area, bg=C_BG1)
        tb2.pack(fill=tk.X, side=tk.TOP)
        self._toolbar = NavigationToolbar2Tk(self._canvas, tb2)
        self._toolbar.update()
        self._toolbar.configure(bg=C_BG1)
        for _w in self._toolbar.winfo_children():
            try:
                _w.configure(bg=C_BG1, fg=C_TEXT,
                              activebackground=C_BG3,
                              activeforeground=C_TEAL,
                              highlightbackground=C_BG1,
                              relief=tk.FLAT)
            except Exception:
                pass
        # message label (bottom of toolbar) also goes dark
        if hasattr(self._toolbar, "_message_label"):
            self._toolbar._message_label.configure(
                bg=C_BG1, fg=C_DIM,
                font=("Courier New", 8))

        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # connect scroll zoom + right-click reset
        self._canvas.mpl_connect("scroll_event",       self._on_scroll)
        self._canvas.mpl_connect("button_press_event", self._on_right_click)

        # - right panel -
        right_panel = tk.Frame(body, bg=C_BG1, width=220)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2), pady=2)
        right_panel.pack_propagate(False)
        self._build_right(right_panel)

    def _build_right(self, parent):

        def sep():
            ttk.Separator(parent, orient="horizontal").pack(
                fill=tk.X, padx=6, pady=3)

        def section(title, color=C_DIM):
            tk.Label(parent, text=title, fg=color, bg=C_BG1,
                     font=("Courier New", 7, "bold"),
                     pady=2, padx=8, anchor="w").pack(fill=tk.X)

        # - NO3 concentration  -  big display -
        ppm_card = tk.Frame(parent, bg=C_BG3, pady=3)
        ppm_card.pack(fill=tk.X, padx=6, pady=(5, 2))
        tk.Label(ppm_card, text="NO3-  CONCENTRATION",
                 fg=C_VIO, bg=C_BG3,
                 font=("Courier New", 7, "bold"),
                 anchor="w", padx=8).pack(fill=tk.X)
        self._lbl_ppm = tk.Label(
            ppm_card, text="--",
            fg=C_AMBER, bg=C_BG3,
            font=("Courier New", 26, "bold"))
        self._lbl_ppm.pack()
        tk.Label(ppm_card, text="ppm", fg=C_DIM, bg=C_BG3,
                 font=("Courier New", 8)).pack(pady=(0, 2))

        # - spectral peak -
        peak_card = tk.Frame(parent, bg=C_BG3, pady=2)
        peak_card.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(peak_card, text="ABSORBANCE @ 220 nm",
                 fg=C_VIO, bg=C_BG3,
                 font=("Courier New", 7, "bold"),
                 anchor="w", padx=8).pack(fill=tk.X)
        self._lbl_peak = tk.Label(
            peak_card, text="--",
            fg=C_TEAL, bg=C_BG3,
            font=("Courier New", 13, "bold"))
        self._lbl_peak.pack(pady=(0, 2))

        sep()

        # - model status -
        ok  = "MODEL  LOADED" if self._pred.ready else "MODEL  NOT FOUND"
        col = C_GREEN         if self._pred.ready else C_RED
        mod_card = tk.Frame(parent, bg=C_BG3, pady=2)
        mod_card.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(mod_card, text=ok, fg=col, bg=C_BG3,
                 font=("Courier New", 8, "bold")).pack(padx=8, pady=2)

        sep()
        section("OPERATING MODE", C_TEAL)

        # mode selector buttons  [M1: AUTO]  [M2: GUI]  [M3: MANUAL]
        _mf = tk.Frame(parent, bg=C_BG1)
        _mf.pack(fill=tk.X, padx=8, pady=(2, 4))
        self._mode_btns = {}
        for _n, _lbl, _bg, _fg in [
            (1, "AUTO",     "#005533", "#00ffb0"),
            (2, "GUI",      "#003355", "#38cfff"),
            (3, "MANUAL",   "#332200", "#ffcc00"),
            (4, "VALIDATE", "#1a1a55", "#9999ff"),
        ]:
            _b = tk.Button(_mf, text=f"M{_n}: {_lbl}",
                           bg=_bg, fg=_fg,
                           activebackground=_bg, activeforeground="white",
                           font=("Courier New", 8, "bold"), relief=tk.FLAT,
                           cursor="hand2", padx=6, pady=5,
                           command=lambda x=_n: self._select_mode(x))
            _b.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
            self._mode_btns[_n] = _b

        sep()
        section("AUTOSAMPLER  (COM10)", C_GREEN)

        # arduino connection status + step label (shared across all modes)
        ard_card = tk.Frame(parent, bg=C_BG3, pady=4)
        ard_card.pack(fill=tk.X, padx=8, pady=3)
        self._lbl_ard_status = tk.Label(
            ard_card, text="[ ]  not connected",
            fg=C_DIM, bg=C_BG3,
            font=("Courier New", 9, "bold"), anchor="w", padx=10)
        self._lbl_ard_status.pack(fill=tk.X)
        self._lbl_ard_step = tk.Label(
            ard_card, text="---",
            fg=C_MUTED, bg=C_BG3,
            font=("Courier New", 8), anchor="w", padx=10)
        self._lbl_ard_step.pack(fill=tk.X, pady=(2, 6))

        # mode-dependent pump controls (rebuilt by _rebuild_mode_controls)
        self._mode_ctrl_frame = tk.Frame(parent, bg=C_BG1)
        self._mode_ctrl_frame.pack(fill=tk.X, padx=8, pady=(2, 3))

        # -- Mode 1 sub-frame: emergency stop only
        self._m1_frame = tk.Frame(self._mode_ctrl_frame, bg=C_BG1)
        self._btn_emg_stop = tk.Button(
            self._m1_frame, text="[!]  EMERGENCY STOP",
            bg="#6b0e0e", fg="#ff7070",
            activebackground="#8f1414", activeforeground="white",
            font=("Courier New", 9, "bold"), relief=tk.FLAT,
            cursor="hand2", pady=8,
            command=self._emergency_stop)
        self._btn_emg_stop.pack(fill=tk.X)

        # -- Mode 2 sub-frame: start + stop
        self._m2_frame = tk.Frame(self._mode_ctrl_frame, bg=C_BG1)
        self._btn_ard_start = tk.Button(
            self._m2_frame, text=">>  START SEQUENCE",
            bg="#006644", fg="#00ffb0",
            activebackground="#008f5e", activeforeground="white",
            font=("Courier New", 9, "bold"), relief=tk.FLAT,
            cursor="hand2", pady=8,
            command=self._start_autosampler)
        self._btn_ard_start.pack(fill=tk.X, pady=(0, 3))
        self._btn_ard_stop = tk.Button(
            self._m2_frame, text="[]  STOP  /  EMERGENCY",
            bg="#6b0e0e", fg="#ff7070",
            activebackground="#8f1414", activeforeground="white",
            font=("Courier New", 9, "bold"), relief=tk.FLAT,
            cursor="hand2", pady=7, state=tk.DISABLED,
            command=self._emergency_stop)
        self._btn_ard_stop.pack(fill=tk.X)

        # -- Mode 3 sub-frame: hint label only
        self._m3_frame = tk.Frame(self._mode_ctrl_frame, bg=C_BG1)
        tk.Label(self._m3_frame,
                 text="Use toolbar:  SET DARK / SET REF / ANALYZE\nNo pump control in Manual mode.",
                 fg=C_DIM, bg=C_BG1,
                 font=("Courier New", 8), justify="center").pack(pady=8)

        # -- Mode 4 sub-frame: validation
        self._m4_frame = tk.Frame(self._mode_ctrl_frame, bg=C_BG1)
        self._val_known_ppm = tk.DoubleVar(value=25.0)
        self._val_ref_var   = tk.DoubleVar(value=0.0)

        # known concentration row
        krow = tk.Frame(self._m4_frame, bg=C_BG1)
        krow.pack(fill=tk.X, pady=(3, 0))
        tk.Label(krow, text="Known ppm:", fg="#9999ff", bg=C_BG1,
                 font=("Courier New", 7)).pack(side=tk.LEFT)
        tk.Entry(krow, textvariable=self._val_known_ppm, width=7,
                 bg=C_BG3, fg=C_TEXT, font=("Courier New", 8),
                 insertbackground=C_TEXT).pack(side=tk.LEFT, padx=4)

        self._lbl_val_result = tk.Label(self._m4_frame,
            text="--", fg="#9999ff", bg=C_BG1,
            font=("Courier New", 7), anchor="w")
        self._lbl_val_result.pack(fill=tk.X, pady=(1, 2))

        def vbtn(txt, cmd):
            return tk.Button(self._m4_frame, text=txt,
                             bg="#1a1a55", fg="#9999ff",
                             activebackground="#2a2a77", activeforeground="white",
                             font=("Courier New", 7, "bold"), relief=tk.FLAT,
                             cursor="hand2", pady=4, command=cmd)

        vbtn("MEASURE STANDARD",       self._run_val_standard).pack(fill=tk.X, pady=1)
        vbtn("RUN x5 REPEATABILITY",   lambda: self._run_repeatability(5)).pack(fill=tk.X, pady=1)
        vbtn("BLANK x10 -> LOD/LOQ",   lambda: self._run_blank_lod(10)).pack(fill=tk.X, pady=1)

        # reference comparison row
        rrow = tk.Frame(self._m4_frame, bg=C_BG1)
        rrow.pack(fill=tk.X, pady=(3, 0))
        tk.Label(rrow, text="Ref method ppm:", fg=C_DIM, bg=C_BG1,
                 font=("Courier New", 7)).pack(side=tk.LEFT)
        tk.Entry(rrow, textvariable=self._val_ref_var, width=6,
                 bg=C_BG3, fg=C_TEXT, font=("Courier New", 8),
                 insertbackground=C_TEXT).pack(side=tk.LEFT, padx=3)

        vbtn("ADD REF COMPARISON POINT", self._add_ref_point).pack(fill=tk.X, pady=1)
        vbtn("EXPORT VALIDATION REPORT", self._export_val_report).pack(fill=tk.X, pady=(3, 1))

        # show initial mode controls (updated after _launch_dialogs selects a mode)
        self._rebuild_mode_controls()

        sep()

        # - event log with clear button -
        log_hdr = tk.Frame(parent, bg=C_BG1)
        log_hdr.pack(fill=tk.X, padx=8)
        tk.Label(log_hdr, text="EVENT LOG",
                 fg=C_TEAL, bg=C_BG1,
                 font=("Courier New", 8, "bold")).pack(side=tk.LEFT, pady=4)
        tk.Button(log_hdr, text="CLEAR",
                  bg="#1a2540", fg=C_TEAL,
                  activebackground=C_BG2, activeforeground="white",
                  font=("Courier New", 7, "bold"), relief=tk.FLAT,
                  cursor="hand2", padx=8, pady=2,
                  command=lambda: (
                      self._log_box.config(state=tk.NORMAL),
                      self._log_box.delete("1.0", tk.END),
                      self._log_box.config(state=tk.DISABLED)
                  )).pack(side=tk.RIGHT, pady=2)

        self._log_box = tk.Text(
            parent, bg="#080b12", fg="#00ffc8",
            font=("Courier New", 8), relief=tk.FLAT,
            padx=10, pady=6, state=tk.NORMAL,
            insertbackground=C_TEXT,
        )
        self._log_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 6))

    # -
    # PLOTS
    # -

    def _build_plots(self):
        plt.rcParams.update(PLOT_RC)
        self._fig.patch.set_facecolor(C_BG0)

        gs = GridSpec(2, 4, figure=self._fig,
                      width_ratios=[1.65, 0.75, 1.3, 1.55],
                      height_ratios=[1.05, 1.0],
                      hspace=0.22, wspace=0.18,
                      left=0.044, right=0.999,
                      top=0.968, bottom=0.058)

        self._ax_live  = self._fig.add_subplot(gs[0, :3])   # live  -  wide
        self._ax_uv    = self._fig.add_subplot(gs[0, 3])    # UV zoom
        self._ax_calib = self._fig.add_subplot(gs[1, :2])   # calibration overlay
        self._ax_trend = self._fig.add_subplot(gs[1, 2])    # ppm trend
        self._ax_bar   = self._fig.add_subplot(gs[1, 3])    # concentration bar


        # keep reference so external methods don't break
        self._ax_samp = self._ax_calib

        def _style(ax):
            ax.grid(True, color="#1e2d50", linewidth=0.8, alpha=1.0)
            ax.tick_params(axis="both", which="both",
                           colors="#ffffff", labelcolor="#ffffff",
                           labelsize=9, length=5, width=1.2, pad=4)
            ax.xaxis.label.set_color("#ffffff")
            ax.yaxis.label.set_color("#ffffff")
            ax.xaxis.label.set_fontsize(9)
            ax.yaxis.label.set_fontsize(9)
            for sp in ax.spines.values():
                sp.set_edgecolor("#3a5080")
                sp.set_linewidth(1.3)
            ax.set_facecolor(C_BG2)
        for ax in self._fig.get_axes():
            _style(ax)

        # - live spectrum -
        ax = self._ax_live
        ax.set_title("LIVE SPECTRUM",
                     color=C_TEAL, fontsize=9.5, fontweight="bold",
                     loc="left", pad=3)
        ax.set_xlabel("Wavelength (nm)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_ylabel("Signal / Absorbance (AU)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_xlim(WL_START, WL_END)
        ax.set_ylim(0, ADC_MAX * 1.05)
        self._ln_live,  = ax.plot(WAVELENGTHS, np.zeros(PIXELS),
                                   color=C_TEAL, lw=2.0)
        self._fill_live  = ax.fill_between(WAVELENGTHS, 0, 0,
                                            color=C_TEAL, alpha=0.13)
        ax.axvspan(200, 240, color=C_VIO, alpha=0.06, zorder=0)
        ax.axvline(220, color=C_VIO,   lw=1.0, ls="--", alpha=0.7,
                   label="NO3- 220 nm")
        ax.axvline(210, color=C_AMBER, lw=1.0, ls="--", alpha=0.7,
                   label="NO2- 210 nm")
        ax.legend(fontsize=8, facecolor=C_BG1, labelcolor="#ffffff",
                  framealpha=0.95, edgecolor="#3a5080", loc="upper right")

        # - UV zoom -
        ax = self._ax_uv
        ax.set_title("UV  200-350 nm",
                     color=C_VIO, fontsize=9, fontweight="bold",
                     loc="left", pad=3)
        ax.set_xlabel("Wavelength (nm)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_ylabel("AU", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_xlim(200, 350)
        ax.set_ylim(-0.05, 2.0)
        self._ln_uv, = ax.plot(WAVELENGTHS, np.zeros(PIXELS),
                                color=C_VIO, lw=2.0)
        ax.axvline(220, color=C_TEAL,  lw=1.0, ls="--", alpha=0.7)
        ax.axvline(210, color=C_AMBER, lw=1.0, ls="--", alpha=0.6)
        ax.axvspan(200, 240, color=C_VIO, alpha=0.08, zorder=0)

        # - calibration overlay (dark + DI ref + sample  -  one plot) -
        ax = self._ax_calib
        ax.set_title("CALIBRATION SPECTRA",
                     color="#ffffff", fontsize=9.5, fontweight="bold",
                     loc="left", pad=3)
        ax.set_xlabel("Wavelength (nm)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_ylabel("Signal / Absorbance (AU)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_xlim(WL_START, WL_END)
        ax.set_ylim(0, 1.0)
        self._ln_dark, = ax.plot(WAVELENGTHS, np.zeros(PIXELS),
                                  color=C_DARK_LINE, lw=1.5, alpha=0.9,
                                  ls="--", label="Dark  (light off)")
        self._ln_ref,  = ax.plot(WAVELENGTHS, np.zeros(PIXELS),
                                  color=C_REF_LINE,  lw=2.0,
                                  label="DI Reference  (Step 3)")
        self._ln_samp, = ax.plot(WAVELENGTHS, np.zeros(PIXELS),
                                  color=C_SAMP_LINE, lw=2.0,
                                  label="Sample  (Step 9)")
        self._ln_diff, = ax.plot(WAVELENGTHS, np.zeros(PIXELS),
                                  color=C_VIO, lw=2.2, ls="-",
                                  label="Differential  (Sample - Ref)")
        ax.axvline(220, color=C_VIO,   lw=0.8, ls="--", alpha=0.5)
        ax.axvline(210, color=C_AMBER, lw=0.8, ls="--", alpha=0.4)
        ax.axvspan(200, 240, color=C_VIO, alpha=0.06, zorder=0)
        self._calib_legend = ax.legend(
            fontsize=7.5, facecolor=C_BG1, labelcolor=C_TEXT,
            framealpha=0.95, edgecolor=C_EDGE, loc="upper right")
        self._calib_status = {"dark": False, "ref": False, "samp": False}
        # placeholder text
        self._calib_hint = ax.text(
            0.5, 0.48, "1. SET DARK   2. Start Sequence\n"
                        "Spectra appear here automatically",
            transform=ax.transAxes, ha="center", va="center",
            color=C_DIM, fontsize=8, style="italic")

        # - ppm trend -
        ax = self._ax_trend
        ax.set_title("NITRATE TREND",
                     color=C_AMBER, fontsize=9.5, fontweight="bold",
                     loc="left", pad=3)
        ax.set_xlabel("Time (s)", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_ylabel("ppm", fontsize=9, color="#ffffff", labelpad=4)
        ax.set_xlim(self._t_hist[0], 0)
        ax.set_ylim(0, 30)
        self._ln_trend,  = ax.plot(self._t_hist, self._ppm_hist,
                                    color=C_AMBER, lw=1.4)
        self._fill_trend  = ax.fill_between(
            self._t_hist, 0,
            np.where(np.isfinite(self._ppm_hist), self._ppm_hist, 0.0),
            color=C_AMBER, alpha=0.15)

        # - concentration bar -
        ax = self._ax_bar
        ax.set_title("NITRATE NOW",
                     color=C_AMBER, fontsize=9.5, fontweight="bold",
                     loc="left", pad=3)
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

    # also update UV panel in update loop
    def _update_uv(self, abs_v):
        self._ln_uv.set_ydata(abs_v)


# -
# PATCH: sync UV panel in _update_ui
# -
_orig_update = App._update_ui

def _patched_update(self):
    _orig_update(self)
    if hasattr(self, '_ln_uv'):
        if self.ref is not None or self.direct_abs.get():
            self._ln_uv.set_ydata(self.last_processed)

App._update_ui = _patched_update


# -
# ENTRY POINT
# -

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1600x950")
    root.state("zoomed")
    App(root)
    root.mainloop()
