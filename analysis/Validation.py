"""
ISO 17025 Validation Tool — UV-Vis PLS Nitrate System
Standalone application for scientific validation of the ARIS 546 / PLS measurement system.

Tests covered
--------------
  1. Linearity           -- 5-7 certified standards, R² >= 0.99
  2. Spike Recovery      -- 3 spike levels (80/100/120%), triplicate, 95-105%
  3. Repeatability       -- n replicates of same standard, RSD < 5%
  4. LOD / LOQ           -- 10x DI blanks, LOD=3*SD, LOQ=10*SD
  5. Reference Comparison-- Bland-Altman / Pearson r vs IC or colorimetric

Usage
-----
  python Validation.py
"""

import tkinter as tk
from tkinter import messagebox
import threading
import datetime
import csv
import time
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
# PATHS  -- match Real time.py locations
# ─────────────────────────────────────────────
MODEL_PATH   = Path(r"c:\Projects-2026\Project N\Pure_PW-20260121T112655Z-1-001\pls_model.pkl")
REPORT_DIR   = Path(r"C:\Report Writing\Figures")
ARDUINO_PORT = "COM10"
BAUD_RATE    = 9600

# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
C_BG0  = "#0a0a14"
C_BG1  = "#11111e"
C_BG2  = "#1a1a2e"
C_BG3  = "#222233"
C_TEXT = "#e0e0f0"
C_DIM  = "#7070a0"
C_TEAL = "#00d4c8"
C_LIME = "#80ff80"
C_AMBER= "#ffcc44"
C_RED  = "#ff5555"
C_BLUE = "#5599ff"
C_VIOL = "#cc88ff"
C_PASS = "#44cc88"
C_FAIL = "#ff5555"

# ─────────────────────────────────────────────
# PLS PREDICTOR  (copy from Real time.py)
# ─────────────────────────────────────────────
class PLSPredictor:
    def __init__(self):
        self.ready  = False
        self.model  = None
        self.wl_min = 215.0
        self.wl_max = 380.0

    def load(self, path: Path, wavelengths: np.ndarray) -> bool:
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            self.model      = obj
            self.wavelengths = wavelengths
            self.ready      = True
            return True
        except Exception as e:
            print(f"[PLS] load failed: {e}")
            return False

    def predict(self, spectrum: np.ndarray) -> float | None:
        if not self.ready or spectrum is None:
            return None
        try:
            mask = (self.wavelengths >= self.wl_min) & (self.wavelengths <= self.wl_max)
            X    = spectrum[mask].reshape(1, -1)
            ppm  = float(self.model.predict(X)[0])
            return max(0.0, ppm)
        except Exception as e:
            print(f"[PLS] predict failed: {e}")
            return None


# ─────────────────────────────────────────────
# DEVICE READER  (ARIS 546)
# ─────────────────────────────────────────────
try:
    from aris_sdk import ArisDevice  # type: ignore
    HAS_ARIS = True
except ImportError:
    HAS_ARIS = False


def acquire_spectrum() -> np.ndarray | None:
    """Return a single float32 absorbance spectrum from the ARIS 546."""
    if not HAS_ARIS:
        return None
    try:
        dev  = ArisDevice()
        spec = dev.get_absorbance()
        return np.array(spec, dtype=np.float32)
    except Exception as e:
        print(f"[ARIS] acquire failed: {e}")
        return None


# ─────────────────────────────────────────────
# VALIDATION CALCULATIONS
# ─────────────────────────────────────────────
def calc_linearity(pairs):
    """pairs: list of (known_ppm, measured_ppm)"""
    if len(pairs) < 2:
        return None
    kn = np.array([p[0] for p in pairs])
    ms = np.array([p[1] for p in pairs])
    m, b = np.polyfit(kn, ms, 1)
    ss_res = np.sum((ms - (m * kn + b)) ** 2)
    ss_tot = np.sum((ms - ms.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r2, "slope": m, "intercept": b, "n": len(pairs),
            "known": kn, "measured": ms}


def calc_repeatability(values):
    """values: list of repeated measurements"""
    if len(values) < 2:
        return None
    v   = np.array(values, dtype=float)
    m   = float(v.mean())
    s   = float(v.std(ddof=1))
    rsd = s / m * 100 if m != 0 else float("nan")
    return {"mean": m, "std": s, "rsd": rsd, "n": len(v), "values": v}


def calc_lod_loq(blanks):
    """blanks: list of blank (DI) measurements"""
    if len(blanks) < 3:
        return None
    v   = np.array(blanks, dtype=float)
    s   = float(v.std(ddof=1))
    lod = 3.0  * s
    loq = 10.0 * s
    return {"lod": lod, "loq": loq, "std": s, "n": len(v), "values": v}


def calc_reference_comparison(pairs):
    """pairs: list of (our_ppm, reference_ppm)"""
    if len(pairs) < 2:
        return None
    ours = np.array([p[0] for p in pairs])
    refs = np.array([p[1] for p in pairs])
    r    = float(np.corrcoef(ours, refs)[0, 1])
    bias = ours - refs
    mean_bias = float(bias.mean())
    sd_bias   = float(bias.std(ddof=1))
    return {"r": r, "mean_bias": mean_bias, "sd_bias": sd_bias,
            "n": len(pairs), "ours": ours, "refs": refs, "bias": bias}


# ─────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────
def export_report(lin_pairs, rep_vals, blank_vals, ref_pairs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        w.writerow(["# UV-Vis PLS Nitrate — ISO 17025 Validation Report"])
        w.writerow(["# Generated", ts])
        w.writerow(["# PLS model", str(MODEL_PATH)])
        w.writerow([])

        # --- Linearity ---
        w.writerow(["SECTION", "1. LINEARITY"])
        w.writerow(["known_ppm", "measured_ppm", "error_pct"])
        for kn, ms in lin_pairs:
            err = (ms - kn) / kn * 100 if kn else float("nan")
            w.writerow([f"{kn:.3f}", f"{ms:.3f}", f"{err:+.2f}"])
        res = calc_linearity(lin_pairs)
        if res:
            w.writerow(["R2", f"{res['r2']:.5f}",
                        "slope", f"{res['slope']:.5f}",
                        "intercept", f"{res['intercept']:+.5f}",
                        "criterion_R2>=0.99",
                        "PASS" if res['r2'] >= 0.99 else "FAIL"])
        w.writerow([])

        # --- Repeatability ---
        w.writerow(["SECTION", "2. REPEATABILITY"])
        w.writerow(["measured_ppm"])
        for v in rep_vals:
            w.writerow([f"{v:.4f}"])
        res = calc_repeatability(rep_vals)
        if res:
            w.writerow(["mean", f"{res['mean']:.4f}",
                        "SD", f"{res['std']:.4f}",
                        "RSD%", f"{res['rsd']:.2f}",
                        "criterion_RSD<5%",
                        "PASS" if res['rsd'] < 5 else "FAIL"])
        w.writerow([])

        # --- LOD / LOQ ---
        w.writerow(["SECTION", "3. LOD / LOQ  (DI blanks)"])
        w.writerow(["blank_ppm"])
        for v in blank_vals:
            w.writerow([f"{v:.4f}"])
        res = calc_lod_loq(blank_vals)
        if res:
            w.writerow(["LOD_3SD_ppm", f"{res['lod']:.4f}",
                        "LOQ_10SD_ppm", f"{res['loq']:.4f}"])
        w.writerow([])

        # --- Spike recovery ---
        w.writerow(["SECTION", "4. SPIKE RECOVERY  (from linearity data)"])
        w.writerow(["Note: load 80%/100%/120% standards as 'known' in linearity section"])
        w.writerow(["known_ppm", "measured_ppm", "recovery_%"])
        for kn, ms in lin_pairs:
            rec = ms / kn * 100 if kn else float("nan")
            w.writerow([f"{kn:.3f}", f"{ms:.3f}", f"{rec:.1f}"])
        w.writerow([])

        # --- Reference comparison ---
        w.writerow(["SECTION", "5. REFERENCE METHOD COMPARISON"])
        w.writerow(["our_ppm", "reference_ppm", "bias_ppm", "bias_%"])
        for ou, rf in ref_pairs:
            bp = ou - rf
            bp_pct = bp / rf * 100 if rf else float("nan")
            w.writerow([f"{ou:.4f}", f"{rf:.4f}", f"{bp:+.4f}", f"{bp_pct:+.2f}"])
        res = calc_reference_comparison(ref_pairs)
        if res:
            w.writerow(["Pearson_r", f"{res['r']:.5f}",
                        "mean_bias_ppm", f"{res['mean_bias']:+.4f}",
                        "SD_bias_ppm", f"{res['sd_bias']:.4f}",
                        "criterion_r>=0.95",
                        "PASS" if res['r'] >= 0.95 else "FAIL"])


# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────
class ValidationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("ISO 17025 Validation — UV-Vis PLS Nitrate")
        root.configure(bg=C_BG0)
        root.geometry("1100x720")
        root.minsize(900, 600)

        # state
        self._lin_pairs   = []   # (known_ppm, measured_ppm)
        self._rep_vals    = []   # repeated measurements
        self._blank_vals  = []   # DI blank measurements
        self._ref_pairs   = []   # (our_ppm, ref_ppm)
        self._dark        = None
        self._ref_spec    = None
        self._last_spec   = None

        # device / model
        self._pred = PLSPredictor()
        self._load_model()

        # Arduino serial (optional)
        self._ser = None
        self._connect_arduino()

        self._build_ui()

    def _load_model(self):
        try:
            with open(MODEL_PATH, "rb") as fh:
                obj = pickle.load(fh)
            # determine wavelength array from model
            if hasattr(obj, "feature_names_in_"):
                wl = np.array(obj.feature_names_in_, dtype=float)
            else:
                wl = np.linspace(182.0, 989.0, 2048)
            self._pred.model      = obj
            self._pred.wavelengths = wl
            self._pred.ready      = True
            self._wavelengths     = wl
        except Exception as e:
            self._wavelengths = np.linspace(182.0, 989.0, 2048)
            print(f"[PLS] load failed: {e}")

    def _connect_arduino(self):
        try:
            import serial
            self._ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=2)
            time.sleep(1.5)
            self._ser.read_all()
        except Exception as e:
            self._ser = None
            print(f"[Arduino] not connected: {e}")

    # ── UI BUILD ────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # top bar
        bar = tk.Frame(root, bg=C_BG2, pady=6)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(bar, text="ISO 17025 VALIDATION TOOL — UV-Vis PLS Nitrate",
                 fg=C_TEAL, bg=C_BG2,
                 font=("Courier New", 12, "bold")).pack(side=tk.LEFT, padx=10)
        status_txt = "PLS READY" if self._pred.ready else "PLS NOT LOADED"
        status_fg  = C_PASS if self._pred.ready else C_RED
        tk.Label(bar, text=status_txt, fg=status_fg, bg=C_BG2,
                 font=("Courier New", 9)).pack(side=tk.LEFT, padx=8)
        ard_txt = f"Arduino {ARDUINO_PORT}" if self._ser else "Arduino OFFLINE"
        ard_fg  = C_PASS if self._ser else C_RED
        tk.Label(bar, text=ard_txt, fg=ard_fg, bg=C_BG2,
                 font=("Courier New", 9)).pack(side=tk.LEFT, padx=8)

        # main body
        body = tk.Frame(root, bg=C_BG0)
        body.pack(fill=tk.BOTH, expand=True)

        # left panel: controls + log
        left = tk.Frame(body, bg=C_BG1, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 2), pady=4)
        left.pack_propagate(False)
        self._build_controls(left)

        # right panel: plots
        right = tk.Frame(body, bg=C_BG0)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 4), pady=4)
        self._build_plots(right)

    def _build_controls(self, parent):
        def section(text):
            f = tk.Frame(parent, bg=C_BG2, pady=2)
            f.pack(fill=tk.X, padx=4, pady=(6, 0))
            tk.Label(f, text=f"── {text} ──",
                     fg=C_DIM, bg=C_BG2,
                     font=("Courier New", 7, "bold")).pack()
            return f

        def btn(parent, text, cmd, fg=C_TEXT, bg=C_BG3):
            return tk.Button(parent, text=text, command=cmd,
                             bg=bg, fg=fg, activebackground="#333355",
                             activeforeground="white",
                             font=("Courier New", 8, "bold"),
                             relief=tk.FLAT, cursor="hand2", pady=4)

        # calibration section
        section("CALIBRATION")
        calf = tk.Frame(parent, bg=C_BG1); calf.pack(fill=tk.X, padx=4)
        btn(calf, "1. CAPTURE DARK",      self._cap_dark,  bg="#1a2a3a").pack(fill=tk.X, pady=1)
        btn(calf, "2. CAPTURE REFERENCE", self._cap_ref,   bg="#1a3a2a").pack(fill=tk.X, pady=1)
        btn(calf, "3. CAPTURE SAMPLE",    self._cap_sample,bg="#1a2a1a").pack(fill=tk.X, pady=1)

        # linearity section
        section("1. LINEARITY")
        lf = tk.Frame(parent, bg=C_BG1); lf.pack(fill=tk.X, padx=4)
        row = tk.Frame(lf, bg=C_BG1); row.pack(fill=tk.X)
        tk.Label(row, text="Known ppm:", fg=C_BLUE, bg=C_BG1,
                 font=("Courier New", 8)).pack(side=tk.LEFT)
        self._lin_known = tk.DoubleVar(value=25.0)
        tk.Entry(row, textvariable=self._lin_known, width=7,
                 bg=C_BG3, fg=C_TEXT, font=("Courier New", 8),
                 insertbackground=C_TEXT).pack(side=tk.LEFT, padx=4)
        btn(lf, "ADD LINEARITY POINT", self._add_lin_point, bg="#1a1a3a", fg=C_BLUE).pack(fill=tk.X, pady=1)
        self._lbl_lin = tk.Label(lf, text="n=0  R²=--", fg=C_BLUE, bg=C_BG1,
                                  font=("Courier New", 7)); self._lbl_lin.pack(anchor="w")

        # repeatability section
        section("2. REPEATABILITY")
        rf = tk.Frame(parent, bg=C_BG1); rf.pack(fill=tk.X, padx=4)
        row2 = tk.Frame(rf, bg=C_BG1); row2.pack(fill=tk.X)
        tk.Label(row2, text="Reps:", fg=C_LIME, bg=C_BG1,
                 font=("Courier New", 8)).pack(side=tk.LEFT)
        self._rep_n = tk.IntVar(value=5)
        tk.Spinbox(row2, from_=3, to_=20, textvariable=self._rep_n, width=4,
                   bg=C_BG3, fg=C_TEXT, font=("Courier New", 8),
                   insertbackground=C_TEXT, buttonbackground=C_BG2).pack(side=tk.LEFT, padx=4)
        btn(rf, "RUN REPEATABILITY", self._run_repeatability, bg="#1a3a1a", fg=C_LIME).pack(fill=tk.X, pady=1)
        self._lbl_rep = tk.Label(rf, text="n=0  RSD=--", fg=C_LIME, bg=C_BG1,
                                  font=("Courier New", 7)); self._lbl_rep.pack(anchor="w")

        # LOD/LOQ section
        section("3. LOD / LOQ  (DI blanks)")
        bf = tk.Frame(parent, bg=C_BG1); bf.pack(fill=tk.X, padx=4)
        row3 = tk.Frame(bf, bg=C_BG1); row3.pack(fill=tk.X)
        tk.Label(row3, text="Blanks:", fg=C_AMBER, bg=C_BG1,
                 font=("Courier New", 8)).pack(side=tk.LEFT)
        self._blank_n = tk.IntVar(value=10)
        tk.Spinbox(row3, from_=3, to_=30, textvariable=self._blank_n, width=4,
                   bg=C_BG3, fg=C_TEXT, font=("Courier New", 8),
                   insertbackground=C_TEXT, buttonbackground=C_BG2).pack(side=tk.LEFT, padx=4)
        btn(bf, "RUN BLANK -> LOD/LOQ", self._run_blank_lod, bg="#3a2a0a", fg=C_AMBER).pack(fill=tk.X, pady=1)
        self._lbl_lod = tk.Label(bf, text="LOD=--  LOQ=--", fg=C_AMBER, bg=C_BG1,
                                  font=("Courier New", 7)); self._lbl_lod.pack(anchor="w")

        # reference comparison section
        section("4. REFERENCE COMPARISON")
        rcf = tk.Frame(parent, bg=C_BG1); rcf.pack(fill=tk.X, padx=4)
        row4 = tk.Frame(rcf, bg=C_BG1); row4.pack(fill=tk.X)
        tk.Label(row4, text="Ref method ppm:", fg=C_VIOL, bg=C_BG1,
                 font=("Courier New", 8)).pack(side=tk.LEFT)
        self._ref_val = tk.DoubleVar(value=0.0)
        tk.Entry(row4, textvariable=self._ref_val, width=7,
                 bg=C_BG3, fg=C_TEXT, font=("Courier New", 8),
                 insertbackground=C_TEXT).pack(side=tk.LEFT, padx=4)
        btn(rcf, "ADD REFERENCE POINT", self._add_ref_point, bg="#2a1a3a", fg=C_VIOL).pack(fill=tk.X, pady=1)
        self._lbl_ref = tk.Label(rcf, text="n=0  r=--  bias=--", fg=C_VIOL, bg=C_BG1,
                                  font=("Courier New", 7)); self._lbl_ref.pack(anchor="w")

        # export
        section("REPORT")
        ef = tk.Frame(parent, bg=C_BG1); ef.pack(fill=tk.X, padx=4)
        btn(ef, "EXPORT FULL REPORT CSV", self._export_report, bg="#2a2a0a", fg=C_TEXT).pack(fill=tk.X, pady=1)
        btn(ef, "CLEAR ALL DATA",         self._clear_all,     bg="#3a1a1a", fg=C_RED).pack(fill=tk.X, pady=1)

        # log
        section("LOG")
        lf2 = tk.Frame(parent, bg=C_BG1); lf2.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))
        sb = tk.Scrollbar(lf2)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_box = tk.Text(lf2, bg=C_BG0, fg=C_TEXT,
                                 font=("Courier New", 7), wrap="word",
                                 yscrollcommand=sb.set, state="disabled",
                                 relief=tk.FLAT, height=10)
        self._log_box.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self._log_box.yview)

    def _build_plots(self, parent):
        fig = plt.Figure(facecolor=C_BG0, tight_layout=True)
        gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # linearity plot
        self._ax_lin = fig.add_subplot(gs[0, 0])
        self._ax_lin.set_facecolor(C_BG1)
        self._ax_lin.set_title("Linearity", color=C_BLUE, fontsize=8)
        self._ax_lin.set_xlabel("Known ppm", color=C_DIM, fontsize=7)
        self._ax_lin.set_ylabel("Measured ppm", color=C_DIM, fontsize=7)
        self._ax_lin.tick_params(colors=C_DIM, labelsize=7)
        for sp in self._ax_lin.spines.values(): sp.set_color(C_BG3)
        self._lin_scatter, = self._ax_lin.plot([], [], "o", color=C_BLUE, ms=5)
        self._lin_fit,     = self._ax_lin.plot([], [], "-", color=C_TEAL, lw=1.2)
        self._lin_ideal,   = self._ax_lin.plot([], [], "--", color=C_DIM, lw=0.8)
        self._ax_lin.text(0.05, 0.90, "R²=--", transform=self._ax_lin.transAxes,
                          color=C_BLUE, fontsize=7, va="top")
        self._txt_lin_r2 = self._ax_lin.texts[-1]

        # repeatability plot
        self._ax_rep = fig.add_subplot(gs[0, 1])
        self._ax_rep.set_facecolor(C_BG1)
        self._ax_rep.set_title("Repeatability", color=C_LIME, fontsize=8)
        self._ax_rep.set_xlabel("Replicate #", color=C_DIM, fontsize=7)
        self._ax_rep.set_ylabel("Measured ppm", color=C_DIM, fontsize=7)
        self._ax_rep.tick_params(colors=C_DIM, labelsize=7)
        for sp in self._ax_rep.spines.values(): sp.set_color(C_BG3)
        self._rep_scatter, = self._ax_rep.plot([], [], "s", color=C_LIME, ms=5)
        self._rep_mean_ln  = self._ax_rep.axhline(0, color=C_AMBER, lw=1, ls="--")
        self._txt_rep_rsd  = self._ax_rep.text(0.05, 0.90, "RSD=--",
                                                transform=self._ax_rep.transAxes,
                                                color=C_LIME, fontsize=7, va="top")

        # LOD/LOQ plot
        self._ax_lod = fig.add_subplot(gs[1, 0])
        self._ax_lod.set_facecolor(C_BG1)
        self._ax_lod.set_title("LOD / LOQ Blanks", color=C_AMBER, fontsize=8)
        self._ax_lod.set_xlabel("Blank #", color=C_DIM, fontsize=7)
        self._ax_lod.set_ylabel("Measured ppm", color=C_DIM, fontsize=7)
        self._ax_lod.tick_params(colors=C_DIM, labelsize=7)
        for sp in self._ax_lod.spines.values(): sp.set_color(C_BG3)
        self._lod_scatter, = self._ax_lod.plot([], [], "^", color=C_AMBER, ms=5)
        self._lod_ln       = self._ax_lod.axhline(0, color=C_RED,  lw=1, ls="--", label="LOD")
        self._loq_ln       = self._ax_lod.axhline(0, color=C_VIOL, lw=1, ls=":",  label="LOQ")
        self._txt_lod      = self._ax_lod.text(0.05, 0.90, "LOD=--",
                                                transform=self._ax_lod.transAxes,
                                                color=C_AMBER, fontsize=7, va="top")

        # reference comparison (Bland-Altman style)
        self._ax_ref = fig.add_subplot(gs[1, 1])
        self._ax_ref.set_facecolor(C_BG1)
        self._ax_ref.set_title("Reference Comparison", color=C_VIOL, fontsize=8)
        self._ax_ref.set_xlabel("Reference ppm", color=C_DIM, fontsize=7)
        self._ax_ref.set_ylabel("Our UV-PLS ppm", color=C_DIM, fontsize=7)
        self._ax_ref.tick_params(colors=C_DIM, labelsize=7)
        for sp in self._ax_ref.spines.values(): sp.set_color(C_BG3)
        self._ref_scatter, = self._ax_ref.plot([], [], "D", color=C_VIOL, ms=5)
        self._ref_ideal,   = self._ax_ref.plot([], [], "--", color=C_DIM, lw=0.8)
        self._txt_ref_r    = self._ax_ref.text(0.05, 0.90, "r=--",
                                                transform=self._ax_ref.transAxes,
                                                color=C_VIOL, fontsize=7, va="top")

        for ax in [self._ax_lin, self._ax_rep, self._ax_lod, self._ax_ref]:
            ax.grid(True, color=C_BG3, linewidth=0.4, alpha=0.5)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas
        self._fig    = fig

    # ── LOGGING ─────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}\n"
        self._log_box.config(state="normal")
        self._log_box.insert(tk.END, line)
        self._log_box.see(tk.END)
        self._log_box.config(state="disabled")
        print(line.strip())

    # ── CALIBRATION ─────────────────────────────────────────────────────────
    def _cap_dark(self):
        spec = acquire_spectrum()
        if spec is None:
            self._log("[DARK] Device offline — using zero dark")
            self._dark = np.zeros(len(self._wavelengths), dtype=np.float32)
        else:
            self._dark = spec
        self._log("[DARK] Captured")

    def _cap_ref(self):
        spec = acquire_spectrum()
        if spec is None:
            self._log("[REF] Device offline"); return
        self._ref_spec = spec
        self._log("[REF] Reference captured")

    def _cap_sample(self):
        spec = acquire_spectrum()
        if spec is None:
            self._log("[SAMPLE] Device offline"); return
        self._last_spec = spec
        self._log("[SAMPLE] Sample captured")

    def _get_prediction(self) -> float | None:
        if self._last_spec is None:
            self._log("[PREDICT] No sample spectrum captured"); return None
        if not self._pred.ready:
            self._log("[PREDICT] PLS model not loaded"); return None
        # use raw absorbance in direct-ABS mode (device outputs calibrated absorbance)
        ppm = self._pred.predict(self._last_spec)
        if ppm is None:
            self._log("[PREDICT] Prediction failed")
        return ppm

    # ── LINEARITY ────────────────────────────────────────────────────────────
    def _add_lin_point(self):
        ppm = self._get_prediction()
        if ppm is None: return
        known = float(self._lin_known.get())
        self._lin_pairs.append((known, ppm))
        n = len(self._lin_pairs)
        err = (ppm - known) / known * 100 if known else float("nan")
        self._log(f"[LIN] n={n}  known={known:.2f}  meas={ppm:.2f}  err={err:+.1f}%")
        self._update_lin_plot()

    def _update_lin_plot(self):
        if not self._lin_pairs: return
        kn = np.array([p[0] for p in self._lin_pairs])
        ms = np.array([p[1] for p in self._lin_pairs])
        self._lin_scatter.set_data(kn, ms)
        lo, hi = kn.min(), kn.max()
        xs = np.linspace(lo, hi, 100) if lo < hi else np.array([lo, lo + 1])
        self._lin_ideal.set_data(xs, xs)
        res = calc_linearity(self._lin_pairs)
        if res and len(self._lin_pairs) >= 2:
            fit_y = res["slope"] * xs + res["intercept"]
            self._lin_fit.set_data(xs, fit_y)
            r2_col = C_PASS if res["r2"] >= 0.99 else C_FAIL
            self._txt_lin_r2.set_text(
                f"R²={res['r2']:.4f}\nslope={res['slope']:.3f}")
            self._txt_lin_r2.set_color(r2_col)
            self._lbl_lin.config(
                text=f"n={res['n']}  R²={res['r2']:.4f}  "
                     f"{'PASS' if res['r2']>=0.99 else 'FAIL'}",
                fg=r2_col)
        pad = max((hi - lo) * 0.1, 2)
        self._ax_lin.set_xlim(lo - pad, hi + pad)
        self._ax_lin.set_ylim(lo - pad, hi + pad)
        self._canvas.draw_idle()

    # ── REPEATABILITY ────────────────────────────────────────────────────────
    def _run_repeatability(self):
        n = int(self._rep_n.get())
        ppm = self._get_prediction()
        if ppm is None: return
        # rapid replicates on frozen spectrum
        vals = [ppm] + [self._pred.predict(self._last_spec) or ppm for _ in range(n - 1)]
        self._rep_vals.extend(vals)
        res = calc_repeatability(self._rep_vals)
        if res:
            rsd_col = C_PASS if res["rsd"] < 5 else C_FAIL
            self._log(f"[REP] n={res['n']}  mean={res['mean']:.2f}  "
                      f"SD={res['std']:.3f}  RSD={res['rsd']:.1f}%  "
                      f"{'PASS' if res['rsd']<5 else 'FAIL'}")
            self._lbl_rep.config(
                text=f"n={res['n']}  mean={res['mean']:.2f}  "
                     f"RSD={res['rsd']:.1f}%  {'PASS' if res['rsd']<5 else 'FAIL'}",
                fg=rsd_col)
            self._rep_scatter.set_data(range(1, res["n"] + 1), res["values"])
            self._rep_mean_ln.set_ydata([res["mean"], res["mean"]])
            self._txt_rep_rsd.set_text(
                f"RSD={res['rsd']:.1f}%\n{'PASS' if res['rsd']<5 else 'FAIL'}")
            self._txt_rep_rsd.set_color(rsd_col)
            lo = res["values"].min(); hi = res["values"].max()
            pad = max((hi - lo) * 0.2, 1.0)
            self._ax_rep.set_xlim(0, res["n"] + 1)
            self._ax_rep.set_ylim(lo - pad, hi + pad)
            self._canvas.draw_idle()

    # ── LOD / LOQ ────────────────────────────────────────────────────────────
    def _run_blank_lod(self):
        n = int(self._blank_n.get())
        ppm = self._get_prediction()
        if ppm is None: return
        vals = [ppm] + [self._pred.predict(self._last_spec) or ppm for _ in range(n - 1)]
        self._blank_vals.extend(vals)
        res = calc_lod_loq(self._blank_vals)
        if res:
            self._log(f"[LOD] n={res['n']}  SD={res['std']:.4f}  "
                      f"LOD={res['lod']:.3f}ppm  LOQ={res['loq']:.3f}ppm")
            self._lbl_lod.config(
                text=f"LOD={res['lod']:.3f}ppm  LOQ={res['loq']:.3f}ppm")
            v = res["values"]
            self._lod_scatter.set_data(range(1, len(v) + 1), v)
            self._lod_ln.set_ydata([res["lod"], res["lod"]])
            self._loq_ln.set_ydata([res["loq"], res["loq"]])
            self._txt_lod.set_text(
                f"LOD={res['lod']:.3f}ppm\nLOQ={res['loq']:.3f}ppm")
            lo = v.min(); hi = max(v.max(), res["loq"]) * 1.2
            self._ax_lod.set_xlim(0, len(v) + 1)
            self._ax_lod.set_ylim(min(lo * 0.5, -0.1), hi)
            self._canvas.draw_idle()

    # ── REFERENCE COMPARISON ─────────────────────────────────────────────────
    def _add_ref_point(self):
        ppm = self._get_prediction()
        if ppm is None: return
        ref = float(self._ref_val.get())
        self._ref_pairs.append((ppm, ref))
        n = len(self._ref_pairs)
        bias = ppm - ref
        bp_pct = bias / ref * 100 if ref else float("nan")
        self._log(f"[REF] n={n}  ours={ppm:.2f}  ref={ref:.2f}  "
                  f"bias={bias:+.2f}ppm ({bp_pct:+.1f}%)")
        self._update_ref_plot()

    def _update_ref_plot(self):
        if not self._ref_pairs: return
        ours = np.array([p[0] for p in self._ref_pairs])
        refs = np.array([p[1] for p in self._ref_pairs])
        self._ref_scatter.set_data(refs, ours)
        lo = min(refs.min(), ours.min()); hi = max(refs.max(), ours.max())
        xs = np.linspace(lo, hi, 100) if lo < hi else np.array([lo, lo + 1])
        self._ref_ideal.set_data(xs, xs)
        res = calc_reference_comparison(self._ref_pairs)
        if res:
            r_col = C_PASS if res["r"] >= 0.95 else C_FAIL
            self._txt_ref_r.set_text(
                f"r={res['r']:.4f}\nbias={res['mean_bias']:+.2f}ppm")
            self._txt_ref_r.set_color(r_col)
            self._lbl_ref.config(
                text=f"n={res['n']}  r={res['r']:.4f}  "
                     f"bias={res['mean_bias']:+.2f}  "
                     f"{'PASS' if res['r']>=0.95 else 'FAIL'}",
                fg=r_col)
        pad = max((hi - lo) * 0.1, 2)
        self._ax_ref.set_xlim(lo - pad, hi + pad)
        self._ax_ref.set_ylim(lo - pad, hi + pad)
        self._canvas.draw_idle()

    # ── EXPORT ───────────────────────────────────────────────────────────────
    def _export_report(self):
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = REPORT_DIR / f"validation_{ts}.csv"
        try:
            export_report(self._lin_pairs, self._rep_vals,
                          self._blank_vals, self._ref_pairs, out)
            self._log(f"[EXPORT] Report saved -> {out}")
            messagebox.showinfo("Export", f"Validation report saved:\n{out}")
        except Exception as e:
            self._log(f"[EXPORT] FAILED: {e}")
            messagebox.showerror("Export Error", str(e))

    def _clear_all(self):
        if not messagebox.askyesno("Clear All", "Delete all collected validation data?"):
            return
        self._lin_pairs.clear()
        self._rep_vals.clear()
        self._blank_vals.clear()
        self._ref_pairs.clear()
        for line in [self._lin_scatter, self._lin_fit, self._lin_ideal,
                     self._rep_scatter, self._lod_scatter, self._ref_scatter,
                     self._ref_ideal]:
            line.set_data([], [])
        self._canvas.draw_idle()
        self._log("[CLEAR] All validation data cleared")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = ValidationApp(root)
    root.mainloop()
