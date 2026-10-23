import usb.core
import usb.util
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import messagebox
import threading
import time
from scipy.signal import savgol_filter

# --- 1. SPECTROMETER DRIVER (With Auto-Recovery) ---
class Aris546ZioDriver:
    def __init__(self):
        self.dev = usb.core.find(idVendor=0x354F)
        if self.dev is None:
            raise ConnectionError("Aris 546 not found.")
        self.dev.set_configuration()
        self.pixels = 2048

    def get_spectrum(self, exp_ms=2.0):
        try:
            exp_us = int(exp_ms * 1000)
            self.dev.write(0x01, struct.pack('<II', 0x1100, exp_us))
            self.dev.write(0x01, struct.pack('<II', 0x0004, 1))
            time.sleep(0.02)
            
            data_reply = self.dev.read(0x81, 16492, timeout=1000)
            
            if len(data_reply) < 8260: return None
            
            spectrum = np.frombuffer(data_reply[68:68 + (self.pixels * 4)], dtype=np.float32)
            return spectrum if spectrum.size == 2048 else None
        
        except usb.core.USBError:
            try: self.dev.clear_halt(0x81)
            except: pass
            return None
        except Exception:
            return None

# --- 2. MAIN GUI APPLICATION ---
class NitrateSpectrometerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nitrate Pro - Realtime")
        self.root.geometry("1500x900")
        self.root.configure(bg='#0F0F0F')

        self.wavelengths = np.linspace(182.98, 988.74, 2048)
        self.dark, self.ref, self.sample = None, None, None
        self.current_spectrum = np.zeros(2048)
        
        try:
            self.hw = Aris546ZioDriver()
            st, clr = "HARDWARE READY ✅", "#00FF41"
        except:
            self.hw, st, clr = None, "OFFLINE MODE ❌", "#FF3131"

        tk.Label(root, text=st, fg=clr, bg='#0F0F0F', font=('Consolas', 12, 'bold')).pack(pady=5)

        self.setup_ui()
        self.setup_plots()
        if self.hw:
            self.start_loop()

    def setup_ui(self):
        main = tk.Frame(self.root, bg='#0F0F0F')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        sidebar = tk.Frame(main, bg='#1A1A1A', padx=10, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        btn_cfg = {"font": ("Segoe UI", 9, "bold"), "width": 22, "height": 2, "fg": "white"}
        tk.Button(sidebar, text="CAPTURE DARK", bg="#37474F", command=self.cap_dark, **btn_cfg).pack(pady=3)
        tk.Button(sidebar, text="CAPTURE REF", bg="#2E7D32", command=self.cap_ref, **btn_cfg).pack(pady=3)
        tk.Button(sidebar, text="CAPTURE SAMPLE", bg="#E65100", command=self.cap_samp, **btn_cfg).pack(pady=3)
        tk.Button(sidebar, text="RUN ANALYSIS", bg="#880E4F", command=self.analyze, **btn_cfg).pack(pady=10)

        self.fig = Figure(figsize=(12, 8), facecolor="#0F0F0F")
        self.canvas = FigureCanvasTkAgg(self.fig, main)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.ax1 = self.fig.add_subplot(221); self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223); self.ax4 = self.fig.add_subplot(224)

    def setup_plots(self):
        self.line1, = self.ax1.plot(self.wavelengths, self.current_spectrum, color="cyan", lw=1)
        self.line_d, = self.ax2.plot(self.wavelengths, self.wavelengths*0, color="#555", label="Dark")
        self.line_r, = self.ax2.plot(self.wavelengths, self.wavelengths*0, color="#00FF41", label="Ref")
        self.line_s, = self.ax2.plot(self.wavelengths, self.wavelengths*0, color="#FF9100", label="Sample")
        self.line_a, = self.ax3.plot(self.wavelengths, self.wavelengths*0, color="#FF00FF", lw=1.5)
        
        for ax, t in zip([self.ax1, self.ax2, self.ax3, self.ax4], ["LIVE FEED", "BASELINE", "ABSORBANCE", "NITRATE PPM"]):
            ax.set_title(t, color="white", fontsize=9, fontweight='bold')
            ax.set_facecolor("#000")
            ax.tick_params(colors="#666", labelsize=8)
            ax.grid(True, alpha=0.1)
        self.ax2.legend(fontsize=7, facecolor="#000", labelcolor="white")
        self.fig.tight_layout()

    def start_loop(self):
        def loop():
            while True:
                data = self.hw.get_spectrum()
                if data is not None:
                    self.current_spectrum = data
                    self.root.after(0, self.update_ui)
                time.sleep(0.05)
        threading.Thread(target=loop, daemon=True).start()

    def update_ui(self):
        self.line1.set_ydata(self.current_spectrum)
        self.ax1.set_ylim(0, np.max(self.current_spectrum) * 1.1 + 10)
        if self.dark is not None: self.line_d.set_ydata(self.dark)
        if self.ref is not None: self.line_r.set_ydata(self.ref)
        display_s = self.sample if self.sample is not None else self.current_spectrum
        self.line_s.set_ydata(display_s)
        self.canvas.draw_idle()

    def analyze(self):
        if self.dark is None or self.ref is None:
            messagebox.showwarning("Error", "Capture Dark and Ref first")
            return
        target = self.sample if self.sample is not None else self.current_spectrum
        ratio = (target - self.dark) / (self.ref - self.dark + 1e-7)
        absorbance = savgol_filter(-np.log10(np.clip(ratio, 0.001, 2.0)), 21, 3)
        self.line_a.set_ydata(absorbance)
        self.ax3.set_xlim(200, 350)
        mask = (self.wavelengths >= 200) & (self.wavelengths <= 350)
        self.ax3.set_ylim(min(absorbance[mask]) - 0.02, max(absorbance[mask]) + 0.05)
        
        ppm = max(0, absorbance[150] * 12.5) 
        self.ax4.clear()
        self.ax4.bar(["Nitrate"], [ppm], color="#880E4F")
        self.ax4.set_title(f"Result: {ppm:.2f} PPM", color="white")
        self.ax4.set_facecolor("#000")
        self.canvas.draw_idle()

    def cap_dark(self): self.dark = self.current_spectrum.copy()
    def cap_ref(self): self.ref = self.current_spectrum.copy()
    def cap_samp(self): self.sample = self.current_spectrum.copy()

if __name__ == "__main__":
    root = tk.Tk()
    app = NitrateSpectrometerGUI(root)
    root.mainloop()