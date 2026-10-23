import usb.core
import usb.util
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import time
import pandas as pd
from datetime import datetime
from scipy.signal import savgol_filter

# --- 1. SPECTROMETER DRIVER ---
class Aris546ZioDriver:
    def __init__(self):
        # Finds the Aris 546 based on the specific Vendor ID
        self.dev = usb.core.find(idVendor=0x354F)
        if self.dev is None:
            raise ConnectionError("Aris 546 not found.")
        self.dev.set_configuration()
        self.pixels = 2048

    def get_spectrum(self, exp_ms=2.0):
        try:
            # Set integration time and trigger
            self.dev.write(0x01, struct.pack('<II', 0x1100, int(exp_ms * 1000)))
            self.dev.write(0x01, struct.pack('<II', 0x0004, 1))
            
            ready = False
            for _ in range(50):
                self.dev.write(0x01, struct.pack('<I', 0x3000) + b'\x00'*4)
                reply = self.dev.read(0x81, 16, timeout=500)
                if len(reply) >= 8:
                    status_val = struct.unpack('<I', reply[4:8])[0]
                    if (status_val >> 8) & 0xFFFF > 0:
                        ready = True
                        break
                time.sleep(0.01)

            if not ready: return None

            self.dev.write(0x01, struct.pack('<I', 0x4000) + b'\x00'*4)
            data_reply = self.dev.read(0x81, 16492, timeout=2000)
            
            if len(data_reply) >= 8260:
                return np.frombuffer(data_reply[68:68 + (self.pixels * 4)], dtype=np.float32)
            return None
        except:
            return None

# --- 2. GUI ---
class NitrateSpectrometerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nitrate Analyzer Pro - LIVE ABSORBANCE")
        self.root.geometry("1500x900")
        self.root.configure(bg='#0F0F0F')

        # X-Axis Data
        self.wavelengths = np.linspace(182.98, 988.74, 2048)
        self.dark, self.ref, self.sample = None, None, None
        self.current_spectrum = np.zeros(2048)

        # Connection Check
        try:
            self.hw = Aris546ZioDriver()
            status, color = "SYSTEM ONLINE ✅", "#00FF41"
        except:
            self.hw, status, color = None, "OFFLINE MODE ❌", "#FF3131"

        tk.Label(root, text=status, fg=color, bg='#0F0F0F',
                 font=('Consolas', 12, 'bold')).pack(pady=5)

        self.setup_ui()
        self.setup_plots()

        if self.hw:
            threading.Thread(target=self.realtime_loop, daemon=True).start()

    def setup_ui(self):
        main = tk.Frame(self.root, bg='#0F0F0F')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Sidebar
        sidebar = tk.Frame(main, bg='#1A1A1A', padx=10, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        btn_style = {"font": ("Segoe UI", 9, "bold"), "width": 22, "height": 2, "fg": "white"}
        tk.Button(sidebar, text="CAPTURE DARK", bg="#37474F", command=self.cap_dark, **btn_style).pack(pady=3)
        tk.Button(sidebar, text="CAPTURE REF", bg="#2E7D32", command=self.cap_ref, **btn_style).pack(pady=3)
        tk.Button(sidebar, text="CAPTURE SAMPLE", bg="#E65100", command=self.cap_samp, **btn_style).pack(pady=3)
        tk.Button(sidebar, text="🔬 RUN ANALYSIS", bg="#880E4F", command=self.analyze, **btn_style).pack(pady=10)

        self.log_box = tk.Text(sidebar, height=10, width=25, bg="#000", fg="#00FF41", font=("Consolas", 9))
        self.log_box.pack(pady=10)

        # Matplotlib Figure
        self.fig = Figure(figsize=(12, 8), facecolor="#0F0F0F")
        self.canvas = FigureCanvasTkAgg(self.fig, main)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, main)
        self.toolbar.update()

        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)

    def setup_plots(self):
        self.line1, = self.ax1.plot(self.wavelengths, self.current_spectrum, color="cyan", lw=1)
        self.line_d, = self.ax2.plot(self.wavelengths, self.wavelengths*0, color="#555", label="Dark")
        self.line_r, = self.ax2.plot(self.wavelengths, self.wavelengths*0, color="#00FF41", label="Ref")
        self.line_s, = self.ax2.plot(self.wavelengths, self.wavelengths*0, color="#FF9100", label="Sample")
        self.line_a, = self.ax3.plot(self.wavelengths, self.wavelengths*0, color="#FF00FF", lw=1.5)

        for ax, title in zip([self.ax1, self.ax2, self.ax3, self.ax4], 
                             ["LIVE ABSORBANCE", "COMPARISON (RAW)", "ANALYSIS WINDOW", "RESULT"]):
            ax.set_title(title, color="white", fontsize=9, fontweight='bold')
            ax.set_facecolor("#000")
            ax.tick_params(colors="#888", labelsize=8)
            ax.grid(True, alpha=0.1)
            
            # Initial y-limits to avoid (0,0) startup
            if ax != self.ax4:
                ax.set_ylim(-0.1, 1.2)

        self.ax2.legend(fontsize=7, facecolor="#000", labelcolor="white")
        self.fig.tight_layout()

        # Bind Mouse Scroll Zoom
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    # --- Interaction Logic ---
    def on_scroll(self, event):
        """Scroll wheel zoom centered on cursor position"""
        if event.inaxes is None: return
        ax = event.inaxes
        base_scale = 1.25
        
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        if event.button == 'up': scale_factor = 1 / base_scale
        elif event.button == 'down': scale_factor = base_scale
        else: scale_factor = 1

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        # Determine relative distance of cursor from axes limits
        rel_x = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([event.xdata - new_width * (1 - rel_x), event.xdata + new_width * rel_x])
        ax.set_ylim([event.ydata - new_height * (1 - rel_y), event.ydata + new_height * rel_y])
        self.canvas.draw_idle()

    def realtime_loop(self):
        while True:
            data = self.hw.get_spectrum()
            if data is not None:
                self.current_spectrum = data
                self.root.after(0, self.update_plots)
            time.sleep(0.05)

    def update_plots(self):
        # 1. Update Live Absorbance
        if self.dark is not None and self.ref is not None:
            ratio = (self.current_spectrum - self.dark) / (self.ref - self.dark + 1e-7)
            abs_data = -np.log10(np.clip(ratio, 0.001, 2.0))
            abs_data = savgol_filter(abs_data, 21, 3)
            self.line1.set_ydata(abs_data)
        
        # 2. Update Comparison (RAW) Lines
        if self.dark is not None: self.line_d.set_ydata(self.dark)
        if self.ref is not None: self.line_r.set_ydata(self.ref)
        
        # Orange line shows live data until 'Capture Sample' is pressed
        display_raw = self.sample if self.sample is not None else self.current_spectrum
        self.line_s.set_ydata(display_raw)

        self.canvas.draw_idle()

    # --- Button Commands ---
    def cap_dark(self):
        self.dark = self.current_spectrum.copy()
        self.log("Dark Captured")
        self.auto_scale_comparison()

    def cap_ref(self):
        self.ref = self.current_spectrum.copy()
        self.log("Ref Captured")
        self.auto_scale_comparison()

    def cap_samp(self):
        self.sample = self.current_spectrum.copy()
        self.log("Sample Captured")
        self.auto_scale_comparison()

    def auto_scale_comparison(self):
        """Scales the comparison plot once when data is captured"""
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw_idle()

    def analyze(self):
        if self.dark is None or self.ref is None:
            messagebox.showwarning("Error", "Need Dark & Ref baseline first."); return
        
        # Use Sample data if captured, otherwise use live
        target = self.sample if self.sample is not None else self.current_spectrum
        ratio = (target - self.dark) / (self.ref - self.dark + 1e-7)
        abs_final = savgol_filter(-np.log10(np.clip(ratio, 0.001, 2.0)), 21, 3)

        self.line_a.set_ydata(abs_final)
        
        # Set viewport for Nitrate detection range
        self.ax3.set_xlim(200, 400)
        mask = (self.wavelengths >= 200) & (self.wavelengths <= 400)
        self.ax3.set_ylim(np.min(abs_final[mask])-0.05, np.max(abs_final[mask])+0.1)

        # Simplified concentration calculation (example coefficient)
        ppm = max(0, abs_final[150] * 10.5)

        self.ax4.clear()
        self.ax4.bar(["Nitrate"], [ppm], color="#880E4F")
        self.ax4.set_title(f"CONC: {ppm:.2f} PPM", color="white", fontweight='bold')
        self.ax4.set_facecolor("#000")
        self.ax4.tick_params(colors="white")

        self.canvas.draw_idle()

    def log(self, msg):
        self.log_box.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_box.see(tk.END)

# --- EXECUTION ---
if __name__ == "__main__":
    root = tk.Tk()
    app = NitrateSpectrometerGUI(root)
    root.mainloop()