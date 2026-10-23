"""
ARIS Demo  -  Automatic Screenshot + Video Capture
===================================================
Launches the PPT demo simulation, takes timed screenshots at key sequence
moments, and records a full MP4 / AVI video of the first two sample cycles.

Output:  C:\\Report Writing\\Figures\\Demo_Captures\\
Run:     python Capture_Demo.py
"""

import threading, time, sys
from pathlib import Path

import numpy as np
import cv2
from PIL import ImageGrab
import tkinter as tk

# ---- import the demo app (same directory) ----------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import Demo_Simulation as _dm

# ---- output folder ---------------------------------------------------------
OUT = Path(r"C:\Report Writing\Figures\Demo_Captures")
OUT.mkdir(parents=True, exist_ok=True)

# ---- screenshot schedule ---------------------------------------------------
# (delay_ms_after_launch, filename_stem, label_for_log)
# Aligned to the demo sequence timing (sequence starts at ~1500 ms):
#   Step1 ends ~3.9s, Ref captured ~7.5s, Sample loaded ~16s, Measured ~16.6s
SCREENSHOTS = [
    ( 3_500, "01_startup_DI_flowing",    "Startup - DI water baseline"),
    ( 9_500, "02_reference_captured",    "DI reference spectrum captured"),
    (14_000, "03_sample_flushing",       "Sample flushing through cell"),
    (17_200, "04_sample_in_NO3_peak",    "Sample in cell - NO3 peak 220nm"),
    (18_500, "05_measurement_result",    "PLS measurement result"),
    (23_500, "06_calib_overlay_complete","Calibration overlay - ref + sample"),
    (40_000, "07_second_sample_trend",   "Second sample - trend building"),
    (63_000, "08_third_sample_full_UI",  "Third sample - full UI"),
]

# Video: record until this many milliseconds after launch
VIDEO_DURATION_MS = 75_000   # ~75 s  (about 4 full sample cycles)
VIDEO_FPS         = 8        # capture rate


# ============================================================================
class CaptureApp(_dm.App):
    """Demo app subclass with automatic screenshot + video capture."""

    def __init__(self, root: tk.Tk):
        super().__init__(root)

        self._frames: list[np.ndarray] = []
        self._recording = True
        self._win_ready = False

        # schedule screenshot moments
        for delay_ms, stem, label in SCREENSHOTS:
            self.root.after(delay_ms, self._take_shot, stem, label)

        # stop recording after VIDEO_DURATION_MS
        self.root.after(VIDEO_DURATION_MS, self._stop_recording)

        # kick off frame-capture loop (runs in main thread via root.after)
        self.root.after(2000, self._capture_frame_loop)  # start after 2 s

        self._log("[CAP] Screenshot + video capture ACTIVE")
        self._log(f"[CAP] Output: {OUT}")

    # ---- window geometry helper ----------------------------------------

    def _window_bbox(self):
        self.root.update_idletasks()
        x  = self.root.winfo_rootx()
        y  = self.root.winfo_rooty()
        w  = self.root.winfo_width()
        h  = self.root.winfo_height()
        return (x, y, x + w, y + h)

    # ---- screenshot -------------------------------------------------------

    def _take_shot(self, stem: str, label: str):
        try:
            bbox = self._window_bbox()
            if bbox[2] - bbox[0] < 200:
                return
            img = ImageGrab.grab(bbox=bbox)
            path = OUT / f"{stem}.png"
            img.save(path)
            self._log(f"[CAP] Screenshot: {stem}.png")
            print(f"  [SHOT] {stem}.png  ({label})")
        except Exception as e:
            self._log(f"[CAP] Screenshot failed: {e}")

    # ---- video frame loop (main thread) -----------------------------------

    def _capture_frame_loop(self):
        if not self._recording or not self.is_running:
            return
        try:
            bbox = self._window_bbox()
            if bbox[2] - bbox[0] > 200:
                img = ImageGrab.grab(bbox=bbox)
                self._frames.append(np.array(img))
        except Exception:
            pass
        self.root.after(int(1000 / VIDEO_FPS), self._capture_frame_loop)

    # ---- stop + save video ------------------------------------------------

    def _stop_recording(self):
        self._recording = False
        n = len(self._frames)
        self._log(f"[CAP] Recording stopped  ({n} frames @ {VIDEO_FPS} fps)")
        print(f"\n  [VIDEO] Captured {n} frames, saving...")
        threading.Thread(target=self._save_video, daemon=True).start()

    def _save_video(self):
        if not self._frames:
            print("  [VIDEO] No frames captured.")
            return

        frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in self._frames]
        h, w = frames_bgr[0].shape[:2]

        # try MP4 first (mp4v), fall back to XVID .avi
        mp4_path = str(OUT / "Demo_Video.mp4")
        avi_path = str(OUT / "Demo_Video.avi")

        saved = False
        for path, fourcc_str, ext in [
            (mp4_path, "mp4v", "mp4"),
            (avi_path, "XVID", "avi"),
        ]:
            try:
                writer = cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*fourcc_str),
                    float(VIDEO_FPS),
                    (w, h)
                )
                if not writer.isOpened():
                    continue
                for frame in frames_bgr:
                    writer.write(frame)
                writer.release()
                size_mb = Path(path).stat().st_size / 1_048_576
                print(f"  [VIDEO] Saved: {Path(path).name}  ({size_mb:.1f} MB)")
                self.root.after(0, lambda p=path: self._log(
                    f"[CAP] Video saved: {Path(p).name}"))
                saved = True
                break
            except Exception as e:
                print(f"  [VIDEO] {ext} failed: {e}")

        if not saved:
            print("  [VIDEO] Could not save video.")

        # print summary
        print(f"\n{'='*50}")
        print(f"  CAPTURE COMPLETE")
        print(f"  Output folder: {OUT}")
        print(f"  Screenshots:   {len(SCREENSHOTS)}")
        print(f"  Video frames:  {len(self._frames)}")
        print(f"{'='*50}")


# ============================================================================
if __name__ == "__main__":
    print(f"\n  ARIS Demo Capture")
    print(f"  Output: {OUT}")
    print(f"  Recording: {VIDEO_DURATION_MS // 1000} seconds of video")
    print(f"  Screenshots: {len(SCREENSHOTS)} timed moments")
    print(f"  Window will close automatically after recording.\n")

    root = tk.Tk()
    root.geometry("1600x950")
    root.state("zoomed")
    app = CaptureApp(root)

    # auto-close after video is saved (add buffer after recording stops)
    root.after(VIDEO_DURATION_MS + 15_000, root.destroy)

    root.mainloop()
    print("\n  Done. Check output folder.")
