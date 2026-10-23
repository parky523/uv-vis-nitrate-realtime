"""
Spectral QC Viewer
------------------
Interactive browser for all .spz spectra in the Most Recent folder.
Use this to identify noisy or baseline-shifted spectra before retraining.

Controls:
  [Prev] / [Next]   - navigate through spectra
  [Sort: Alpha]     - sort alphabetically
  [Sort: Noise]     - worst noise first  (SG1 std in 390-450 nm)
  [Sort: Baseline]  - worst baseline offset first (mean abs 400-450 nm)
  Click any line in the overview panel  - jump to that spectrum
"""
import sys, os, re, warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from scipy.signal import savgol_filter
from glob import glob

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Projects-2026\Project N\Pure_PW-20260121T112655Z-1-001\Pure_PW\Most Recent"

WV_MODEL_START    = 215   # nm — PLS model region (shaded)
WV_MODEL_END      = 380
WV_BASELINE_START = 400   # nm — should be flat near 0 for clean porewater
WV_BASELINE_END   = 450

EXCLUDE = {
    "PW_29", "PW_2",
    "PW_1F1",
    "PW_2F", "PW_2F0", "PW_2F0.1", "PW_2F0.2", "PW_2F2",
    "PW_4F0", "PW_4F2",
    "PW_7F0",
    "PW_29.1", "PW_29.2", "PW_29.3",
    "PW_51A", "PW_42E", "PW_51B", "PW_45", "PW_1",
}

TARGET_MAP = {
    "PW_04":"27","PW_1":"43","PW_10":"44","PW_11":"26","PW_12":"34","PW_13":"52",
    "PW_14":"50","PW_15":"56","PW_16":"7","PW_17":"73","PW_17A":"82","PW_17B":"12",
    "PW_18":"42","PW_19":"64","PW_1B":"5","PW_1C":"46","PW_1D":"21","PW_1E":"47",
    "PW_1F1":"27","PW_2":"80","PW_20":"46","PW_20A":"27","PW_20B":"33","PW_20C":"28",
    "PW_21":"44","PW_22A":"24","PW_22B":"24","PW_22C":"24","PW_22D":"28","PW_22E":"42",
    "PW_22F":"26","PW_22G":"76","PW_22H":"23","PW_22I":"30","PW_22J":"22","PW_23":"42",
    "PW_23A":"46","PW_24":"90","PW_24A":"25","PW_24B":"22","PW_24C":"4","PW_24D":"12",
    "PW_24E":"25","PW_24F":"23","PW_26":"32","PW_27":"41","PW_28":"45","PW_28A":"34",
    "PW_28B":"52","PW_28C":"32","PW_29":"74","PW_29.1":"26","PW_29.2":"29","PW_29.3":"25",
    "PW_2B":"11","PW_2C":"28","PW_2D":"4","PW_2E":"20","PW_2F":"25","PW_2F0":"26",
    "PW_2F0.1":"43","PW_2F0.2":"28","PW_2F2":"40","PW_3":"27","PW_30":"46","PW_30A":"36",
    "PW_31":"51","PW_31A":"13","PW_31B":"10","PW_31C":"10","PW_31D":"11","PW_31E":"11",
    "PW_31F":"18","PW_31G":"8","PW_31H":"16","PW_32":"48","PW_32A":"46","PW_36":"53",
    "PW_36A":"75","PW_37":"44","PW_37A":"54","PW_38":"56","PW_38A":"24","PW_38B":"12",
    "PW_38C":"12","PW_39":"47","PW_39A":"90","PW_39B":"38","PW_39C":"35","PW_39D":"23",
    "PW_39E":"23","PW_39E.2":"19","PW_39F":"11","PW_39G":"9","PW_39I":"12","PW_39J":"54",
    "PW_3A":"82","PW_3B":"5","PW_40":"30","PW_40A":"11","PW_40B":"11","PW_40C":"9",
    "PW_40D":"9","PW_40E":"18","PW_41":"68","PW_42":"90","PW_42A":"37","PW_42B":"29",
    "PW_42C":"29","PW_42D":"21","PW_42E":"20","PW_42F":"26","PW_42I":"36","PW_42J":"22",
    "PW_43":"73","PW_44":"41","PW_45":"24","PW_45A":"72","PW_45B":"10","PW_45C":"6",
    "PW_45D":"8","PW_45E":"9","PW_45G":"11","PW_46":"46","PW_47":"22","PW_47A":"32",
    "PW_49":"74","PW_4F0":"35","PW_4F2":"24","PW_5":"54","PW_50":"56","PW_51":"77",
    "PW_51A":"136","PW_51B":"32","PW_51C":"90","PW_51D":"18","PW_51E":"20","PW_52":"50",
    "PW_52A":"24","PW_52B":"27","PW_52C":"18","PW_52D":"33","PW_53":"23","PW_53A":"117",
    "PW_53B":"58","PW_54.2":"35","PW_54.3":"38","PW_6":"41","PW_6A":"50","PW_7":"44",
    "PW_7A":"60","PW_7F0":"66","PW_8":"53","PW_9":"48","PW_HS1":"42",
}

# ─── LOAD + METRICS ──────────────────────────────────────────────────────────

def load_spz(fp):
    df = pd.read_csv(fp, sep="\t", skiprows=25, header=None, encoding="latin-1")
    wv = df.iloc[:, 0].values.astype(float)
    ab = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).values
    return wv, ab


def compute_metrics(wv, ab):
    """
    noise_sg   : std of SG1 derivative in 390-450 nm.
                 Higher => noisier spectrum. Threshold ~0.0015 is a rough red flag.
    base_mean  : mean absorbance in 400-450 nm.
                 Should be near 0. |base_mean| > 0.05 suggests a baseline offset.
    base_std   : std of absorbance in 400-450 nm.
                 High => fluctuating baseline.
    min_model  : minimum raw absorbance in 215-380 nm.
                 Strongly negative values can indicate lamp/reference issues.
    """
    mask_b = (wv >= WV_BASELINE_START) & (wv <= WV_BASELINE_END)
    if mask_b.sum() > 1:
        base_vals = ab[mask_b]
        base_mean = float(np.mean(base_vals))
        base_std  = float(np.std(base_vals))
    else:
        base_mean = base_std = 0.0

    mask_n = (wv >= 390) & (wv <= 450)
    if mask_n.sum() > 11:
        sg = savgol_filter(ab[mask_n], window_length=11, polyorder=2, deriv=1)
        noise_sg = float(np.std(sg))
    else:
        noise_sg = 0.0

    mask_m = (wv >= WV_MODEL_START) & (wv <= WV_MODEL_END)
    min_model = float(np.min(ab[mask_m])) if mask_m.sum() > 0 else 0.0

    return dict(noise_sg=noise_sg, base_mean=base_mean,
                base_std=base_std, min_model=min_model)


print("Loading spectra...")
spectra = []
spz_files = sorted(glob(os.path.join(DATA_PATH, "*.spz")))
for fp in spz_files:
    sid = os.path.splitext(os.path.basename(fp))[0]
    if sid.lower() == "a":
        continue
    try:
        wv, ab = load_spz(fp)
        m = compute_metrics(wv, ab)
        if sid in TARGET_MAP and sid not in EXCLUDE:
            cat = "model"
            ic  = float(TARGET_MAP[sid])
        elif sid in EXCLUDE:
            cat = "excluded"
            ic  = float(TARGET_MAP.get(sid, 0))
        else:
            cat = "other"
            ic  = None
        spectra.append(dict(name=sid, wv=wv, ab=ab, cat=cat, ic=ic, **m))
    except Exception as e:
        print(f"  [WARN] {sid}: {e}")

print(f"Loaded {len(spectra)} spectra  "
      f"({sum(1 for s in spectra if s['cat']=='model')} model  "
      f"{sum(1 for s in spectra if s['cat']=='excluded')} excluded  "
      f"{sum(1 for s in spectra if s['cat']=='other')} other)")

# ─── SORT STATE ──────────────────────────────────────────────────────────────
sort_orders = {
    "alpha":    sorted(range(len(spectra)), key=lambda i: spectra[i]["name"]),
    "noise":    sorted(range(len(spectra)), key=lambda i: -spectra[i]["noise_sg"]),
    "baseline": sorted(range(len(spectra)), key=lambda i: -abs(spectra[i]["base_mean"])),
}
state = {"sort": "alpha", "pos": 0}

def get_order():
    return sort_orders[state["sort"]]

def current():
    return spectra[get_order()[state["pos"]]]

# ─── THRESHOLDS for color-coding quality ─────────────────────────────────────
NOISE_WARN  = np.percentile([s["noise_sg"]  for s in spectra], 75)
NOISE_BAD   = np.percentile([s["noise_sg"]  for s in spectra], 90)
BASE_WARN   = np.percentile([abs(s["base_mean"]) for s in spectra], 75)
BASE_BAD    = np.percentile([abs(s["base_mean"]) for s in spectra], 90)

def quality_color(s):
    """Red=bad, orange=warn, green=ok."""
    n, b = s["noise_sg"], abs(s["base_mean"])
    if n >= NOISE_BAD or b >= BASE_BAD:
        return "#ff4444"
    if n >= NOISE_WARN or b >= BASE_WARN:
        return "#ffaa00"
    return "#44cc44"

# ─── FIGURE ──────────────────────────────────────────────────────────────────
BG_DARK  = "#0f0f1e"
BG_PANEL = "#12122a"
FG_TEXT  = "#ddddee"
ACCENT   = "#7777ff"

CAT_COL  = {"model": "#5599ff", "excluded": "#ff8833", "other": "#888899"}

fig = plt.figure(figsize=(17, 10), facecolor=BG_DARK)
gs  = gridspec.GridSpec(
    2, 2, figure=fig,
    height_ratios=[1.5, 1],
    hspace=0.40, wspace=0.28,
    left=0.06, right=0.97, top=0.93, bottom=0.10,
)

ax_over = fig.add_subplot(gs[0, :])   # top: all spectra
ax_det  = fig.add_subplot(gs[1, 0])   # bottom-left: selected detail
ax_qual = fig.add_subplot(gs[1, 1])   # bottom-right: quality bar chart

for ax in (ax_over, ax_det, ax_qual):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=FG_TEXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#334")

# ── Overview: all spectra ─────────────────────────────────────────────────────
ov_lines = {}   # name -> Line2D (for pick)
for s in spectra:
    ln, = ax_over.plot(
        s["wv"], s["ab"],
        color=CAT_COL[s["cat"]], lw=0.5, alpha=0.25,
        picker=4,                          # 4 pt pick radius
        label=s["name"],
    )
    ov_lines[s["name"]] = ln

ax_over.axvspan(WV_MODEL_START, WV_MODEL_END, color="#ffffff", alpha=0.04, zorder=0)
ax_over.axvline(WV_MODEL_START, color=ACCENT, lw=0.8, ls="--", alpha=0.6)
ax_over.axvline(WV_MODEL_END,   color=ACCENT, lw=0.8, ls="--", alpha=0.6)

sel_ov, = ax_over.plot([], [], color="#ff3333", lw=2.0, zorder=10)
ax_over.set_xlim(180, 500)
ax_over.set_xlabel("Wavelength (nm)", color=FG_TEXT, fontsize=9)
ax_over.set_ylabel("Absorbance (AU)", color=FG_TEXT, fontsize=9)
ax_over.set_title(
    "All Spectra — blue=model  orange=excluded  grey=other  |  click a line to select",
    color=FG_TEXT, fontsize=9, pad=4,
)

# legend patch for categories
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor=CAT_COL["model"],    label="In model"),
    Patch(facecolor=CAT_COL["excluded"], label="Excluded"),
    Patch(facecolor=CAT_COL["other"],    label="Other/dilution"),
    Patch(facecolor="#ff3333",           label="Selected"),
]
ax_over.legend(handles=legend_els, loc="upper right",
               facecolor="#1a1a30", edgecolor="#444", labelcolor=FG_TEXT,
               fontsize=7, framealpha=0.8)

# ── Detail: selected spectrum ─────────────────────────────────────────────────
# faint background of all spectra
for s in spectra:
    ax_det.plot(s["wv"], s["ab"], color="#2a2a45", lw=0.4, alpha=0.6)

ax_det.axvspan(WV_MODEL_START, WV_MODEL_END, color="#ffffff", alpha=0.05, zorder=0)
ax_det.axvline(WV_MODEL_START, color=ACCENT, lw=0.8, ls="--", alpha=0.5)
ax_det.axvline(WV_MODEL_END,   color=ACCENT, lw=0.8, ls="--", alpha=0.5)
ax_det.axhline(0, color="#555577", lw=0.6, ls=":")

sel_det, = ax_det.plot([], [], color="#ff5555", lw=1.8, zorder=10)
ax_det.set_xlabel("Wavelength (nm)", color=FG_TEXT, fontsize=9)
ax_det.set_ylabel("Absorbance (AU)", color=FG_TEXT, fontsize=9)

info_box = ax_det.text(
    0.98, 0.97, "",
    transform=ax_det.transAxes, va="top", ha="right",
    color="#ffff99", fontsize=7.5, fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.4", fc="#000011", ec="#445", alpha=0.85),
    zorder=20,
)

# ── Quality bar chart ─────────────────────────────────────────────────────────
# Sort by noise descending so worst are on the left
bar_order  = sorted(range(len(spectra)), key=lambda i: -spectra[i]["noise_sg"])
bar_specs  = [spectra[i] for i in bar_order]
bar_vals   = [s["noise_sg"] for s in bar_specs]
bar_cols   = [quality_color(s) for s in bar_specs]
bar_names  = [s["name"] for s in bar_specs]
N_bar      = len(bar_specs)

ax_qual.bar(range(N_bar), bar_vals, color=bar_cols, width=1.0, align="center")
sel_vline = ax_qual.axvline(-1, color="#ffffff", lw=1.5, ls="-", alpha=0.8, zorder=5)

ax_qual.set_xlim(-0.5, N_bar - 0.5)
ax_qual.set_xlabel("Samples (sorted: worst noise first)", color=FG_TEXT, fontsize=8)
ax_qual.set_ylabel("SG1 noise score", color=FG_TEXT, fontsize=8)
ax_qual.set_title(
    "Noise quality  red=bad  orange=warn  green=ok  |  click bar to select",
    color=FG_TEXT, fontsize=8, pad=4,
)
ax_qual.tick_params(labelbottom=False)

# Threshold lines
ax_qual.axhline(NOISE_WARN, color="#ffaa00", lw=0.8, ls="--", alpha=0.7)
ax_qual.axhline(NOISE_BAD,  color="#ff4444", lw=0.8, ls="--", alpha=0.7)

# ─── UPDATE DISPLAY ──────────────────────────────────────────────────────────

def update_display():
    s = current()

    # overview highlight
    sel_ov.set_data(s["wv"], s["ab"])
    # bring selected line to front in overview
    for name, ln in ov_lines.items():
        ln.set_alpha(0.07 if name != s["name"] else 0.9)
        ln.set_linewidth(0.5 if name != s["name"] else 1.5)

    # detail
    sel_det.set_data(s["wv"], s["ab"])

    # quality marker
    try:
        bar_pos = bar_names.index(s["name"])
        sel_vline.set_xdata([bar_pos, bar_pos])
    except ValueError:
        sel_vline.set_xdata([-1, -1])

    # info text
    ic_str  = f"{s['ic']:.0f} ppm" if s["ic"] is not None else "N/A"
    qcol    = quality_color(s)
    flag    = (" !! NOISY"   if s["noise_sg"]       >= NOISE_BAD  else
               " ! warn"     if s["noise_sg"]       >= NOISE_WARN else "")
    bflag   = (" !! OFFSET"  if abs(s["base_mean"]) >= BASE_BAD   else
               " ! warn"     if abs(s["base_mean"]) >= BASE_WARN  else "")

    txt = (
        f"Sample : {s['name']}\n"
        f"Status : {s['cat'].upper()}\n"
        f"IC ref : {ic_str}\n"
        f"─────────────────\n"
        f"Noise  : {s['noise_sg']:.5f}{flag}\n"
        f"Base   : {s['base_mean']:+.4f}{bflag}\n"
        f"BaseSD : {s['base_std']:.4f}\n"
        f"MinAbs : {s['min_model']:.4f}\n"
        f"─────────────────\n"
        f"Thresholds (75/90 pct)\n"
        f"Noise  : {NOISE_WARN:.5f} / {NOISE_BAD:.5f}\n"
        f"Base   : {BASE_WARN:.4f} / {BASE_BAD:.4f}"
    )
    info_box.set_text(txt)

    order = get_order()
    n_tot = len(order)
    ax_det.set_title(
        f"Selected: {s['name']}  ({state['pos']+1}/{n_tot})  "
        f"sorted by: {state['sort']}",
        color=FG_TEXT, fontsize=8,
    )

    fig.canvas.draw_idle()

# ─── NAVIGATION BUTTONS ───────────────────────────────────────────────────────
btn_ax = {
    "prev":  fig.add_axes([0.06, 0.01, 0.07, 0.04]),
    "next":  fig.add_axes([0.14, 0.01, 0.07, 0.04]),
    "alpha": fig.add_axes([0.25, 0.01, 0.09, 0.04]),
    "noise": fig.add_axes([0.35, 0.01, 0.09, 0.04]),
    "base":  fig.add_axes([0.45, 0.01, 0.11, 0.04]),
}
btn_style = dict(color="#1e1e3e", hovercolor="#2e2e5e")

btns = {k: Button(ax, label, **btn_style) for k, (ax, label) in {
    "prev":  (btn_ax["prev"],  "< Prev"),
    "next":  (btn_ax["next"],  "Next >"),
    "alpha": (btn_ax["alpha"], "Sort: Alpha"),
    "noise": (btn_ax["noise"], "Sort: Noise"),
    "base":  (btn_ax["base"],  "Sort: Baseline"),
}.items()}

for b in btns.values():
    b.label.set_color(FG_TEXT)
    b.label.set_fontsize(8)

def on_prev(event):
    state["pos"] = (state["pos"] - 1) % len(spectra)
    update_display()

def on_next(event):
    state["pos"] = (state["pos"] + 1) % len(spectra)
    update_display()

def make_sort_cb(key):
    def cb(event):
        state["sort"] = key
        state["pos"]  = 0
        update_display()
    return cb

btns["prev"].on_clicked(on_prev)
btns["next"].on_clicked(on_next)
btns["alpha"].on_clicked(make_sort_cb("alpha"))
btns["noise"].on_clicked(make_sort_cb("noise"))
btns["base"].on_clicked(make_sort_cb("baseline"))

# ─── CLICK INTERACTIONS ───────────────────────────────────────────────────────

def on_pick(event):
    """Click a line in the overview to select that spectrum."""
    label = event.artist.get_label()
    idx = next((i for i, s in enumerate(spectra) if s["name"] == label), None)
    if idx is None:
        return
    order = get_order()
    try:
        state["pos"] = order.index(idx)
    except ValueError:
        state["pos"] = 0
    update_display()

fig.canvas.mpl_connect("pick_event", on_pick)


def on_bar_click(event):
    """Click a bar in the quality chart to jump to that spectrum."""
    if event.inaxes is not ax_qual:
        return
    bar_pos = int(round(event.xdata)) if event.xdata is not None else -1
    if 0 <= bar_pos < N_bar:
        clicked_name = bar_names[bar_pos]
        idx = next((i for i, s in enumerate(spectra) if s["name"] == clicked_name), None)
        if idx is None:
            return
        order = get_order()
        try:
            state["pos"] = order.index(idx)
        except ValueError:
            state["pos"] = 0
        update_display()

fig.canvas.mpl_connect("button_press_event", on_bar_click)

# keyboard navigation
def on_key(event):
    if event.key in ("left", "a"):
        on_prev(None)
    elif event.key in ("right", "d"):
        on_next(None)

fig.canvas.mpl_connect("key_press_event", on_key)

# ─── INITIAL DRAW ─────────────────────────────────────────────────────────────
update_display()

fig.suptitle(
    "Spectral QC Viewer  —  use Prev/Next or arrow keys to navigate  |  "
    "sort by Noise to find worst spectra first",
    color=FG_TEXT, fontsize=9, y=0.97,
)

plt.show()
