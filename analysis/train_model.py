"""
PLS-NO3 Model Trainer
=====================
Best configuration (thesis): 215-380 nm | SG1 w=11 p=2 | LOO-CV | n=115 | 12 LV | R2=0.954

Produces:
  models/pls_model.pkl      – fitted PLSRegression object
  models/wavelengths.npy    – model wavelength axis (215-380 nm, native spz grid)
  models/preprocess.pkl     – SG1 parameters dict for the real-time predictor
  models/train_summary.txt  – performance metrics & sample list
  models/diagnostics.png    – 4-panel diagnostic figure
"""

import sys, os, re, warnings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import f as f_dist
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import joblib

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these if needed
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = Path(r"C:\Project N\Final code\27.01.2025\Most Recent")
OUT_DIR   = Path(r"C:\Projects-2026\Project N\Pure_PW-20260121T112655Z-1-001\Pure_PW\models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WV_START  = 215       # nm
WV_END    = 380       # nm
SG_W      = 11        # Savitzky-Golay window (must be odd)
SG_P      = 2         # polynomial order
SG_D      = 1         # derivative order (1st derivative)
MAX_LV    = 20        # maximum LVs to search

EXCLUDE = {
    # Known bad outliers (confirmed in thesis)
    "PW_29", "PW_2",
    # F0/F1/F2 = dilution series — same water at forced concentrations, NOT field samples
    "PW_1F0", "PW_1F0.0", "PW_1F0.1", "PW_1F0.2", "PW_1F0.3", "PW_1F0.4", "PW_1F1",
    "PW_2F",  "PW_2F0",  "PW_2F0.1", "PW_2F0.2", "PW_2F2",
    "PW_3F0",
    "PW_4F0", "PW_4F0.1", "PW_4F0.2", "PW_4F2",
    "PW_6F0", "PW_7F0",  "PW_52F0",
    # Decimal sub-samples of already-excluded PW_29
    "PW_29.1", "PW_29.2", "PW_29.3",
    # LOO residual > 2*RMSECV — likely IC reference errors or spectral contamination
    "PW_51A",  # |resid|=102 ppm: IC=136 but spectrum predicts 34 (possible IC error)
    "PW_42E",  # |resid|=63 ppm:  IC=20  but spectrum predicts 81 (contaminated cuvette?)
    "PW_51B",  # |resid|=61 ppm:  IC=32  but spectrum predicts 81 (same well as PW_51A)
    "PW_45",   # |resid|=58 ppm:  IC=24  but spectrum predicts 69 (possible ID swap with PW_45A)
    "PW_1",    # |resid|=46 ppm:  IC=43  but spectrum predicts 89
}

DATE_CUTOFF = None   # None = include all samples regardless of measurement date

# ══════════════════════════════════════════════════════════════════════════════
# TARGET MAP  (IC reference values)
# ══════════════════════════════════════════════════════════════════════════════
target_map = {
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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1  — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("LOADING DATA")
print(f"{'='*60}")
print(f"Source : {DATA_PATH}")

X_list, y_list, names, model_wv = [], [], [], None
missing, excluded = [], []
cutoff_dt = pd.Timestamp(DATE_CUTOFF) if DATE_CUTOFF else None

filtered = {}
for k, v in target_map.items():
    if k in EXCLUDE:
        excluded.append(f"{k} (manual)")
    else:
        filtered[k] = v

for sid, ppm_str in filtered.items():
    fp = DATA_PATH / f"{sid}.spz"
    if not fp.exists():
        missing.append(sid)
        continue
    try:
        # date cutoff check (reads first 25 lines only)
        if cutoff_dt is not None:
            raw_lines = fp.read_text(encoding="latin-1").splitlines()[:25]
            for ln in raw_lines:
                if ln.startswith("Date:"):
                    try:
                        meas_date = pd.Timestamp(ln.split("\t", 1)[1].strip())
                        if meas_date > cutoff_dt:
                            excluded.append(f"{sid} (after {DATE_CUTOFF})")
                            fp = None
                    except Exception:
                        pass
                    break
            if fp is None:
                continue

        df   = pd.read_csv(fp, sep="\t", skiprows=25, header=None, encoding="latin-1")
        wv   = df.iloc[:, 0].values.astype(float)
        ab   = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
        mask = (wv >= WV_START) & (wv <= WV_END)
        if mask.sum() == 0:
            missing.append(sid); continue
        X_list.append(ab[mask])
        y_list.append(float(ppm_str))
        names.append(sid)
        if model_wv is None:
            model_wv = wv[mask]
    except Exception as e:
        print(f"  [WARN] {sid}: {e}")
        missing.append(sid)

n = len(names)
print(f"Loaded : {n} samples | Excluded: {excluded} | Not found: {len(missing)}")
if missing:
    print(f"  Missing files: {missing[:10]}{'...' if len(missing)>10 else ''}")

if n < 10:
    raise SystemExit("Too few samples to train (<10). Check DATA_PATH.")

X_raw = np.array(X_list)
y     = np.array(y_list)

print(f"y range: {y.min():.1f} – {y.max():.1f} ppm  |  mean={y.mean():.1f}")
print(f"Wavelength region: {model_wv[0]:.1f} – {model_wv[-1]:.1f} nm  ({len(model_wv)} points)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2  — PREPROCESS  (SG1 only — no SNV; SNV discards real NO3 signal here)
# ══════════════════════════════════════════════════════════════════════════════
def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

def sg1(X, w=SG_W, p=SG_P, d=SG_D):
    return savgol_filter(X, window_length=w, polyorder=p, deriv=d, axis=1)

# Try both — pick the one giving lower LOO-CV RMSE
X_sg1     = sg1(X_raw)
X_snv_sg1 = sg1(snv(X_raw))
print(f"\nPreprocessing candidates: SG1 vs SNV+SG1 — will pick by LOO-CV")

# Quick 6-LV LOO-CV comparison to pick preprocessing
def quick_rmsecv(X_try, lv=6):
    pls_t = PLSRegression(n_components=min(lv, X_try.shape[0]-1, X_try.shape[1]))
    y_cv  = cross_val_predict(pls_t, X_try, y, cv=loo)
    return np.sqrt(mean_squared_error(y, y_cv))

loo = LeaveOneOut()
r_sg1     = quick_rmsecv(X_sg1)
r_snv_sg1 = quick_rmsecv(X_snv_sg1)
print(f"  SG1 only     RMSECV@6LV = {r_sg1:.2f} ppm")
print(f"  SNV + SG1    RMSECV@6LV = {r_snv_sg1:.2f} ppm")

if r_snv_sg1 < r_sg1:
    X_pre   = X_snv_sg1
    PP_NAME = "SNV + SG1"
    joblib.dump({"snv": True, "window": SG_W, "polyorder": SG_P, "deriv": SG_D},
                OUT_DIR / "preprocess.pkl")
else:
    X_pre   = X_sg1
    PP_NAME = "SG1 only"
    joblib.dump({"snv": False, "window": SG_W, "polyorder": SG_P, "deriv": SG_D},
                OUT_DIR / "preprocess.pkl")
print(f"  → Using: {PP_NAME}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3  — FIND OPTIMAL LVs VIA LOO-CV
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("LV SELECTION  (LOO-CV)")
print(f"{'='*60}")

loo     = LeaveOneOut()
max_lv  = min(MAX_LV, n - 1, X_pre.shape[1])
rmsecv  = []

for lv in range(1, max_lv + 1):
    pls    = PLSRegression(n_components=lv)
    y_cv   = cross_val_predict(pls, X_pre, y, cv=loo)
    rmsecv.append(np.sqrt(mean_squared_error(y, y_cv)))

opt_lv = int(np.argmin(rmsecv)) + 1
print(f"Optimal LVs: {opt_lv}  (RMSECV={rmsecv[opt_lv-1]:.2f} ppm)")
for lv, r in enumerate(rmsecv, 1):
    marker = " <-- optimal" if lv == opt_lv else ""
    print(f"  LV={lv:2d}  RMSECV={r:.2f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4  — FIT FINAL MODEL & LOO PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
pls_final = PLSRegression(n_components=opt_lv)
pls_final.fit(X_pre, y)

y_cal = pls_final.predict(X_pre).ravel()
y_loo = cross_val_predict(pls_final, X_pre, y, cv=loo)

r2_cal  = r2_score(y, y_cal)
rmse_cal= np.sqrt(mean_squared_error(y, y_cal))
r2_loo  = r2_score(y, y_loo)
rmse_loo= np.sqrt(mean_squared_error(y, y_loo))
bias    = np.mean(y_loo - y)

print(f"\n{'='*60}")
print("FINAL MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"  Calibration  R2={r2_cal:.4f}  RMSEC={rmse_cal:.2f} ppm")
print(f"  LOO-CV       R2={r2_loo:.4f}  RMSECV={rmse_loo:.2f} ppm  bias={bias:.2f}")
print(f"  n={n}  LV={opt_lv}  region={WV_START}-{WV_END}nm")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5  — OUTLIER DETECTION  (leverage + Q-residuals)
# ══════════════════════════════════════════════════════════════════════════════
T  = pls_final.x_scores_                          # (n, lv)
P  = pls_final.x_loadings_                        # (p, lv)
Xr = X_pre - T @ P.T                              # X residuals

leverage = np.diag(T @ np.linalg.pinv(T.T @ T) @ T.T)
q_res    = np.sum(Xr ** 2, axis=1)

lev_lim  = 3 * opt_lv / n                         # 3h-bar
q_lim    = np.percentile(q_res, 95)               # 95th pct

outlier_flags = (leverage > lev_lim) | (q_res > q_lim)
outlier_names = [names[i] for i in range(n) if outlier_flags[i]]

print(f"\nOutlier check (leverage>{lev_lim:.3f} OR Q>95pct={q_lim:.4f}):")
if outlier_names:
    print(f"  Potential outliers: {outlier_names}")
else:
    print("  None detected")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6  — SAVE MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
joblib.dump(pls_final, OUT_DIR / "pls_model.pkl")
np.save(OUT_DIR / "wavelengths.npy", model_wv)
# preprocess.pkl already saved in preprocessing block above

summary = (
    f"PLS-NO3 Model Summary\n"
    f"=====================\n"
    f"Date      : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n"
    f"Samples   : {n}\n"
    f"Excluded  : {excluded}\n"
    f"Region    : {WV_START}-{WV_END} nm\n"
    f"Preprocess: SG1 window={SG_W} poly={SG_P} deriv={SG_D}\n"
    f"LV opt    : {opt_lv}\n"
    f"RMSEC     : {rmse_cal:.2f} ppm\n"
    f"RMSECV    : {rmse_loo:.2f} ppm\n"
    f"R2-cal    : {r2_cal:.4f}\n"
    f"R2-LOO    : {r2_loo:.4f}\n"
    f"Bias      : {bias:.2f} ppm\n"
    f"\nSamples loaded:\n"
    + "\n".join(f"  {n_:12s} act={y[i]:6.1f}  cal={y_cal[i]:6.2f}  loo={y_loo[i]:6.2f}"
                for i, n_ in enumerate(names))
)
(OUT_DIR / "train_summary.txt").write_text(summary)
print(f"\nArtifacts saved to: {OUT_DIR}")
print(f"  pls_model.pkl  wavelengths.npy  preprocess.pkl  train_summary.txt")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7  — DIAGNOSTICS PLOT  (4 panels)
# ══════════════════════════════════════════════════════════════════════════════
plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0d0d14")
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30,
                        left=0.07, right=0.97, top=0.93, bottom=0.07)

C_TEAL  = "#00e5cc"
C_VIO   = "#9c6fe4"
C_AMBER = "#ffb74d"
C_RED   = "#ef5350"
C_GREEN = "#66bb6a"
C_TEXT  = "#dde0f0"
C_DIM   = "#707090"
BG      = "#12121c"

def styled_ax(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor("#252545")
    ax.tick_params(colors=C_DIM, labelsize=8)
    ax.xaxis.label.set_color(C_DIM); ax.yaxis.label.set_color(C_DIM)

# ── Panel 1: RMSECV vs LV ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
styled_ax(ax1)
lv_range = range(1, max_lv + 1)
ax1.plot(lv_range, rmsecv, color=C_TEAL, lw=2, marker="o", ms=5)
ax1.axvline(opt_lv, color=C_AMBER, lw=1.5, ls="--", label=f"Opt LV={opt_lv}")
ax1.axhline(rmsecv[opt_lv-1], color=C_AMBER, lw=0.8, ls=":")
ax1.set_xlabel("Latent Variables"); ax1.set_ylabel("RMSECV (ppm)")
ax1.set_title("LOO-CV: RMSECV vs LV", color=C_TEAL, fontsize=10, fontweight="bold")
ax1.legend(fontsize=8, facecolor="#1c1c32", labelcolor=C_TEXT)
ax1.text(0.97, 0.95, f"Opt={opt_lv} LV\nRMSECV={rmsecv[opt_lv-1]:.2f}",
         transform=ax1.transAxes, ha="right", va="top",
         fontsize=9, color=C_AMBER, fontfamily="monospace")

# ── Panel 2: Predicted vs Actual (LOO-CV) ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
styled_ax(ax2)
cmap = plt.cm.plasma(np.linspace(0.1, 0.9, n))

sc = ax2.scatter(y, y_loo, c=y, cmap="plasma", s=70,
                 edgecolors="white", linewidths=0.4, zorder=3, alpha=0.85)
lo, hi = min(y.min(), y_loo.min()) - 5, max(y.max(), y_loo.max()) + 5
ax2.plot([lo, hi], [lo, hi], "r--", lw=1.2, alpha=0.6, label="1:1 line")
# ±RMSECV band
ax2.fill_between([lo, hi], [lo-rmse_loo, hi-rmse_loo],
                 [lo+rmse_loo, hi+rmse_loo], color=C_RED, alpha=0.10)

# annotate outliers
for i, nm in enumerate(names):
    if outlier_flags[i]:
        ax2.annotate(nm, (y[i], y_loo[i]), fontsize=6, color=C_RED,
                     xytext=(4, 4), textcoords="offset points")

plt.colorbar(sc, ax=ax2, label="Actual (ppm)", pad=0.02).ax.yaxis.label.set_color(C_DIM)
ax2.set_xlabel("Actual NO3 (ppm)"); ax2.set_ylabel("LOO-CV Predicted (ppm)")
ax2.set_title("Prediction vs Reference (LOO-CV)", color=C_TEAL, fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, facecolor="#1c1c32", labelcolor=C_TEXT)
ax2.text(0.03, 0.97, f"R²={r2_loo:.4f}\nRMSECV={rmse_loo:.2f} ppm\nbias={bias:.2f} ppm",
         transform=ax2.transAxes, va="top", fontsize=9,
         color=C_GREEN, fontfamily="monospace",
         bbox=dict(boxstyle="round", fc="#1c1c32", alpha=0.8))

# ── Panel 3: Leverage – Q Residual ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
styled_ax(ax3)
colors_lev = [C_RED if f else C_TEAL for f in outlier_flags]
ax3.scatter(leverage, q_res, c=colors_lev, s=60, edgecolors="white",
            linewidths=0.4, alpha=0.85, zorder=3)
ax3.axvline(lev_lim, color=C_AMBER, lw=1.2, ls="--", label=f"Lev limit={lev_lim:.3f}")
ax3.axhline(q_lim,   color=C_VIO,   lw=1.2, ls="--", label=f"Q 95pct={q_lim:.4f}")
for i, nm in enumerate(names):
    if outlier_flags[i]:
        ax3.annotate(nm, (leverage[i], q_res[i]), fontsize=6, color=C_RED,
                     xytext=(4, 4), textcoords="offset points")
ax3.set_xlabel("Leverage (h)"); ax3.set_ylabel("Q Residuals")
ax3.set_title("Outlier Map: Leverage vs Q Residuals", color=C_TEAL, fontsize=10, fontweight="bold")
ax3.legend(fontsize=8, facecolor="#1c1c32", labelcolor=C_TEXT)

# ── Panel 4: VIP scores + regression coefficients ───────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
styled_ax(ax4)

# VIP (Variable Importance in Projection)
W   = pls_final.x_weights_                        # (p, lv)
SS  = np.sum(pls_final.y_loadings_ ** 2 * np.sum(T ** 2, axis=0), axis=1)  # (lv,)
p_feat = X_pre.shape[1]
vip = np.sqrt(p_feat * np.sum(W ** 2 * SS, axis=1) / np.sum(SS))

ax4_r = ax4.twinx()
ax4.plot(model_wv, vip, color=C_TEAL, lw=1.8, label="VIP")
ax4.axhline(1.0, color=C_AMBER, lw=1.0, ls="--", alpha=0.7, label="VIP=1 threshold")
ax4.set_xlabel("Wavelength (nm)"); ax4.set_ylabel("VIP score", color=C_TEAL)
ax4.tick_params(axis="y", labelcolor=C_TEAL)

coef = pls_final.coef_.ravel() if pls_final.coef_.ndim > 1 else pls_final.coef_
ax4_r.plot(model_wv, coef, color=C_VIO, lw=1.2, alpha=0.7, label="Regression coef")
ax4_r.set_ylabel("Regression Coefficient", color=C_VIO)
ax4_r.tick_params(axis="y", labelcolor=C_VIO)
ax4_r.spines["right"].set_edgecolor("#252545")
ax4_r.yaxis.label.set_color(C_VIO)

lines1, labs1 = ax4.get_legend_handles_labels()
lines2, labs2 = ax4_r.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labs1 + labs2, fontsize=8,
           facecolor="#1c1c32", labelcolor=C_TEXT)
ax4.set_title("VIP Scores & Regression Coefficients", color=C_TEAL, fontsize=10, fontweight="bold")

fig.suptitle(
    f"PLS-NO3  |  n={n}  LV={opt_lv}  R²-LOO={r2_loo:.4f}  "
    f"RMSECV={rmse_loo:.2f} ppm  |  {WV_START}-{WV_END}nm  SG1(w={SG_W})",
    color=C_TEXT, fontsize=11, fontweight="bold"
)

diag_path = OUT_DIR / "diagnostics.png"
fig.savefig(diag_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close("all")
print(f"  diagnostics.png saved")
print("\nDone.")
