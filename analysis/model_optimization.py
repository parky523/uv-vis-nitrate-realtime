"""
model_optimization.py
=====================
Systematic PLS-R optimisation for MSc thesis:
  Step 1 -- Parameter grid search (spectral range x preprocessing)
  Step 2 -- Iterative outlier removal (LOO-CV residual screening)
  Step 3 -- Train/Test split evaluation (70/30 stratified)
  Step 4 -- Full-dataset final model comparison

Outputs to Report_Figures/Optimization/

Parthasarathy Shanmugasundaram -- HFU / Hahn-Schickard, 2026
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy.stats import norm, linregress
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                     StratifiedShuffleSplit)
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# =============================================================================
#  PATHS & GLOBAL SETTINGS
# =============================================================================
DATA_PATH = r"C:\Project N\Final code\27.01.2025\Most Recent"
OUT_DIR   = r"C:\Projects-2026\Project N\Pure_PW-20260121T112655Z-1-001\Pure_PW\Report_Figures\Optimization"
os.makedirs(OUT_DIR, exist_ok=True)

FULL_START       = 190
FULL_END         = 500
SG_WINDOW        = 11
SG_POLY          = 2
MAX_LV           = 15
MAX_REMOVE_FRAC  = 0.30   # stop removal at 30 % of original n
DPI              = 300
TARGET_R2        = 0.95   # target LOO-CV R2 for outlier removal loop

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 11,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.fontsize': 9,
})

# =============================================================================
#  SAMPLE MAP (full confirmed dataset)
# =============================================================================
ppm_map = {
    "PW_04": "27 ppm",  "PW_1":   "43 ppm",  "PW_10":  "44 ppm",  "PW_11":  "26 ppm",
    "PW_12": "34 ppm",  "PW_13":  "52 ppm",  "PW_14":  "50 ppm",  "PW_15":  "56 ppm",
    "PW_16": "7 ppm",   "PW_17":  "73 ppm",  "PW_17A": "82 ppm",  "PW_17B": "12 ppm",
    "PW_18": "42 ppm",  "PW_19":  "64 ppm",  "PW_1B":  "5 ppm",   "PW_1C":  "46 ppm",
    "PW_1D": "21 ppm",  "PW_1E":  "47 ppm",  "PW_1F1": "27 ppm",  "PW_2":   "80 ppm",
    "PW_20": "46 ppm",  "PW_20A": "27 ppm",  "PW_20B": "33 ppm",  "PW_20C": "28 ppm",
    "PW_21": "44 ppm",  "PW_22A": "24 ppm",  "PW_22B": "24 ppm",  "PW_22C": "24 ppm",
    "PW_22D":"28 ppm",  "PW_22E": "42 ppm",  "PW_22F": "26 ppm",  "PW_22G": "76 ppm",
    "PW_22H":"23 ppm",  "PW_22I": "30 ppm",  "PW_22J": "22 ppm",  "PW_23":  "42 ppm",
    "PW_23A":"46 ppm",  "PW_24":  "90 ppm",  "PW_24A": "25 ppm",  "PW_24B": "22 ppm",
    "PW_24C":"4 ppm",   "PW_24D": "12 ppm",  "PW_24E": "25 ppm",  "PW_24F": "23 ppm",
    "PW_26": "32 ppm",  "PW_27":  "41 ppm",  "PW_28":  "45 ppm",  "PW_28A": "34 ppm",
    "PW_28B":"52 ppm",  "PW_28C": "32 ppm",  "PW_29":  "74 ppm",  "PW_29.1":"26 ppm",
    "PW_29.2":"29 ppm", "PW_29.3":"25 ppm",  "PW_2B":  "11 ppm",  "PW_2C":  "28 ppm",
    "PW_2D": "4 ppm",   "PW_2E":  "20 ppm",  "PW_2F":  "25 ppm",  "PW_2F0": "26 ppm",
    "PW_2F0.1":"43 ppm","PW_2F0.2":"28 ppm", "PW_2F2": "40 ppm",  "PW_3":   "27 ppm",
    "PW_30": "46 ppm",  "PW_30A": "36 ppm",  "PW_31":  "51 ppm",  "PW_31A": "13 ppm",
    "PW_31B":"10 ppm",  "PW_31C": "10 ppm",  "PW_31D": "11 ppm",  "PW_31E": "11 ppm",
    "PW_31F":"18 ppm",  "PW_31G": "8 ppm",   "PW_31H": "16 ppm",  "PW_32":  "48 ppm",
    "PW_32A":"46 ppm",  "PW_36":  "53 ppm",  "PW_36A": "75 ppm",  "PW_37":  "44 ppm",
    "PW_37A":"54 ppm",  "PW_38":  "56 ppm",  "PW_38A": "24 ppm",  "PW_38B": "12 ppm",
    "PW_38C":"12 ppm",  "PW_39":  "47 ppm",  "PW_39A": "90 ppm",  "PW_39B": "38 ppm",
    "PW_39C":"35 ppm",  "PW_39D": "23 ppm",  "PW_39E": "23 ppm",  "PW_39E.2":"19 ppm",
    "PW_39F":"11 ppm",  "PW_39G": "9 ppm",   "PW_39I": "12 ppm",  "PW_39J": "54 ppm",
    "PW_3A": "82 ppm",  "PW_3B":  "5 ppm",   "PW_40":  "30 ppm",  "PW_40A": "11 ppm",
    "PW_40B":"11 ppm",  "PW_40C": "9 ppm",   "PW_40D": "9 ppm",   "PW_40E": "18 ppm",
    "PW_41": "68 ppm",  "PW_42":  "90 ppm",  "PW_42A": "37 ppm",  "PW_42B": "29 ppm",
    "PW_42C":"29 ppm",  "PW_42D": "21 ppm",  "PW_42E": "20 ppm",  "PW_42F": "26 ppm",
    "PW_42I":"36 ppm",  "PW_42J": "22 ppm",  "PW_43":  "73 ppm",  "PW_44":  "41 ppm",
    "PW_45": "24 ppm",  "PW_45A": "72 ppm",  "PW_45B": "10 ppm",  "PW_45C": "6 ppm",
    "PW_45D":"8 ppm",   "PW_45E": "9 ppm",   "PW_45G": "11 ppm",  "PW_46":  "46 ppm",
    "PW_47": "22 ppm",  "PW_47A": "32 ppm",  "PW_49":  "74 ppm",  "PW_4F0": "35 ppm",
    "PW_4F2":"24 ppm",  "PW_5":   "54 ppm",  "PW_50":  "56 ppm",  "PW_51":  "77 ppm",
    "PW_51A":"136 ppm", "PW_51B": "32 ppm",  "PW_51C": "90 ppm",  "PW_51D": "18 ppm",
    "PW_51E":"20 ppm",  "PW_52":  "50 ppm",  "PW_52A": "24 ppm",  "PW_52B": "27 ppm",
    "PW_52C":"18 ppm",  "PW_52D": "33 ppm",  "PW_53":  "23 ppm",  "PW_53A": "117 ppm",
    "PW_53B":"58 ppm",  "PW_54.2":"35 ppm",  "PW_54.3":"38 ppm",  "PW_6":   "41 ppm",
    "PW_6A": "50 ppm",  "PW_7":   "44 ppm",  "PW_7A":  "60 ppm",  "PW_7F0": "66 ppm",
    "PW_8":  "53 ppm",  "PW_9":   "48 ppm",  "PW_HS1": "42 ppm",
}
exclude_list = {
    'PW_29',  'PW_2',   'PW_2D',  'PW_16',  'PW_1B',  'PW_42E',
    'PW_24C', 'PW_45',  'PW_51B', 'PW_31G', 'PW_2F0', 'PW_39I',
    'PW_45D', 'PW_45C', 'PW_31H', 'PW_42J', 'PW_51D', 'PW_31B',
    'PW_1F1', 'PW_31A', 'PW_24D', 'PW_28A', 'PW_3B',  'PW_45E',
    'PW_31C',
}
filtered_map = {k: v for k, v in ppm_map.items() if k not in exclude_list}

# =============================================================================
#  LOAD SPECTRA
# =============================================================================
print("Loading spectra ...")
specs, y_vals, ids = [], [], []
wl_axis = None

for sid, target in filtered_map.items():
    fp = os.path.join(DATA_PATH, f"{sid}.spz")
    if not os.path.exists(fp):
        continue
    try:
        df   = pd.read_csv(fp, sep='\t', skiprows=25, header=None, encoding='latin-1')
        wvs  = df.iloc[:, 0].values
        ints = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        mask = (wvs >= FULL_START) & (wvs <= FULL_END)
        spec = ints[mask]
        if len(spec) > 20 and not np.all(np.isnan(spec)):
            specs.append(spec)
            y_vals.append(float(re.findall(r"\d+", target)[0]))
            ids.append(sid)
            if wl_axis is None:
                wl_axis = wvs[mask]
    except Exception as e:
        print(f"  [error] {sid}: {e}")

spectra_all = np.array(specs)
y_all       = np.array(y_vals)
n_all       = len(ids)
print(f"  Loaded: {n_all} samples | Range: {y_all.min():.0f}-{y_all.max():.0f} ppm\n")

# =============================================================================
#  PREPROCESSING FUNCTIONS
# =============================================================================
def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)

def msc(X):
    mu = X.mean(axis=0)
    X_msc = np.zeros_like(X)
    for i in range(X.shape[0]):
        a, b = np.polyfit(mu, X[i], 1)
        X_msc[i] = (X[i] - b) / (a + 1e-12)
    return X_msc

def sg1(X):
    return savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=1)

def sg2(X):
    return savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=2)

def preprocess(X, method):
    if method == 'SNV+SG1':  return sg1(snv(X))
    if method == 'MSC+SG1':  return sg1(msc(X))
    if method == 'SNV+SG2':  return sg2(snv(X))
    if method == 'SG1 only': return sg1(X)
    if method == 'SNV only': return snv(X)
    if method == 'Raw':      return X.copy()
    raise ValueError(f"Unknown preprocessing: {method}")

# LV scan helper
def lv_scan(X_pre, y, max_lv=MAX_LV):
    mlv = min(max_lv, X_pre.shape[0] - 2, X_pre.shape[1])
    rmsecv, r2cv = [], []
    for nlv in range(1, mlv + 1):
        y_cv = cross_val_predict(PLSRegression(n_components=nlv),
                                 X_pre, y, cv=LeaveOneOut()).ravel()
        rmsecv.append(float(np.sqrt(mean_squared_error(y, y_cv))))
        r2cv.append(float(r2_score(y, y_cv)))
    return np.array(rmsecv), np.array(r2cv), list(range(1, mlv + 1))

# =============================================================================
#  STEP 1 -- PARAMETER GRID SEARCH
# =============================================================================
print("Step 1: Parameter grid search ...")

spectral_ranges = [
    (200, 400, '200-400 nm'),
    (200, 500, '200-500 nm'),
    (215, 380, '215-380 nm'),
    (190, 400, '190-400 nm'),
]
preproc_methods = ['SNV+SG1', 'MSC+SG1', 'SNV+SG2', 'SG1 only', 'SNV only']

grid_rows = []
best_r2cv = -np.inf
best_cfg  = {}

for wl_lo, wl_hi, rng_lbl in spectral_ranges:
    for prep in preproc_methods:
        mask_rng = (wl_axis >= wl_lo) & (wl_axis <= wl_hi)
        X_rng    = spectra_all[:, mask_rng]
        X_pre    = preprocess(X_rng, prep)
        rmsecv, r2cv, lv_ax = lv_scan(X_pre, y_all)
        opt_lv   = int(np.argmin(rmsecv)) + 1
        r2_best  = r2cv[opt_lv - 1]
        rmse_best = rmsecv[opt_lv - 1]
        print(f"  {rng_lbl:12s} | {prep:10s} | LV={opt_lv:2d} | "
              f"R2CV={r2_best:.4f}  RMSECV={rmse_best:.2f}")
        grid_rows.append({'Range': rng_lbl, 'Preprocessing': prep,
                          'Opt_LV': opt_lv, 'R2CV': r2_best, 'RMSECV': rmse_best,
                          'rmsecv_arr': rmsecv, 'r2cv_arr': r2cv, 'lv_ax': lv_ax})
        if r2_best > best_r2cv:
            best_r2cv = r2_best
            best_cfg  = {'range': (wl_lo, wl_hi, rng_lbl), 'prep': prep,
                         'lv': opt_lv, 'r2cv': r2_best, 'rmsecv': rmse_best,
                         'rmsecv_arr': rmsecv, 'r2cv_arr': r2cv, 'lv_ax': lv_ax}

print(f"\n  Best: {best_cfg['range'][2]} + {best_cfg['prep']} | "
      f"LV={best_cfg['lv']} | R2CV={best_cfg['r2cv']:.4f}  "
      f"RMSECV={best_cfg['rmsecv']:.2f} ppm\n")

# =============================================================================
#  FIGURE O1 -- Parameter Grid Heatmap + LV Curves
# =============================================================================
print("Fig O1: Parameter grid heatmap ...")

r2_grid = np.array([[row['R2CV'] for row in grid_rows
                      if row['Preprocessing'] == prep]
                     for prep in preproc_methods])
rmse_grid = np.array([[row['RMSECV'] for row in grid_rows
                        if row['Preprocessing'] == prep]
                       for prep in preproc_methods])
lv_grid   = np.array([[row['Opt_LV'] for row in grid_rows
                        if row['Preprocessing'] == prep]
                       for prep in preproc_methods])
rng_labels = [r[2] for r in spectral_ranges]

fig1o, axes1o = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'wspace': 0.35})
fig1o.suptitle(
    'Parameter Grid Search: Spectral Range x Preprocessing\n'
    f'PLS-R LOO-CV  |  n = {n_all} samples',
    fontsize=12, fontweight='bold'
)

for ax, data, cmap, title, fmt in [
    (axes1o[0], r2_grid,   'RdYlGn', '(a)  LOO-CV R2',    '{:.3f}'),
    (axes1o[1], rmse_grid, 'RdYlGn_r','(b)  LOO-CV RMSECV (ppm)', '{:.1f}'),
]:
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=data.min(), vmax=data.max())
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    ax.set_xticks(range(len(rng_labels)))
    ax.set_xticklabels(rng_labels, fontsize=9, rotation=12)
    ax.set_yticks(range(len(preproc_methods)))
    ax.set_yticklabels(preproc_methods, fontsize=9)
    ax.set_title(title, loc='left', fontsize=10)
    for i in range(len(preproc_methods)):
        for j in range(len(rng_labels)):
            val = data[i, j]
            lv_v = lv_grid[i, j]
            txt_color = 'black' if 0.3 < (val - data.min()) / (data.max() - data.min() + 1e-9) < 0.8 else 'white'
            ax.text(j, i, fmt.format(val) + f'\nLV={lv_v}',
                    ha='center', va='center', fontsize=8, color=txt_color)

fig1o.savefig(os.path.join(OUT_DIR, 'FigO1_Parameter_Grid.png'), dpi=DPI, bbox_inches='tight')
print("  Saved: FigO1_Parameter_Grid.png")
plt.close(fig1o)

# =============================================================================
#  STEP 2 -- ITERATIVE OUTLIER REMOVAL (best params)
# =============================================================================
print("Step 2: Iterative outlier removal ...")

wl_lo_b, wl_hi_b, rng_lbl_b = best_cfg['range']
prep_b = best_cfg['prep']
mask_b = (wl_axis >= wl_lo_b) & (wl_axis <= wl_hi_b)

# Build best-preprocessed full dataset
X_full_b = preprocess(spectra_all[:, mask_b], prep_b)
wl_b     = wl_axis[mask_b]

max_remove = int(MAX_REMOVE_FRAC * n_all)
cur_idx    = np.arange(n_all)
removal_log = []   # list of dicts: {step, n, r2cv, rmsecv, removed_id}

print(f"  Using: {rng_lbl_b} + {prep_b}  |  max remove = {max_remove}")

for step in range(max_remove + 1):
    X_cur = X_full_b[cur_idx]
    y_cur = y_all[cur_idx]

    # LV scan for current subset
    rmsecv_s, r2cv_s, lv_s = lv_scan(X_cur, y_cur)
    opt_lv_s = int(np.argmin(rmsecv_s)) + 1
    r2cv_s_best = r2cv_s[opt_lv_s - 1]
    rmsecv_s_best = rmsecv_s[opt_lv_s - 1]

    # LOO-CV predictions for current best LV
    y_cv_s = cross_val_predict(PLSRegression(n_components=opt_lv_s),
                               X_cur, y_cur, cv=LeaveOneOut()).ravel()
    resids = np.abs(y_cv_s - y_cur)
    worst_local = int(np.argmax(resids))
    worst_id    = ids[cur_idx[worst_local]]

    removal_log.append({
        'step':    step,
        'n':       len(cur_idx),
        'r2cv':    r2cv_s_best,
        'rmsecv':  rmsecv_s_best,
        'opt_lv':  opt_lv_s,
        'removed': worst_id,
        'removed_ppm': float(y_cur[worst_local]),
        'removed_resid': float(resids[worst_local]),
        'cur_idx': cur_idx.copy(),
        'y_cv':    y_cv_s.copy(),
    })

    print(f"  Step {step:2d}: n={len(cur_idx):3d}  R2CV={r2cv_s_best:.4f}  "
          f"RMSECV={rmsecv_s_best:.2f}  LV={opt_lv_s}  "
          f"[next remove: {worst_id} ({y_cur[worst_local]:.0f} ppm, err={resids[worst_local]:.1f})]")

    if r2cv_s_best >= TARGET_R2:
        print(f"  *** Target R2CV >= {TARGET_R2} reached at step {step} ***")
        break

    cur_idx = np.delete(cur_idx, worst_local)

# Find best step (max R2CV — may be last step or where it first plateaus)
best_step   = int(np.argmax([r['r2cv'] for r in removal_log]))
best_result = removal_log[best_step]
print(f"\n  Best at step {best_step}: n={best_result['n']}  "
      f"R2CV={best_result['r2cv']:.4f}  RMSECV={best_result['rmsecv']:.2f} ppm  "
      f"LV={best_result['opt_lv']}")

# Clean dataset indices
clean_idx  = best_result['cur_idx']
X_clean    = X_full_b[clean_idx]
y_clean    = y_all[clean_idx]
ids_clean  = [ids[i] for i in clean_idx]
n_clean    = len(clean_idx)
removed_ids = [r['removed'] for r in removal_log[:best_step]]

# Re-fit final LOO on clean set
opt_lv_clean = best_result['opt_lv']
pls_clean = PLSRegression(n_components=opt_lv_clean)
pls_clean.fit(X_clean, y_clean)
y_cal_clean = pls_clean.predict(X_clean).ravel()
y_cv_clean  = best_result['y_cv']

r2_cal_c  = r2_score(y_clean, y_cal_clean)
rmse_cal_c = float(np.sqrt(mean_squared_error(y_clean, y_cal_clean)))
r2_cv_c   = best_result['r2cv']
rmse_cv_c = best_result['rmsecv']

print(f"  Final model: n={n_clean}  Cal R2={r2_cal_c:.4f}  LOO-CV R2={r2_cv_c:.4f}  "
      f"RMSECV={rmse_cv_c:.2f} ppm  LV={opt_lv_clean}\n")

# =============================================================================
#  FIGURE O2 -- Outlier Removal Curve
# =============================================================================
print("Fig O2: Outlier removal curve ...")

steps_arr  = [r['step']   for r in removal_log]
r2cv_arr   = [r['r2cv']   for r in removal_log]
rmsecv_arr = [r['rmsecv'] for r in removal_log]
n_arr      = [r['n']      for r in removal_log]

fig2o, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5.5),
                                     gridspec_kw={'wspace': 0.38})
fig2o.suptitle(
    f'Iterative Outlier Removal via LOO-CV Residual Screening\n'
    f'{rng_lbl_b} + {prep_b}  |  Optimal LV determined at each step',
    fontsize=11, fontweight='bold'
)

# (a) R2CV vs samples removed
ax2a.plot(steps_arr, r2cv_arr, 'o-', color='#1a6faf', linewidth=2.0,
          markersize=6, markerfacecolor='white', markeredgewidth=1.8)
ax2a.axhline(TARGET_R2, color='#e31a1c', linewidth=1.5, linestyle='--',
             label=f'Target R$^2$CV = {TARGET_R2}')
ax2a.axvline(best_step, color='#27ae60', linewidth=1.5, linestyle=':',
             label=f'Best step: {best_step} removed\nR$^2$CV={best_result["r2cv"]:.4f}')
ax2a.scatter([best_step], [best_result['r2cv']], color='#27ae60', s=120, zorder=5)
ax2a.set_xlabel('Samples Removed (cumulative)')
ax2a.set_ylabel('LOO-CV R$^2$')
ax2a.set_title('(a)  R$^2$CV vs. samples removed', loc='left', fontsize=10)
ax2a.legend(fontsize=8)
ax2a.set_xlim(-0.5, len(steps_arr) - 0.5)
ax2a.xaxis.set_minor_locator(AutoMinorLocator(1))
ax2a.yaxis.set_minor_locator(AutoMinorLocator(4))

# (b) RMSECV vs samples removed
ax2b_r = ax2b.twinx()
ax2b.plot(steps_arr, rmsecv_arr, 's-', color='#c0392b', linewidth=2.0,
          markersize=6, markerfacecolor='white', markeredgewidth=1.8, label='RMSECV')
ax2b_r.plot(steps_arr, n_arr, '^--', color='#7f8c8d', linewidth=1.2,
            markersize=5, alpha=0.6, label='n remaining')
ax2b.axvline(best_step, color='#27ae60', linewidth=1.5, linestyle=':')
ax2b.scatter([best_step], [best_result['rmsecv']], color='#c0392b', s=120, zorder=5)
ax2b.set_xlabel('Samples Removed (cumulative)')
ax2b.set_ylabel('RMSECV (ppm)', color='#c0392b')
ax2b_r.set_ylabel('n remaining', color='#7f8c8d')
ax2b.set_title('(b)  RMSECV and n remaining vs. samples removed', loc='left', fontsize=10)
lines1, labels1 = ax2b.get_legend_handles_labels()
lines2, labels2 = ax2b_r.get_legend_handles_labels()
ax2b.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower left')
ax2b.set_xlim(-0.5, len(steps_arr) - 0.5)
ax2b.xaxis.set_minor_locator(AutoMinorLocator(1))
ax2b.yaxis.set_minor_locator(AutoMinorLocator(4))

fig2o.savefig(os.path.join(OUT_DIR, 'FigO2_Outlier_Removal.png'), dpi=DPI, bbox_inches='tight')
print("  Saved: FigO2_Outlier_Removal.png")
plt.close(fig2o)

# =============================================================================
#  FIGURE O3 -- Best Model LOO-CV (after outlier removal)
# =============================================================================
print("Fig O3: Best model after outlier removal ...")

ppm_lo3 = y_clean.min() - 5
ppm_hi3 = y_clean.max() + 5
cmap3   = plt.cm.plasma
norm3   = mcolors.Normalize(vmin=y_clean.min(), vmax=y_clean.max())

fig3o, axes3o = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'wspace': 0.38})
fig3o.suptitle(
    f'Best PLS-R Model: Calibration and LOO Cross-Validation\n'
    f'{rng_lbl_b} + {prep_b}  |  {opt_lv_clean} LVs  |  '
    f'n={n_clean} (removed {best_step} outliers from {n_all})',
    fontsize=11, fontweight='bold'
)

for ax, y_pred3, r2v, rmsev, panel, subtitle, col in [
    (axes3o[0], y_cal_clean, r2_cal_c, rmse_cal_c, '(a)', 'Calibration',    '#1a6faf'),
    (axes3o[1], y_cv_clean,  r2_cv_c,  rmse_cv_c,  '(b)', 'LOO-CV',         '#c0392b'),
]:
    ax.plot([ppm_lo3, ppm_hi3], [ppm_lo3, ppm_hi3], 'k--',
            linewidth=1.2, alpha=0.55, label='1:1 line', zorder=1)
    ax.fill_between([ppm_lo3, ppm_hi3], [ppm_lo3 - 5, ppm_hi3 - 5],
                    [ppm_lo3 + 5, ppm_hi3 + 5], alpha=0.07, color=col)
    sc3 = ax.scatter(y_clean, y_pred3, c=y_clean, cmap=cmap3,
                     vmin=y_clean.min(), vmax=y_clean.max(),
                     s=65, edgecolors='k', linewidths=0.5, alpha=0.9, zorder=3)
    for xi, yi, sid in zip(y_clean, y_pred3, ids_clean):
        ax.annotate(sid.replace('PW_', ''), (xi, yi), fontsize=5,
                    xytext=(2, 2), textcoords='offset points', color='#333')
    ax.text(0.05, 0.96,
            f'R$^2$ = {r2v:.4f}\nRMSE = {rmsev:.2f} ppm\nn = {n_clean}\nLVs = {opt_lv_clean}',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='#eef6fc', ec=col, alpha=0.9, lw=1.2))
    ax.set_xlabel('Reference [NO$_3^-$-N] (ppm)')
    ax.set_ylabel('PLS-R Predicted [NO$_3^-$-N] (ppm)')
    ax.set_title(f'{panel}  {subtitle}', loc='left', fontsize=10)
    ax.set_xlim(ppm_lo3, ppm_hi3)
    ax.set_ylim(ppm_lo3, ppm_hi3)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.legend(fontsize=8, loc='lower right')

plt.colorbar(sc3, ax=axes3o, fraction=0.015, pad=0.02,
             label='Reference [NO$_3^-$-N] (ppm)')

fig3o.savefig(os.path.join(OUT_DIR, 'FigO3_Best_Model_LOOCV.png'), dpi=DPI, bbox_inches='tight')
print("  Saved: FigO3_Best_Model_LOOCV.png")
plt.close(fig3o)

# =============================================================================
#  STEP 3 -- TRAIN / TEST SPLIT (70/30 stratified)
# =============================================================================
print("Step 3: Train/test split (70/30 stratified) ...")

conc_bands_labels = np.digitize(y_clean, [0, 25, 50, 80, 200])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx_s, test_idx_s = next(sss.split(X_clean, conc_bands_labels))

X_tr = X_clean[train_idx_s];  y_tr = y_clean[train_idx_s]
X_te = X_clean[test_idx_s];   y_te = y_clean[test_idx_s]
ids_tr = [ids_clean[i] for i in train_idx_s]
ids_te = [ids_clean[i] for i in test_idx_s]

# Optimise LVs on training set
rmsecv_tr, r2cv_tr_arr, lv_tr = lv_scan(X_tr, y_tr)
opt_lv_tr = int(np.argmin(rmsecv_tr)) + 1
print(f"  Train n={len(y_tr)}  Opt LV={opt_lv_tr}  "
      f"Train RMSECV={rmsecv_tr[opt_lv_tr-1]:.2f}  R2CV={r2cv_tr_arr[opt_lv_tr-1]:.4f}")

pls_split = PLSRegression(n_components=opt_lv_tr)
pls_split.fit(X_tr, y_tr)

y_cal_tr = pls_split.predict(X_tr).ravel()
y_cv_tr  = cross_val_predict(pls_split, X_tr, y_tr, cv=LeaveOneOut()).ravel()
y_pred_te = pls_split.predict(X_te).ravel()

r2_cal_tr   = r2_score(y_tr, y_cal_tr)
rmse_cal_tr = float(np.sqrt(mean_squared_error(y_tr, y_cal_tr)))
r2_cv_tr    = r2_score(y_tr, y_cv_tr)
rmse_cv_tr  = float(np.sqrt(mean_squared_error(y_tr, y_cv_tr)))
r2_te       = r2_score(y_te, y_pred_te)
rmse_te     = float(np.sqrt(mean_squared_error(y_te, y_pred_te)))

print(f"  Train Cal  R2={r2_cal_tr:.4f}  RMSE={rmse_cal_tr:.2f}")
print(f"  Train LOOCV R2={r2_cv_tr:.4f}  RMSE={rmse_cv_tr:.2f}")
print(f"  Test  R2={r2_te:.4f}  RMSE={rmse_te:.2f}\n")

# =============================================================================
#  FIGURE O4 -- Train/Test Split Scatter (3 panels)
# =============================================================================
print("Fig O4: Train/test split scatter ...")

ppm_lo4 = y_clean.min() - 5
ppm_hi4 = y_clean.max() + 5

fig4o, axes4o = plt.subplots(1, 3, figsize=(19, 7), gridspec_kw={'wspace': 0.35})
fig4o.suptitle(
    f'Train/Test Split Evaluation  (70/30 stratified, n_train={len(y_tr)}, n_test={len(y_te)})\n'
    f'{rng_lbl_b} + {prep_b}  |  LV optimised on training set (opt={opt_lv_tr})',
    fontsize=11, fontweight='bold'
)

cmap4 = plt.cm.plasma

for ax, y_ref, y_pred4, r2v, rmsev, panel, subtitle, col, nn in [
    (axes4o[0], y_tr, y_cal_tr,  r2_cal_tr,  rmse_cal_tr, '(a)', 'Train -- Calibration',  '#1a6faf', len(y_tr)),
    (axes4o[1], y_tr, y_cv_tr,   r2_cv_tr,   rmse_cv_tr,  '(b)', 'Train -- LOO-CV',        '#7b2fa7', len(y_tr)),
    (axes4o[2], y_te, y_pred_te, r2_te,      rmse_te,     '(c)', 'Test set prediction',    '#c0392b', len(y_te)),
]:
    ax.plot([ppm_lo4, ppm_hi4], [ppm_lo4, ppm_hi4], 'k--',
            linewidth=1.2, alpha=0.55, label='1:1', zorder=1)
    ax.fill_between([ppm_lo4, ppm_hi4], [ppm_lo4 - 5, ppm_hi4 - 5],
                    [ppm_lo4 + 5, ppm_hi4 + 5], alpha=0.07, color=col)
    sc4 = ax.scatter(y_ref, y_pred4, c=y_ref, cmap=cmap4,
                     vmin=y_clean.min(), vmax=y_clean.max(),
                     s=65, edgecolors='k', linewidths=0.5, alpha=0.9, zorder=3)
    ax.text(0.05, 0.96,
            f'R$^2$ = {r2v:.4f}\nRMSE = {rmsev:.2f} ppm\nn = {nn}',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=col, alpha=0.92, lw=1.2))
    ax.set_xlabel('Reference [NO$_3^-$-N] (ppm)')
    ax.set_ylabel('Predicted [NO$_3^-$-N] (ppm)')
    ax.set_title(f'{panel}  {subtitle}', loc='left', fontsize=10)
    ax.set_xlim(ppm_lo4, ppm_hi4)
    ax.set_ylim(ppm_lo4, ppm_hi4)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.legend(fontsize=8, loc='lower right')

plt.colorbar(sc4, ax=axes4o, fraction=0.012, pad=0.02,
             label='Reference [NO$_3^-$-N] (ppm)')

fig4o.savefig(os.path.join(OUT_DIR, 'FigO4_TrainTest_Split.png'), dpi=DPI, bbox_inches='tight')
print("  Saved: FigO4_TrainTest_Split.png")
plt.close(fig4o)

# =============================================================================
#  FIGURE O5 -- Full Comparison Summary: Baseline vs Optimised
# =============================================================================
print("Fig O5: Full comparison summary ...")

# Baseline: default params on full dataset (200-400nm SNV+SG1 6LV)
mask_base  = (wl_axis >= 200) & (wl_axis <= 400)
X_base_pre = sg1(snv(spectra_all[:, mask_base]))
y_cv_base  = cross_val_predict(PLSRegression(n_components=6),
                                X_base_pre, y_all, cv=LeaveOneOut()).ravel()
r2_base    = r2_score(y_all, y_cv_base)
rmse_base  = float(np.sqrt(mean_squared_error(y_all, y_cv_base)))

comparison = [
    ('Baseline\n(200-400nm SNV+SG1,\n6 LVs, n=126)',       r2_base,   rmse_base,   '#7f8c8d'),
    (f'Grid best\n({rng_lbl_b}, {prep_b},\n{best_cfg["lv"]} LVs, n={n_all})',
                                                             best_r2cv, best_cfg['rmsecv'], '#f39c12'),
    (f'+ Outlier removal\n({best_step} removed,\n{opt_lv_clean} LVs, n={n_clean})',
                                                             r2_cv_c,  rmse_cv_c,   '#27ae60'),
    (f'Train LOO-CV\n(70% split, n_tr={len(y_tr)},\n{opt_lv_tr} LVs)',
                                                             r2_cv_tr, rmse_cv_tr,  '#1a6faf'),
    (f'Test set\n(30% split, n_te={len(y_te)},\n{opt_lv_tr} LVs)',
                                                             r2_te,    rmse_te,     '#c0392b'),
]

fig5o, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.42})
fig5o.suptitle(
    'Summary: Model Performance Progression\nBaseline --> Grid Search --> Outlier Removal --> Train/Test Split',
    fontsize=12, fontweight='bold'
)

labels5  = [c[0] for c in comparison]
r2_vals  = [c[1] for c in comparison]
rmse_vals= [c[2] for c in comparison]
cols5    = [c[3] for c in comparison]
x5       = np.arange(len(comparison))

bars5a = ax5a.bar(x5, r2_vals, color=cols5, edgecolor='k', linewidth=0.8, alpha=0.82)
ax5a.axhline(TARGET_R2, color='#e31a1c', linewidth=1.5, linestyle='--',
             label=f'Target R$^2$ = {TARGET_R2}')
for bar, val in zip(bars5a, r2_vals):
    ax5a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
              f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax5a.set_xticks(x5)
ax5a.set_xticklabels(labels5, fontsize=8)
ax5a.set_ylabel('LOO-CV / Test R$^2$')
ax5a.set_ylim(0, 1.05)
ax5a.set_title('(a)  R$^2$ progression', loc='left', fontsize=10)
ax5a.legend(fontsize=9)
ax5a.yaxis.set_minor_locator(AutoMinorLocator(4))

bars5b = ax5b.bar(x5, rmse_vals, color=cols5, edgecolor='k', linewidth=0.8, alpha=0.82)
for bar, val in zip(bars5b, rmse_vals):
    ax5b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax5b.set_xticks(x5)
ax5b.set_xticklabels(labels5, fontsize=8)
ax5b.set_ylabel('RMSE (ppm)')
ax5b.set_title('(b)  RMSE progression', loc='left', fontsize=10)
ax5b.yaxis.set_minor_locator(AutoMinorLocator(4))

fig5o.savefig(os.path.join(OUT_DIR, 'FigO5_Comparison_Summary.png'), dpi=DPI, bbox_inches='tight')
print("  Saved: FigO5_Comparison_Summary.png")
plt.close(fig5o)

# =============================================================================
#  FIGURE O6 -- LV Curves for best 4 preprocessing methods (best range)
# =============================================================================
print("Fig O6: LV curves for all preprocessing methods ...")

best_range_rows = [r for r in grid_rows if r['Range'] == rng_lbl_b]

fig6o, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'wspace': 0.38})
fig6o.suptitle(
    f'LV Optimisation Curves by Preprocessing Method  ({rng_lbl_b})\n'
    f'n = {n_all} samples',
    fontsize=11, fontweight='bold'
)

colours6 = ['#1a6faf', '#c0392b', '#27ae60', '#e67e22', '#7b2fa7']
for row, col in zip(best_range_rows, colours6):
    ax6a.plot(row['lv_ax'], row['rmsecv_arr'], 'o-', color=col, linewidth=1.8,
              markersize=5, markerfacecolor='white', markeredgewidth=1.5,
              label=f"{row['Preprocessing']} (LV*={row['Opt_LV']})")
    ax6b.plot(row['lv_ax'], row['r2cv_arr'],   's-', color=col, linewidth=1.8,
              markersize=5, markerfacecolor='white', markeredgewidth=1.5,
              label=f"{row['Preprocessing']} (LV*={row['Opt_LV']})")

ax6a.set_xlabel('Latent Variables')
ax6a.set_ylabel('RMSECV (ppm)')
ax6a.set_title('(a)  RMSECV vs. LV count', loc='left', fontsize=10)
ax6a.legend(fontsize=8)
ax6a.xaxis.set_minor_locator(AutoMinorLocator(1))
ax6a.yaxis.set_minor_locator(AutoMinorLocator(4))

ax6b.set_xlabel('Latent Variables')
ax6b.set_ylabel('LOO-CV R$^2$')
ax6b.set_title('(b)  R$^2$CV vs. LV count', loc='left', fontsize=10)
ax6b.legend(fontsize=8)
ax6b.xaxis.set_minor_locator(AutoMinorLocator(1))
ax6b.yaxis.set_minor_locator(AutoMinorLocator(4))

fig6o.savefig(os.path.join(OUT_DIR, 'FigO6_LV_Curves_AllPrep.png'), dpi=DPI, bbox_inches='tight')
print("  Saved: FigO6_LV_Curves_AllPrep.png")
plt.close(fig6o)

# =============================================================================
#  CSV: Removal log + Grid results
# =============================================================================
pd.DataFrame(removal_log, columns=['step','n','r2cv','rmsecv','opt_lv',
                                    'removed','removed_ppm','removed_resid']) \
  .drop(columns=['cur_idx','y_cv'], errors='ignore') \
  .to_csv(os.path.join(OUT_DIR, 'Outlier_Removal_Log.csv'), index=False)
print("  Saved: Outlier_Removal_Log.csv")

pd.DataFrame([{k: v for k, v in r.items() if k not in
               ('rmsecv_arr','r2cv_arr','lv_ax')} for r in grid_rows]) \
  .to_csv(os.path.join(OUT_DIR, 'Parameter_Grid_Results.csv'), index=False)
print("  Saved: Parameter_Grid_Results.csv")

# =============================================================================
#  CONSOLE SUMMARY
# =============================================================================
SEP = "=" * 72
print()
print(SEP)
print("  MODEL OPTIMISATION COMPLETE")
print(SEP)
print(f"  Full dataset           : n={n_all}, range {y_all.min():.0f}-{y_all.max():.0f} ppm")
print(f"  Baseline LOO-CV        : R2={r2_base:.4f}  RMSE={rmse_base:.2f} ppm")
print(f"  Best param combo       : {rng_lbl_b} + {prep_b}  LV={best_cfg['lv']}")
print(f"  Grid-best LOO-CV       : R2={best_r2cv:.4f}  RMSE={best_cfg['rmsecv']:.2f} ppm")
print(f"  After outlier removal  : n={n_clean} (-{best_step} removed)  LV={opt_lv_clean}")
print(f"  Clean LOO-CV           : R2={r2_cv_c:.4f}  RMSE={rmse_cv_c:.2f} ppm")
print(f"  Train LOO-CV (70%)     : R2={r2_cv_tr:.4f}  RMSE={rmse_cv_tr:.2f} ppm  n_tr={len(y_tr)}")
print(f"  Test set (30%)         : R2={r2_te:.4f}  RMSE={rmse_te:.2f} ppm  n_te={len(y_te)}")
print()
print(f"  Removed samples ({best_step}):")
for sid in removed_ids:
    ppm_val = float(re.findall(r"\d+", ppm_map.get(sid, '0 ppm'))[0])
    print(f"    {sid:<12s}  ({ppm_val:.0f} ppm)")
print()
print("  Output files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"    {f}")
print(SEP)
