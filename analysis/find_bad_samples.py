"""
Greedy bad-sample finder for the PLS-NO3 model.
Iteratively removes the sample with the largest LOO residual,
recomputes RMSECV, keeps removing as long as RMSECV improves.
Stops when no further improvement (or fewer than 80 samples remain).
"""
import sys, warnings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# ── same config as train_model.py ─────────────────────────────────────────────
DATA_PATH = Path(r"C:\Project N\Final code\27.01.2025\Most Recent")
WV_START, WV_END = 215, 380
SG_W, SG_P, SG_D = 11, 2, 1
MAX_LV = 15
MIN_SAMPLES = 80   # stop removing if we'd go below this

BASE_EXCLUDE = {
    "PW_29", "PW_2",
    "PW_1F0", "PW_1F0.0", "PW_1F0.1", "PW_1F0.2", "PW_1F0.3", "PW_1F0.4", "PW_1F1",
    "PW_2F",  "PW_2F0",  "PW_2F0.1", "PW_2F0.2", "PW_2F2",
    "PW_3F0", "PW_4F0", "PW_4F0.1", "PW_4F0.2", "PW_4F2",
    "PW_6F0", "PW_7F0",  "PW_52F0",
    "PW_29.1", "PW_29.2", "PW_29.3",
}

# ── load SPZ files (same logic as train_model.py) ─────────────────────────────
target_map = {
    "PW_1":4,"PW_1B":5,"PW_1C":46,"PW_1D":21,"PW_1E":47,
    "PW_10":44,"PW_11":26,"PW_12":34,"PW_13":52,"PW_14":50,"PW_15":56,
    "PW_16":7,"PW_17":73,"PW_17A":82,"PW_17B":12,"PW_18":42,"PW_19":64,
    "PW_20":46,"PW_20B":33,"PW_20C":28,
    "PW_21":44,"PW_22A":24,"PW_22B":24,"PW_22C":24,"PW_22D":28,"PW_22E":42,
    "PW_22F":26,"PW_22G":76,"PW_22H":23,"PW_22I":30,"PW_22J":22,
    "PW_23":42,"PW_23A":46,"PW_24":90,"PW_24A":25,"PW_24B":22,"PW_24C":4,
    "PW_24D":12,"PW_24E":25,"PW_24F":23,"PW_26":32,"PW_27":41,"PW_28":45,
    "PW_28A":34,"PW_28B":52,"PW_28C":32,"PW_2B":11,"PW_2C":28,"PW_2D":4,
    "PW_2E":20,"PW_30":46,"PW_30A":36,"PW_31":51,"PW_31A":13,"PW_31B":10,
    "PW_31C":10,"PW_31D":11,"PW_31E":11,"PW_31F":18,"PW_31G":8,"PW_31H":16,
    "PW_32":48,"PW_36":53,"PW_36A":75,"PW_37":44,"PW_37A":54,"PW_38":56,
    "PW_38A":24,"PW_38B":12,"PW_38C":12,"PW_39":47,"PW_39A":90,"PW_39B":38,
    "PW_39C":35,"PW_39D":23,"PW_39E":23,"PW_39E.2":19,"PW_39F":11,"PW_39G":9,
    "PW_39I":12,"PW_39J":54,"PW_3A":82,"PW_3B":5,"PW_40":30,"PW_40A":11,
    "PW_40B":11,"PW_40C":9,"PW_40D":9,"PW_40E":18,"PW_41":68,"PW_42":90,
    "PW_42A":37,"PW_42B":29,"PW_42C":29,"PW_42D":21,"PW_42E":20,"PW_42F":26,
    "PW_42I":36,"PW_42J":22,"PW_43":73,"PW_44":41,"PW_45":24,"PW_45A":72,
    "PW_45B":10,"PW_45C":6,"PW_45D":8,"PW_45E":9,"PW_45G":11,"PW_46":46,
    "PW_47":22,"PW_47A":32,"PW_49":74,"PW_5":54,"PW_50":56,"PW_51":77,
    "PW_51A":136,"PW_51B":32,"PW_51C":90,"PW_51D":18,"PW_51E":20,"PW_52":50,
    "PW_52A":24,"PW_52B":27,"PW_52C":18,"PW_52D":33,"PW_53":23,"PW_53A":117,
    "PW_53B":58,"PW_54.2":35,"PW_54.3":38,"PW_6":41,"PW_6A":50,"PW_7":44,
    "PW_7A":60,"PW_8":53,"PW_9":48,"PW_HS1":42,
    "PW_04":43,"PW_20A":46,"PW_3":82,"PW_32A":48,
}

import pandas as pd

def load_all():
    samples, names, ys = [], [], []
    wv_ref = None
    for sid, ppm in target_map.items():
        if sid in BASE_EXCLUDE:
            continue
        fp = DATA_PATH / f"{sid}.spz"
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, sep="\t", skiprows=25, header=None, encoding="latin-1")
            wv = df.iloc[:, 0].values.astype(float)
            ab = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
            mask = (wv >= WV_START) & (wv <= WV_END)
            if mask.sum() < 10:
                continue
            ab_region = ab[mask]
            if not np.all(np.isfinite(ab_region)):
                continue
            if wv_ref is None:
                wv_ref = wv[mask]
            x = savgol_filter(ab_region, SG_W, SG_P, SG_D)
            samples.append(x)
            names.append(sid)
            ys.append(float(ppm))
        except Exception as e:
            pass
    return np.array(samples), np.array(ys), names, wv_ref

def loo_rmsecv(X, y, n_lv):
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        idx = list(range(n))
        idx.pop(i)
        Xtr, ytr = X[idx], y[idx]
        Xte = X[i:i+1]
        m = PLSRegression(n_components=n_lv)
        m.fit(Xtr, ytr)
        preds[i] = float(m.predict(Xte).ravel()[0])
    resid = y - preds
    rmsecv = np.sqrt(np.mean(resid**2))
    r2 = r2_score(y, preds)
    return rmsecv, r2, preds

def best_lv(X, y, max_lv=MAX_LV):
    best_rmse, best_lv_ = np.inf, 1
    for lv in range(1, min(max_lv, len(y)-1)+1):
        r, _, _ = loo_rmsecv(X, y, lv)
        if r < best_rmse:
            best_rmse, best_lv_ = r, lv
    return best_lv_, best_rmse

# ── main greedy loop ───────────────────────────────────────────────────────────
print("Loading data...")
X_all, y_all, names_all, wv = load_all()
n_total = len(y_all)
print(f"  {n_total} samples loaded")

active = list(range(n_total))
removed = []
iteration = 0

lv0, rmse0 = best_lv(X_all[active], y_all[active])
print(f"\nBaseline: n={len(active)}, LV={lv0}, RMSECV={rmse0:.2f} ppm\n")
print(f"{'Iter':>4}  {'Removed sample':<14} {'|Resid|':>8}  {'n':>4}  {'LV':>3}  {'RMSECV':>8}  {'R2-LOO':>7}  {'Delta':>8}")
print("-" * 72)

current_rmse = rmse0
current_lv = lv0

while True:
    X_act = X_all[active]
    y_act = y_all[active]
    names_act = [names_all[i] for i in active]

    # compute LOO residuals with current best LV
    _, _, preds = loo_rmsecv(X_act, y_act, current_lv)
    resid = np.abs(y_act - preds)
    worst_idx = int(np.argmax(resid))
    worst_name = names_act[worst_idx]
    worst_resid = resid[worst_idx]

    # try removing worst sample
    trial_active = [i for j, i in enumerate(active) if j != worst_idx]
    if len(trial_active) < MIN_SAMPLES:
        print(f"\nStopping: would drop below {MIN_SAMPLES} samples.")
        break

    trial_lv, trial_rmse = best_lv(X_all[trial_active], y_all[trial_active])
    _, trial_r2, _ = loo_rmsecv(X_all[trial_active], y_all[trial_active], trial_lv)
    delta = trial_rmse - current_rmse

    iteration += 1
    marker = " <-- IMPROVED" if delta < 0 else ""
    print(f"{iteration:>4}  {worst_name:<14} {worst_resid:>8.1f}  {len(trial_active):>4}  "
          f"{trial_lv:>3}  {trial_rmse:>8.2f}  {trial_r2:>7.4f}  {delta:>+8.2f}{marker}")

    if delta < 0:
        removed.append((worst_name, worst_resid, delta))
        active = trial_active
        current_rmse = trial_rmse
        current_lv = trial_lv
    else:
        print(f"\nStopping: removing '{worst_name}' (|resid|={worst_resid:.1f}) "
              f"did not improve RMSECV ({delta:+.2f}).")
        break

print(f"\n{'='*60}")
print(f"RESULT: {len(active)} samples kept, {len(removed)} removed")
print(f"Final: LV={current_lv}, RMSECV={current_rmse:.2f} ppm")
_, final_r2, _ = loo_rmsecv(X_all[active], y_all[active], current_lv)
print(f"       R2-LOO={final_r2:.4f}")
print(f"\nSamples to add to EXCLUDE set:")
for name, resid, delta in removed:
    print(f"  \"{name}\",  # |LOO resid|={resid:.1f} ppm, RMSECV delta={delta:+.2f}")
