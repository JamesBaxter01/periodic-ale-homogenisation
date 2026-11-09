"""History criterion: departure of the stationary cell from the transient one against pulse duration T.

Usage: python sweep.py CRITERION_FOLDER
Reads figure4_criterion_<solid>.npz (13 pulses, 0.2-200 s) and figure4_stationary.npz (the stationary
cell, the same for both skeletons). For each solid and T it reports
  dq    max |Q/Q0 - Q_st(e)/Q_st(0)|            flux departure, fraction of the rest flux
  dP    max |P33 - P33_st(e)| / max|P33_st|      stress departure, fraction of the stationary peak
  dm    max |m - m_st(e)| / (m_st(0) - m_st(emax)) fluid-content departure, fraction of the content change
  W/U   energy lost per pulse / work stored at peak compression (the stationary cell loses none)
  tau   relaxation time fitted to the flux departure in the recovery hold
plus T*dq and the local log-log slopes, and the T at which each measure falls to 10, 5, 2 and 1 %.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

FOLDER = Path(sys.argv[1])
PAPER = FOLDER.parent
st = np.load(PAPER / "figure4_stationary.npz")
e_s = st["compression"]
Qs = CubicSpline(e_s, st["flux"][:, 0] / st["flux"][0, 0])
Ps = CubicSpline(e_s, st["stress"][:, 2, 2])
ms = CubicSpline(e_s, st["fluid_content"])
P_PEAK = np.max(np.abs(st["stress"][:, 2, 2]))


def load(solid):
    d = np.load(FOLDER / f"figure4_criterion_{solid}.npz")
    cols = list(d["columns"])
    meta = json.loads(str(d["metadata"]))["settings"]
    out = []
    for j, T in enumerate(d["durations"]):
        r = d["rows"][d["rows"][:, cols.index("duration_index")] == j]
        out.append((float(T), {c: r[:, cols.index(c)] for c in cols}))
    return out, meta


def measures(T, h, emax):
    t, e, P, m, stage = h["time"], h["compression"], h["P33"], h["fluid_content"], h["stage"]
    q = h["Q1"] / h["Q1"][0]
    dq = q - Qs(e)
    dP = np.max(np.abs(P - Ps(e))) / P_PEAK
    dm = np.max(np.abs(m - ms(e))) / abs(ms(0.0) - ms(emax))
    pulse = stage == 0
    W = np.trapezoid(-P[pulse], e[pulse])
    load_e = np.linspace(0, emax, 401)
    U = np.trapezoid(-Ps(load_e), load_e)
    # tau: log-linear fit of |dq| in the hold, from 0.1 T after the pulse, above 50x the settle floor
    hold = (stage == 1) & (t >= T * 1.1)
    y = np.abs(dq[hold])
    ok = y > 5e-4
    tau = np.nan
    if ok.sum() >= 5 and np.all(np.sign(dq[hold][ok]) == np.sign(dq[hold][ok][0])):
        s = np.polyfit(t[hold][ok], np.log(y[ok]), 1)[0]
        tau = -1 / s if s < 0 else np.nan
    return dict(dq=np.max(np.abs(dq)), dP=dP, dm=dm, WU=W / U, W=W, tau=tau, dq_end=dq[-1])


def crossing(T, v, level):
    """T at which v (falling) first drops below level, log-log interpolated."""
    for i in range(1, len(T)):
        if v[i - 1] >= level > v[i]:
            a = (np.log(level) - np.log(v[i - 1])) / (np.log(v[i]) - np.log(v[i - 1]))
            return float(np.exp(np.log(T[i - 1]) + a * (np.log(T[i]) - np.log(T[i - 1]))))
    return np.nan


results = {}
for solid in ("elastic", "viscoelastic"):
    runs, meta = load(solid)
    emax = meta["compression"]
    rows = [(T, measures(T, h, emax)) for T, h in runs]
    results[solid] = rows
    T = np.array([r[0] for r in rows])
    print(f"\n{solid}  (compression {emax}, steps {meta['steps']}, recovery {meta['recovery_fraction']})")
    print(f"{'T':>8s} {'dq':>8s} {'T*dq':>8s} {'slope':>6s} {'dP':>8s} {'slope':>6s} {'dm':>8s} {'W/U':>8s} {'T*W/U':>7s} {'slope':>6s} {'tau':>6s} {'dq_end':>9s}")
    names = ("dq", "dP", "WU")
    v = {n: np.array([r[1][n] for r in rows]) for n in names}
    sl = {n: np.r_[np.nan, np.diff(np.log(v[n])) / np.diff(np.log(T))] for n in names}
    for i, (Ti, m) in enumerate(rows):
        print(f"{Ti:8.3f} {m['dq']:8.4f} {Ti*m['dq']:8.4f} {sl['dq'][i]:6.2f} {m['dP']:8.4f} {sl['dP'][i]:6.2f} "
              f"{m['dm']:8.4f} {m['WU']:8.4f} {Ti*m['WU']:7.3f} {sl['WU'][i]:6.2f} {m['tau']:6.3f} {m['dq_end']:+9.2e}")
    for n in names:
        print(f"  {n:3s} falls below " + ", ".join(f"{100*l:g}% at T = {crossing(T, v[n], l):.3g} s" for l in (0.1, 0.05, 0.02, 0.01)))
np.save(Path(__file__).with_name("sweep_results.npy"), results, allow_pickle=True)
