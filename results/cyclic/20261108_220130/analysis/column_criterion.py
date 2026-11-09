"""History criterion at the column level: stationary-cell FE² against FE² with history and resolved FSI.

Usage: python column_criterion.py STUDY_FOLDER [STUDY_FOLDER ...]
Each STUDY_FOLDER holds <skeleton>_P<P>/{fe2,resolved}.npz and stationary_P<P>/fe2_stationary.npz
(submit_cyclic.sh). Over the last cycle, per model:
  W_osc   work of the oscillating traction, (t - t_max/2) ds over the cycle (J/m^2). The column still drifts
          under the mean traction t_max/2 at long periods; this removes that work, which is consolidation,
          not cyclic dissipation (the loop area of compare_cyclic.py keeps it).
  drift   settlement change over the cycle, and its work (t_max/2) * drift
  amp     amplitude of the fundamental Fourier component of the settlement (immune to the drift)
  lag     phase lag of that component behind the traction (deg)
Then: missed = (W_osc,res - W_osc,stat) / W_osc,res, the energy the stationary cells miss; history excess
(W_osc,hist - W_osc,res) / W_osc,res; the largest settlement difference stationary - history over the run
relative to the history amplitude (the history effect alone, free of the first-order homogenisation error).
"""
import json
import sys
from pathlib import Path

import numpy as np

def series(path):
    with np.load(path, allow_pickle=True) as d:
        return d["time"], d["settlement"]


def last_cycle(t, s, T, cycles, traction):
    k = cycles - 1
    g = np.linspace(k * T, (k + 1) * T, 801)
    if g[-1] > t[-1] * (1 + 1e-9):
        return None
    sg = np.interp(g, t, s)
    F = 0.5 * traction * (1 - np.cos(2 * np.pi * g / T))
    ph = np.exp(-2j * np.pi * (g[:-1] - k * T) / T)
    cs, cf = np.sum(sg[:-1] * ph), np.sum(F[:-1] * ph)
    n = len(g) - 1
    return dict(W_osc=np.trapezoid(F - traction / 2, sg), loop=np.trapezoid(F, sg), drift=sg[-1] - sg[0],
                amp=2 * abs(cs) / n * 2,  # peak-to-peak of the fundamental
                lag=np.degrees((np.angle(cf) - np.angle(cs) + np.pi) % (2 * np.pi) - np.pi))


def main(folders):
    print(f"{'case':24s} {'T (s)':>7s} {'model':10s} {'W_osc':>8s} {'loop':>8s} {'drift':>7s} {'amp':>7s} {'lag':>6s}"
          "   (mJ/m^2, mm, deg)")
    summary = []
    for folder in map(Path, folders):
        for run in sorted(folder.glob("*_P*")):
            if run.name.startswith("stationary") or not (run / "fe2.npz").exists():
                continue
            skeleton, P = run.name.rsplit("_P", 1)
            T = float(P)
            stat = folder / f"stationary_P{P}" / "fe2_stationary.npz"
            files = dict(history=run / "fe2.npz", stationary=stat, resolved=run / "resolved.npz")
            with np.load(files["history"], allow_pickle=True) as d:
                meta = json.loads(str(d["metadata"]))
            traction, cycles = meta["macro"]["traction"], meta["cyclic"]["cycles"]
            m, raw = {}, {}
            for name, f in files.items():
                if f.exists():
                    raw[name] = series(f)
                    m[name] = last_cycle(*raw[name], T, cycles, traction)
            for name, v in m.items():
                if v:
                    print(f"{run.name:24s} {T:7.1f} {name:10s} {1e3 * v['W_osc']:8.3f} {1e3 * v['loop']:8.3f} "
                          f"{1e3 * v['drift']:7.3f} {1e3 * v['amp']:7.3f} {v['lag']:6.2f}")
            if all(m.get(k) for k in ("history", "stationary", "resolved")):
                Wr, Ws, Wh = (m[k]["W_osc"] for k in ("resolved", "stationary", "history"))
                th, sh = raw["history"]
                ds = np.interp(th, *raw["stationary"]) - sh
                summary.append((run.name, T, (Wr - Ws) / Wr, (Wh - Wr) / Wr,
                                np.abs(ds).max() / m["history"]["amp"], m["history"]["lag"] - m["stationary"]["lag"]))
    print(f"\n{'case':24s} {'T (s)':>7s} {'missed':>8s} {'hist exc':>9s} {'max|s_st-s_h|/amp':>18s} {'lag diff':>9s}")
    for name, T, missed, excess, ds, dlag in sorted(summary, key=lambda r: (r[0].split('_')[0], r[1])):
        print(f"{name:24s} {T:7.1f} {missed:8.1%} {excess:+9.1%} {ds:18.2%} {dlag:9.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
