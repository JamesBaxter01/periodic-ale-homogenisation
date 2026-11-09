"""Compare the 40-step pulses (figure4_criterion_steps40.npz) with the 100-step ones of the paper.

Usage: python compare_steps.py STEPS40.npz [CRITERION_ELASTIC.npz]
References (100 steps): results/paper/figure4_cycles.npz (elastic and viscoelastic, T = 2 s, every
2nd step), figure4_rates.npz (viscoelastic, every step), and, once the elastic sweep is back,
figure4_criterion_elastic.npz (elastic T = 20 s). Stationary cell: figure4_stationary.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path("/Users/jamesbaxter/Documents/University/PhD/Code/periodic-ale-homogenisation/results/paper")
st = np.load(PAPER / "figure4_stationary.npz")
EPS_S, Q_S, P_S = st["compression"], st["flux"][:, 0] / st["flux"][0, 0], st["stress"][:, 2, 2]


def histories(path):
    """{(material, T): dict of arrays} for every history in a figure4_cycles-style file."""
    d = np.load(path)
    cols = list(d["columns"])
    rows = d["rows"]
    steps = json.loads(str(d["metadata"]))["settings"]["steps"]
    out = {}
    for m, material in enumerate(d["materials"]):
        for j, T in enumerate(d["durations"]):
            k = (rows[:, cols.index("material")] == m) & (rows[:, cols.index("duration_index")] == j)
            h = {c: rows[k, cols.index(c)] for c in ("time", "compression", "Q1", "fluid_content", "P33")}
            h["steps"] = steps
            out[(str(material), round(float(T), 6))] = h
    return out


def metrics(h, T):
    t, e, q, P, m = h["time"], h["compression"], h["Q1"] / h["Q1"][0], h["P33"], h["fluid_content"]
    dq = q - np.interp(e, EPS_S, Q_S)
    pulse = t <= T * (1 + 1e-9)
    stress = -P
    i = np.argmax(stress)
    return dict(
        dep_max=np.max(np.abs(dq)), dep_end=dq[-1], peak=stress[i], t_peak=t[i] / T,
        tension=max(0.0, -stress.min()),
        loop=np.trapezoid(stress[pulse], e[pulse]),  # energy lost per pulse (J/m^3), stationary: 0
        content_end=m[-1])


def main():
    new = histories(sys.argv[1])
    refs = {}
    for f in ("figure4_cycles.npz", "figure4_rates.npz"):
        refs.update(histories(PAPER / f))
    if len(sys.argv) > 2:
        refs.update(histories(sys.argv[2]))
    names = ["dep_max", "dep_end", "peak", "t_peak", "tension", "loop", "content_end"]
    print(f"{'solid':13s} {'T':>5s} {'steps':>5s} " + " ".join(f"{n:>11s}" for n in names))
    for key in sorted(new):
        rows = [(new[key]["steps"], metrics(new[key], key[1]))]
        if key in refs:
            rows.append((refs[key]["steps"], metrics(refs[key], key[1])))
        for steps, mt in rows:
            print(f"{key[0]:13s} {key[1]:5g} {steps:5d} " + " ".join(f"{mt[n]:11.4f}" for n in names))
        if len(rows) == 2:
            a, b = rows[0][1], rows[1][1]
            print(f"{'':13s} {'':5s} {'diff':>5s} " + " ".join(
                f"{(a[n] - b[n]) / max(abs(b[n]), 1e-12):+10.1%} " for n in names))
        else:
            print(f"{'':13s} {'':5s}  no 100-step reference yet (comes from the elastic sweep)")


if __name__ == "__main__":
    main()
