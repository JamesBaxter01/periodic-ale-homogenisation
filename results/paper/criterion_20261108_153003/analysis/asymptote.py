"""1/T coefficients of the stationary-cell departures, whether C/T bounds the whole sweep, and a figure."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

F = Path(sys.argv[1]); OUT = Path(sys.argv[2])
st = np.load(F.parent / "figure4_stationary.npz"); es = st["compression"]
Qs = CubicSpline(es, st["flux"][:, 0] / st["flux"][0, 0]); Ps = CubicSpline(es, st["stress"][:, 2, 2])
Ppk = np.abs(st["stress"][:, 2, 2]).max(); EMAX = es.max()
U = np.trapezoid(-Ps(np.linspace(0, EMAX, 401)), np.linspace(0, EMAX, 401))
TAU, MU_G = 0.7, 0.02
res = {}
for solid in ("elastic", "viscoelastic"):
    d = np.load(F / f"figure4_criterion_{solid}.npz"); c = list(d["columns"]); R = d["rows"]
    Ts, rows = d["durations"], []
    for j, T in enumerate(Ts):
        a = R[R[:, c.index("duration_index")] == j]; r = a[a[:, c.index("stage")] == 0]
        e = r[:, c.index("compression")]; P = r[:, c.index("P33")]
        dq = r[:, c.index("Q1")] / a[0, c.index("Q1")] - Qs(e); dP = (P - Ps(e)) / Ppk
        n = len(e) // 2; i = np.arange(n + 1); k = 2 * n - i
        rows.append([np.abs(dq).max(), np.abs((dq[i] - dq[k]) / 2).max(), np.abs(dP).max(),
                     np.trapezoid(-P, e) / U])
    res[solid] = (Ts, np.array(rows))
names = ["flux (all)", "flux (rate-odd)", "stress", "energy W/U"]
print(f"tau = {TAU} s, mu_f/G = {MU_G} s;  coefficient C = T*X at T = 200 s;  X <= C/T over the sweep?")
for solid, (Ts, X) in res.items():
    for m, nm in enumerate(names):
        C = Ts[-1] * X[-1, m]
        ratio = (Ts * X[:, m]).max() / C
        print(f"  {solid:12s} {nm:16s} C = {C:.3f} s = {C/TAU:.2f} tau = {C/MU_G:5.1f} mu_f/G;"
              f"  max T*X / C = {ratio:.3f};  1% at T = {C/0.01:6.1f} s = {C/0.01/TAU:5.0f} tau")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, constrained_layout=True)
cols = {"flux (rate-odd)": "#1f5fa8", "stress": "#555555", "energy W/U": "#8fb3dc"}
for a, (solid, (Ts, X)) in zip(ax, res.items()):
    x = Ts / TAU
    for m, nm in enumerate(names):
        if nm == "flux (all)":
            a.loglog(x, X[:, m], ":", color=cols["flux (rate-odd)"], lw=1, label="flux (with floor)")
            continue
        C = Ts[-1] * X[-1, m]
        a.loglog(x, X[:, m], "o-", color=cols[nm], ms=3.5, lw=1.4, label=nm)
        a.loglog(x, C / Ts, "--", color=cols[nm], lw=0.8)
    a.axhline(0.01, color="k", lw=0.5); a.set_title(f"{solid} skeleton"); a.set_xlabel(r"$T/\tau$  ($\tau = 0.7$ s)")
    a.grid(True, which="major", lw=0.3)
ax[0].set_ylabel("stationary-cell departure"); ax[1].legend(fontsize=8, frameon=False)
fig.savefig(OUT, dpi=160); print("saved", OUT)
