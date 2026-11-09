"""Split the departure at each compression into the rate-odd part (loading minus unloading) and the
even part (their mean): first order in 1/T is odd; a floor independent of T is even."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
F = Path(sys.argv[1]); st = np.load(F.parent / "figure4_stationary.npz")
es = st["compression"]; Qs = CubicSpline(es, st["flux"][:, 0] / st["flux"][0, 0]); Ps = CubicSpline(es, st["stress"][:, 2, 2])
Ppk = np.abs(st["stress"][:, 2, 2]).max()
print("stationary compression grid:", es.min(), es.max(), len(es), " Q_s range", Qs(es).min(), Qs(es).max())
for solid in ("elastic", "viscoelastic"):
    d = np.load(F / f"figure4_criterion_{solid}.npz"); c = list(d["columns"]); R = d["rows"]
    print(f"\n{solid}:  T, max|odd dq|, max|even dq|, e at max even, T*odd;   same for dP")
    for j, T in enumerate(d["durations"]):
        r = R[(R[:, c.index("duration_index")] == j) & (R[:, c.index("stage")] == 0)]
        t, e = r[:, c.index("time")], r[:, c.index("compression")]
        dq = r[:, c.index("Q1")] / R[R[:, c.index("duration_index")] == j][0, c.index("Q1")] - Qs(e)
        dP = (r[:, c.index("P33")] - Ps(e)) / Ppk
        n = len(t) // 2                      # loading 0..n, unloading n..end, mirror pairs i <-> 2n - i
        i = np.arange(0, n + 1); k = 2 * n - i
        out = []
        for x in (dq, dP):
            odd, even = (x[i] - x[k]) / 2, (x[i] + x[k]) / 2
            out += [np.abs(odd).max(), np.abs(even).max(), e[i][np.argmax(np.abs(even))], T * np.abs(odd).max()]
        print(f"{T:8.3f}  {out[0]:.2e} {out[1]:+.2e} e={out[2]:.3f} {out[3]:.3f}   {out[4]:.2e} {out[5]:.2e} e={out[6]:.3f} {out[7]:.3f}")
