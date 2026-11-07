"""Where FE² dissipates more than resolved FSI per cycle (the energy-excess sentence of Section 5.3.2).

Press Run (or python examples/multiscale/cyclic_dissipation_split.py [STUDY]).

For each cyclic run (skeleton x period) of the study folder, over the last cycle:
- loop energy  W = closed integral of tau d(settlement)  (should reproduce Table 6);
- split by segment: total axial nominal stress = tau everywhere in the confined column, so the work on a
  segment per cycle is the loop of tau against its compression. Segments run between cell centres
  (resolved: solid-averaged cell displacement; FE2: cell average of the macroscopic displacement),
  plus the base-to-centre-of-cell-1 and centre-of-cell-10-to-top segments;
- FE2 dissipation of the uniform fluid dilatation lambda_p = mdot_f/m_f: (2/3) mu_f mdot_f^2 / m_f per
  unit reference volume, integrated over the column and the cycle. Energies per unit area, mJ/m^2.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/cyclic/20261026_201037"
sys.path.insert(0, str(ROOT))
from multiscale.column import Column1D, lagrange, lobatto_nodes  # noqa: E402


def loop(tau, x):
    return float(np.sum(0.5 * (tau[1:] + tau[:-1]) * np.diff(x)))


def fe2_cell_mean_u(col, y, ell, ncell):
    """Cell averages of the macroscopic axial displacement u(Z), for each time row of y."""
    zq, wq = np.polynomial.legendre.leggauss(8)
    out = np.zeros((len(y), ncell))
    nodes = lobatto_nodes(col.ku)
    for i in range(ncell):
        a, b = i * ell, (i + 1) * ell
        Z = a + (zq + 1) / 2 * (b - a)
        W = wq * (b - a) / 2
        for Zk, Wk in zip(Z, W):
            e = min(np.searchsorted(col.edges, Zk, side="right") - 1, col.elements - 1)
            xi = 2 * (Zk - col.edges[e]) / col.lengths[e] - 1
            N, _ = lagrange(nodes, np.array([xi]))
            dofs = e * col.ku + np.arange(col.ku + 1)
            out[:, i] += Wk * (y[:, dofs] @ N[0]) / ell
    return out


def segments(u_cells, settle):
    """Compression of each segment (base, cell centres 1..10, top): u_lower - u_upper; u is negative down."""
    u = np.column_stack([np.zeros(len(settle)), u_cells, -settle])
    return -(np.diff(u, axis=1))


def split(study=STUDY):
    """One row per cyclic run: (skeleton, period, W_res, W_fe2, seg_res, seg_fe2, D_lambda), energies in J/m^2."""
    rows = []
    for skeleton in ("elastic", "viscoelastic"):
        for P in ("2", "0.2"):
            folder = Path(study) / f"{skeleton}_P{P}"
            f = np.load(folder / "fe2.npz", allow_pickle=False)
            r = np.load(folder / "resolved.npz", allow_pickle=False)
            meta = json.loads(str(f["metadata"]))
            macro, cyc = meta["macro"], meta["cyclic"]
            T, n = cyc["period"], cyc["steps_per_cycle"]
            H, ncell = macro["height"], 10
            ell = H / ncell
            mu_f = meta["cell"].get("mu_f", 1.0)
            col = Column1D(height=H, area=macro["area"], elements=macro["elements"], traction=lambda t: 0.0,
                           stress_scale=1.0, modulus=1.0, displacement_order=macro["displacement_order"],
                           pressure_order=macro["pressure_order"], gauss_points=macro["gauss_points"],
                           grading=macro["grading"])
            t = f["time"]
            last = slice(len(t) - n - 1, len(t))
            tau = macro["traction"] / 2 * (1 - np.cos(2 * np.pi * t / T))
            # loop energies (settlement positive in compression)
            W_fe2 = loop(tau[last], f["settlement"][last])
            W_res = loop(tau[last], r["settlement"][last])
            s_fe2 = segments(fe2_cell_mean_u(col, f["y"], ell, ncell), f["settlement"])
            s_res = segments(r["cell_displacement"], r["settlement"])
            seg_fe2 = np.array([loop(tau[last], s_fe2[last, k]) for k in range(s_fe2.shape[1])])
            seg_res = np.array([loop(tau[last], s_res[last, k]) for k in range(s_res.shape[1])])
            # lambda_p dissipation in FE2, per unit area, last cycle
            m = f["outputs"][:, :, 9]
            mdot = np.diff(m, axis=0) / np.diff(t)[:, None]
            m_mid = 0.5 * (m[1:] + m[:-1])
            rate = (2.0 / 3.0) * mu_f * np.sum(col.weights[None, :] / macro["area"] * mdot**2 / m_mid, axis=1)
            steps = np.arange(len(t) - 1)[-n:]
            D_lambda = float(np.sum(rate[steps] * np.diff(t)[steps]))
            rows.append((skeleton, P, W_res, W_fe2, seg_res, seg_fe2, D_lambda))
    return rows


def main(study=STUDY):
    mJ = 1e3
    for skeleton, P, W_res, W_fe2, seg_res, seg_fe2, D_lambda in split(study):
        d = (seg_fe2 - seg_res) * mJ
        print(f"\n{skeleton:12s} T={P:>3s} s  loop energy resolved {W_res*mJ:6.2f}  FE2 {W_fe2*mJ:6.2f}  excess {(W_fe2-W_res)*mJ:5.2f} mJ/m^2")
        print(f"   segment sums: resolved {seg_res.sum()*mJ:6.2f}  FE2 {seg_fe2.sum()*mJ:6.2f}")
        print("   excess by segment (base..top, mJ/m^2): " + " ".join(f"{x:+.2f}" for x in d))
        print(f"   top half-cell excess {d[-1]:+.2f}, rest {d[:-1].sum():+.2f};  FE2 lambda_p dissipation {D_lambda*mJ:.3f} mJ/m^2")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else STUDY)
