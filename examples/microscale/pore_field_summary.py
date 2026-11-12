"""Summary of the Section 5.2 mechanism check (figure4_pore_fields.py): what inside the cell carries its departure
from the stationary cell during the Figure 4 pulse.

Press Run (or python examples/microscale/pore_field_summary.py [RUN]). Reads RUN/figure4_pore_fields*.npz and the
stationary cell (DATA/figure4_stationary.npz) and writes RUN/pore_fields.md with, for each solid and pulse:
- every tenth step: the axial stress and its departure from the stationary cell at the same compression, split into
  solid elastic (minus the stationary stress), Kelvin-Voigt, fluid viscous (flow within the pores, deviatoric) and
  fluid viscous (uniform dilatation lambda_p); the fluid content and flux against the stationary cell;
- the dissipation above the steady background flow (step 0), integrated over the pulse and hold, by mechanism;
- the peak stress, the largest tension of the pulse and its split.
The claims of Section 5.2 that these numbers support are listed at the end.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "paper"
RUN = DATA / "pore_fields_20261112_180711"
PARTS = ["solid_elastic", "solid_kv", "fluid_pressure", "fluid_viscous_dev", "fluid_viscous_vol"]
MECHANISMS = {"fluid_dev": "flow within the pores", "fluid_vol": "uniform dilatation", "skeleton": "skeleton"}


def load(run):
    files = sorted(glob.glob(str(run / "figure4_pore_fields*.npz")))
    if not files:
        raise SystemExit(f"No figure4_pore_fields*.npz in {run}; run figure4_pore_fields.py (run.py pore-fields)")
    cases = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        cols = list(d["columns"])
        for m, material in enumerate(d["materials"]):
            for j, T in enumerate(d["durations"]):
                r = d["rows"][(d["rows"][:, 0] == m) & (d["rows"][:, 1] == j)]
                r = r[np.argsort(r[:, cols.index("step")])]
                cases.append((str(material), float(T), {c: r[:, i] for i, c in enumerate(cols)}))
    return cases


def summary(material, T, r, s):
    sc, sm, sP, sQ = s["compression"], s["fluid_content"], s["stress"][:, 2, 2], s["flux"][:, 0]
    c = r["compression"]
    Pst, mst, Qst = np.interp(c, sc, sP), np.interp(c, sc, sm), np.interp(c, sc, sQ) / sQ[0]
    lines = [f"## {material.capitalize()} skeleton, T = {T:g} s", "",
             "| t/T | P33 (Pa) | P33 − stationary | solid elastic − stationary | Kelvin-Voigt | fluid, pore flow "
             "| fluid, dilatation | m_f − stationary | Q/Q0 | stationary Q/Q0 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k in range(0, len(c), 10):
        lines.append(f"| {r['step'][k] / 100:.2f} | {r['P33'][k]:+.2f} | {r['P33'][k] - Pst[k]:+.2f} "
                     f"| {r['P33_solid_elastic'][k] - Pst[k]:+.2f} | {r['P33_solid_kv'][k]:+.2f} "
                     f"| {r['P33_fluid_viscous_dev'][k]:+.2f} | {r['P33_fluid_viscous_vol'][k]:+.2f} "
                     f"| {r['fluid_content'][k] - mst[k]:+.4f} | {r['Q1'][k] / r['Q1'][0]:.3f} | {Qst[k]:.3f} |")
    excess = {k: np.trapezoid(r[f"D_{k}"] - r[f"D_{k}"][0], r["step"]) for k in MECHANISMS}
    total = sum(excess.values())
    peak = int(np.argmin(r["P33"]))
    pulse = r["step"] <= 100
    tension = int(np.argmax(np.where(pulse, r["P33"], -np.inf)))
    fluid = r["P33_fluid_viscous_dev"][tension] + r["P33_fluid_viscous_vol"][tension]
    lines += ["", "Dissipation above the steady background flow, pulse and hold: "
              + ", ".join(f"{name} {100 * excess[k] / total:.1f} %" for k, name in MECHANISMS.items()) + ".",
              f"Peak compressive stress {-r['P33'][peak]:.2f} Pa at t/T = {r['step'][peak] / 100:.2f} "
              f"(stationary peak {-sP.min():.2f} Pa).",
              f"Largest tension in the pulse {r['P33'][tension]:.2f} Pa at t/T = {r['step'][tension] / 100:.2f}: "
              f"solid elastic {r['P33_solid_elastic'][tension]:+.2f}, Kelvin-Voigt {r['P33_solid_kv'][tension]:+.2f}, "
              f"fluid viscous {fluid:+.2f} Pa; fluid content {r['fluid_content'][tension] - mst[tension]:+.4f} "
              f"against the stationary cell.",
              f"Pore-pressure fluctuation at t/T = 0.25: {r['pf_min'][r['step'] == 25][0]:.3g} to "
              f"{r['pf_max'][r['step'] == 25][0]:.3g} (solver units; background {r['pf_min'][0]:.3g} to "
              f"{r['pf_max'][0]:.3g}).", ""]
    return lines


CLAIMS = """## What these numbers support (Section 5.2)

- The pores lag behind the imposed deformation: under load the cell holds more fluid than the stationary cell at the
  same compression, and less on unloading.
- The fluid must flow within the pores to follow: most of the extra fluid dissipation is deviatoric flow within the
  pores; the rest is the uniform dilatation lambda_p, by which fluid leaves or enters the cell.
- The lag adds to the stress: the fluid the cell still holds makes the solid carry more of the compression (the solid
  elastic part), and the fluid's viscous stress adds the rest. The stress peaks above the stationary one, just before
  mid-pulse, and the flux decreases less while the pores still hold more fluid (T = 2 s).
- On unloading the cell has not yet drawn its fluid back in, and the stress becomes tensile.
- Skeleton viscosity (T = 0.2 s) adds its stress directly, carrying the higher peak and most of the tension, and
  stiffens the pore walls against the pressure that drives the pore flow: the pore pressure is larger, yet the fluid
  content and flux stay closer to the stationary cell.
- The pore-pressure fluctuation has zero fluid mean, so it reaches the homogenised stress only through the solid walls
  (its own part, fluid pressure, is zero to round-off).
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", type=Path, default=RUN, help=f"run folder (default {RUN.relative_to(ROOT)})")
    a = p.parse_args()
    stationary = np.load(DATA / "figure4_stationary.npz", allow_pickle=True)
    lines = ["# Section 5.2 mechanism check: the pulse of Figure 4 split by mechanism", "",
             "Stresses in Pa, positive in tension; departures at the same compression as the stationary cell "
             "(figure4_stationary.npz). Written by examples/microscale/pore_field_summary.py.", ""]
    for material, T, r in load(a.run):
        lines += summary(material, T, r, stationary)
    lines.append(CLAIMS)
    out = a.run / "pore_fields.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
