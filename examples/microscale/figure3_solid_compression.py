"""How far the solid is compressed at the softest, highest-pressure case of Figure 3 (the end of Section 5.1).

Press Run (a few minutes). Takes the cell of figure3_biot.py at K = 10 Pa and p = 30 Pa through its ramp and hold, for
each boundary condition, and after every accepted step measures the fraction of the solid's volume with J_s < THRESHOLD
and the smallest J_s at its vertices. The fixed cell has no equilibrium here, so its record is the last
accepted step before its solver fails. Saves OUTPUT, read by examples/paper_numbers.py.

THRESHOLD is where the hydrostatic stress of the solid law stops increasing under compression at this K and G.
"""

import contextlib
import io
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import figure3_biot as biot  # noqa: E402
from geometry import square_geo  # noqa: E402
from homogenisation_microscale import FSISession  # noqa: E402
from ngsolve import CoefficientFunction, Det, Grad, Id, IfPos, Integrate, SetNumThreads, TaskManager  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
K = 10.0                          # solid bulk modulus (Pa)
PRESSURE = 30.0                   # pore pressure (Pa)
THRESHOLD = 0.5                   # J_s of the fraction reported
OUTPUT = ROOT / "results" / "paper" / "figure3_solid_compression.json"
QUADRATURE = 8
# ---------------------------------------------------------------------------


def compression(result):
    """(volume fraction of the solid with J_s < THRESHOLD, smallest J_s sampled at the solid's vertices). The cell is held undeformed (F = I), so the map to the current configuration is X + w."""
    mesh, w = result["mesh"], result["mesh_displacement"]
    mesh.UnsetDeformation()
    try:
        J = Det(Id(3) + Grad(w))
        solid = mesh.Materials("solid")
        volume = Integrate(CoefficientFunction(1.0), mesh, definedon=solid, order=QUADRATURE)
        below = Integrate(IfPos(THRESHOLD - J, 1.0, 0.0), mesh, definedon=solid, order=QUADRATURE)
        solid_vertices = sorted({v.nr for e in mesh.Elements() if e.mat == "solid" for v in e.vertices})
        X = np.clip(np.array([mesh.vertices[n].point for n in solid_vertices]), 1e-9, 1 - 1e-9)
        return below / volume, float(np.min(J(mesh(*X.T))))
    finally:
        mesh.SetDeformation(w)


def run(solver, boundary):
    dt = biot.DURATION / biot.STEPS
    records = []

    def accept(r, stage, time):
        fraction, smallest = compression(r)
        mean = float(np.trace(r["homogenised_response"]["P_total"]) / 3)
        records.append(dict(stage=stage, time=time, fraction_below=fraction, smallest_J=smallest, alpha=-mean / PRESSURE))
        print(f"{boundary:8s} {stage:5s} t = {time:6.2f} s: fraction below {THRESHOLD:g}: {100 * fraction:.2f} %, "
              f"smallest J_s {smallest:.3f}, alpha {-mean / PRESSURE:.4f}", flush=True)

    def solve(**load):
        with contextlib.redirect_stdout(io.StringIO()):
            r = solver(K=K, **load)
        if any(not h[-1].get("converged", False) for h in r["newton_histories"]):
            raise RuntimeError("unconverged")
        return r

    r = solve(dt=dt, t_final=biot.DURATION, P_target=PRESSURE)
    accept(r, "ramp", biot.DURATION)
    time, stable, previous = biot.DURATION, 0, records[-1]["alpha"]
    try:
        for _ in range(biot.HOLD_MAX_STEPS):
            r = solve(dt=dt, t_final=dt, P_target=PRESSURE, initial_state=r["restart_state"])
            time += dt
            accept(r, "hold", time)
            alpha = records[-1]["alpha"]
            stable = stable + 1 if abs(alpha - previous) <= biot.HOLD_TOLERANCE * abs(alpha) else 0
            previous = alpha
            if stable >= biot.HOLD_STABLE:
                return dict(status="equilibrated", records=records)
    except RuntimeError as error:
        return dict(status=f"failed in the hold: {type(error).__name__}", records=records)
    return dict(status="not stationary", records=records)


def main():
    if biot.COMPILE_TANGENT:
        biot.compiler_environment()
    SetNumThreads(biot.THREADS)
    cell = dict(G=biot.G, eta_s=0.0, zeta_s=0.0, geometry=partial(square_geo, biot.CHANNEL_WIDTH),
                rve_length=biot.RVE_LENGTH, maxh=biot.MAXH, order=biot.ORDER, quadrature_order=biot.QUADRATURE_ORDER,
                mesh_mu=biot.MESH_MU, mesh_lame_lambda=biot.MESH_LAME_LAMBDA, homogenise=True, print_iterations=False)
    results = {}
    with TaskManager():
        for boundary in biot.BOUNDARIES:
            with FSISession(**cell, mu_f=biot.MU_F, F_target=np.eye(3), G_target=np.zeros(3), exterior=boundary,
                            tangent_method=None, compile_tangent=biot.COMPILE_TANGENT) as solver:
                results[boundary] = run(solver, boundary)
            last = results[boundary]["records"][-1]
            print(f"{boundary}: {results[boundary]['status']}; last accepted state t = {last['time']:.2f} s, "
                  f"{100 * last['fraction_below']:.2f} % below {THRESHOLD:g}, smallest J_s {last['smallest_J']:.3f}", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(dict(K=K, G=biot.G, pressure=PRESSURE, threshold=THRESHOLD, boundaries=results), indent=1) + "\n")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
