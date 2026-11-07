"""Replay cells of an FE² consolidation run exactly and record their solid surface at every step (Figure 7 data).

Press Run, optionally with point names: python consolidation_cell_replay.py bottom
Reads the FE² history RUN/fe2.npz (consolidation_multiscale.py), picks the Gauss point nearest each of POINTS and
drives one cell per point through the point's saved inputs and time steps exactly as the run did (step 0 from rest at
the run's first dt, then one step per accepted macro step from the committed state). The run saves each point's
inputs, outputs and tangents but not the cell's fields; the replay rebuilds them.

Writes RUN/cell_replay_<name>.npz: time, dt, the point's inputs, the replayed outputs and the run's ("run_outputs"),
the largest difference between the two relative to each output's largest value ("replay_mismatch": the replay is the
run's cell to the cell's Newton tolerance), the solid surface (reference vertices "surface_vertices", triangles
"surface_triangles": channel walls and the solid's periodic faces) and the cell displacement w at those vertices at
every step ("surface_displacement", in cell lengths: affine plus fluctuation, relative to the cell centre).
plot_paper_cells.py draws Figure 7 from it. About an hour per point on a laptop (four threads).

--run FOLDER replays another run's fe2.npz (default RUN). --storage also records, at every step, the rate of the
storage moment K_rate (dK/dt, Section 2.5) and the flux Q0 from the cell's diagnostics ("storage_moment_rate",
"diagnostic_flux", m/s) and prints |K_rate|/|Q0| at STORAGE_TIME and its range over STORAGE_RANGE (the
storage-moment sentence of Section 5.3.1: python consolidation_cell_replay.py top --storage --run <elastic step run>).
"""

import contextlib
import io
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
RUN = ROOT / "results" / "consolidation" / "full_viscoelastic_step_20261026_201037"
POINTS = {"bottom": 0.0, "top": 1.0}  # name: height Z (m); the nearest Gauss point is replayed
DEFAULT = ["bottom"]                  # the points replayed without names on the command line (Figure 7)
THREADS = 4
STORAGE_TIME = 0.01               # --storage: the ratio |K_rate|/|Q0| is printed at this time (s) ...
STORAGE_RANGE = (0.1, 3.0)        # ... and its smallest and largest value over this interval (s)
# ---------------------------------------------------------------------------


def solid_surface(mesh):
    """Reference vertices and the solid's surface triangles: channel walls and the solid's periodic faces."""
    from ngsolve import BND

    X = np.array([v.point for v in mesh.vertices])
    triangles = np.array([[v.nr for v in e.vertices] for e in mesh.Elements(BND)
                          if e.mat == "interface" or e.mat.endswith("_solid")])
    return X, triangles


def replay(name, Z, storage=False):
    from ngsolve import TaskManager
    from geometry import square_geo
    from homogenisation_microscale import FSISession

    with np.load(RUN / "fe2.npz") as d:
        metadata = json.loads(str(d["metadata"]))
        times, dts = d["time"], d["dt"]
        point = int(np.argmin(np.abs(d["Z_points"] - Z)))
        inputs, saved = d["inputs"][:, point], d["outputs"][:, point]
        Z_point = float(d["Z_points"][point])
    settings = dict(metadata["cell"])
    settings.pop("compile_tangent", None)
    settings.update(geometry=partial(square_geo, metadata.get("channel_width", 0.33)), tangent_method=None)
    if storage:
        settings["diagnostics"] = True  # the cell's power and storage diagnostics, with K_rate and Q0
    print(f"{name}: point {point} at Z = {Z_point:.4f} m, {len(times) - 1} steps", flush=True)

    outputs, displacement, surface, moment, flux = [], [], None, [], []
    started = time.perf_counter()
    scale = np.maximum(np.max(np.abs(saved), axis=0), 1e-300)
    with TaskManager(), FSISession(**settings) as run:
        state = None
        for n in range(len(times)):
            dt = dts[1] if n == 0 else dts[n]  # step 0: the rest state, evaluated at the run's first dt
            row = inputs[n]
            with contextlib.redirect_stdout(io.StringIO()):
                r = run(F_target=row[:9].reshape(3, 3), P_target=float(row[9]), G_target=row[10:].copy(),
                        dt=float(dt), t_final=float(dt), initial_state=state, homogenise=True)
            state = r["restart_state"]
            mesh, w = r["mesh"], r["mesh_displacement"]
            mesh.UnsetDeformation()
            try:
                if surface is None:
                    surface = solid_surface(mesh)
                X = surface[0]
                values = w(mesh(*np.clip(X, 1e-9, 1 - 1e-9).T))
                displacement.append(np.asarray(values).reshape(len(X), 3).astype(np.float32))
            finally:
                mesh.SetDeformation(w)
            outputs.append(np.asarray(r["homogenised_response"]["outputs"], dtype=float))
            if storage:
                audit = r["homogenised_response"]["hill_mandel_diagnostics"]
                moment.append(np.asarray(audit["K_rate"], dtype=float).ravel())
                flux.append(np.asarray(audit["Q0"], dtype=float).ravel())
            if n % 25 == 0 or n == len(times) - 1:
                elapsed = time.perf_counter() - started
                so_far = np.max(np.abs(np.array(outputs) - saved[:n + 1]) / scale)
                print(f"  {name}: step {n}/{len(times) - 1}, t = {times[n]:.3g} s, mismatch so far {so_far:.1e}, "
                      f"about {elapsed / (n + 1) * (len(times) - 1 - n) / 60:.0f} min left", flush=True)

    outputs = np.array(outputs)
    mismatch = float(np.max(np.abs(outputs - saved) / scale))
    print(f"{name}: replayed outputs differ from the run's by at most {mismatch:.2e} (relative to each output's "
          f"largest value)", flush=True)
    X, triangles = surface
    info = dict(run=str(RUN.relative_to(ROOT) if RUN.is_relative_to(ROOT) else RUN), point=point, Z=Z_point, cell=settings | {"geometry": "square_geo"},
                note="replay of the FE² cell through its saved inputs and time steps")
    extra = {}
    if storage:
        extra = dict(storage_moment_rate=np.array(moment), diagnostic_flux=np.array(flux))
        with np.errstate(divide="ignore", invalid="ignore"):  # step 0 is the rest state: 0/0
            ratio = np.linalg.norm(extra["storage_moment_rate"], axis=1) / np.linalg.norm(extra["diagnostic_flux"], axis=1)
        inside = (times >= STORAGE_RANGE[0]) & (times <= STORAGE_RANGE[1])
        n = int(np.argmin(np.abs(times - STORAGE_TIME)))
        span = (f"{ratio[inside].min():.3f} to {ratio[inside].max():.3f}" if inside.any() else "no steps")
        print(f"{name}: |K_rate|/|Q0| = {ratio[n]:.3f} at t = {times[n]:.3g} s; {span} for "
              f"{STORAGE_RANGE[0]:g} <= t <= {STORAGE_RANGE[1]:g} s", flush=True)
    path = RUN / f"cell_replay_{name}.npz"
    np.savez_compressed(path, metadata=np.array(json.dumps(info, default=str)), time=times, dt=dts, inputs=inputs,
                        outputs=outputs, run_outputs=saved, replay_mismatch=mismatch, surface_vertices=X,
                        surface_triangles=triangles, surface_displacement=np.array(displacement), **extra)
    print(f"Saved {path} ({(time.perf_counter() - started) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    import argparse

    from ngsolve import SetNumThreads

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("points", nargs="*", help=f"points to replay, from {list(POINTS)} (default {DEFAULT})")
    parser.add_argument("--run", type=Path, help=f"FE² run folder with fe2.npz (default {RUN})")
    parser.add_argument("--storage", action="store_true", help="also record K_rate and Q0 (cell diagnostics)")
    parser.add_argument("--threads", type=int, default=THREADS, help=f"NGSolve threads (default {THREADS})")
    args = parser.parse_args()
    if set(args.points) - set(POINTS):
        parser.error(f"unknown points {sorted(set(args.points) - set(POINTS))}; choose from {list(POINTS)}")
    if args.run:
        RUN = args.run.resolve()
    SetNumThreads(args.threads)
    for name in args.points or DEFAULT:
        replay(name, POINTS[name], args.storage)
