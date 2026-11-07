"""Fully resolved consolidation of the confined 1 m column: 10 square-channel cells, no homogenisation.

Press Run. Every setting, including the smoke test, is in the block below.
(Also runs as python run.py consolidation-resolved [--smoke] [--resume].)

The same column as consolidation_multiscale.py, resolved down to the pores:
CELLS cells of length RVE_LENGTH stacked to HEIGHT, periodic sideways, with an
impermeable symmetry plane at the bottom (normal velocity zero, sliding free:
exactly half of a column of 2 x CELLS cells drained at both ends, which is the
FE² base condition with no artificial layer). On top a rigid, porous,
frictionless platen carries
the traction (ramped over RAMP_TIME, then held) and lets the fluid drain. The
equations, fields, mesher and scaling are the periodic cell solver's
(resolved/column.py), so the comparison with FE² measures homogenisation.

Time steps: TIME_GRID = path of an FE² history (consolidation_multiscale.py
output) solves exactly its accepted times, the fairest comparison; None uses
the FE² example's schedule (DT_INITIAL growing by DT_GROWTH per step) and
stops when the pore pressure has drained to PRESSURE_TOLERANCE of its peak.

Saves OUTPUT, rewritten after every step, and OUTPUT's _checkpoint file.
RESUME = True continues a stopped run (e.g. after WALLTIME) from it.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale", ROOT / "verification"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry import square_column  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
# Material, as consolidation_multiscale.py and the paper figures (elastic solid).
K, G = 50.0, 50.0                 # solid bulk and shear moduli (Pa)
MU_F = 1.0                        # fluid viscosity (Pa s)
SKELETON = "elastic"              # "elastic" or "viscoelastic": the solid viscosities below
SKELETONS = dict(elastic=(0.0, 0.0), viscoelastic=(2.0, 1.0))   # (eta_s, zeta_s) in Pa s, as the paper figures
CHANNEL_WIDTH = 0.33
MAXH, ORDER = 0.09, 2             # per cell, as the FE² cells
MESH_MU, MESH_LAME_LAMBDA = 0.01, 0.05

# Column and load
HEIGHT = 1.0                      # column height (m)
CELLS = 10                        # cells stacked over the height
RVE_LENGTH = HEIGHT / CELLS       # cell length (m)
TRACTION = 5.0                    # compressive platen traction (Pa): about 4.5 % compression after the ramp, 8 % drained
RAMP_TIME = 10.0                  # traction ramp (s)

# Time steps
TIME_GRID = None                  # FE² history (.npz) whose times to solve, or None for the schedule below
TIME_REFINE = 1                   # with TIME_GRID: split each of its intervals into this many equal steps
TIME_GRID_STEPS = None            # with TIME_GRID: only its first this many steps (None: all; for smoke checks)
DT_INITIAL = 1e-4                 # first step (s)
DT_GROWTH = 1.05                  # each step 5 % longer than the last
DT_MAX = 500.0                    # largest step (s)
PRESSURE_TOLERANCE = 1e-3         # schedule only: stop when max |p| <= this fraction of its peak
T_MAX = 2.0e4                     # schedule only: give up after this time (s)
NEWTON_TOLERANCE = 1e-8           # residual relative to the step's unbalanced load

SNAPSHOTS = ()                    # times (s) at which to save the full fields
THREADS = 8                       # NGSolve threads
WALLTIME = None                   # seconds: stop cleanly (checkpointed) after the step that passes it
RESUME = False                    # continue OUTPUT from its checkpoint

OUTPUT = ROOT / "results" / "resolved" / "consolidation.npz"

# Smoke test: True runs the same column on a coarse mesh through the ramp with dt growing 1.5x
# per step to t = 20 s, as the FE² smoke test, so the two can be compared (about 1 minute).
# A functionality check, not data. TIME_GRID still applies.
SMOKE_TEST = False
SMOKE = dict(MAXH=0.4, T_MAX=20.0, DT_GROWTH=1.5)
SMOKE_OUTPUT = ROOT / "results" / "resolved_smoke" / "consolidation.npz"

# Step load: True applies the full TRACTION at t = 0+ instead of the ramp (as consolidation_multiscale.py
# --step; with TIME_GRID the FE² run's log-spaced times are solved, else the schedule starts at 1e-6 s).
STEP_LOAD = False
STEP = dict(RAMP_TIME=0.0, DT_INITIAL=1e-6)

# Cyclic load, as consolidation_multiscale.py: the raised-cosine traction TRACTION (1 - cos(2 pi t / P)) / 2 of
# period CYCLIC_PERIOD (s), from zero. Run it on the FE² run's time grid (TIME_GRID), which ends after its cycles.
CYCLIC_PERIOD = None
# ---------------------------------------------------------------------------


def traction(t):
    if CYCLIC_PERIOD:
        return 0.5 * TRACTION * (1.0 - np.cos(2.0 * np.pi * t / CYCLIC_PERIOD))
    return TRACTION * min(t / RAMP_TIME, 1.0) if RAMP_TIME > 0 else (TRACTION if t > 0 else 0.0)


def drained(history):
    """True after the ramp once the pore pressure is below PRESSURE_TOLERANCE of its peak."""
    pressure = np.asarray(history["max_pressure"])
    return history["time"][-1] > RAMP_TIME and pressure[-1] <= PRESSURE_TOLERANCE * pressure.max()


def command_line():
    """Optional overrides for run.py; pressing Run needs none."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="set SMOKE_TEST = True")
    mode.add_argument("--full", action="store_true", help="set SMOKE_TEST = False (whatever the file says)")
    parser.add_argument("--resume", action="store_true", help="set RESUME = True")
    parser.add_argument("--threads", type=int, help=f"NGSolve threads (default {THREADS})")
    parser.add_argument("--walltime", type=float, help="seconds before a clean, checkpointed stop")
    parser.add_argument("--time-grid", type=Path, help="FE² history whose times to solve")
    parser.add_argument("--time-refine", type=int, help=f"split each time-grid interval (default {TIME_REFINE})")
    parser.add_argument("--grid-steps", type=int, help="solve only the first N steps of the time grid")
    parser.add_argument("--maxh", type=float, help=f"mesh size per cell (default {MAXH})")
    parser.add_argument("--cyclic", type=float, metavar="PERIOD", help="set CYCLIC_PERIOD (s): cyclic traction")
    parser.add_argument("--snapshots", type=float, nargs="+", help="times (s) at which to save the full fields")
    parser.add_argument("--output", type=Path, help="history file")
    parser.add_argument("--skeleton", choices=sorted(SKELETONS), help=f"set SKELETON (default {SKELETON})")
    parser.add_argument("--step", action="store_true", help="set STEP_LOAD = True (step traction)")
    args = parser.parse_args()
    settings = globals()
    if args.skeleton:
        settings["SKELETON"] = args.skeleton
    if args.step:
        settings["STEP_LOAD"] = True
    if args.smoke or args.full:
        settings["SMOKE_TEST"] = args.smoke
    settings["RESUME"] = RESUME or args.resume
    for name, value in (("THREADS", args.threads), ("WALLTIME", args.walltime), ("TIME_GRID", args.time_grid),
                        ("TIME_REFINE", args.time_refine), ("TIME_GRID_STEPS", args.grid_steps), ("MAXH", args.maxh),
                        ("SNAPSHOTS", tuple(args.snapshots) if args.snapshots else None),
                        ("CYCLIC_PERIOD", args.cyclic)):
        if value is not None:
            settings[name] = value
    if args.output:
        settings["SMOKE_OUTPUT" if SMOKE_TEST or args.smoke else "OUTPUT"] = args.output


def main():
    global OUTPUT, RVE_LENGTH
    eta_s, zeta_s = SKELETONS[SKELETON]
    if SMOKE_TEST:
        globals().update(SMOKE)
        OUTPUT = SMOKE_OUTPUT
        RVE_LENGTH = HEIGHT / CELLS
    if STEP_LOAD:
        globals().update(STEP)
    if CYCLIC_PERIOD and TIME_GRID is None:
        raise SystemExit("a cyclic load needs TIME_GRID (--time-grid): the FE² run's times, which end after its cycles")
    from ngsolve import Mesh, SetNumThreads
    from periodic_fsi.inputs import _resolve_elastic_parameters
    from resolved.column import ResolvedColumn
    from resolved.run import ResolvedRunOptions, ScheduleOptions, run
    from state_io import solver_sha256

    SetNumThreads(THREADS)
    E, nu = _resolve_elastic_parameters(K=K, G=G)
    mesh_file = OUTPUT.with_name(OUTPUT.stem + "_mesh.vol.gz")
    if not RESUME and OUTPUT.exists():
        kept = OUTPUT.with_name(f"{OUTPUT.stem}_{time.strftime('%Y%m%d-%H%M%S', time.localtime(OUTPUT.stat().st_mtime))}")
        for path in (OUTPUT, OUTPUT.with_name(OUTPUT.stem + "_checkpoint.npz"), mesh_file):
            if path.exists():
                path.rename(path.with_name(path.name.replace(OUTPUT.stem, kept.name, 1)))
        print(f"Kept the previous run as {kept.name}*")
    times = None
    if TIME_GRID is not None:
        with np.load(TIME_GRID, allow_pickle=False) as fe2:
            times = fe2["time"][:None if TIME_GRID_STEPS is None else TIME_GRID_STEPS + 1].copy()
        if TIME_REFINE > 1:  # TIME_REFINE equal steps in every interval of the grid
            fractions = np.arange(1, TIME_REFINE + 1) / TIME_REFINE
            times = np.r_[times[0], (times[:-1, None] + np.diff(times)[:, None] * fractions).ravel()]
    settings = dict(K=K, G=G, mu_f=MU_F, eta_s=eta_s, zeta_s=zeta_s, channel_width=CHANNEL_WIDTH, maxh=MAXH,
                    order=ORDER, mesh_mu=MESH_MU, mesh_lame_lambda=MESH_LAME_LAMBDA, height=HEIGHT, cells=CELLS,
                    cell_length=RVE_LENGTH, traction=TRACTION, ramp_time=RAMP_TIME, load="cyclic" if CYCLIC_PERIOD else ("step" if STEP_LOAD else "ramp"), cyclic_period=CYCLIC_PERIOD,
                    time_grid=str(TIME_GRID), time_refine=TIME_REFINE, time_grid_steps=TIME_GRID_STEPS,
                    dt_initial=DT_INITIAL, dt_growth=DT_GROWTH, dt_max=DT_MAX, pressure_tolerance=PRESSURE_TOLERANCE)
    # TaskManager is held only inside the column's assembly methods (resolved/column.py, _parallel).
    mesh = None
    if RESUME:
        mesh = Mesh(str(mesh_file))
        mesh.Curve(ORDER)
    column = ResolvedColumn(square_column(CELLS, CHANNEL_WIDTH) if mesh is None else None, cells=CELLS, E=E,
                            nu=nu, mu_f=MU_F, eta_s=eta_s, zeta_s=zeta_s, maxh=MAXH, order=ORDER,
                            mesh_mu=MESH_MU, mesh_lame_lambda=MESH_LAME_LAMBDA, mesh=mesh)
    if not RESUME:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        column.mesh.ngmesh.Save(str(mesh_file))
    print(f"Resolved consolidation: {CELLS} cells, {column.mesh.ne} elements, {column.X.ndof} DOFs "
          f"({column.size} after the platen elimination), {THREADS} threads")
    options = ResolvedRunOptions(
        cell_length=RVE_LENGTH, times=times, output=OUTPUT, newton_tolerance=NEWTON_TOLERANCE,
        walltime=WALLTIME, resume=RESUME, snapshots=tuple(SNAPSHOTS),
        schedule=None if times is not None else ScheduleOptions(
            dt_initial=DT_INITIAL, growth=DT_GROWTH, dt_max=DT_MAX, t_max=T_MAX, events=(RAMP_TIME,)),
        metadata=dict(example="consolidation_resolved", smoke=SMOKE_TEST, settings=settings,
                      solver_sha256=solver_sha256(ROOT / "homogenisation_microscale.py"),
                      started=time.strftime("%Y-%m-%d %H:%M:%S")))
    column.solver_threads = THREADS  # MUMPS factorisation threads (the job pins BLAS/OpenMP to 1 otherwise)
    history = run(column, traction, options, stop=None if times is not None else drained)
    print(f"Saved {OUTPUT}")
    return history


if __name__ == "__main__":
    command_line()
    main()
