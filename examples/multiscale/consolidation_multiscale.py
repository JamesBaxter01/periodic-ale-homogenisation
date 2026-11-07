"""Multiscale (FE²) consolidation of a confined 1 m column, with a periodic FSI cell at every material point.

Press Run. Every setting, including the smoke test, is in the block below.
(Also runs as python run.py consolidation-fe2 [--smoke]; --smoke sets SMOKE_TEST.)

The column (height 1 m, cross-section one cell, 0.1 m x 0.1 m) is fixed and
sealed at the bottom and drained at the top, where a compressive traction is
ramped up over RAMP_TIME and then held. 10 macro elements graded towards the
drained top (GRADING), cubic displacement / quadratic pressure, 4 Gauss
points each: 40 cells of length 0.1 m. The run
stops once the pore pressure has drained to PRESSURE_TOLERANCE of its peak.
dt starts at DT_INITIAL (1e-4 s) and grows by at most DT_GROWTH (5 %) per
step, cut by the local error check (multiscale/driver.py) if needed: about
330 steps, 176 of them in the ramp. Material as in the paper figures.

Saves OUTPUT (an existing file is kept with a date suffix): the history of
every accepted step, rewritten after each one. SMOKE_TEST = True applies the
SMOKE settings instead (a coarse cell, faster dt growth, stops at 20 s; a
functionality check, not data) and saves to SMOKE_OUTPUT.
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale", ROOT / "verification"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry import square_geo  # noqa: E402
from multiscale.column import Column1D  # noqa: E402
from multiscale.driver import RunOptions, TimeStepOptions, run  # noqa: E402
from multiscale.interface import build_model  # noqa: E402
from multiscale.newton import NewtonOptions  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
# Material model at every point: "fe" (the periodic FSI cell set up below), "linear" or "finite_strain"
# (closed-form Biot laws), or "package.module:Class" for any other MaterialModel (e.g. a surrogate),
# built as Class(n_points, **MODEL_SETTINGS) (multiscale/interface.py).
# "stationary_fe" is the same cell without its transient history: each evaluation is the cell's
# equilibrated response to the current inputs (multiscale/stationary_cell.py).
MODEL = "fe"
MODEL_SETTINGS = {}               # keywords for a MODEL other than "fe" and "stationary_fe"
CELL_MODELS = ("fe", "stationary_fe")

# Material, as in the paper figures (Figures 4 and 6, elastic solid).
K, G = 50.0, 50.0                 # solid bulk and shear moduli (Pa)
MU_F = 1.0                        # fluid viscosity (Pa s)
SKELETON = "elastic"              # "elastic" or "viscoelastic": the solid viscosities below
SKELETONS = dict(elastic=(0.0, 0.0), viscoelastic=(2.0, 1.0))   # (eta_s, zeta_s) in Pa s, as the paper figures
CHANNEL_WIDTH = 0.33
MAXH, ORDER, QUADRATURE_ORDER = 0.09, 2, None
MESH_MU, MESH_LAME_LAMBDA = 0.01, 0.05
COMPILE_TANGENT = True
# Cell Newton: converged when residual <= max(tolerance, relative_tolerance * initial residual).
# The default absolute 1e-8 is met before the first correction for the tiny increments of the first
# steps (dt = 1e-4 s), leaving ~1e-3 relative error in the outputs; a relative target fixes it. The
# absolute floor must stay above round-off (~1e-14 for the cell, higher on Aire's OpenBLAS): at 1e-14
# the line search stalled there and every cell failed at t = 14 s of the Aire smoke, whatever the dt.
CELL_NEWTON = dict(tolerance=1e-12, relative_tolerance=1e-8)

# Macro problem
HEIGHT = 1.0                      # column height (m)
RVE_LENGTH = 0.1                  # cell length (m): the microstructure, 10 cells over the height
ELEMENTS = 10                     # macro elements
GRADING = 3.0                     # bottom element length / top element length (1 = uniform); towards the drain
GAUSS_POINTS = 4                  # material points (cells) per element
DISPLACEMENT_ORDER, PRESSURE_ORDER = 3, 2
TRACTION = 5.0                    # compressive traction at the top (Pa): about 4.5 % compression after the ramp, 8 % drained
RAMP_TIME = 10.0                  # traction ramp (s); H^2/c_v of the cell is about 1600 s

# Time stepping and stopping
DT_INITIAL = 1e-4                 # first step (s)
DT_GROWTH = 1.05                  # each step at most 5 % longer than the last
DT_MIN, DT_MAX = 1e-6, 500.0      # step bounds (s)
STEP_TOLERANCE = 2e-3             # local error per step, a safeguard: with 5 % growth it rarely binds
TIME_STEP_CONTROL = True          # False: no error control, every step DT_GROWTH times the last (log-spaced)
PRESSURE_TOLERANCE = 1e-3         # stop when max |p| <= this fraction of its peak
T_MAX = 2.0e4                     # give up after this time (s)
NEWTON_TOLERANCE = 1e-8           # scaled macro residual (load and its content change)

# Parallel cells: WORKERS processes (0 = all in this one), each with THREADS_PER_WORKER NGSolve
# threads; keep WORKERS * THREADS_PER_WORKER within the machine's cores.
WORKERS = 6
THREADS_PER_WORKER = 1

OUTPUT = ROOT / "results" / "multiscale" / "consolidation.npz"

# Smoke test: True runs a coarse cell with no compiled tangent through the ramp and 10 s of the
# hold, with dt growing up to 1.5x per step (about 30 steps, ~10 minutes with 6 workers).
SMOKE_TEST = False
SMOKE = dict(MAXH=0.4, COMPILE_TANGENT=False, T_MAX=20.0, DT_GROWTH=1.5)
SMOKE_OUTPUT = ROOT / "results" / "multiscale_smoke" / "consolidation.npz"

# Step load: True applies the full TRACTION at t = 0+ instead of the ramp, on a log-spaced grid from
# 1e-6 s (DT_GROWTH per step: 5 % gives ~47 steps per decade, ~400 steps to drain). The error control
# would hold dt at DT_MIN: the drained-boundary flux is singular at t = 0 (falls like 1/sqrt(t)).
STEP_LOAD = False
STEP = dict(RAMP_TIME=0.0, DT_INITIAL=1e-6, DT_MIN=1e-8, TIME_STEP_CONTROL=False)

# Cyclic load: a period CYCLIC_PERIOD (s) applies the raised-cosine traction TRACTION (1 - cos(2 pi t / P)) / 2,
# from zero, for CYCLES cycles on a uniform grid of STEPS_PER_CYCLE steps per cycle (no error control),
# stopping after the last cycle. None: the ramp or step load above.
CYCLIC_PERIOD = None
CYCLES = 5
STEPS_PER_CYCLE = 40
# ---------------------------------------------------------------------------


def traction(t):
    if CYCLIC_PERIOD:
        return 0.5 * TRACTION * (1.0 - np.cos(2.0 * np.pi * t / CYCLIC_PERIOD))
    return TRACTION * min(t / RAMP_TIME, 1.0) if RAMP_TIME > 0 else (TRACTION if t > 0 else 0.0)


def cyclic_settings():
    """Time-grid settings of the cyclic load: uniform steps, no error control, CYCLES periods."""
    dt = CYCLIC_PERIOD / STEPS_PER_CYCLE
    return dict(RAMP_TIME=0.0, DT_INITIAL=dt, DT_MIN=dt, DT_MAX=dt, DT_GROWTH=1.0, TIME_STEP_CONTROL=False,
                T_MAX=CYCLES * CYCLIC_PERIOD)


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
    parser.add_argument("--compile-tangent", action=argparse.BooleanOptionalAction,
                        help="override COMPILE_TANGENT (also in smoke mode); every worker compiles, a few GB each")
    parser.add_argument("--workers", type=int, help=f"cell processes (default {WORKERS}; 0 = this process)")
    parser.add_argument("--threads", type=int, help=f"NGSolve threads per worker (default {THREADS_PER_WORKER})")
    parser.add_argument("--output", type=Path, help="history file")
    parser.add_argument("--skeleton", choices=sorted(SKELETONS), help=f"set SKELETON (default {SKELETON})")
    parser.add_argument("--step", action="store_true", help="set STEP_LOAD = True (step traction, log-spaced dt)")
    parser.add_argument("--cyclic", type=float, metavar="PERIOD", help="set CYCLIC_PERIOD (s): cyclic traction")
    parser.add_argument("--cycles", type=int, help=f"set CYCLES (default {CYCLES})")
    parser.add_argument("--steps-per-cycle", type=int, help=f"set STEPS_PER_CYCLE (default {STEPS_PER_CYCLE})")
    parser.add_argument("--model", help=f"set MODEL (default {MODEL}; e.g. stationary_fe)")
    parser.add_argument("--walltime", type=float, metavar="SECONDS",
                        help="checkpoint to <output>_checkpoint/ and stop before this job's wall time, for a next job")
    parser.add_argument("--resume", action="store_true", help="continue from <output>_checkpoint/ if it exists")
    parser.add_argument("--cell-relative-tolerance", type=float, metavar="TOL",
                        help=f"set CELL_NEWTON's relative_tolerance (default {CELL_NEWTON['relative_tolerance']:g}); "
                             "tighter for small steps, where the macro storage term divides the cells' error by dt")
    parser.add_argument("--newton-tolerance", type=float, metavar="TOL",
                        help=f"set NEWTON_TOLERANCE, the scaled macro residual (default {NEWTON_TOLERANCE:g}); "
                             "looser for stationary cells at small dt, whose outputs carry a floor set by the "
                             "cells' tolerance divided by dt")
    args = parser.parse_args()
    settings = globals()
    for name, value in (("CYCLIC_PERIOD", args.cyclic), ("CYCLES", args.cycles), ("MODEL", args.model),
                        ("STEPS_PER_CYCLE", args.steps_per_cycle)):
        if value is not None:
            settings[name] = value
    if args.step:
        settings["STEP_LOAD"] = True
    if args.skeleton:
        settings["SKELETON"] = args.skeleton
    if args.smoke or args.full:
        settings["SMOKE_TEST"] = args.smoke
    if args.compile_tangent is not None:
        settings["COMPILE_OVERRIDE"] = args.compile_tangent
    if args.workers is not None:
        settings["WORKERS"] = args.workers
    if args.threads:
        settings["THREADS_PER_WORKER"] = args.threads
    if args.output:
        settings["OUTPUT" if not SMOKE_TEST else "SMOKE_OUTPUT"] = args.output
    settings["WALLTIME"], settings["RESUME"] = args.walltime, args.resume
    if args.cell_relative_tolerance is not None:
        settings["CELL_NEWTON"] = dict(CELL_NEWTON, relative_tolerance=args.cell_relative_tolerance)
    if args.newton_tolerance is not None:
        settings["NEWTON_TOLERANCE"] = args.newton_tolerance


COMPILE_OVERRIDE = None  # set by --compile-tangent / --no-compile-tangent
WALLTIME, RESUME = None, False  # set by --walltime / --resume: segments of one run in successive jobs


def main():
    global OUTPUT, COMPILE_TANGENT
    eta_s, zeta_s = SKELETONS[SKELETON]
    if SMOKE_TEST:
        globals().update(SMOKE)
        OUTPUT = SMOKE_OUTPUT
    if STEP_LOAD:
        globals().update(STEP)
    if CYCLIC_PERIOD:
        globals().update(cyclic_settings())
    if COMPILE_OVERRIDE is not None:
        COMPILE_TANGENT = COMPILE_OVERRIDE
    workers, threads = WORKERS, THREADS_PER_WORKER

    if MODEL in CELL_MODELS and COMPILE_TANGENT:
        from paper_common import compiler_environment

        compiler_environment()
    from state_io import solver_sha256

    checkpoint = OUTPUT.with_name(f"{OUTPUT.stem}_checkpoint") if WALLTIME or RESUME else None
    resuming = RESUME and checkpoint.exists()
    if OUTPUT.exists() and not resuming:
        kept = OUTPUT.with_name(f"{OUTPUT.stem}_{time.strftime('%Y%m%d-%H%M%S', time.localtime(OUTPUT.stat().st_mtime))}.npz")
        OUTPUT.rename(kept)
        print(f"Kept the previous {OUTPUT.name} as {kept.name}")

    problem = Column1D(height=HEIGHT, area=RVE_LENGTH**2, elements=ELEMENTS, traction=traction,
                       stress_scale=TRACTION, modulus=G, displacement_order=DISPLACEMENT_ORDER,
                       pressure_order=PRESSURE_ORDER, gauss_points=GAUSS_POINTS, grading=GRADING)
    cell = dict(K=K, G=G, mu_f=MU_F, eta_s=eta_s, zeta_s=zeta_s, geometry=partial(square_geo, CHANNEL_WIDTH),
                rve_length=RVE_LENGTH, maxh=MAXH, order=ORDER, quadrature_order=QUADRATURE_ORDER,
                mesh_mu=MESH_MU, mesh_lame_lambda=MESH_LAME_LAMBDA, compile_tangent=COMPILE_TANGENT,
                newton_options=CELL_NEWTON, print_iterations=False)
    recorded = {k: v for k, v in cell.items() if k != "geometry"}
    metadata = dict(example="consolidation_multiscale", smoke=SMOKE_TEST, skeleton=SKELETON, model=MODEL,
                    load="cyclic" if CYCLIC_PERIOD else ("step" if STEP_LOAD else "ramp"),
                    cyclic=dict(period=CYCLIC_PERIOD, cycles=CYCLES, steps_per_cycle=STEPS_PER_CYCLE) if CYCLIC_PERIOD else None,
                    model_settings=MODEL_SETTINGS, cell=recorded, channel_width=CHANNEL_WIDTH,
                    macro=dict(height=HEIGHT, area=RVE_LENGTH**2, elements=ELEMENTS, grading=GRADING,
                               gauss_points=GAUSS_POINTS,
                               displacement_order=DISPLACEMENT_ORDER, pressure_order=PRESSURE_ORDER,
                               traction=TRACTION, ramp_time=RAMP_TIME, pressure_tolerance=PRESSURE_TOLERANCE),
                    workers=workers, threads_per_worker=threads,
                    solver_sha256=solver_sha256(ROOT / "homogenisation_microscale.py"),
                    started=time.strftime("%Y-%m-%d %H:%M:%S"))
    options = RunOptions(t_max=T_MAX, output=OUTPUT, metadata=metadata,
                         newton=NewtonOptions(tolerance=NEWTON_TOLERANCE),
                         time_steps=TimeStepOptions(dt_initial=DT_INITIAL, dt_min=DT_MIN, dt_max=DT_MAX,
                                                    tolerance=STEP_TOLERANCE, growth=DT_GROWTH,
                                                    events=() if CYCLIC_PERIOD else (RAMP_TIME,),
                                                    controlled=TIME_STEP_CONTROL),
                         checkpoint=checkpoint, resume=resuming, wall_limit=WALLTIME)
    print(f"{MODEL} model, consolidation: {ELEMENTS} elements x {GAUSS_POINTS} cells = {problem.n_points} cells, "
          f"maxh {MAXH}, {workers} workers x {threads} threads")
    settings = dict(cell, workers=workers, threads_per_worker=threads) if MODEL in CELL_MODELS else dict(MODEL_SETTINGS)
    with build_model(MODEL, problem.n_points, **settings) as model:
        history = run(problem, model, options, stop=None if CYCLIC_PERIOD else drained)
        if getattr(model, "calls", 0):
            print(f"{model.calls} cell solves, {model.cell_seconds / model.calls:.2f} s each")
    print(f"Saved {OUTPUT}")
    return history


if __name__ == "__main__":
    command_line()
    main()
