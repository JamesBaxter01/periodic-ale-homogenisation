"""Paper Figure 4 data: compression pulse, flow and recovery under a background pressure gradient.

Press Run. Saves results/paper/figure4_cycles.npz; plot it with plot_paper_figures.py.
--rates runs the same pulse at the nine durations of the Section 5.2 rate sweep (RATES: viscoelastic
solid only, every step sampled) and saves results/paper/figure4_rates.npz instead; plot_paper_figures.py
prints its largest flux departures from the stationary cell.

For the elastic and the viscoelastic solid:
1. hold the cell at F = I under PRESSURE_GRADIENT, in steps of SETTLE_DT, until the
   flux, fluid content and mesh displacement change by less than SETTLING_TOLERANCE in
   STABLE_CHECKS successive checks (every CHECK_EVERY steps, at least MINIMUM_INITIAL_TIME);
then, for every pulse duration T, from that settled state, in steps of T / STEPS:
2. compress with a raised-cosine pulse 0 -> COMPRESSION -> 0 over STEPS steps;
3. hold for RECOVERY_FRACTION * STEPS more steps.
Flux, fluid content and nominal stress are sampled every SAMPLE_EVERY steps.
One FSISession per material serves all durations; outputs only (tangent_method=None);
compiled tangents. The flow settles once per solid at a fixed step (settling at T / STEPS needed thousands of steps at T = 0.2 s and
too few checks within INITIAL_HOLD_MAX_TIME at long T).
"""

import contextlib
import io
import sys
import time
from functools import partial

import numpy as np

from paper_common import accumulate, command_line, compiler_environment, save_results
from ngsolve import SetNumThreads, TaskManager
from geometry import square_geo
from homogenisation_microscale import FSISession, format_profile_table

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
K, G = 50.0, 50.0                 # solid bulk and shear moduli (Pa)
MU_F = 1.0                        # fluid viscosity (Pa s)
ETA_S, ZETA_S = 2.0, 1.0          # solid viscosities of the viscoelastic solid (Pa s)
CHANNEL_WIDTH = 0.33
RVE_LENGTH = 1.0
MAXH, ORDER, QUADRATURE_ORDER = 0.09, 2, None
MESH_MU, MESH_LAME_LAMBDA = 0.01, 0.05

DURATIONS = [2.0, 0.2]            # pulse durations T (s), as the verification and the cyclic study
COMPRESSION = 0.2                 # peak compression of the pulse
STEPS = 100                       # steps per pulse
RECOVERY_FRACTION = 0.5           # hold after the pulse, as a fraction of STEPS
SAMPLE_EVERY = 2
PRESSURE_GRADIENT = [-1.0, 0.0, 0.0]  # background spatial pressure gradient (Pa/m)

SETTLE_DT = 0.02                  # settling: step (s), the same for every duration
INITIAL_HOLD_MAX_TIME = 10.0      # settling: give up after this time (s)
MINIMUM_INITIAL_TIME = 1.0        # settling: hold at least this long (s)
CHECK_EVERY = 10                  # settling: check every this many steps
SETTLING_TOLERANCE = 1e-5         # settling: relative change of flux and content, absolute of mesh
STABLE_CHECKS = 3                 # settling: successive checks below the tolerance

SOLIDS = ["elastic", "viscoelastic"]  # the solids run, from MATERIALS below
NAME = "figure4_cycles"           # saved as results/paper/<NAME>.npz

COMPILE_TANGENT = True
THREADS = 4
# ---------------------------------------------------------------------------

# --smoke (run.py figure4 --smoke): a coarse functionality check, not paper data.
SMOKE = dict(DURATIONS=[0.2], STEPS=2, COMPRESSION=0.001, SAMPLE_EVERY=1, CHECK_EVERY=1,
             MINIMUM_INITIAL_TIME=0.1, STABLE_CHECKS=2, SETTLING_TOLERANCE=1e-3, MAXH=0.7,
             COMPILE_TANGENT=False)
# --rates: the rate sweep of Section 5.2, nine durations from 0.2 to 20 s, viscoelastic solid, every step.
RATES = dict(DURATIONS=np.geomspace(0.2, 20.0, 9).tolist(), SOLIDS=["viscoelastic"], SAMPLE_EVERY=1,
             NAME="figure4_rates")

MATERIALS = {"elastic": (0.0, 0.0), "viscoelastic": (ETA_S, ZETA_S)}
COLUMNS = ["material", "duration_index", "time", "compression", "Q1", "Q2", "Q3", "fluid_content",
           *[f"P{i}{j}" for i in range(1, 4) for j in range(1, 4)], "stage"]


def values(r):
    a = r["homogenised_response"]
    return np.asarray(a["flux"]).copy(), float(a["fluid_content"]), np.asarray(a["P_total"]).copy()


def advance(run, dt, F, state, observe, profile):
    with contextlib.redirect_stdout(io.StringIO()):
        r = run(dt=dt, t_final=dt, F_target=F, P_target=0.0, G_target=np.asarray(PRESSURE_GRADIENT),
                initial_state=state, homogenise=observe)
    accumulate(profile, r["profiling"])
    return r


def settle(run, rest, profile):
    """1. Settle the flow at F = I in steps of SETTLE_DT: (state, flux, fluid content, stress, hold time)."""
    dt = SETTLE_DT
    started = time.perf_counter()
    r = advance(run, dt, np.eye(3), rest, True, profile)
    state = r["restart_state"]
    previous_flux, previous_content, _ = values(r)
    previous_mesh = np.asarray(state["mesh_displacement"]).copy()
    stable, settled = 0, False
    maximum = int(np.ceil(INITIAL_HOLD_MAX_TIME / dt))
    for step in range(1, maximum + 1):
        observe = step % CHECK_EVERY == 0 or step == maximum
        r = advance(run, dt, np.eye(3), state, observe, profile)
        state = r["restart_state"]
        if observe:
            flux, content, P = values(r)
            mesh = np.asarray(state["mesh_displacement"])
            changes = (np.linalg.norm(flux - previous_flux)
                       / max(np.linalg.norm(flux), np.linalg.norm(previous_flux), 1e-12),
                       abs(content - previous_content) / max(abs(content), abs(previous_content), 1e-12),
                       np.max(abs(mesh - previous_mesh)))
            stable = stable + 1 if max(changes) < SETTLING_TOLERANCE else 0
            previous_flux, previous_content, previous_mesh = flux.copy(), content, mesh.copy()
            settled = (step + 1) * dt >= MINIMUM_INITIAL_TIME and stable >= STABLE_CHECKS
            print(f"    settling: t = {(step + 1) * dt:.4g} s, largest change {max(changes):.2e}", flush=True)
        if settled:
            break
    if not settled:
        raise RuntimeError("Initial flow did not settle; increase INITIAL_HOLD_MAX_TIME")
    if not np.isfinite(flux).all() or flux[0] <= 0:
        raise RuntimeError("Invalid initial flux")
    print(f"    settled after {(step + 1) * dt:.4g} s ({time.perf_counter() - started:.1f} s)", flush=True)
    return state, flux, content, P, (step + 1) * dt


def pulse(run, settled, m, j, duration, rows, profile):
    """2-3. Pulse and recovery hold from the settled state, in steps of duration / STEPS."""
    dt = duration / STEPS
    state, flux, content, P, _ = settled

    def record(t, compression, stage):
        rows.append([m, j, t, compression, *flux, content, *P.ravel(), stage])

    record(0.0, 0.0, 0)
    started = time.perf_counter()
    total = STEPS + int(round(RECOVERY_FRACTION * STEPS))
    for step in range(1, total + 1):
        compression = 0.5 * COMPRESSION * (1 - np.cos(2 * np.pi * step / STEPS)) if step <= STEPS else 0.0
        observe = step % SAMPLE_EVERY == 0 or step in (STEPS, total)
        r = advance(run, dt, np.diag([1.0, 1.0, 1.0 - compression]), state, observe, profile)
        state = r["restart_state"]
        if observe:
            flux, content, P = values(r)
            record(step * dt, compression, int(step > STEPS))
        per_step = (time.perf_counter() - started) / step
        line = (f"    pulse: step {step}/{total}, t = {step * dt:.4g} s, compression {100 * compression:.2f}%, "
                f"{per_step:.2f} s/step, about {per_step * (total - step):.0f} s left")
        if sys.stdout.isatty():
            print("\r" + line.ljust(110), end="", flush=True)
        elif step % 10 == 0 or step == total:
            print(line, flush=True)
    if sys.stdout.isatty():
        print("\r" + " " * 110 + "\r", end="")
    print(f"    pulse and recovery: {time.perf_counter() - started:.1f} s", flush=True)


def main():
    if COMPILE_TANGENT:
        compiler_environment()
    SetNumThreads(THREADS)
    rows, initial, holds, profile = [], [], [], {}
    started_all = time.perf_counter()
    options = dict(K=K, G=G, mu_f=MU_F, rve_length=RVE_LENGTH, maxh=MAXH, order=ORDER,
                   quadrature_order=QUADRATURE_ORDER, mesh_mu=MESH_MU, mesh_lame_lambda=MESH_LAME_LAMBDA,
                   geometry=partial(square_geo, CHANNEL_WIDTH), print_iterations=False, Profiling=True,
                   tangent_method=None, compile_tangent=COMPILE_TANGENT)
    solids = {name: MATERIALS[name] for name in SOLIDS}
    for m, (material, (eta, zeta)) in enumerate(solids.items()):
        with TaskManager(), FSISession(**options, eta_s=eta, zeta_s=zeta) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                rest = run(dt=SETTLE_DT, t_final=SETTLE_DT, F_target=np.eye(3), P_target=0.0,
                           G_target=np.zeros(3))["restart_state"]
            print(f"{material}: settling in steps of {SETTLE_DT:g} s", flush=True)
            settled = settle(run, rest, profile)
            for j, duration in enumerate(DURATIONS):
                print(f"{material}, T = {duration:g} s:", flush=True)
                initial.append([m, j, *settled[1]])
                holds.append([m, j, settled[4]])
                pulse(run, settled, m, j, duration, rows, profile)
    seconds = time.perf_counter() - started_all
    print("\n" + format_profile_table(profile, title="Profiling summary, all histories"))
    settings = dict(K=K, G=G, mu_f=MU_F, eta_s=ETA_S, zeta_s=ZETA_S, channel_width=CHANNEL_WIDTH,
                    rve_length=RVE_LENGTH, maxh=MAXH, order=ORDER, quadrature_order=QUADRATURE_ORDER,
                    mesh_mu=MESH_MU, mesh_lame_lambda=MESH_LAME_LAMBDA, durations=DURATIONS,
                    compression=COMPRESSION, steps=STEPS, recovery_fraction=RECOVERY_FRACTION,
                    sample_every=SAMPLE_EVERY, pressure_gradient=PRESSURE_GRADIENT,
                    settle_dt=SETTLE_DT, initial_hold_max_time=INITIAL_HOLD_MAX_TIME,
                    minimum_initial_time=MINIMUM_INITIAL_TIME,
                    check_every=CHECK_EVERY, settling_tolerance=SETTLING_TOLERANCE,
                    stable_checks=STABLE_CHECKS, solids=SOLIDS, compile_tangent=COMPILE_TANGENT, threads=THREADS)
    save_results(NAME, settings, seconds, rows=np.asarray(rows), initial_flux=np.asarray(initial),
                 initial_hold=np.asarray(holds), durations=np.asarray(DURATIONS), columns=np.array(COLUMNS),
                 materials=np.array(list(solids)))


if __name__ == "__main__":
    command_line(globals(), SMOKE, {"rates": ("the Section 5.2 rate sweep (RATES), saved as figure4_rates.npz", RATES)})
    main()
