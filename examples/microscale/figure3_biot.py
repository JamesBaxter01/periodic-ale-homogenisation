"""Paper Figure 3 data: secant Biot coefficient against solid bulk modulus, fixed and periodic boundaries.

Press Run. Saves results/paper/figure3_biot.npz; plot it with plot_paper_figures.py.

For every bulk modulus K and pore pressure p, an elastic cell held at F = I has
p ramped up over STEPS steps, then held until it is stationary: the secant Biot
coefficient alpha = -tr(P)/(3p) of the homogenised nominal stress P is taken
once it changes by less than HOLD_TOLERANCE (relative) in HOLD_STABLE
successive hold steps. This matches Sandstrom et al. (2016), who evaluate alpha
for the stationary problem; stopping at the end of the ramp underestimates it
(by 1.8% at K = 10 Pa, p = 30 Pa). The hold uses the ramp's step, grown by
HOLD_GROWTH per step. A case whose solve fails is rerun from rest with 2, then 4
(RETRIES) times finer steps, ramp and hold alike (hold tolerance per unit time
unchanged): the equilibrated elastic cell does not depend on the path. A case that
still fails is saved as NaN (refinement 0) and the sweep goes on.

Each boundary condition has one FSISession, which serves every (K, p) case (each
a new history at rest on the same cell), outputs only (tangent_method=None), with
compiled tangents (COMPILE_TANGENT): "periodic" is the cell used in FE², "fixed"
the same cell with exterior="fixed" (velocity and relative mesh velocity zero on
the exterior faces), the boundary conditions of Sandstrom et al.
"""

import contextlib
import io
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
G = 50.0                          # solid shear modulus (Pa)
BULK_MODULI = np.geomspace(10.0, 1.0e4, 20).tolist()  # solid bulk moduli K (Pa): 20 values, log-spaced 10 to 10^4
PRESSURES = [1.0, 10.0, 20.0, 30.0]  # pore pressures p (Pa)
MU_F = 1.0                        # fluid viscosity (Pa s); the solid is elastic
CHANNEL_WIDTH = 0.33
RVE_LENGTH = 1.0
MAXH, ORDER, QUADRATURE_ORDER = 0.09, 2, None
MESH_MU, MESH_LAME_LAMBDA = 0.01, 0.05

DURATION = 1.0                    # pressure ramp duration (s)
STEPS = 5                         # steps of the ramp
HOLD_TOLERANCE = 1e-6             # stationary when alpha changes by less than this (relative) ...
HOLD_STABLE = 2                   # ... in this many successive hold steps
HOLD_GROWTH = 1.0                 # each hold step this much longer than the last; 1 = the ramp's step
                                  # (2.0 made GMRES stall on the first longer step at K = 10 Pa, p = 30 Pa)
HOLD_MAX_STEPS = 40               # give up (error) after this many hold steps
RETRIES = 2                       # a failed case is rerun with 2**k times finer steps, k = 1 ... RETRIES
                                  # (26 Oct 2026: fixed boundary, K = 10 Pa, p = 30 Pa failed at STEPS = 5)
BOUNDARIES = ["fixed", "periodic"]

COMPILE_TANGENT = True            # native tangent assembly once compiled
THREADS = 4
# ---------------------------------------------------------------------------

# --smoke (run.py figure3 --smoke): a coarse functionality check, not paper data.
SMOKE = dict(BULK_MODULI=[50.0], PRESSURES=[1.0], STEPS=2, MAXH=0.7, COMPILE_TANGENT=False)


def equilibrated(solver, K, boundary, pressure, profile, refine=1):
    """Ramp to the pressure, then hold until alpha is stationary, in steps refine times finer than STEPS.

    Returns (mean stress, alpha at the end of the ramp, hold time, hold steps,
    largest solid velocity in the last step)."""
    dt = DURATION / (STEPS * refine)

    def solve(**load):
        with contextlib.redirect_stdout(io.StringIO()):
            r = solver(K=K, **load)
        if any(not h[-1].get("converged", False) for h in r["newton_histories"]):
            raise RuntimeError(f"Unconverged: {boundary}, K={K:g}, p={pressure:g}")
        if "profiling" in r:
            accumulate(profile, r["profiling"])
        return r

    r = solve(dt=dt, t_final=DURATION, P_target=pressure)
    mean = np.trace(r["homogenised_response"]["P_total"]) / 3
    alpha_ramp = -mean / pressure
    hold_time, stable = 0.0, 0
    for step in range(1, HOLD_MAX_STEPS * refine + 1):
        dt *= HOLD_GROWTH
        r = solve(dt=dt, t_final=dt, P_target=pressure, initial_state=r["restart_state"])
        hold_time += dt
        previous, mean = mean, np.trace(r["homogenised_response"]["P_total"]) / 3
        stable = stable + 1 if abs(mean - previous) <= HOLD_TOLERANCE / refine * abs(mean) else 0
        if stable >= HOLD_STABLE:
            state = r["restart_state"]
            velocity = float(np.abs(state["unknowns"][0]).max() * state["velocity_scale"])
            return mean, alpha_ramp, hold_time, step, velocity
    raise RuntimeError(f"Not stationary after {HOLD_MAX_STEPS} hold steps: {boundary}, "
                       f"K={K:g}, p={pressure:g}; increase HOLD_MAX_STEPS")


def main():
    if COMPILE_TANGENT:
        compiler_environment()
    SetNumThreads(THREADS)
    shape = (len(BOUNDARIES), len(PRESSURES), len(BULK_MODULI))
    mean, alpha_ramp, hold_time, hold_steps, velocity = (np.full(shape, np.nan) for _ in range(5))
    refinement = np.zeros(shape, dtype=int)  # the step refinement each case needed; 0: failed
    profile = {}
    started_all = time.perf_counter()
    total = len(BOUNDARIES) * len(PRESSURES) * len(BULK_MODULI)
    done = 0
    cell = dict(G=G, eta_s=0.0, zeta_s=0.0, geometry=partial(square_geo, CHANNEL_WIDTH), rve_length=RVE_LENGTH,
                maxh=MAXH, order=ORDER, quadrature_order=QUADRATURE_ORDER, mesh_mu=MESH_MU,
                mesh_lame_lambda=MESH_LAME_LAMBDA, homogenise=True, print_iterations=False)
    with TaskManager():
        for b, boundary in enumerate(BOUNDARIES):
            with FSISession(**cell, mu_f=MU_F, F_target=np.eye(3), G_target=np.zeros(3), exterior=boundary,
                            tangent_method=None, compile_tangent=COMPILE_TANGENT, Profiling=True) as solver:
                for k, modulus in enumerate(BULK_MODULI):
                    for j, pressure in enumerate(PRESSURES):
                        started = time.perf_counter()
                        for attempt in range(RETRIES + 1):
                            refine = 2 ** attempt
                            try:
                                (mean[b, j, k], alpha_ramp[b, j, k], hold_time[b, j, k], hold_steps[b, j, k],
                                 velocity[b, j, k]) = equilibrated(solver, float(modulus), boundary,
                                                                   float(pressure), profile, refine)
                            except RuntimeError as error:
                                more = f"rerunning with {2 * refine}x finer steps" if attempt < RETRIES else "saved as NaN"
                                print(f"{boundary:8s} K={modulus:9.3f} Pa, p={pressure:4g} Pa: failed at "
                                      f"{refine}x steps ({type(error).__name__}: {error}); {more}", flush=True)
                                continue
                            refinement[b, j, k] = refine
                            break
                        done += 1
                        if not refinement[b, j, k]:
                            continue
                        print(f"{boundary:8s} K={modulus:9.3f} Pa, p={pressure:4g} Pa: "
                              f"alpha={-mean[b, j, k] / pressure:.5f} (end of ramp {alpha_ramp[b, j, k]:.5f}), "
                              f"held {hold_time[b, j, k]:.1f} s in {int(hold_steps[b, j, k])} steps"
                              f"{f' ({refinement[b, j, k]}x finer steps)' if refinement[b, j, k] > 1 else ''}  "
                              f"({time.perf_counter() - started:.1f} s, {done}/{total})", flush=True)
    seconds = time.perf_counter() - started_all
    if profile:
        print("\n" + format_profile_table(profile, title="Profiling summary"))
    settings = dict(G=G, bulk_moduli=BULK_MODULI, pressures=PRESSURES, mu_f=MU_F, channel_width=CHANNEL_WIDTH,
                    rve_length=RVE_LENGTH, maxh=MAXH, order=ORDER, quadrature_order=QUADRATURE_ORDER,
                    mesh_mu=MESH_MU, mesh_lame_lambda=MESH_LAME_LAMBDA, duration=DURATION, steps=STEPS,
                    hold_tolerance=HOLD_TOLERANCE, hold_stable=HOLD_STABLE, hold_growth=HOLD_GROWTH,
                    hold_max_steps=HOLD_MAX_STEPS, retries=RETRIES, boundaries=BOUNDARIES,
                    compile_tangent=COMPILE_TANGENT,
                    threads=THREADS)
    pressures = np.asarray(PRESSURES)
    save_results("figure3_biot", settings, seconds, bulk_moduli=np.asarray(BULK_MODULI), pressures=pressures,
                 boundaries=np.array(BOUNDARIES), mean_stress=mean,
                 alpha_secant=-mean / pressures[None, :, None], alpha_end_of_ramp=alpha_ramp,
                 hold_time=hold_time, hold_steps=hold_steps, final_solid_velocity=velocity, refinement=refinement)


if __name__ == "__main__":
    command_line(globals(), SMOKE)
    main()
