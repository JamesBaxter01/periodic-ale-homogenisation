"""Time loop of the resolved column: the FE² schedule (or its exact time grid), checkpoints, resume.

Steps follow either an explicit grid of times (``times``, e.g. the accepted
times of an FE² run, so both solve the same time steps) or the FE² example's
schedule: dt starts at dt_initial and grows by ``growth`` per step up to
dt_max, landing exactly on every event. A failed step is retried with half the
interval (on a grid, the grid point is reached in two steps), at most
``max_halvings`` times in a row before the run stops with its checkpoint.

After every accepted step the history and a checkpoint (state, mesh
displacement, solid F, platen position) are written atomically; ``resume``
continues from the checkpoint with the saved mesh. ``walltime`` stops cleanly
after the step that passes it. ``snapshots`` saves the full fields (state, mesh displacement) at the
first accepted step at or after each requested time, as <output>_snapshot_<k>.npz, for drawing them.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from .solver import StepFailure, solve_step


@dataclass
class ScheduleOptions:
    dt_initial: float
    growth: float = 1.05
    dt_max: float = np.inf
    dt_min: float = 1e-8
    t_max: float = np.inf
    events: tuple = ()


@dataclass
class ResolvedRunOptions:
    cell_length: float
    schedule: ScheduleOptions | None = None
    times: np.ndarray | None = None  # explicit grid, starting at 0
    output: Path | None = None
    metadata: dict = field(default_factory=dict)
    newton_tolerance: float = 1e-8
    walltime: float | None = None  # seconds
    max_halvings: int = 10  # consecutive failed tries of one step before the run stops
    resume: bool = False
    verbose: bool = True
    snapshots: tuple = ()  # times (s) at which to save the full fields


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _history_arrays(history, metadata):
    arrays = {key: np.asarray(value) for key, value in history.items()}
    return dict(metadata=np.array(json.dumps(metadata, default=str)), **arrays)


def run(column, traction, options, stop=None):
    """Advance the column under the platen traction ``traction(t)`` (Pa, compressive positive)."""
    log = (lambda *a: print(*a, flush=True)) if options.verbose else (lambda *a: None)
    started = perf_counter()
    ell = options.cell_length
    output = Path(options.output) if options.output else None
    checkpoint = output.with_name(output.stem + "_checkpoint.npz") if output else None
    metadata = dict(options.metadata, schedule=asdict(options.schedule) if options.schedule else None,
                    explicit_times=options.times is not None, newton_tolerance=options.newton_tolerance,
                    unknowns=column.size, full_dofs=column.X.ndof, elements=column.mesh.ne, status="running")

    if options.resume:
        with np.load(checkpoint, allow_pickle=False) as saved:
            column.state.vec.FV().NumPy()[:] = saved["state"]
            column.displacement.vec.FV().NumPy()[:] = saved["displacement"]
            column.F_old.vec.FV().NumPy()[:] = saved["F_old"]
            column.platen_displacement = float(saved["platen_displacement"])
            y, t, dt_previous = saved["y"].copy(), float(saved["time"]), float(saved["dt"])
            history = {key[9:]: list(saved[key]) for key in saved.files if key.startswith("history__")}
        log(f"resumed at t = {t:.6g} s")
    else:
        y, t, dt_previous = np.zeros(column.size), 0.0, None
        rest = column.summary(ell)
        history = dict(time=[0.0], dt=[0.0], newton_iterations=[0], wall_seconds=[0.0], outflow_rate=[0.0],
                       cumulative_outflow=[0.0], **{k: [v] for k, v in rest.items()})
    V0 = history["fluid_volume"][0]

    def save(status=None):
        if status:
            metadata["status"] = status
        if output:
            _atomic_npz(output, **_history_arrays(history, metadata))
            _atomic_npz(checkpoint, state=column.state.vec.FV().NumPy(), displacement=column.displacement.vec.FV().NumPy(),
                        F_old=column.F_old.vec.FV().NumPy(), platen_displacement=column.platen_displacement, y=y,
                        time=t, dt=dt_previous if dt_previous is not None else 0.0,
                        **{"history__" + k: np.asarray(v) for k, v in history.items()})

    if options.times is not None:
        targets = [float(s) for s in np.asarray(options.times) if s > t * (1 + 1e-12) + 1e-300]
        next_dt = None
    else:
        sched = options.schedule
        if not np.isfinite(sched.t_max) and stop is None:
            raise ValueError("a schedule needs t_max or a stop criterion")
        events = sorted(e for e in sched.events if e > t) + ([sched.t_max] if np.isfinite(sched.t_max) else [])
        next_dt = sched.dt_initial if dt_previous is None else min(dt_previous * sched.growth, sched.dt_max)
    halvings = 0
    while True:
        if options.times is not None:
            if not targets:
                break
            target = targets[0]
        else:
            if t >= sched.t_max * (1 - 1e-12):
                break
            target = t + next_dt
            upcoming = [e for e in events if e > t * (1 + 1e-12)]
            if upcoming and target >= upcoming[0] * (1 - 1e-9):
                target = upcoming[0]
        dt = target - t
        if dt <= 0:
            raise RuntimeError("non-increasing time grid")
        log(f"t = {target:.6g} s (dt = {dt:.4g} s)")
        step_started = perf_counter()
        column.set_step(dt)
        column.begin_step()
        guess = y.copy()
        if dt_previous:
            ratio = dt / dt_previous  # velocities are increments per step: rescale with dt
            for block in ("u", "platen", "delta"):
                guess[column.blocks[block]] *= ratio
        try:
            y_new, norms = solve_step(column, guess, traction(target) / column.mu_s, tolerance=options.newton_tolerance,
                                      log=log if options.verbose else None)
            column.accept(y_new)
        except (StepFailure, RuntimeError) as error:
            column.set_reduced(y)
            halvings += 1
            if halvings > options.max_halvings or dt / 2 < (options.schedule.dt_min if options.schedule else 1e-12):
                save(f"failed at t={target:g} after {halvings} tries: {error}")
                raise
            log(f"  step failed ({error}); halving")
            if options.times is not None:
                targets.insert(0, t + dt / 2)
            else:
                next_dt = dt / 2
            continue
        y, t, dt_previous, halvings = y_new, target, dt, 0
        if options.times is not None:
            if abs(targets[0] - t) <= 1e-12 * max(t, 1.0):
                targets.pop(0)
        else:
            next_dt = min(dt * sched.growth, sched.dt_max)
        row = column.summary(ell)
        outflow = V0 - row["fluid_volume"]
        values = dict(time=t, dt=dt, newton_iterations=len(norms) - 1, wall_seconds=perf_counter() - started,
                      outflow_rate=(outflow - history["cumulative_outflow"][-1]) / dt, cumulative_outflow=outflow, **row)
        for key, value in values.items():
            history[key].append(value)
        log(f"  accepted: {len(norms) - 1} Newton, settlement {row['settlement']:.4e} m, max pressure "
            f"{row['max_pressure']:.4e} Pa, {perf_counter() - step_started:.1f} s")
        save()
        for k, wanted in enumerate(options.snapshots):
            path = output.with_name(f"{output.stem}_snapshot_{k:02d}.npz") if output else None
            if path is not None and t >= wanted * (1 - 1e-9) and not path.exists():
                _atomic_npz(path, time=t, requested=wanted, state=column.state.vec.FV().NumPy(),
                            displacement=column.displacement.vec.FV().NumPy(),
                            platen_displacement=column.platen_displacement, cell_length=ell)
                log(f"  snapshot {k} saved (t = {t:.6g} s)")
        if stop is not None and stop(history):
            save("stopped")
            break
        if options.walltime and perf_counter() - started > options.walltime:
            save("walltime")
            log("walltime reached: checkpoint saved; resume to continue")
            break
    if metadata["status"] == "running":
        save("complete")
    metadata["wall_seconds"] = perf_counter() - started
    log(f"finished ({metadata['status']}) at t = {t:.6g} s, {len(history['time']) - 1} steps, "
        f"{metadata['wall_seconds']:.0f} s")
    history["metadata"] = metadata
    return history
