"""Time loop: adaptive dt, macro Newton per step, commit on acceptance, append-only history.

The step size is chosen for accuracy with backward Euler's local error
estimate, which costs no extra solves: the accepted solution z_n is compared
with its linear extrapolation from the two previous accepted steps,

    err = dt_n / (dt_n + dt_(n-1)) * max |z_n - z_pred| / scale,

over the macro unknowns and all material-point outputs, each group scaled by
the problem's ``error_scales(rest_tangent)`` (the size the load gives it), or
by its largest change so far for groups the problem does not scale. Scaling
by the change so far alone fails at the start, when nothing has changed yet. A step with err > tolerance is redone with a
smaller dt; the next dt is dt * clip(safety * sqrt(tolerance / err), shrink,
growth). The estimate restarts at t = 0 and at every event (a kink in the
load, e.g. the end of a ramp): the first two steps after one are not
controlled. A failed step (Newton or a material point) is retried with
dt * shrink. The model is committed only for accepted steps.

With ``controlled=False`` the estimate is still recorded but never rejects a step or sets dt:
every accepted step is ``growth`` times longer than the last (up to dt_max), a log-spaced grid.
For a step load, whose response is singular at t = 0 (the drained-boundary flux falls like
1/sqrt(t)), the controller would hold dt at dt_min; a geometric grid resolves every decade alike.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from .interface import MaterialPointFailure
from .newton import MacroStepFailure, NewtonOptions, solve_step


@dataclass
class TimeStepOptions:
    dt_initial: float
    dt_min: float
    dt_max: float
    tolerance: float = 1e-2
    growth: float = 1.5
    shrink: float = 0.5
    safety: float = 0.9
    max_retries: int = 8  # consecutive failures or rejections of one step
    events: tuple = ()  # times every step must land on (load kinks)
    controlled: bool = True  # False: no error control, dt grows by ``growth`` every step (log-spaced grid)


@dataclass
class RunOptions:
    t_max: float
    time_steps: TimeStepOptions
    newton: NewtonOptions = field(default_factory=NewtonOptions)
    output: Path | None = None
    metadata: dict = field(default_factory=dict)
    verbose: bool = True
    checkpoint: Path | None = None  # folder for the committed state of the driver and every cell
    checkpoint_seconds: float = 1800.0  # wall time between checkpoints
    resume: bool = False  # continue from ``checkpoint`` if it exists
    wall_limit: float | None = None  # seconds of this job: checkpoint and stop before another step could overrun
    save_tangents: bool = True  # keep every accepted step's material tangents (n, 13, 13) in the history


class _ErrorEstimate:
    """Backward Euler local error from the last two accepted steps since the last event."""

    def __init__(self, problem, rest_tangent):
        self.problem = problem
        self.fixed = problem.error_scales(rest_tangent)
        self.points = []  # (t, y, outputs) since the last event, newest last
        self.output0 = None
        self.changes = {}

    def restart(self, t, y, outputs):
        self.points = [(t, y, outputs)]

    def _scales(self, y, outputs):
        p = self.problem
        groups = dict(stress=outputs[:, :9] - self.output0[:, :9], content=outputs[:, 9] - self.output0[:, 9],
                      flux=outputs[:, 10:], u=p.displacement(y), p=p.pressure(y))
        for name, values in groups.items():
            self.changes[name] = self.fixed.get(name) or max(self.changes.get(name, 0.0), float(np.max(np.abs(values))))
        return self.changes

    def estimate(self, t, y, outputs):
        """None while fewer than two steps since the last event, else the scaled error."""
        if len(self.points) < 2:
            self._scales(y, outputs)
            return None
        (t2, y2, o2), (t1, y1, o1) = self.points[-2:]
        dt, dt_prev = t - t1, t1 - t2
        extrapolate = lambda a, b: a + dt * (a - b) / dt_prev  # noqa: E731
        weight = dt / (dt + dt_prev)
        scales = self._scales(y, outputs)
        p = self.problem
        parts = dict(stress=(outputs[:, :9], extrapolate(o1[:, :9], o2[:, :9])),
                     content=(outputs[:, 9], extrapolate(o1[:, 9], o2[:, 9])),
                     flux=(outputs[:, 10:], extrapolate(o1[:, 10:], o2[:, 10:])),
                     u=(p.displacement(y), extrapolate(p.displacement(y1), p.displacement(y2))),
                     p=(p.pressure(y), extrapolate(p.pressure(y1), p.pressure(y2))))
        self.groups = {name: weight * float(np.max(np.abs(value - predicted))) / scales[name]
                       for name, (value, predicted) in parts.items() if scales[name] > 0}
        return max(self.groups.values(), default=0.0)

    def accept(self, t, y, outputs):
        self.points = (self.points + [(t, y, outputs)])[-2:]


def _save(path, history, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in history.items()}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, default=str)), **arrays)
    os.replace(temporary, path)


def _write_checkpoint(folder, model, state):
    """Write the cells' committed histories and the driver state to ``folder``, replacing it only
    once the new checkpoint is complete (``folder``.old survives a crash between the renames)."""
    folder = Path(folder)
    temporary, previous = folder.with_name(folder.name + ".tmp"), folder.with_name(folder.name + ".old")
    shutil.rmtree(temporary, ignore_errors=True)
    (temporary / "cells").mkdir(parents=True)
    model.save_checkpoint(temporary / "cells")
    with (temporary / "driver.pkl").open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.rmtree(previous, ignore_errors=True)
    if folder.exists():
        folder.rename(previous)
    temporary.rename(folder)
    shutil.rmtree(previous, ignore_errors=True)


def _read_checkpoint(folder, model):
    """The driver state of the newest complete checkpoint, after loading its cells into ``model``."""
    folder = Path(folder)
    if not (folder / "driver.pkl").exists():
        folder = folder.with_name(folder.name + ".old")
    if not (folder / "driver.pkl").exists():
        return None
    with (folder / "driver.pkl").open("rb") as stream:
        state = pickle.load(stream)
    model.load_checkpoint(folder / "cells")
    return state


def run(problem, model, options, stop=None):
    """Advance from rest until ``stop(history)`` is true or t_max; return the history dict.

    ``stop`` sees the history after every accepted step. History arrays have
    one row per accepted time, starting with the state at rest (t = 0).

    The history keeps every accepted step's inputs, outputs and (with
    ``save_tangents``) material tangents. With ``options.checkpoint`` and a model
    that supports it, the committed state (driver and every cell) is written there every ``checkpoint_seconds`` of wall time and when the run
    ends; ``resume=True`` continues from it (a missing checkpoint starts from
    rest). With ``wall_limit`` the run checkpoints and returns with status
    "wall limit" once another step might not finish in time, for the next job
    to resume. The limit is on this call's wall time, as each job of a chain has
    its own; the history's ``wall_seconds`` count from the run's start.
    """
    ts = options.time_steps
    log = (lambda *a: print(*a, flush=True)) if options.verbose else (lambda *a, **k: None)
    events = sorted(e for e in ts.events if 0 < e < options.t_max) + [options.t_max]
    metadata = dict(options.metadata, time_steps=asdict(ts), newton=asdict(options.newton), t_max=options.t_max,
                    status="running")
    started = job_started = perf_counter()  # started moves back by the time of earlier jobs on a resume

    # Checkpointing is a model capability (the FE cell has it); a model without it is not checkpointed.
    checkpointing = options.checkpoint is not None and hasattr(model, "save_checkpoint")
    if options.checkpoint is not None and not checkpointing:
        if options.resume:
            raise ValueError("resume needs a model with save_checkpoint / load_checkpoint")
        log("the material model has no checkpoint support: not checkpointing")
    saved = None
    if options.resume and checkpointing:
        saved = _read_checkpoint(options.checkpoint, model)
    if saved is not None:
        history, t, dt, y, old_content = (saved[k] for k in ("history", "t", "dt", "y", "old_content"))
        estimate = _ErrorEstimate.__new__(_ErrorEstimate)
        estimate.__dict__.update(saved["estimate"], problem=problem)
        # Each resume records the Newton settings it ran with, as a segment may change them.
        metadata["resumed"] = saved["metadata"].get("resumed", []) + [
            dict(t=t, steps=len(history["time"]) - 1, newton=asdict(options.newton))]
        started -= history["wall_seconds"][-1]
        log(f"resumed from {options.checkpoint} at t = {t:.6g} s after {len(history['time']) - 1} steps")
    else:
        # Rest state: one evaluation at the unloaded inputs gives the initial fluid content.
        y = problem.initial_state()
        rest = model.evaluate(problem.point_inputs(y), ts.dt_initial)
        model.commit()
        old_content = rest.outputs[:, 9].copy()
        R0 = problem.residual(y, rest.outputs, old_content, ts.dt_initial, 0.0)
        rest_norm = float(np.max(np.abs(R0[problem.free] / problem.row_scale(ts.dt_initial)[problem.free])))
        log(f"rest state: scaled residual {rest_norm:.2e}, fluid volume {problem.fluid_volume(rest.outputs):.6g} m^3")
        if rest_norm > options.newton.tolerance:
            raise RuntimeError("the unloaded cells are not in macro equilibrium")
        estimate = _ErrorEstimate(problem, rest.tangent)
        estimate.output0 = rest.outputs.copy()
        estimate.restart(0.0, y, rest.outputs)

        V0 = problem.fluid_volume(rest.outputs)
        history = dict(time=[0.0], dt=[0.0], y=[y.copy()], inputs=[problem.point_inputs(y)], outputs=[rest.outputs],
                       outflow_rate=[0.0], cumulative_outflow=[0.0], fluid_volume=[V0], mass_balance_error=[0.0],
                       newton_iterations=[0], scaled_residual=[rest_norm], error_estimate=[np.nan], rejected=[0],
                       cell_calls=[getattr(model, "calls", 0)], wall_seconds=[perf_counter() - started])
        if options.save_tangents:
            history["tangents"] = [rest.tangent]
        if "equilibrium" in rest.info:  # memoryless cells: the largest change over their last step (stress, content, flux)
            history["cell_equilibrium"] = [np.max(rest.info["equilibrium"], axis=0)]
        for key, value in problem.summary(y, R0).items():
            history[key] = [value]
        history.update(problem.point_coordinates())
        t, dt = 0.0, ts.dt_initial
    V0 = history["fluid_volume"][0]

    def checkpoint():
        if not checkpointing:
            return
        state = dict(history=history, t=t, dt=dt, y=y, old_content=old_content, metadata=metadata,
                     estimate={k: v for k, v in estimate.__dict__.items() if k != "problem"})
        _write_checkpoint(options.checkpoint, model, state)
        log(f"  checkpoint at t = {t:.6g} s")

    last_checkpoint = perf_counter()
    slowest_step = 0.0
    while t < options.t_max * (1 - 1e-12):
        if options.wall_limit is not None and perf_counter() - job_started + 1.5 * slowest_step > options.wall_limit:
            metadata["status"] = "wall limit"
            break
        step_started = perf_counter()
        event = next(e for e in events if e > t * (1 + 1e-12))
        failures = rejections = 0
        while True:
            step_dt = min(dt, ts.dt_max)
            if event - (t + step_dt) <= 1e-9 * step_dt or t + step_dt > event:
                step_dt = event - t  # land exactly on the event
            elif event - (t + step_dt) < 0.2 * step_dt:
                step_dt = 0.5 * (event - t)  # avoid a sliver step before the event
            t_new = t + step_dt
            log(f"t = {t_new:.6g} s (dt = {step_dt:.4g} s)")
            try:
                step = solve_step(problem, model, y, old_content, step_dt, t_new, problem.row_scale(step_dt),
                                  options.newton, log if options.verbose else None)
            except (MacroStepFailure, MaterialPointFailure) as error:
                failures += 1
                log(f"  step failed ({error}); retrying with dt = {step_dt * ts.shrink:.4g} s")
                if failures > ts.max_retries or step_dt * ts.shrink < ts.dt_min:
                    metadata["status"] = f"failed at t={t_new:g}: {error}"
                    if options.output:
                        _save(options.output, history, metadata)
                    raise
                dt = step_dt * ts.shrink
                continue
            error = estimate.estimate(t_new, step.y, step.outputs)
            if (ts.controlled and error is not None and error > ts.tolerance and step_dt > ts.dt_min
                    and rejections < ts.max_retries):
                rejections += 1
                dt = max(step_dt * max(ts.shrink, ts.safety * np.sqrt(ts.tolerance / error)), ts.dt_min)
                worst = max(estimate.groups, key=estimate.groups.get)
                log(f"  rejected: error estimate {error:.2e} > {ts.tolerance:g} (from {worst}); dt -> {dt:.4g} s")
                continue
            break

        model.commit()
        rate = problem.outflow_rate(step.residual)
        volume = problem.fluid_volume(step.outputs)
        cumulative = history["cumulative_outflow"][-1] + step_dt * rate
        row = dict(time=t_new, dt=step_dt, y=step.y.copy(), inputs=step.inputs, outputs=step.outputs,
                   outflow_rate=rate, cumulative_outflow=cumulative, fluid_volume=volume,
                   mass_balance_error=volume - V0 + cumulative, newton_iterations=len(step.norms) - 1,
                   scaled_residual=step.norms[-1], error_estimate=np.nan if error is None else error,
                   rejected=rejections + failures, cell_calls=getattr(model, "calls", 0),
                   wall_seconds=perf_counter() - started, **problem.summary(step.y, step.residual))
        if "tangents" in history:  # the converged step's d outputs / d inputs: training data for a surrogate
            row["tangents"] = step.tangents
        if "cell_equilibrium" in history:
            row["cell_equilibrium"] = np.max(step.info["equilibrium"], axis=0)
        for key, value in row.items():
            history[key].append(value)
        t, y, old_content = t_new, step.y, step.outputs[:, 9].copy()
        if abs(t - event) <= 1e-9 * max(event, 1.0):
            t = event
            estimate.restart(t, y, step.outputs)
        else:
            estimate.accept(t, y, step.outputs)
        if not ts.controlled:
            dt = step_dt * ts.growth
        elif error is not None:
            dt = step_dt * float(np.clip(ts.safety * np.sqrt(ts.tolerance / max(error, 1e-300)),
                                         ts.shrink, ts.growth))
        else:
            dt = step_dt
        dt = float(np.clip(dt, ts.dt_min, ts.dt_max))
        log(f"  accepted: {len(step.norms) - 1} Newton, error {row['error_estimate']:.2e}, "
            + ", ".join(f"{k} {v:.4g}" for k, v in problem.summary(y, step.residual).items()
                        if np.ndim(v) == 0)
            + f", cell calls {row['cell_calls']}, {row['wall_seconds']:.0f} s")
        if options.output:
            _save(options.output, history, metadata)
        slowest_step = max(slowest_step, perf_counter() - step_started)
        if stop is not None and stop(history):
            metadata["status"] = "stopped"
            break
        if perf_counter() - last_checkpoint >= options.checkpoint_seconds:
            checkpoint()
            last_checkpoint = perf_counter()
    else:
        metadata["status"] = "t_max"
    checkpoint()
    metadata["wall_seconds"] = perf_counter() - started
    if options.output:
        _save(options.output, history, metadata)
    log(f"finished ({metadata['status']}) at t = {t:.6g} s after {len(history['time']) - 1} steps, "
        f"{metadata['wall_seconds']:.0f} s")
    history["metadata"] = metadata
    return history
