"""Rerun a finished FE² consolidation run with another material model and compare it with the saved run.

    python examples/multiscale/replay_fe2.py results/consolidation/full_elastic_step_20261026_201037/fe2.npz
    python examples/multiscale/replay_fe2.py <fe2.npz> --model homogenisation_ml.material:SurrogateModel \
        --setting checkpoint=/path/to/rno.pt

Rebuilds the run's macro problem, load and time stepping from the history's metadata, runs it with
the chosen material model and compares settlement, pressures and outflow with the saved run (on the
saved times; a model that cuts steps runs on its own grid and is interpolated). Works for any
consolidation_multiscale.py run (ramp, step or cyclic; "fe" or "stationary_fe").

The default model, "replay" (multiscale/replay_model.py), answers every point with the stored outputs
and tangent of its step, linearised about the stored inputs, with no cell solves. Its rerun lands on
the stored solution to the macro Newton tolerance, which shows the saved inputs, outputs, tangents and
time grid are complete and consistent: the ceiling for a surrogate trained on them. Any other model
(a surrogate, given as "package.module:Class" with --setting key=value for its settings) is scored
against the same run.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import consolidation_multiscale as example  # noqa: E402
from multiscale.column import Column1D  # noqa: E402
from multiscale.driver import RunOptions, TimeStepOptions, run  # noqa: E402
from multiscale.interface import build_model  # noqa: E402
from multiscale.newton import NewtonOptions  # noqa: E402

COMPARED = ("settlement", "max_pressure", "base_pressure", "cumulative_outflow")


def problem_and_options(saved, output=None):
    """The saved run's Column1D problem and RunOptions, rebuilt from its metadata."""
    metadata = json.loads(str(saved["metadata"]))
    if metadata.get("example") != "consolidation_multiscale":
        raise ValueError("not a consolidation_multiscale.py history")
    macro, cyclic = metadata["macro"], metadata.get("cyclic")
    # The example's own load function, with this run's load.
    example.TRACTION, example.RAMP_TIME = macro["traction"], macro["ramp_time"]
    example.CYCLIC_PERIOD = cyclic["period"] if cyclic else None
    problem = Column1D(height=macro["height"], area=macro["area"], elements=macro["elements"],
                       traction=example.traction, stress_scale=macro["traction"], modulus=metadata["cell"]["G"],
                       displacement_order=macro["displacement_order"], pressure_order=macro["pressure_order"],
                       gauss_points=macro["gauss_points"], grading=macro["grading"])
    time_steps = dict(metadata["time_steps"], events=tuple(metadata["time_steps"]["events"]))
    options = RunOptions(t_max=metadata["t_max"], time_steps=TimeStepOptions(**time_steps),
                         newton=NewtonOptions(**metadata["newton"]), output=output, verbose=False,
                         metadata=metadata)
    return problem, options, metadata


def replay(history_file, output=None, model="replay", settings=None, quiet=False):
    """Rerun the saved history with ``model`` (build_model kind) up to its last saved time.

    Returns (history, comparison): comparison[key] = dict(max_error, max_saved, relative) for the
    COMPARED quantities on the saved times, plus the run's status. A run that fails partway is
    compared up to where it got.
    """
    say = (lambda *a: None) if quiet else print
    saved = np.load(history_file)
    problem, options, metadata = problem_and_options(saved, output)
    options.metadata = dict(metadata, model=model, model_settings={k: str(v) for k, v in (settings or {}).items()},
                            replayed=str(history_file))
    settings = dict(history=history_file) if model == "replay" and settings is None else dict(settings or {})
    t_end = saved["time"][-1]
    latest = {}

    def stop(history):
        latest["history"] = history
        return history["time"][-1] >= t_end * (1 - 1e-12)

    started = perf_counter()
    status = "completed"
    with build_model(model, problem.n_points, **settings) as material:
        try:
            history = run(problem, material, options, stop=stop)
        except Exception as error:  # a surrogate may fail partway: compare what it did
            history, status = latest.get("history"), f"failed: {error}"
            if history is None:
                raise
        offsets = np.array(getattr(material, "offsets", []))
    seconds = perf_counter() - started

    time = np.array(history["time"])
    n = len(time)
    covered = saved["time"] <= time[-1] * (1 + 1e-12)
    comparison = dict(status=status, seconds=seconds, steps=n - 1, saved_steps=len(saved["time"]) - 1,
                      t_reached=float(time[-1]), t_saved=float(t_end))
    say(f"{history_file}: {metadata['model']} run, {metadata.get('load')} load, {metadata['skeleton']} skeleton; "
        f"rerun with {model}")
    say(f"{status}: {n - 1} steps (saved {len(saved['time']) - 1}) to t = {time[-1]:.6g} s in {seconds:.1f} s "
        f"(original {metadata['wall_seconds'] / 3600:.1f} h), {sum(history['newton_iterations'])} Newton iterations "
        f"(original {saved['newton_iterations'][covered].sum()}), {sum(history['rejected'])} cut steps "
        f"(original {saved['rejected'][covered].sum()})")
    same_grid = n <= len(saved["time"]) and np.allclose(time, saved["time"][:n], rtol=1e-12, atol=0)
    if len(offsets):
        say(f"inputs at the committed evaluations: max |x - x_saved| = {offsets.max():.2e}")
    if same_grid:
        y, y0 = np.array(history["y"]), saved["y"][:n]
        for name, part in (("u", slice(0, problem.n_u)), ("p", slice(problem.n_u, None))):
            say(f"{name}: max |rerun - saved| = {np.max(np.abs(y[:, part] - y0[:, part])):.2e} "
                f"(max |saved| {np.max(np.abs(y0[:, part])):.2e})")
    else:
        say("time grid differs from the saved one: quantities interpolated onto the saved times")
    for key in COMPARED:
        a = np.interp(saved["time"][covered], time, np.array(history[key]))
        b = saved[key][covered]
        error, size = float(np.max(np.abs(a - b))), float(np.max(np.abs(saved[key])))
        comparison[key] = dict(max_error=error, max_saved=size, relative=error / size if size else float("nan"))
        say(f"{key}: max |rerun - saved| = {error:.2e} (max |saved| {size:.2e}, {100 * error / size:.3g} %)")
    return history, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("history", type=Path, nargs="+", help="fe2.npz (or fe2_stationary.npz) of finished runs")
    parser.add_argument("--output", type=Path, help="save the rerun history here (one history only)")
    parser.add_argument("--model", default="replay", help="material model (build_model kind; default replay)")
    parser.add_argument("--setting", action="append", default=[], metavar="KEY=VALUE",
                        help="a setting of --model (repeatable); values are strings")
    args = parser.parse_args()
    settings = dict(item.split("=", 1) for item in args.setting) if args.setting else None
    if args.output and len(args.history) > 1:
        parser.error("--output takes one history")
    for path in args.history:
        replay(path, args.output, args.model, settings)
        print()


if __name__ == "__main__":
    main()
