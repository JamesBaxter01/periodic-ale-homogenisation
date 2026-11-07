"""Plot a saved FE² consolidation history (consolidation_multiscale.py); no solver runs.

Press Run. Reads DATA and writes to OUTPUT, in each of FORMATS:

- consolidation_settlement: top settlement against time;
- consolidation_pressure: pore pressure at the sealed base and the largest nodal pressure;
- consolidation_profiles: pore pressure along the column at PROFILE_TIMES;
- consolidation_steps: time step size and macro Newton iterations of every step;
- consolidation_balance: fluid volume balance error (outflow against the change in content).

Also prints a summary of the run. Works on a run still in progress (the file
is rewritten after every accepted step).
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multiscale.column import Column1D, lagrange, lobatto_nodes  # noqa: E402
from plot_paper_figures import FIGSIZE, STYLE, colours, finish  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
SMOKE_TEST = True                 # plot the smoke-test run instead of the production run
DATA = ROOT / "results" / "multiscale" / "consolidation.npz"
SMOKE_DATA = ROOT / "results" / "multiscale_smoke" / "consolidation.npz"
OUTPUT = ROOT / "results" / "multiscale" / "figures"
SMOKE_OUTPUT = ROOT / "results" / "multiscale_smoke" / "figures"
FORMATS = ["pdf"]
PROFILE_TIMES = [1.0, 10.0, 50.0, 200.0, 800.0, 2500.0]  # (s) in the ramp, at its end and while draining
LOG_TIME = True                   # logarithmic time axes
TIME_RANGE = (0.1, None)          # time-axis limits (s); None = the run's first step / end
# ---------------------------------------------------------------------------


def load(path):
    with np.load(path, allow_pickle=False) as data:
        history = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    return history, metadata


def column(metadata):
    m = metadata["macro"]
    return Column1D(height=m["height"], area=m["area"], elements=m["elements"], traction=lambda t: 0.0,
                    stress_scale=m["traction"], modulus=metadata["cell"]["G"],
                    displacement_order=m["displacement_order"], pressure_order=m["pressure_order"],
                    gauss_points=m["gauss_points"], grading=m.get("grading", 1.0))


def pressure_profile(problem, y, samples=20):
    """Pore pressure along the column from the element shape functions (Z, p)."""
    xi = np.linspace(-1, 1, samples)
    N, _ = lagrange(lobatto_nodes(problem.kp), xi)
    Z, p = [], []
    for e in range(problem.elements):
        dofs = problem.n_u + e * problem.kp + np.arange(problem.kp + 1)
        Z.append(problem.edges[e] + (xi + 1) / 2 * problem.lengths[e])
        p.append(N @ y[dofs])
    return np.concatenate(Z), np.concatenate(p)


def new_figure():
    return plt.subplots(figsize=FIGSIZE, layout="constrained")


def time_axis(ax, time):
    ax.set_xlabel("Time (s)")
    if LOG_TIME:
        ax.set_xscale("log")
        ax.set_xlim(TIME_RANGE[0] or time[1], TIME_RANGE[1] or time[-1])


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        path = folder / f"{name}.{extension}"
        fig.savefig(path)
        print(f"  {path}")
    plt.close(fig)


def main():
    data, folder = (SMOKE_DATA, SMOKE_OUTPUT) if SMOKE_TEST else (DATA, OUTPUT)
    h, metadata = load(data)
    problem = column(metadata)
    t = h["time"]
    ramp = metadata["macro"]["ramp_time"]
    traction = metadata["macro"]["traction"]

    print(f"{data}\n  status: {metadata['status']}, {len(t) - 1} steps to t = {t[-1]:.6g} s, "
          f"{int(h['cell_calls'][-1])} cell solves, {h['wall_seconds'][-1] / 60:.1f} min")
    print(f"  settlement {h['settlement'][-1]:.4e} m, base pressure {h['base_pressure'][-1]:.4e} Pa "
          f"(peak max pressure {h['max_pressure'].max():.4e} Pa), traction {traction:g} Pa")
    print(f"  dt from {h['dt'][1]:.3g} to {h['dt'][-1]:.3g} s, {int(h['rejected'].sum())} rejected or failed tries, "
          f"Newton iterations per step {int(h['newton_iterations'][1:].min())}-{int(h['newton_iterations'][1:].max())}, "
          f"max |volume balance error| {np.abs(h['mass_balance_error']).max():.2e} m^3")

    plt.rcParams.update(STYLE)
    c = colours(3)

    fig, ax = new_figure()
    ax.plot(t[1:], 1e3 * h["settlement"][1:], color=c[0], linewidth=2.5, label="FE²")
    ax.axvline(ramp, color="grey", linestyle=":", linewidth=1, label="end of ramp")
    ax.set_ylabel("Settlement (mm)")
    time_axis(ax, t)
    finish(ax)
    save(fig, folder, "consolidation_settlement")

    fig, ax = new_figure()
    ax.plot(t[1:], h["base_pressure"][1:] / traction, color=c[0], linewidth=2.5, label="sealed base")
    ax.plot(t[1:], h["max_pressure"][1:] / traction, color=c[2], linewidth=2.5, linestyle="--",
            label="largest nodal value")
    ax.axvline(ramp, color="grey", linestyle=":", linewidth=1, label="end of ramp")
    ax.set_ylabel("Pore pressure / traction")
    time_axis(ax, t)
    finish(ax, loc="upper left")
    save(fig, folder, "consolidation_pressure")

    fig, ax = new_figure()
    # The nearest accepted time to each requested one, skipping times the run has not reached.
    picks = [int(np.argmin(np.abs(t - s))) for s in PROFILE_TIMES if s <= t[-1] * 1.0001]
    for colour, i in zip(colours(len(picks)), picks):
        Z, p = pressure_profile(problem, h["y"][i])
        ax.plot(p / traction, Z, color=colour, linewidth=2.5, label=f"t = {t[i]:.3g} s")
    ax.set_xlabel("Pore pressure / traction")
    ax.set_ylabel("Height Z (m)")
    ax.set_ylim(0, problem.height)
    finish(ax, loc="upper left")
    ax.get_legend().set_bbox_to_anchor((1.02, 1.0))  # outside the axes: the curves fill the plot
    save(fig, folder, "consolidation_profiles")

    fig, ax = new_figure()
    ax.plot(t[1:], h["dt"][1:], color=c[0], linewidth=2.5, marker="o", markersize=3, label="dt")
    ax.set_yscale("log")
    ax.set_ylabel("Time step (s)")
    time_axis(ax, t)
    right = ax.twinx()
    right.step(t[1:], h["newton_iterations"][1:], color=c[2], where="post", linewidth=1.5, label="Newton iterations")
    right.set_ylabel("Macro Newton iterations")
    right.set_ylim(0, max(4, h["newton_iterations"].max() + 1))
    lines = ax.get_lines() + right.get_lines()
    finish(ax, lines, [line.get_label() for line in lines])
    save(fig, folder, "consolidation_steps")

    fig, ax = new_figure()
    ax.plot(t[1:], np.abs(h["mass_balance_error"][1:]) / h["fluid_volume"][0], color=c[0], linewidth=2.5,
            label="|content change + outflow| / initial content")
    ax.set_yscale("log")
    ax.set_ylabel("Relative volume balance error")
    time_axis(ax, t)
    finish(ax)
    save(fig, folder, "consolidation_balance")


if __name__ == "__main__":
    main()
