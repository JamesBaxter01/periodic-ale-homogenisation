"""Compare the FE² and the fully resolved consolidation runs; no solver runs.

Press Run. Reads FE2 (consolidation_multiscale.py output) and RESOLVED
(consolidation_resolved.py output) and writes to OUTPUT, in each of FORMATS:

- comparison_settlement and comparison_outflow: platen settlement and the
  cumulative volume drained through the top against time;
- comparison_cell_pressure: the pore pressure of CELLS_SHOWN cells against time;
- comparison_profiles: cell pressures along the column at PROFILE_TIMES;
- comparison_metrics.json: the largest differences, relative to the resolved
  maximum of each quantity.

Per-cell values are compared like for like. Resolved: the fluid pressure
averaged over the current fluid volume of the cell, and the fluid volume per
reference cell volume. FE²: the macro pressure averaged over the cell's
interval [k l, (k+1) l] weighted by the fluid volume, i.e. the content m times
the cell's fluid area fraction against height (about 70 % of the fluid lies in
the middle third, in the horizontal channels), and m averaged over it (m
interpolated within each element through its Gauss-point values). Without the
area fraction the top cell, where the pressure profile is steep and curved,
reads several per cent lower than the resolved average for that reason alone.
Differences contain homogenisation error (largest near the drained top while
the drainage layer is thinner than a cell), the top cell's platen constraint
and discretisation error; this script reports them, it does not judge them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from numpy.polynomial import legendre  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multiscale.column import lagrange, lobatto_nodes  # noqa: E402
from plot_consolidation_multiscale import column, load  # noqa: E402
from plot_paper_figures import FIGSIZE, STYLE, colours, finish  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
FE2 = ROOT / "results" / "multiscale" / "consolidation.npz"
RESOLVED = ROOT / "results" / "resolved" / "consolidation.npz"
OUTPUT = ROOT / "results" / "comparison"
FORMATS = ["pdf"]
CELLS_SHOWN = None                # cell indices from the bottom, or None for bottom, middle and top
PROFILE_TIMES = [1.0, 10.0, 50.0, 200.0, 800.0, 2500.0]  # (s) in the ramp, at its end and while draining
LOG_TIME = True
TIME_RANGE = (0.1, None)          # time-axis limits (s); None = the runs' common start / end
# ---------------------------------------------------------------------------


def fluid_fraction_bands(channel_width):
    """Fluid area fraction of the square-channel cell against height, as (bands, fractions).

    Bands are fractions of the cell height. The vertical channel (area w^2) runs the full
    height; the two horizontal channels add 2w - 2w^2 in the middle band [(1-w)/2, (1+w)/2].
    """
    w = float(channel_width)
    a, b = (1 - w) / 2, (1 + w) / 2
    return [(0.0, a), (a, b), (b, 1.0)], [w * w, 2 * w - w * w, w * w]


def fe2_cell_averages(problem, y, outputs, cell_length, cells, channel_width, points=8):
    """Cell pressure and mean content of the FE² solution over each cell interval.

    The pressure is weighted as the resolved average is, by the fluid volume: the macro
    content m times the cell's fluid area fraction against height (most of the fluid is in
    the middle band). The content is m averaged over the interval.
    """
    xi_q, w_q = legendre.leggauss(points)
    gauss, _ = legendre.leggauss(len(problem.weights) // problem.elements)
    Np_nodes = lobatto_nodes(problem.kp)
    bands, fractions = fluid_fraction_bands(channel_width)
    pressure, content = np.zeros(cells), np.zeros(cells)
    for k in range(cells):
        weighted = fluid = mass = 0.0
        for (lo, hi), fraction in zip(bands, fractions):
            lower, upper = (k + lo) * cell_length, (k + hi) * cell_length
            for e in range(problem.elements):
                left, right = max(lower, problem.edges[e]), min(upper, problem.edges[e + 1])
                if right <= left:
                    continue
                Z = 0.5 * (left + right) + 0.5 * (right - left) * xi_q
                local = 2 * (Z - problem.edges[e]) / problem.lengths[e] - 1
                dofs = problem.n_u + e * problem.kp + np.arange(problem.kp + 1)
                p = lagrange(Np_nodes, local)[0] @ y[dofs]
                block = outputs[e * len(gauss):(e + 1) * len(gauss), 9]
                m = lagrange(gauss, local)[0] @ block  # through the element's Gauss-point contents
                weights = 0.5 * (right - left) * w_q
                weighted += fraction * (weights @ (m * p))
                fluid += fraction * (weights @ m)
                mass += weights @ m
        pressure[k] = weighted / fluid
        content[k] = mass / cell_length
    return pressure, content


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        fig.savefig(OUTPUT / f"{name}.{extension}")
    print(f"  {OUTPUT / name}.{{{','.join(FORMATS)}}}")
    plt.close(fig)


def main():
    fe2, fe2_meta = load(FE2)
    with np.load(RESOLVED, allow_pickle=False) as data:
        res = {key: data[key] for key in data.files if key != "metadata"}
        res_meta = json.loads(str(data["metadata"]))
    problem = column(fe2_meta)
    rs = res_meta["settings"]
    cells, ell = int(rs["cells"]), float(rs["cell_length"])
    channel_width = float(rs["channel_width"])
    checks = dict(height=(problem.height, rs["height"]), traction=(fe2_meta["macro"]["traction"], rs["traction"]),
                  channel_width=(fe2_meta.get("channel_width", channel_width), channel_width),
                  ramp_time=(fe2_meta["macro"]["ramp_time"], rs["ramp_time"]),
                  cell_length=(fe2_meta["cell"]["rve_length"], ell), K=(fe2_meta["cell"]["K"], rs["K"]),
                  G=(fe2_meta["cell"]["G"], rs["G"]), mu_f=(fe2_meta["cell"]["mu_f"], rs["mu_f"]))
    different = {k: v for k, v in checks.items() if not np.isclose(*v)}
    if different:
        raise ValueError(f"the runs are not the same physical problem: {different}")

    t_end = min(fe2["time"][-1], res["time"][-1])
    t = res["time"][(res["time"] > 0) & (res["time"] <= t_end)]
    on = lambda h, key: np.interp(t, h["time"], h[key])  # noqa: E731
    fe2_cells = np.array([fe2_cell_averages(problem, y, o, ell, cells, channel_width)
                          for y, o in zip(fe2["y"], fe2["outputs"])])
    fe2_p = np.array([np.interp(t, fe2["time"], fe2_cells[:, 0, k]) for k in range(cells)]).T
    res_p = np.array([np.interp(t, res["time"], res["cell_pressure"][:, k]) for k in range(cells)]).T

    metrics = dict(common_end_time=float(t_end), fe2=str(FE2), resolved=str(RESOLVED))
    for name, a, b in (("settlement", on(fe2, "settlement"), on(res, "settlement")),
                       ("cumulative_outflow", on(fe2, "cumulative_outflow"), on(res, "cumulative_outflow")),
                       ("cell_pressure", fe2_p, res_p)):
        scale = float(np.max(np.abs(b)))
        metrics[name] = dict(max_abs_difference=float(np.max(np.abs(a - b))),
                             relative_to_resolved_max=float(np.max(np.abs(a - b)) / scale) if scale else None)
    late = t >= 2 * fe2_meta["macro"]["ramp_time"]
    metrics["cell_pressure_after_twice_ramp"] = float(np.max(np.abs(fe2_p[late] - res_p[late])) /
                                                      np.max(np.abs(res_p))) if late.any() else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    plt.rcParams.update(STYLE)
    c = colours(3)

    def axis(ax):
        ax.set_xlabel("Time (s)")
        if LOG_TIME:
            ax.set_xscale("log")
            ax.set_xlim(TIME_RANGE[0] or t[t > 0][0], TIME_RANGE[1] or t[-1])

    for key, label, scale, name in (("settlement", "Settlement (mm)", 1e3, "comparison_settlement"),
                                    ("cumulative_outflow", "Drained volume (mm$^3$)", 1e9, "comparison_outflow")):
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        ax.plot(t, scale * on(res, key), color=c[0], linewidth=2.5, label="resolved")
        ax.plot(t, scale * on(fe2, key), color=c[2], linewidth=2.5, linestyle="--", label="FE²")
        ax.set_ylabel(label)
        axis(ax)
        finish(ax)
        save(fig, name)

    traction = rs["traction"]
    shown = CELLS_SHOWN if CELLS_SHOWN is not None else sorted({0, cells // 2, cells - 1})
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for colour, k in zip(colours(len(shown)), shown):
        ax.plot(t, res_p[:, k] / traction, color=colour, linewidth=2.5, label=f"cell {k + 1}, resolved")
        ax.plot(t, fe2_p[:, k] / traction, color=colour, linewidth=2.0, linestyle="--", label=f"cell {k + 1}, FE²")
    ax.set_ylabel("Cell pore pressure / traction")
    axis(ax)
    finish(ax, loc="upper left")
    save(fig, "comparison_cell_pressure")

    times = [s for s in PROFILE_TIMES if s <= t[-1] * 1.0001]  # skip times a run in progress has not reached
    centres = (np.arange(cells) + 0.5) * ell
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for colour, s in zip(colours(len(times)), times):
        i = int(np.argmin(np.abs(t - s)))
        ax.plot(res_p[i] / traction, centres, color=colour, linewidth=2.0, marker="o", markersize=4,
                label=f"t = {t[i]:.3g} s")
        ax.plot(fe2_p[i] / traction, centres, color=colour, linewidth=1.5, linestyle="--", marker="s", markersize=3)
    ax.set_xlabel("Cell pore pressure / traction  (resolved solid, FE² dashed)")
    ax.set_ylabel("Height Z (m)")
    ax.set_ylim(0, cells * ell)
    finish(ax, loc="lower left")
    save(fig, "comparison_profiles")


def command_line():
    """Optional overrides (used by the HPC comparison job); pressing Run needs none."""
    global FE2, RESOLVED, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fe2", type=Path)
    parser.add_argument("--resolved", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    FE2, RESOLVED, OUTPUT = args.fe2 or FE2, args.resolved or RESOLVED, args.output or OUTPUT


if __name__ == "__main__":
    command_line()
    main()
