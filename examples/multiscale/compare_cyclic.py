"""Cyclic consolidation: FE² with history, FE² with stationary cells, and the resolved column.

Press Run (or python examples/multiscale/compare_cyclic.py FOLDER [FOLDER ...]). Each run folder (written
by hpc/submit_cyclic.sh) holds fe2.npz (FE², cells carry their transient history), resolved.npz (the
resolved column on that run's time grid) and, for the stationary cells, either fe2_stationary.npz or the
path in the file `stationary` (one stationary run serves both skeletons: its response has no viscous part).

For each model, over the last cycle (the earlier ones are printed too): the settlement amplitude and mean,
its phase lag behind the traction (fundamental Fourier components), and the energy dissipated per cycle and
unit area, the area of the traction-settlement loop. Against the resolved column, over all steps: the
largest settlement difference relative to the resolved settlement amplitude, and the largest cell-pressure
differences over the traction for cells 1 to n-1 and the top cell (Table 4 conventions).

Writes, per folder, FOLDER/comparison/cyclic.pdf (settlement against time, the last-cycle loops, the
pressure of a mid-height cell) and FOLDER/comparison/cyclic.md; with several folders also OUTPUT/cyclic.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale", ROOT / "examples" / "multiscale"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_consolidation import fe2_cell_averages  # noqa: E402
from plot_consolidation_multiscale import column, load  # noqa: E402
from plot_paper_figures import FIGSIZE, LINEWIDTH, STYLE, colours  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
FOLDERS = sorted((ROOT / "results" / "cyclic").glob("*_P*")) if (ROOT / "results" / "cyclic").is_dir() else []
OUTPUT = ROOT / "results" / "cyclic"
MID_CELL = 4                      # cell (from the base, 0-based) whose pressure is drawn
CELL_START = 0.0                  # pressure statistics from this time (s); the cyclic load starts smoothly
# ---------------------------------------------------------------------------

MODELS = {"history": "FE², history", "stationary": "FE², stationary cells", "resolved": "Resolved"}


def fe2_history(path):
    history, meta = load(path)
    problem = column(meta)
    ell = meta["cell"]["rve_length"]
    cells = int(round(problem.height / ell))
    width = meta.get("channel_width", 0.33)
    p = np.array([fe2_cell_averages(problem, y, o, ell, cells, width)[0] for y, o in zip(history["y"], history["outputs"])])
    return dict(time=history["time"], settlement=history["settlement"], pressure=p), meta, cells


def read(folder):
    folder = Path(folder)
    runs = {}
    runs["history"], meta, cells = fe2_history(folder / "fe2.npz")
    stationary = folder / "fe2_stationary.npz"
    if not stationary.exists() and (folder / "stationary").exists():
        stationary = Path((folder / "stationary").read_text().strip())
        if not stationary.exists():  # an absolute path from another machine: the same run in this study folder
            stationary = folder.parent / stationary.parent.name / stationary.name
    if stationary.exists():
        runs["stationary"] = fe2_history(stationary)[0]
    runs = {k: v for k, v in runs.items() if len(v["time"]) > 1}  # a run that failed before its first step
    if (folder / "resolved.npz").exists():
        with np.load(folder / "resolved.npz", allow_pickle=False) as d:
            runs["resolved"] = dict(time=d["time"], settlement=d["settlement"], pressure=d["cell_pressure"])
    macro, cyclic = meta["macro"], meta.get("cyclic") or {}
    skeleton = "viscoelastic" if meta["cell"].get("eta_s") or meta["cell"].get("zeta_s") else "elastic"
    return dict(runs=runs, traction=macro["traction"], period=cyclic.get("period"), cycles=cyclic.get("cycles"),
                skeleton=skeleton, cells=cells, name=folder.name)


def load_at(case, t):
    return 0.5 * case["traction"] * (1 - np.cos(2 * np.pi * t / case["period"]))


def cycle_metrics(case, history, k):
    """Settlement amplitude, mean, phase lag (degrees) and dissipated energy per area (J/m^2) in cycle k."""
    P = case["period"]
    t, s = history["time"], history["settlement"]
    grid = np.linspace(k * P, (k + 1) * P, 401)
    if grid[-1] > t[-1] * (1 + 1e-9):
        return None
    settle = np.interp(grid, t, s)
    force = load_at(case, grid)
    phase = np.exp(-2j * np.pi * (grid[:-1] - k * P) / P)
    lag = np.angle(np.sum((force[:-1] - force.mean()) * phase)) - np.angle(np.sum((settle[:-1] - settle.mean()) * phase))
    energy = float(np.trapezoid(force, settle))  # loop area: traction work per unit area over the cycle
    return dict(amplitude=float(settle.max() - settle.min()), mean=float(settle.mean()),
                lag=float(np.degrees((lag + np.pi) % (2 * np.pi) - np.pi)), energy=energy)


def differences(case, model):
    """Largest differences from the resolved column, on the resolved times."""
    ref, other = case["runs"]["resolved"], case["runs"][model]
    t = ref["time"][(ref["time"] > CELL_START) & (ref["time"] <= other["time"][-1] * (1 + 1e-12))]
    if t.size == 0:
        return None
    amplitude = np.ptp(ref["settlement"])
    settle = np.max(np.abs(np.interp(t, other["time"], other["settlement"]) - np.interp(t, ref["time"], ref["settlement"])))
    p = lambda h: np.array([np.interp(t, h["time"], h["pressure"][:, c]) for c in range(case["cells"])]).T  # noqa: E731
    error = np.abs(p(other) - p(ref)) / case["traction"]
    return dict(settlement=float(settle / amplitude), lower=float(error[:, :-1].max()), top=float(error[:, -1].max()))


def figure(case, folder):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 2.2, FIGSIZE[1]), layout="constrained")
    palette = dict(zip(MODELS, colours(3)))
    style = dict(history=dict(linestyle="-"), stationary=dict(linestyle="--"), resolved=dict(linestyle="none", marker="o", markersize=3))
    P, n = case["period"], case["cycles"]
    for model, history in case["runs"].items():
        kw = dict(color=palette[model], label=MODELS[model], linewidth=LINEWIDTH, **style[model])
        axes[0].plot(history["time"] / P, 1e3 * history["settlement"], **kw)
        last = (history["time"] >= (n - 1) * P - 1e-12)
        axes[1].plot(load_at(case, history["time"][last]), 1e3 * history["settlement"][last], **kw)
        axes[2].plot(history["time"] / P, history["pressure"][:, MID_CELL] / case["traction"], **kw)
    axes[0].set(xlabel="t / P", ylabel="Settlement (mm)")
    axes[1].set(xlabel="Traction (Pa)", ylabel="Settlement (mm)", title=f"Cycle {n}")
    axes[2].set(xlabel="t / P", ylabel=f"Cell {MID_CELL + 1} pressure / traction")
    axes[0].legend(frameon=False)
    fig.suptitle(f"{case['skeleton']} skeleton, period {P:g} s")
    out = Path(folder) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf",):
        fig.savefig(out / f"cyclic.{extension}", dpi=200)
    plt.close(fig)


def report(case):
    lines = [f"## {case['name']}: {case['skeleton']} skeleton, period {case['period']:g} s, {case['cycles']} cycles", "",
             "| Model | Cycle | Amplitude (mm) | Mean (mm) | Phase lag (deg) | Energy per cycle (mJ/m^2) |", "|---|---|---|---|---|---|"]
    for model, history in case["runs"].items():
        for k in range(case["cycles"]):
            m = cycle_metrics(case, history, k)
            if m:
                lines.append(f"| {MODELS[model]} | {k + 1} | {1e3 * m['amplitude']:.4f} | {1e3 * m['mean']:.4f} | "
                             f"{m['lag']:.2f} | {1e3 * m['energy']:.4f} |")
    if "resolved" in case["runs"]:
        lines += ["", "| Model | Max settlement difference / resolved amplitude | Cells 1 to n-1, max |p - p_res| / traction | Top cell |",
                  "|---|---|---|---|"]
        for model in ("history", "stationary"):
            if model in case["runs"]:
                d = differences(case, model)
                if d:
                    lines.append(f"| {MODELS[model]} | {100 * d['settlement']:.2f} % | {d['lower']:.4f} | {d['top']:.4f} |")
    return "\n".join(lines) + "\n"


def main(folders):
    texts = []
    for folder in folders:
        case = read(folder)
        figure(case, folder)
        text = report(case)
        (Path(folder) / "comparison" / "cyclic.md").write_text(text)
        print(text)
        texts.append(text)
    if len(folders) > 1:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "cyclic.md").write_text("# Cyclic consolidation: history against stationary cells\n\n" + "\n".join(texts))
        print(f"Wrote {OUTPUT / 'cyclic.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", type=Path, help="run folders (default FOLDERS)")
    parser.add_argument("--output", type=Path, help=f"folder for the joint summary (default {OUTPUT})")
    args = parser.parse_args()
    OUTPUT = args.output or OUTPUT
    main(args.folders or FOLDERS)
