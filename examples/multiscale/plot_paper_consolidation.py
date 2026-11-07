"""Paper Figure 6, consolidation under a step load: FE² against resolved, elastic and viscoelastic.

Press Run. Reads the run folders RUNS (results/consolidation/<run>/ with fe2.npz and, if present,
resolved.npz; one per solid) and writes to OUTPUT, in each of FORMATS, one file per panel:

- figure_6_settlement: top settlement against time, every run;
- figure_6_pressure_profiles: cell pore pressure along the column at PROFILE_TIMES, for the runs in
  PROFILE_RUNS (one panel each, titled only if there are several; every run in RUNS if none of them
  is in PROFILE_RUNS yet).

Also writes the LaTeX of the paper figure, settlement (a) and profiles (b) side by side, to
OUTPUT/figure_6_consolidation.tex (LATEX_NAME): CAPTION below, filled from the runs' metadata and
the profile times actually plotted (paper_latex.py).

Both figures: FE² lines, resolved markers (MODEL_STYLE). Settlement: one colour per solid.
Profiles: one colour per time; resolved markers are cell averages at the cell centres, the FE² line
is the macroscale pressure field (PROFILE_FE2). Cell pressures elsewhere are compared like for like
(compare_consolidation.fe2_cell_averages). A resolved run in progress
is drawn up to its last step; a run without one is drawn FE² only.
"""

import argparse
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
from plot_consolidation_multiscale import column, load, pressure_profile  # noqa: E402
from plot_paper_figures import FIGSIZE, LINEWIDTH, MARKERSIZE, SOLID_COLOURS, STYLE, colours, finish, solid_colour  # noqa: E402
import paper_latex as tex  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
# Also works for a ramp load, e.g. the elastic 10 s ramp quoted in Section 5.3.1 (full_elastic_20261026_201037) with --suffix.
RUNS = [ROOT / "results" / "consolidation" / "full_elastic_step_20261026_201037",       # elastic, step load
        ROOT / "results" / "consolidation" / "full_viscoelastic_step_20261026_201037"]  # viscoelastic, step load
OUTPUT = ROOT / "results" / "paper" / "figures"
FORMATS = ["pdf"]
PROFILE_TIMES = [10.0, 50.0, 200.0, 800.0, 2000.0]  # (s): end of the ramp, then through the drainage
PROFILE_RUNS = ["elastic"]        # solids whose pressure profiles are drawn (one panel each)
TIME_RANGE = (0.1, None)          # log time-axis limits (s); None = the runs' end
# Step load (RAMP_TIME = 0 in the runs): these replace PROFILE_TIMES, PROFILE_RUNS and TIME_RANGE. The
# solids differ in the first ~0.1 s (elastic: cell pressure overshoot, fast undrained settlement; viscoelastic:
# solid viscosity delays both), then drain alike.
STEP_PROFILE_TIMES = PROFILE_TIMES  # the ramp figure's times
STEP_PROFILE_RUNS = ["viscoelastic"]
STEP_TIME_RANGE = (1e-6, None)
MARKERS_PER_DECADE = 5           # resolved settlement markers per decade of time
MODEL_STYLE = {"resolved": dict(linestyle="none", marker="o", markersize=MARKERSIZE, zorder=3),  # resolved: markers
               "FE²": dict(linestyle="-", linewidth=LINEWIDTH)}                       # FE²: lines (both figures)
PROFILE_FE2 = "field"           # FE² line: "field" = macro pressure p(Z), "cells" = cell averages joined
PROFILE_LEGEND = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0))  # profile legend (matplotlib legend
                                 # keywords; bbox_to_anchor in axes fraction, > 1 = outside the frame)

# LaTeX caption of the paper figure (settlement and profiles side by side): <<name>> is filled from the runs
# (see latex_figure).
CAPTION = (r"Consolidation under <<loading>>. (a)~Top settlement for the <<solids>><<ramp_mark>>. (b)~Pore pressure, "
           r"normalised by the traction, against height $Z$ for <<profile_solids>> at $t=<<times>>$. Lines: FE$^2$; "
           r"markers: resolved FSI, averaged over each cell's fluid volume and plotted at the cell centre.")
SUBCAPTIONS = ("Top settlement.", "Pore pressure profiles.")
LATEX_NAME = "figure_6_consolidation"   # OUTPUT/<LATEX_NAME>.tex
# Appended to every file name and LaTeX label (e.g. "_ramp": figure_6_consolidation_ramp.tex,
# fig:consolidation-ramp), so several versions can sit side by side; --suffix on the command line.
SUFFIX = ""
PANEL_WIDTHS = (0.46, 0.51)             # text-width fractions: the same height (the profiles are 1.1x wider)
# ---------------------------------------------------------------------------


def skeleton(metadata):
    cell = metadata["cell"]
    return "viscoelastic" if cell.get("eta_s") or cell.get("zeta_s") else "elastic"


def read(folder):
    """One run: FE² and (if present) resolved histories with cell pressures, on their own times."""
    fe2, meta = load(folder / "fe2.npz")
    problem = column(meta)
    ell = meta["cell"]["rve_length"]
    cells = int(round(problem.height / ell))
    width = meta.get("channel_width", 0.33)
    area = meta["macro"]["area"]
    run = dict(label=skeleton(meta), ramp=meta["macro"]["ramp_time"], traction=meta["macro"]["traction"],
               height=problem.height, ell=ell, cells=cells, eta_s=meta["cell"].get("eta_s", 0.0),
               zeta_s=meta["cell"].get("zeta_s", 0.0))
    p = np.array([fe2_cell_averages(problem, y, o, ell, cells, width)[0] for y, o in zip(fe2["y"], fe2["outputs"])])
    run["FE²"] = dict(time=fe2["time"], settlement=fe2["settlement"], drainage=fe2["cumulative_outflow"] / area,
                      pressure=p, y=fe2["y"], problem=problem)
    resolved = folder / "resolved.npz"
    if resolved.exists():
        with np.load(resolved, allow_pickle=False) as data:
            run["resolved"] = dict(time=data["time"], settlement=data["settlement"],
                                   drainage=data["cumulative_outflow"] / area, pressure=data["cell_pressure"])
    return run


def at_time(time, values, t):
    """values (one row per step) linearly interpolated to time t, so profiles are at exactly PROFILE_TIMES."""
    i = int(np.clip(np.searchsorted(time, t), 1, len(time) - 1))
    w = (t - time[i - 1]) / (time[i] - time[i - 1])
    return (1 - w) * np.asarray(values[i - 1]) + w * np.asarray(values[i])


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name += SUFFIX
    for extension in FORMATS:
        fig.savefig(OUTPUT / f"{name}.{extension}")
    print(f"  {OUTPUT / name}.{{{','.join(FORMATS)}}}")
    plt.close(fig)


def step_load(runs):
    return all(run["ramp"] <= 0 for run in runs)


def time_axis(ax, runs):
    end = TIME_RANGE[1] or max(run["FE²"]["time"][-1] for run in runs)
    if not step_load(runs):
        ax.axvline(runs[0]["ramp"], color="grey", linestyle=":", linewidth=1, label="End of ramp")
    ax.set_xscale("log")
    ax.set_xlim(TIME_RANGE[0], end)
    ax.set_xlabel("Time (s)")


def models(run):
    return [(name, style) for name, style in (("resolved", "-"), ("FE²", "--")) if name in run]


def main():
    global PROFILE_TIMES, PROFILE_RUNS, TIME_RANGE
    runs = [read(Path(folder)) for folder in RUNS]
    if step_load(runs):
        PROFILE_TIMES, PROFILE_RUNS, TIME_RANGE = STEP_PROFILE_TIMES, STEP_PROFILE_RUNS, STEP_TIME_RANGE
    plt.rcParams.update(STYLE)
    palette = ([solid_colour(run["label"]) for run in runs] if all(run["label"] in SOLID_COLOURS for run in runs)
               else colours(max(len(runs), 2)))   # each solid in its colour of Figures 4 and 8

    for key, scale, ylabel, name in (("settlement", 1e3, "Settlement (mm)", "figure_6_settlement"),):
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        handles, labels = [], []
        for colour, run in zip(palette, runs):
            for model, _ in models(run):
                h = run[model]
                if model == "resolved":  # markers evenly spaced in log time, on the resolved history
                    end = h["time"][-1]
                    at = np.geomspace(TIME_RANGE[0], end, max(2, int(MARKERS_PER_DECADE * np.log10(end / TIME_RANGE[0])) + 1))
                    if 0 < run["ramp"] <= end:  # and one at the end of the ramp, nearest log-spaced one replaced
                        at[np.argmin(np.abs(np.log(at / run["ramp"])))] = run["ramp"]
                    ax.plot(at, scale * np.interp(at, h["time"], h[key]), color=colour, **MODEL_STYLE[model])
                else:
                    ax.plot(h["time"][1:], scale * h[key][1:], color=colour, **MODEL_STYLE[model])
            if len(runs) > 1:
                handles.append(plt.Line2D([], [], color=colour, linewidth=LINEWIDTH))
                labels.append(f"{run['label'].capitalize()} skeleton")
        time_axis(ax, runs)
        for model in ("resolved", "FE²"):
            if any(model in run for run in runs):
                handles.append(plt.Line2D([], [], color="black", **MODEL_STYLE[model]))
                labels.append(model.capitalize() if model == "resolved" else model)
        if not step_load(runs):
            handles.append(plt.Line2D([], [], color="grey", linestyle=":", linewidth=1))
            labels.append("End of ramp")
        ax.set_ylabel(ylabel)
        finish(ax, handles, labels, loc="upper left")
        save(fig, name)

    profiled = [run for run in runs if run["label"] in PROFILE_RUNS]
    if not profiled:
        print(f"  NOTE: no run in RUNS has a solid in PROFILE_RUNS = {PROFILE_RUNS}; profiles drawn for every run")
        profiled = runs
    shown = {}
    fig, axes = plt.subplots(1, len(profiled), figsize=(FIGSIZE[0] * (0.6 + 0.5 * len(profiled)), FIGSIZE[1]),
                             sharey=True, layout="constrained", squeeze=False)
    for ax, run in zip(axes[0], profiled):
        centres = (np.arange(run["cells"]) + 0.5) * run["ell"]
        handles, labels = [], []
        for colour, s in zip(colours(len(PROFILE_TIMES)), PROFILE_TIMES):
            drawn = None
            for model, _ in models(run):
                h = run[model]
                if s > h["time"][-1] * (1 + 1e-9):
                    continue  # not reached yet (a run in progress)
                if model == "FE²" and PROFILE_FE2 == "field":
                    Z, p = pressure_profile(h["problem"], at_time(h["time"], h["y"], s))
                else:
                    Z, p = centres, at_time(h["time"], h["pressure"], s)
                ax.plot(p / run["traction"], Z, color=colour, **MODEL_STYLE[model])
                if drawn is None:
                    drawn = s
                    shown.setdefault(run["label"], []).append(float(drawn))
            if drawn is not None:
                handles.append(plt.Line2D([], [], color=colour, linewidth=LINEWIDTH))
                labels.append(f"t = {drawn:g} s" if drawn >= 0.1 else f"t = {1e3 * drawn:g} ms")
        for model, _ in models(run):
            handles.append(plt.Line2D([], [], color="black", **MODEL_STYLE[model]))
            labels.append(model.capitalize() if model == "resolved" else model)
        ax.set_xlabel("Pore pressure / traction")
        if len(profiled) > 1:
            ax.set_title(f"{run['label'].capitalize()} skeleton", fontsize=12)
        ax.set_ylim(0, run["height"])
        finish(ax, handles, labels, **PROFILE_LEGEND)
    axes[0][0].set_ylabel("Height Z (m)")
    save(fig, "figure_6_pressure_profiles")
    latex_figures(runs, profiled, shown)


def solids(runs):
    """'elastic and viscoelastic skeletons' or 'the viscoelastic skeleton'."""
    names = []
    for run in runs:
        names.append(run["label"])
    return f"the {names[0]} skeleton" if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] + " skeletons"


def latex_figures(runs, profiled, shown):
    run = runs[0]
    for r in runs:
        if "resolved" not in r:
            print(f"  NOTE: the {r['label']} run has no resolved history yet; the caption describes both models")
    times = sorted({t for r in profiled for t in shown.get(r["label"], [])})
    caption = tex.fill(CAPTION, cells=tex.words(run["cells"]), height=tex.number(run["height"]),
                       ell=tex.number(run["ell"]), solids=solids(runs), profile_solids=solids(profiled),
                       loading=(rf"a step traction of {tex.quantity(run['traction'], 'Pa')}" if step_load(runs)
                                else rf"a traction ramped linearly to {tex.quantity(run['traction'], 'Pa')} over {tex.quantity(run['ramp'], 's')}"),
                       ramp_mark="" if step_load(runs) else "; the dotted line marks the end of the ramp",
                       times=tex.listing(times, "s")[1:-1])
    label = SUFFIX.replace("_", "-")
    panels = [("figure_6_settlement" + SUFFIX, SUBCAPTIONS[0], "fig:consolidation-settlement" + label),
              ("figure_6_pressure_profiles" + SUFFIX, SUBCAPTIONS[1], "fig:consolidation-pressure" + label)]
    source = "examples/multiscale/plot_paper_consolidation.py from " + ", ".join(tex.relative(f, ROOT) for f in RUNS)
    tex.write(OUTPUT / f"{LATEX_NAME}{SUFFIX}.tex", tex.figure("fig:consolidation" + label, caption, panels, width=list(PANEL_WIDTHS)),
              source)


def command_line():
    global RUNS, OUTPUT, SUFFIX
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run folders, one per solid (default RUNS)")
    parser.add_argument("--output", type=Path, help=f"figure folder (default {OUTPUT})")
    parser.add_argument("--suffix", help="appended to every file name and label, e.g. _ramp (default SUFFIX)")
    args = parser.parse_args()
    RUNS = args.runs or RUNS
    OUTPUT = args.output or OUTPUT
    SUFFIX = SUFFIX if args.suffix is None else args.suffix


if __name__ == "__main__":
    command_line()
    main()
