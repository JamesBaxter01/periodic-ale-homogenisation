"""Paper figure of the cyclic consolidation study: FE² with history, FE² with stationary cells, resolved FSI.

Press Run (or python examples/multiscale/plot_paper_cyclic.py [STUDY]). Reads the study folder STUDY written by
hpc/submit_cyclic.sh: <skeleton>_P<T>/fe2.npz (FE², cells carry their transient history) and resolved.npz,
and stationary_P<T>/fe2_stationary.npz (FE² with stationary cells, one run per period: its response has no
viscous part, so it serves both skeletons). A stationary run is drawn only if it records its cells'
equilibrium (cell_equilibrium, written since the direct stationary cell); the earlier runs, whose cells
were not equilibrated, are skipped. Writes to OUTPUT, in each of FORMATS, one file per panel:

- figure_8_settlement_T<T>s: settlement of the top against t/T over every cycle;
- figure_8_loop_T<T>s: traction against settlement over the last cycle (the loop area is the work
  dissipated per cycle and unit area).

FE² lines, resolved markers, one colour per solid (as the consolidation figure); stationary cells grey.
Annotated as in the paper: the FE² transient-cell loops shaded (energy dissipated per cycle), and in the
settlement panels the phase lag of each skeleton's FE² settlement behind the last traction peak, as an arrow
labelled in degrees (Table 6). --plain (or ANNOTATE = False) draws the figure without them.
Also writes the LaTeX (OUTPUT/figure_8_cyclic.tex): CAPTION below, filled from the runs' metadata, and Table 6
(TABLES/table_6_cyclic.tex): the settlement amplitude, phase lag and energy per cycle of each model over the last cycle,
at the periods of STUDY (0.2 and 2 s, also in the figure) and of LONG_STUDY (20 and 200 s, the same column and
cells, submit_cyclic.sh --periods "20 200").
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

import paper_latex as tex  # noqa: E402
from compare_cyclic import cycle_metrics, fe2_history, load_at  # noqa: E402
from plot_paper_figures import (ANNOTATION_FONTSIZE, FIGSIZE, LINEWIDTH, MARKERSIZE, SHADE_ALPHA, STYLE,  # noqa: E402
                                finish, solid_colour)

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
STUDY = ROOT / "results" / "cyclic" / "20261026_201037"
OUTPUT = ROOT / "results" / "paper" / "figures"
TABLES = ROOT / "results" / "paper" / "tables"
FORMATS = ["pdf"]
ANNOTATE = True                      # shaded loops and lag arrows, as in the paper
LAG_ROWS = (0.07, 0.15)              # annotated: heights of the lag arrows above the highest peak, per skeleton, as a fraction of it
SKELETONS = ("elastic", "viscoelastic")
PERIODS = (2.0, 0.2)                 # one row each, as Figure 4
STATIONARY_STYLE = dict(color="0.25", linewidth=LINEWIDTH * 0.75, linestyle=(0, (4, 2)))  # stationary cells
MARKERS_PER_CYCLE = 10               # resolved markers on the settlement panels
LOOP_MARKERS = 20                    # resolved markers on the last-cycle loops
LEGEND_PANEL = "figure_8_settlement_T2s"
LEGEND_HEADROOM = 1.45            # its y range is stretched by this factor to keep the legend clear of the curves
LATEX_NAME = "figure_8_cyclic"
CAPTION = (r"Cyclic consolidation under the traction $<<half>>\,\mathrm{Pa}\,[1-\cos(2\pi t/T)]$ (<<rows>>). "
           r"<<settlement_panels>>~Top settlement; <<loop_panels>>~traction against settlement over the last cycle. "
           r"Lines: FE$^2$ with transient cells; markers: resolved FSI; one colour per skeleton.<<stationary>>"
           r"<<annotations>>")
ANNOTATION_CAPTION = (r" Shaded loops: energy dissipated per cycle; arrows in <<settlement_panels>>: phase lag of the "
                      r"FE$^2$ settlement behind the last traction peak (dotted).")
STATIONARY_CAPTION = r" Grey dashed: FE$^2$ with stationary cells."
TABLE_PERIODS = (0.2, 2.0, 20.0, 200.0)   # Table 6, in this order; each read from STUDY or LONG_STUDY
LONG_STUDY = ROOT / "results" / "cyclic" / "20261108_220130"   # the longer periods, T = 20 and 200 s
TABLE_CAPTION = (r"Last-cycle settlement amplitude, phase lag behind the traction and energy dissipated per cycle under "
                 r"cyclic consolidation (Figure~\ref{fig:cyclic} for $T=0.2$ and $2\,\mathrm s$). One stationary run "
                 r"serves both skeletons at each period, since stationary cells have no skeleton viscosity. At $T=20$ "
                 r"and $200\,\mathrm s$ the energy also includes work done on the continuing settlement.")
# ---------------------------------------------------------------------------

LABELS = ("Elastic", "Viscoelastic")


def stationary_run(study, period):
    """The stationary-cell FE² history for this period, or None (missing, or from before the direct cell)."""
    path = Path(study) / f"stationary_P{period:g}" / "fe2_stationary.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "cell_equilibrium" not in data.files:
            print(f"Skipped {path}: no cell_equilibrium record (a run from before the direct stationary cell)")
            return None
    run = fe2_history(path)[0]
    return run if len(run["time"]) > 1 else None


def read(study, periods=PERIODS):
    cases, meta = {}, None
    for period in periods:
        for skeleton in SKELETONS:
            folder = Path(study) / f"{skeleton}_P{period:g}"
            run, meta_run, _ = fe2_history(folder / "fe2.npz")
            with np.load(folder / "resolved.npz", allow_pickle=False) as d:
                resolved = dict(time=d["time"], settlement=d["settlement"])
            cases[skeleton, period] = dict(history=run, resolved=resolved)
            if skeleton == "viscoelastic":
                meta = meta_run
        cases["stationary", period] = stationary_run(study, period)
    return cases, meta


def save(fig, name):
    if name != LEGEND_PANEL and fig.axes[0].get_legend() is not None:
        fig.axes[0].get_legend().remove()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        fig.savefig(OUTPUT / f"{name}.{extension}")
        print(f"  {OUTPUT / f'{name}.{extension}'}")
    plt.close(fig)


def figures(cases, meta):
    cyclic = meta["cyclic"]
    traction, cycles = meta["macro"]["traction"], cyclic["cycles"]
    palette = [solid_colour(skeleton) for skeleton in SKELETONS]
    markers = ("o", "s")
    for T in PERIODS:
        case = dict(traction=traction, period=T)
        for kind in ("settlement", "loop"):
            fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
            handles = []
            stationary = cases["stationary", T]
            if stationary is not None:
                x, y = _xy(kind, stationary, case, cycles)
                # Dashed and on top: in the settlement panels it lies close to the transient-cell curves.
                ax.plot(x, y, **STATIONARY_STYLE, zorder=4)
                handles.append((plt.Line2D([], [], **STATIONARY_STYLE), "FE$^2$, stationary cells"))
            for i, skeleton in enumerate(SKELETONS):
                run = cases[skeleton, T]
                x, y = _xy(kind, run["history"], case, cycles)
                ax.plot(x, y, color=palette[i], linewidth=LINEWIDTH, zorder=2)
                if ANNOTATE and kind == "loop":  # the enclosed area: energy dissipated per cycle
                    ax.fill(x, y, color=palette[i], alpha=SHADE_ALPHA, linewidth=0, zorder=1)
                x, y = _xy(kind, run["resolved"], case, cycles, markers=True)
                ax.plot(x, y, color=palette[i], linestyle="none", marker=markers[i], markersize=MARKERSIZE,
                        markerfacecolor="white", markeredgewidth=1.5, zorder=3)
                handles.append((plt.Line2D([], [], color=palette[i], linewidth=LINEWIDTH), f"{LABELS[i]}, FE$^2$"))
                handles.append((plt.Line2D([], [], color=palette[i], linestyle="none", marker=markers[i],
                                           markersize=MARKERSIZE, markerfacecolor="white", markeredgewidth=1.5),
                                f"{LABELS[i]}, resolved"))
            if kind == "settlement":
                if ANNOTATE:
                    lag_arrows(ax, cases, case, cycles, palette)
                ax.set_xlabel(r"Normalised time $t/T$")
                ax.set_ylabel("Settlement (mm)")
                if f"figure_8_{kind}_T{T:g}s" == LEGEND_PANEL:  # headroom for the legend above the cycles
                    ax.set_ylim(top=LEGEND_HEADROOM * ax.get_ylim()[1])
            else:
                ax.set_xlabel("Settlement (mm)")
                ax.set_ylabel("Traction (Pa)")
            finish(ax, *zip(*handles), loc="upper center", ncol=2)
            save(fig, f"figure_8_{kind}_T{T:g}s")


def lag_arrows(ax, cases, case, cycles, palette):
    """The phase lag of each skeleton's FE² settlement behind the last traction peak (t/T = cycles - 1/2): a dotted
    line at the traction peak and, above the curves, an arrow from it to t/T + lag/360 labelled with the lag."""
    T, peak = case["period"], cycles - 0.5
    runs = [cases[k, T]["history"] for k in SKELETONS] + [cases["stationary", T]] * (cases["stationary", T] is not None)
    top = max(1e3 * np.max(run["settlement"]) for run in runs)
    ax.set_ylim(top=max(ax.get_ylim()[1], top * (1 + LAG_ROWS[-1] + 0.07)))
    ax.axvline(peak, color="0.6", linewidth=0.8, linestyle=":", zorder=0)
    for i, skeleton in enumerate(SKELETONS):
        lag = cycle_metrics(case, cases[skeleton, T]["history"], cycles - 1)["lag"]
        y, end = top * (1 + LAG_ROWS[i]), peak + lag / 360
        ax.annotate("", xy=(end, y), xytext=(peak, y), zorder=5,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", color=palette[i], linewidth=1.2,
                                    shrinkA=0, shrinkB=0))
        ax.text(end + 0.04, y, rf"${lag:.0f}^\circ$", color=palette[i], fontsize=ANNOTATION_FONTSIZE, ha="left",
                va="center")


def _xy(kind, run, case, cycles, markers=False):
    """Settlement (mm) against t/T over all cycles, or traction against settlement over the last cycle; with
    markers, resampled to evenly spaced times."""
    T = case["period"]
    t, s = run["time"], 1e3 * run["settlement"]
    if kind == "settlement":
        if markers:
            grid = np.linspace(0.0, min(cycles * T, t[-1]), cycles * MARKERS_PER_CYCLE + 1)
            return grid / T, np.interp(grid, t, s)
        return t / T, s
    start = (cycles - 1) * T
    if markers:
        grid = np.linspace(start, start + T, LOOP_MARKERS + 1)[:-1]
        grid = grid[grid <= t[-1]]
        return np.interp(grid, t, s), load_at(case, grid)
    last = t >= start - 1e-12
    return s[last], load_at(case, t[last])


def stationary_caption(cases):
    """STATIONARY_CAPTION if any period has a stationary run, naming the periods while some are still missing."""
    present = [T for T in PERIODS if cases["stationary", T] is not None]
    if not present:
        return ""
    if len(present) == len(PERIODS):
        return STATIONARY_CAPTION
    shown = " and ".join(tex.quantity(T, "s").replace("$", "$T=", 1) for T in present)
    return STATIONARY_CAPTION[:-1] + f" (shown for {shown})."


def latex(cases, meta):
    cell, cyclic = meta["cell"], meta["cyclic"]
    letters = "abcdefgh"
    rows = "; ".join(f"{('top', 'bottom')[j]}, {tex.quantity(T, 's').replace('$', '$T=', 1)}"
                     for j, T in enumerate(PERIODS))
    settlement_panels = "(" + ", ".join(letters[2 * j] for j in range(len(PERIODS))) + ")"
    annotations = ANNOTATION_CAPTION.replace("<<settlement_panels>>", settlement_panels) if ANNOTATE else ""
    caption = tex.fill(CAPTION, half=tex.number(meta["macro"]["traction"] / 2), cycles=tex.words(cyclic["cycles"]),
                       steps=cyclic["steps_per_cycle"], settlement_panels=settlement_panels, annotations=annotations,
                       loop_panels="(" + ", ".join(letters[2 * j + 1] for j in range(len(PERIODS))) + ")",
                       rows=rows, eta_s=tex.number(cell["eta_s"]), zeta_s=tex.number(cell["zeta_s"]),
                       stationary=stationary_caption(cases))
    panels = []
    for T in PERIODS:
        duration = tex.quantity(T, "s").replace("$", "$T=", 1)
        panels += [(f"figure_8_settlement_T{T:g}s", f"Settlement, {duration}.", f"fig:cyclic-settlement-T{T:g}"),
                   (f"figure_8_loop_T{T:g}s", f"Last cycle, {duration}.", f"fig:cyclic-loop-T{T:g}")]
    tex.write(OUTPUT / f"{LATEX_NAME}.tex", tex.figure("fig:cyclic", caption, panels, placement="tb", width=0.49),
              f"examples/multiscale/plot_paper_cyclic.py from {tex.relative(STUDY, ROOT)}")


def table_6(meta):
    """TABLES/table_6_cyclic.tex: each model over the last cycle, one row per skeleton and period, each period
    read from whichever of STUDY and LONG_STUDY holds it."""
    traction, cycles = meta["macro"]["traction"], meta["cyclic"]["cycles"]
    cases = {}
    for period in TABLE_PERIODS:
        study = next(s for s in (STUDY, LONG_STUDY) if (Path(s) / f"elastic_P{period:g}").is_dir())
        cases.update(read(study, (period,))[0])
    units = r"(mm) & ($^\circ$) & ($\mathrm{mJ\,m^{-2}}$)"
    lines = [r"\begin{table}[pos=tb]", rf"    \caption{{{TABLE_CAPTION}}}", r"    \label{tab:cyclic}", r"    \centering",
             r"    \small", r"    \renewcommand{\arraystretch}{1.0}",
             r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrrrrr@{}}", r"        \toprule",
             r"        & & \multicolumn{3}{c}{Resolved} & \multicolumn{3}{c}{FE$^2$, transient cells} & "
             r"\multicolumn{3}{c}{FE$^2$, stationary cells}\\",
             r"        \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
             r"        Skeleton & $T$ & Ampl. & Lag & Energy & Ampl. & Lag & Energy & Ampl. & Lag & Energy\\",
             rf"        & (s) & {units} & {units} & {units}\\", r"        \midrule"]
    for period in TABLE_PERIODS:
        for skeleton, label in zip(SKELETONS, LABELS):
            case = dict(period=period, traction=traction)
            cells = [f"{label} & {period:g}"]
            for history in (cases[skeleton, period]["resolved"], cases[skeleton, period]["history"], cases["stationary", period]):
                m = cycle_metrics(case, history, cycles - 1)
                cells.append(f"{1e3 * m['amplitude']:.2f} & {m['lag']:.2f} & {1e3 * m['energy']:.2f}")
            lines.append("        " + " & ".join(cells) + r"\\")
    lines += [r"        \bottomrule", r"    \end{tabular*}", r"\end{table}"]
    tex.write(TABLES / "table_6_cyclic.tex", "\n".join(lines) + "\n",
              f"examples/multiscale/plot_paper_cyclic.py from {tex.relative(STUDY, ROOT)} and "
              f"{tex.relative(LONG_STUDY, ROOT)}")


def main():
    global STUDY, ANNOTATE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", nargs="?", type=Path, help=f"study folder (default {STUDY})")
    parser.add_argument("--plain", action="store_true", help="the figure without the shaded loops and lag arrows")
    args = parser.parse_args()
    ANNOTATE = ANNOTATE and not args.plain
    if args.study:
        STUDY = args.study.resolve()
    cases, meta = read(STUDY)
    with plt.rc_context(STYLE):
        figures(cases, meta)
    latex(cases, meta)
    table_6(meta)


if __name__ == "__main__":
    main()
