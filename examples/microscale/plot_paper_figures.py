"""Paper Figures 3-4 and Table 3 from the saved data; no solver runs.

Press Run. Reads figure3_biot.npz, figure4_cycles.npz and figure4_stationary.npz from DATA
(written by figure3_biot.py, figure4_cycles.py and figure4_stationary.py) and writes one figure
per panel to OUTPUT in each of FORMATS: figure_3_biot_<boundary> and
figure_4_<flux|stress>_T<T>s. Also writes the LaTeX of each paper figure (figure environment,
panels, caption and label) to OUTPUT/figure_<n>_<name>.tex: the captions are the CAPTION_*
templates below, filled with the settings saved with the data (paper_latex.py). A missing data
file is skipped.

Also writes TABLES/table_3_criterion.tex: the departure of the cell from the stationary cell (flux,
axial stress, energy dissipated per pulse) at the pulse durations TABLE_3_DURATIONS, from the sweep of
the same pulse over 0.2 to 200 s in DATA/CRITERION_4 (figure4_cycles.py --criterion, both solids). It
also prints, for the earlier nine-duration sweep DATA/figure4_rates (viscoelastic, 0.2 to 20 s, which the
criterion sweep reproduces), the largest flux departure from the stationary cell over the pulse and hold.

With --annotated (or ANNOTATE = True) it draws only Figure 4, with the departure from the stationary cell shaded
and the hold labelled, to figure_4_<flux|stress>_T<T>s_annotated; nothing else is written.

From the command line (run.py plot): plot_paper_figures.py [DATA] [--output DIR] [--format FMT] [--annotated].
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import paper_latex as tex  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
DATA = ROOT / "results" / "paper"
OUTPUT = ROOT / "results" / "paper" / "figures"
TABLES = ROOT / "results" / "paper" / "tables"
FORMATS = ["pdf"]
ANNOTATE = False   # Figure 4 only, shaded departure from the stationary cell, to <name>_annotated

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "pdf.fonttype": 42,
}
FIGSIZE = (6, 4)
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINEWIDTH = 3.0   # every data line in the paper figures (also used by plot_paper_consolidation.py)
MARKERSIZE = 6.0  # every data marker
SOLID_COLOURS = {"elastic": 0.08, "viscoelastic": 0.5}   # viridis positions: dark purple, teal
MARKER_STEPS = 5  # Figure 4: saved steps between markers on one curve (0.1 T at 100 steps per pulse)
SHADE_ALPHA = 0.15  # annotated figures: translucent fill in the solid's colour
ANNOTATION_FONTSIZE = 12
# The panels that keep a legend; each figure's legend covers all of its panels in the paper.
LEGENDS = {"figure_3_biot_fixed", "figure_4_flux_T2s"}

# Approximate readings of Sandström et al. (2016), Figure 5(b), https://doi.org/10.1016/j.cma.2016.05.021
SANDSTROM_K = np.array([10.0, 100.0, 1000.0, 10000.0])
SANDSTROM_ALPHA = {
    1.0: [0.30, 0.50, 0.85, 0.98],
    10.0: [0.32, 0.52, 0.85, 0.98],
    20.0: [0.36, 0.53, 0.85, 0.98],
    30.0: [0.39, 0.55, 0.85, 0.98],
}
# Their crosses: a white face inside a thin outline in the pressure's colour, to stand out on the curves beneath.
SANDSTROM_MARKER = {"marker": "X", "markersize": 10, "markeredgewidth": 1.6, "markerfacecolor": "white",
                    "linestyle": "none"}

# LaTeX captions: <<name>> is filled from the data's saved settings (see the latex_figure_* functions).
CAPTION_3 = (r"Equilibrated secant Biot coefficient $\alpha$ of the elastic cell held at $\overline{\bm F}=\bm I$, "
             r"against solid bulk modulus $K$, for pore pressures $\overline p=<<pressures>>$. (a)~Fixed boundaries, "
             r"reproducing the fixed-displacement case of \citeauthor{Sandstrom2016}~\cite{Sandstrom2016}; (b)~periodic "
             r"boundaries. Crosses: values read from their Figure~5(b), coloured by pressure.<<excluded>>")
# Added to CAPTION_3 for a case saved as NaN (the fixed cell at K = 10 Pa, p = 30 Pa).
EXCLUDED_3 = r" The fixed cell does not equilibrate at $K=<<K>>$ and $\overline p=<<p>>$, so (a) has no computed value there."
SUBCAPTIONS_3 = {"fixed": ("Fixed boundary conditions.", "fig:rve-biot-fixed"),
                 "periodic": ("Periodic boundary conditions.", "fig:rve-biot-periodic")}
CAPTION_4 = (r"Flux <<flux_panels>> and axial nominal stress $-\overline P_{33}$ <<stress_panels>> of the cell during "
             r"the compression pulse of Section~\ref{sec:cell-history} and the following hold; the vertical dashed line marks "
             r"the end of the pulse. <<rows>>. $Q$ is the referential flux along the pressure gradient, normalised by "
             r"its initial value. Dashed with circles: elastic skeleton; solid line with squares: viscoelastic "
             r"skeleton.<<stationary>>")
# Grey line in every Figure 4 panel: the stationary cell (figure4_stationary.py), if its data exist.
STATIONARY_4 = "figure4_stationary.npz"
STATIONARY_CAPTION_4 = r" Grey: stationary cell."
LABELS_4 = {2.0: ("fig:flux-long", "fig:stress-long"), 0.2: ("fig:flux-short", "fig:stress-short")}
# Table 3: the pulse of Figure 4 at thirteen durations from 0.2 to 200 s, both solids (figure4_cycles.py
# --criterion; one file per solid, figure4_criterion_<solid>.npz), and the durations shown.
CRITERION_4 = "criterion_20261108_153003"
TABLE_3_DURATIONS = (0.2, 2.0, 20.0, 200.0)
CAPTION_TABLE_3 = (r"Departure of the single cell from the stationary response, in per cent. Flux: largest departure, "
                   r"relative to the initial flux. Stress: largest departure of the axial stress, relative to the "
                   r"stationary peak. Energy: dissipated per pulse, relative to the work needed to compress the "
                   r"stationary cell.")
# The earlier sweep of the pulse at nine durations (viscoelastic solid, figure4_cycles.py --rates).
RATES_4 = "figure4_rates"
# ---------------------------------------------------------------------------


def load(name):
    """DATA/<name>.npz, or None if it is missing."""
    path = DATA / f"{name}.npz"
    if not path.is_file():
        print(f"Skipped {name}: no {name}.npz in {DATA}")
        return None
    with np.load(path, allow_pickle=False) as f:
        print(f"Plotting {path}")
        return {k: f[k].copy() for k in f.files}


def save(fig, name):
    if name not in LEGENDS:
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = "_annotated" if ANNOTATE else ""
    for extension in FORMATS:
        path = OUTPUT / f"{name}{suffix}.{extension}"
        fig.savefig(path, bbox_inches=None)
        print(f"  {path}")
    plt.close(fig)


def colours(n):
    """Viridis without the pale yellow end, for contrast on white."""
    return plt.get_cmap("viridis")(np.linspace(0.08, 0.78, n))


def solid_colour(solid):
    """The colour of a solid in every paper figure (Figures 4, 6 and 8): SOLID_COLOURS along viridis."""
    return plt.get_cmap("viridis")(SOLID_COLOURS[solid])


def new_figure():
    return plt.subplots(figsize=FIGSIZE, layout="constrained")


def curve(ax, x, y, colour, index, label=None, markevery=1, linestyle="-", zorder=2):
    """A line with markers; the markers are drawn above every line (zorder 4), so that one curve's line never
    hides another's markers. An empty line carries the legend entry."""
    marker = MARKERS[index % len(MARKERS)]
    ax.plot(x, y, color=colour, linestyle=linestyle, linewidth=LINEWIDTH, zorder=zorder)
    ax.plot(x, y, color=colour, linestyle="none", marker=marker, markersize=MARKERSIZE, markevery=markevery,
            zorder=4)
    ax.plot([], [], color=colour, linestyle=linestyle, linewidth=LINEWIDTH, marker=marker, markersize=MARKERSIZE,
            label=label)


def finish(ax, handles=None, labels=None, loc="upper left", title=None, **legend_options):
    ax.tick_params(which="both", direction="in", top=True, right=True, length=5, width=0.6)
    ax.tick_params(which="minor", length=3)
    ax.grid(False)
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, loc=loc, title=title, title_fontsize=12, frameon=True, fancybox=False, framealpha=1,
                       edgecolor="black", labelspacing=0.6, borderpad=0.6, handlelength=2.5, handletextpad=0.7,
                       **legend_options)
    legend.get_frame().set_linewidth(0.6)


def figure_3(a):
    """One figure per boundary condition: secant Biot coefficient against solid bulk modulus."""
    pressures = a["pressures"]
    for i, boundary in enumerate(str(b) for b in a["boundaries"]):
        fig, ax = new_figure()
        for j, (p, c) in enumerate(zip(pressures, colours(len(pressures)))):
            curve(ax, a["bulk_moduli"], a["alpha_secant"][i, j], c, j, label=rf"$p={p:g}\,\mathrm{{Pa}}$")
            if float(p) in SANDSTROM_ALPHA:
                ax.plot(SANDSTROM_K, SANDSTROM_ALPHA[float(p)], color=c, zorder=5, **SANDSTROM_MARKER)
        ax.set_xscale("log")
        ax.set_xlabel(r"Solid bulk modulus $K\;(\mathrm{Pa})$")
        ax.set_ylabel(r"Secant Biot coefficient $\alpha$")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color="0.3", **SANDSTROM_MARKER))
        labels.append("Sandström et al. (2016)")
        finish(ax, handles, labels, loc="lower right")
        save(fig, f"figure_3_biot_{boundary}")


def load_stationary_4():
    path = DATA / STATIONARY_4
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        return {key: d[key] for key in ("compression", "flux", "fluid_content", "stress")}


def stationary_4(stationary, name, compression):
    """The stationary cell's Q/Q0 or -P33 at the given compressions (linear in its continuation grid)."""
    if name == "flux":
        return np.interp(compression, stationary["compression"], stationary["flux"][:, 0]) / stationary["flux"][0, 0]
    return -np.interp(compression, stationary["compression"], stationary["stress"][:, 2, 2])


def figure_4(a):
    """For each duration, one figure of normalised flux and one of the axial nominal stress against t/T. With
    ANNOTATE, the gap between each solid's curve and the stationary cell is shaded in the solid's colour and the
    hold after the vertical end-of-pulse line is labelled."""
    col = {str(name): i for i, name in enumerate(a["columns"])}
    stationary = load_stationary_4()
    # The two materials nearly coincide at long T: the elastic dash is drawn over the viscoelastic line, and
    # their markers (every MARKER_STEPS saved steps) are staggered so that they do not land on each other.
    colour = [solid_colour(name) for name in ("elastic", "viscoelastic")]
    styles = {0: ("Elastic skeleton", dict(linestyle=(0, (4, 2)), zorder=3), 0),
              1: ("Viscoelastic skeleton", dict(linestyle="-", zorder=2), 2)}
    for j, T in enumerate(a["durations"]):
        for name in ("flux", "stress"):
            fig, ax = new_figure()
            for material in (1, 0):
                label, style, first = styles[material]
                rows = a["rows"][(a["rows"][:, col["material"]] == material)
                                 & (a["rows"][:, col["duration_index"]] == j)]
                initial = a["initial_flux"][(a["initial_flux"][:, 0] == material)
                                            & (a["initial_flux"][:, 1] == j), 2].item()
                y = rows[:, col["Q1"]] / initial if name == "flux" else -rows[:, col["P33"]]
                curve(ax, rows[:, col["time"]] / T, y, colour[material], material, label=label,
                      markevery=slice(first, None, MARKER_STEPS), **style)
                if ANNOTATE and stationary is not None:  # departure from the stationary cell, at the same compression
                    ax.fill_between(rows[:, col["time"]] / T, y, stationary_4(stationary, name, rows[:, col["compression"]]),
                                    color=colour[material], alpha=SHADE_ALPHA, linewidth=0, zorder=0)
            if stationary is not None:
                # read at the compression of each saved step
                rows = a["rows"][(a["rows"][:, col["material"]] == 0) & (a["rows"][:, col["duration_index"]] == j)]
                y = stationary_4(stationary, name, rows[:, col["compression"]])
                ax.plot(rows[:, col["time"]] / T, y, color="0.55", linewidth=2.0, zorder=1, label="Stationary cell")
            ax.axvline(1.0, color="0.6", linewidth=0.8, linestyle="--", zorder=1)
            if ANNOTATE:  # in the empty corner: below the flux curves, above the stress curves
                ax.text(1.03, 0.05 if name == "flux" else 0.93, "hold", transform=ax.get_xaxis_transform(),
                        ha="left", va="center", fontsize=ANNOTATION_FONTSIZE, color="0.35")
            ax.set_xlabel(r"Normalised time $t/T$")
            if name == "flux":
                ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle="--", zorder=1)
                ax.set_ylabel(r"$Q / Q_{\mathrm{initial}}$")
            else:
                ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":", zorder=1)
                ax.set_ylabel(r"$-\overline{P}_{33}$ (Pa)")
            finish(ax, loc="best")
            save(fig, f"figure_4_{name}_T{T:g}s")


def criterion_departures():
    """Departures of the transient cell from the stationary cell in the criterion sweep (DATA/CRITERION_4), as
    fractions, keyed by (solid, T), or None if the data are missing:
      flux    max |Q/Q0 - Q_st/Q_st,0| over the pulse and hold, the stationary cell read at the same compression;
      stress  max |P33 - P33_st| over the pulse and hold, over the stationary cell's peak |P33|;
      energy  the work dissipated over the pulse, over the work needed to compress the stationary cell (which
              dissipates none)."""
    from scipy.interpolate import CubicSpline

    stationary = load_stationary_4()
    folder = DATA / CRITERION_4
    if stationary is None or not folder.is_dir():
        print(f"Skipped the criterion sweep: no stationary cell data or no {folder}")
        return None
    c_s = stationary["compression"]
    stress_s = CubicSpline(c_s, stationary["stress"][:, 2, 2])
    peak = np.max(np.abs(stationary["stress"][:, 2, 2]))
    load_c = np.linspace(0.0, c_s.max(), 401)
    stored = np.trapezoid(-stress_s(load_c), load_c)
    out = {}
    for path in sorted(folder.glob("figure4_criterion_*.npz")):
        solid = path.stem.removeprefix("figure4_criterion_")
        if solid not in SOLID_COLOURS:  # figure4_criterion_steps40: the time-step check
            continue
        with np.load(path, allow_pickle=False) as d:
            rows, durations = d["rows"], d["durations"]
            col = {str(name): i for i, name in enumerate(d["columns"])}
        for j, T in enumerate(durations):
            r = rows[rows[:, col["duration_index"]] == j]
            c, stress = r[:, col["compression"]], r[:, col["P33"]]
            q = r[:, col["Q1"]] / r[0, col["Q1"]]
            pulse = r[:, col["stage"]] == 0
            out[solid, float(T)] = dict(flux=float(np.max(np.abs(q - stationary_at(stationary, c)[0]))),
                                        stress=float(np.max(np.abs(stress - stress_s(c))) / peak),
                                        energy=float(np.trapezoid(-stress[pulse], c[pulse]) / stored))
    return out


def table_3():
    """TABLES/table_3_criterion.tex: flux, stress and energy departures (per cent) at TABLE_3_DURATIONS."""
    departures = criterion_departures()
    if departures is None:
        return
    percent = lambda x: f"{100 * x:.1f}" if 100 * x >= 1 else f"{100 * x:.2f}"
    rows = []
    for T in TABLE_3_DURATIONS:
        cells = [f"{T:g}"]
        for solid in ("elastic", "viscoelastic"):
            d = next(v for (s, t), v in departures.items() if s == solid and np.isclose(t, T))
            cells += [percent(d["flux"]), percent(d["stress"]), percent(d["energy"])]
        rows.append(" & ".join(cells) + r" \\")
    text = "\n".join([
        r"\begin{table}[pos=htb]", r"\centering", rf"\caption{{{CAPTION_TABLE_3}}}", r"\label{tab:criterion}",
        r"\small", r"\setlength{\tabcolsep}{1.1em}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r" & \multicolumn{3}{c}{Elastic skeleton} & \multicolumn{3}{c}{Viscoelastic skeleton} \\",
        r"\cmidrule(lr){2-4} \cmidrule(l){5-7}",
        r"$T$ (s) & Flux & Stress & Energy & Flux & Stress & Energy \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n"
    tex.write(TABLES / "table_3_criterion.tex", text,
              f"examples/microscale/plot_paper_figures.py from {tex.relative(DATA / CRITERION_4, ROOT)}")


def load_rates_4():
    """The rate campaign RATES_4.npz (figure4_cycles.py --rates), sorted by duration, or None if it is missing."""
    if not (DATA / f"{RATES_4}.npz").is_file():
        print(f"Skipped {RATES_4}: no {RATES_4}.npz in {DATA}")
        return None
    with np.load(DATA / f"{RATES_4}.npz", allow_pickle=False) as d:
        rows, columns = d["rows"], [str(c) for c in d["columns"]]
        durations, metadata = d["durations"], json.loads(str(d["metadata"]))
    c = {name: i for i, name in enumerate(columns)}
    paths = []
    for j, period in enumerate(durations):
        r = rows[rows[:, c["duration_index"]] == j]
        paths.append(dict(period=float(period), time=r[:, c["time"]], compression=r[:, c["compression"]],
                          flux=r[:, c["Q1"]], fluid_content=r[:, c["fluid_content"]]))
    print(f"Reading {DATA / f'{RATES_4}.npz'} ({len(paths)} durations)")
    return dict(manifest=metadata, paths=sorted(paths, key=lambda p: p["period"]))


def stationary_at(stationary, c):
    """The stationary cell's Q/Q0 and porosity at compressions c (cubic spline through its continuation grid)."""
    from scipy.interpolate import CubicSpline

    q = CubicSpline(stationary["compression"], stationary["flux"][:, 0] / stationary["flux"][0, 0])(c)
    return q, CubicSpline(stationary["compression"], stationary["fluid_content"])(c) / (1 - c)


def departure(compression, flux, fluid_content, stationary):
    """Largest |Q/Q0 - Q_s/Q_s0| and |phi - phi_s|/phi0 along a trajectory, the stationary cell read at c(t)."""
    c = np.asarray(compression)
    q_s, phi_s = stationary_at(stationary, c)
    phi = fluid_content / (1 - c)
    return float(np.max(np.abs(flux / flux[0] - q_s))), float(np.max(np.abs(phi - phi_s)) / phi[0])


def rates_summary(rates):
    """Print the largest flux departure from the stationary cell at every duration of the rate campaign."""
    stationary = load_stationary_4()
    if stationary is None:
        print("Skipped the rate summary: no stationary cell data")
        return
    print("Largest |Q/Q0 - Q_stationary/Q_stationary,0| over the pulse and hold, viscoelastic solid:")
    for p in rates["paths"]:
        flux, _ = departure(p["compression"], p["flux"], p["fluid_content"], stationary)
        print(f"  T = {p['period']:7.3g} s   {100 * flux:5.2f} %")


def settings(a):
    """The run settings saved with the data."""
    return json.loads(str(a["metadata"]))["settings"]


def save_latex(name, text):
    tex.write(OUTPUT / f"{name}.tex", text, f"examples/microscale/plot_paper_figures.py from {tex.relative(DATA, ROOT)}")


def latex_figure_3(a):
    s = settings(a)
    missing = np.argwhere(np.isnan(a["alpha_secant"]))
    excluded = "".join(tex.fill(EXCLUDED_3, K=tex.quantity(a["bulk_moduli"][k], "Pa")[1:-1],
                                p=tex.quantity(a["pressures"][j], "Pa")[1:-1]) for b, j, k in missing)
    caption = tex.fill(CAPTION_3, G=tex.number(s["G"]), mu_f=tex.number(s["mu_f"]),
                       pressures=tex.listing(s["pressures"], "Pa")[1:-1],
                       duration=tex.quantity(s["duration"], "s"), steps=tex.words(s["steps"]),
                       hold_tolerance=f"${tex.number(s['hold_tolerance'])}$", hold_stable=tex.words(s["hold_stable"]),
                       excluded=excluded)
    panels = [(f"figure_3_biot_{b}", *SUBCAPTIONS_3[str(b)]) for b in a["boundaries"]]
    save_latex("figure_3_biot", tex.figure("fig:rve-biot-response", caption, panels, placement="tb"))


def latex_figure_4(a):
    s = settings(a)
    durations = [float(T) for T in a["durations"]]
    letters = "abcdefgh"
    hold = s["recovery_fraction"]
    rows = "; ".join(f"{('top', 'bottom')[j] if len(durations) == 2 else f'row {j + 1}'}, "
                     f"{tex.quantity(T, 's').replace('$', '$T=', 1)}" for j, T in enumerate(durations))
    caption = tex.fill(CAPTION_4, gradient=tex.quantity(np.linalg.norm(s["pressure_gradient"]), r"Pa\,m^{-1}"),
                       half_compression=tex.number(s["compression"] / 2), compression=tex.percent(s["compression"]),
                       hold="$T/2$" if hold == 0.5 else f"${tex.number(hold)}T$",
                       flux_panels="(" + ", ".join(letters[2 * j] for j in range(len(durations))) + ")",
                       stress_panels="(" + ", ".join(letters[2 * j + 1] for j in range(len(durations))) + ")",
                       rows=rows[0].upper() + rows[1:], steps=s["steps"], eta_s=tex.number(s["eta_s"]),
                       zeta_s=tex.number(s["zeta_s"]),
                       stationary=STATIONARY_CAPTION_4 if load_stationary_4() is not None else "")
    panels = []
    for T in durations:
        flux_label, stress_label = LABELS_4.get(T, (f"fig:flux-T{T:g}", f"fig:stress-T{T:g}"))
        duration = tex.quantity(T, "s").replace("$", "$T=", 1)
        panels += [(f"figure_4_flux_T{T:g}s", f"Normalised flux, {duration}.", flux_label),
                   (f"figure_4_stress_T{T:g}s", f"Axial stress, {duration}.", stress_label)]
    save_latex("figure_4_cycles", tex.figure("fig:compression_cycle", caption, panels, placement="tb", width=0.49))


def main():
    import argparse

    global DATA, OUTPUT, FORMATS, ANNOTATE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="?", type=Path, help=f"folder with the .npz files (default {DATA})")
    p.add_argument("--output", type=Path, help=f"folder for the figures (default {OUTPUT})")
    p.add_argument("--format", choices=["pdf", "png", "svg"], help=f"one format (default {FORMATS})")
    p.add_argument("--annotated", action="store_true", help="only Figure 4, annotated, to <name>_annotated")
    a = p.parse_args()
    ANNOTATE = ANNOTATE or a.annotated
    if a.data:
        DATA = a.data
        if not a.output:
            OUTPUT = a.data / "figures"
    if a.output:
        OUTPUT = a.output
    if a.format:
        FORMATS = [a.format]
    with plt.rc_context(STYLE):
        if ANNOTATE:
            data = load("figure4_cycles")
            if data is not None:
                figure_4(data)
            return
        for name, draw, latex in (("figure3_biot", figure_3, latex_figure_3),
                                  ("figure4_cycles", figure_4, latex_figure_4)):
            data = load(name)
            if data is not None:
                draw(data)
                latex(data)
        table_3()
        rates = load_rates_4()
        if rates is not None:
            rates_summary(rates)


if __name__ == "__main__":
    main()
