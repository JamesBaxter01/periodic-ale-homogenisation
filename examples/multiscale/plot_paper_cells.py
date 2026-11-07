"""Paper Figure 7: one FE² cell of the consolidation column in 3D at three times; no solver runs.

Press Run. Reads DATA, a cell replayed through every step of its point (consolidation_cell_replay.py: the cell at the
sealed base of the viscoelastic step-load column), and writes to OUTPUT, in each of FORMATS, one file per panel with
no text, figure_7_cell_<name> for the PANELS (name, time): by default the three phases of the settlement, early (the
solid's delayed compression), undrained (compressed, the pore pressure still carrying the load) and drained.

Each cell is drawn as the solid surface (channel walls and the solid's periodic faces) in its deformed configuration
at true scale, with its element edges, coloured by the cell displacement |u| / l (affine plus fluctuation, relative to
the cell centre) on one colour range, and shaded by the faces' orientation (SHADE). The cell axes are drawn beside
the first cell and the colour bar beside the last; every file is cropped to the same box, so the cells keep one scale
when set side by side at one width.

Also writes the LaTeX, the three images in a row as subfigures (a)-(c) (OUTPUT/figure_7_cells.tex): CAPTION below, filled from the replay
and the times actually drawn (paper_latex.py). The time, height and state of each cell are printed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import paper_latex as tex  # noqa: E402
from plot_paper_figures import FIGSIZE, STYLE  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
RUN = ROOT / "results" / "consolidation" / "full_viscoelastic_step_20261026_201037"
DATA = RUN / "cell_replay_bottom.npz"   # consolidation_cell_replay.py output
OUTPUT = ROOT / "results" / "paper" / "figures"
FORMATS = ["pdf"]
DPI = 300                          # of any rasterised parts
# (file name, time (s)): the stored step nearest each time; None = the run's end. The settlement's three phases.
PANELS = [("early", 0.01), ("undrained", 10.0), ("drained", None)]
VIEW = (22, -55)                   # elevation, azimuth (degrees)
ZOOM = 1.05                        # of the 3D axes (larger clips the least compressed cell)
COLOURMAP = "viridis"
EDGE_COLOUR = (0, 0, 0, 0.35)      # element edges (RGBA); None = no edges
EDGE_WIDTH = 0.15
SHADE = 0.3                        # darkening of faces turned from LIGHT (0 = flat colours, exactly the colour bar's)
LIGHT = (45, -25)                  # light direction: elevation, azimuth (degrees)
# Cell-axis arrows beside the first cell: labels of the cell's x, y, z axes (z along the column) and the inset
# [left, bottom, width, height] in figure fraction; None draws none.
TRIAD = dict(labels=(r"$x$", r"$y$", r"$z$"), box=(0.0, 0.03, 0.22, 0.22))
LATEX_NAME = "figure_7_cells"
PANEL_WIDTH = 0.32                 # of the text width, each image
CAPTION = (r"The cell at the sealed base of the viscoelastic column of Figure~\ref{fig:consolidation} in the three "
           r"stages of settlement: (a)~early, while skeleton viscosity delays the compression; (b)~undrained, with the "
           r"pore pressure still <<p_undrained>> of the traction; (c)~drained. Solid phase only, deformed at true scale "
           r"and coloured by the displacement $|\bm u|/\ell$ relative to the cell centre; arrows show the cell axes "
           r"($z$ along the column).")
SUBCAPTIONS = {"early": "Early", "undrained": "Undrained", "drained": "Drained"}   # (a)-(c), with t and F_33
# ---------------------------------------------------------------------------


def draw(X, triangles, displacement, clim, colour_bar, axes):
    """One cell: faces coloured by |u| (mean of their vertices), shaded by their orientation to LIGHT (either side
    lit alike: the triangles' orientation is not consistent)."""
    fig = plt.figure(figsize=(FIGSIZE[1] * 1.1, FIGSIZE[1]))
    ax = fig.add_axes([0.0, 0.0, 0.78, 1.0], projection="3d")
    corners = (X + displacement)[triangles]
    mappable = plt.cm.ScalarMappable(matplotlib.colors.Normalize(*clim), COLOURMAP)
    colours = mappable.to_rgba(np.linalg.norm(displacement, axis=1)[triangles].mean(axis=1))
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    e, a = np.radians(LIGHT)
    lit = np.abs(normal @ [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]) / np.linalg.norm(normal, axis=1)
    colours[:, :3] *= (1 - SHADE * (1 - lit))[:, None]
    faces = Poly3DCollection(corners, facecolors=colours, linewidths=EDGE_WIDTH,
                             edgecolor=EDGE_COLOUR if EDGE_COLOUR else "face")
    ax.add_collection3d(faces)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1), zoom=ZOOM)
    ax.view_init(*VIEW)
    ax.set_axis_off()
    if TRIAD and axes:
        triad = fig.add_axes(TRIAD["box"], projection="3d")
        triad.set_facecolor("none")
        for axis, text in enumerate(TRIAD["labels"]):
            d = np.eye(3)[axis]
            triad.quiver(0, 0, 0, *d, color="black", linewidth=1.2, arrow_length_ratio=0.2)
            triad.text(*(1.35 * d), text, fontsize=12, ha="center", va="center")
        triad.set_xlim(0, 1)
        triad.set_ylim(0, 1)
        triad.set_zlim(0, 1)
        triad.set_box_aspect((1, 1, 1))
        triad.view_init(*VIEW)
        triad.set_axis_off()
    bar = fig.colorbar(mappable, cax=fig.add_axes([0.8, 0.2, 0.035, 0.6]), label=r"$|\mathbf{u}|/\ell$")
    bar.ax.set_visible(colour_bar)  # one colour bar (the last panel's); the space is kept so the cells match
    return fig


def crop_box(figures):
    """The union of the figures' contents (inches), with the triad labels, which the tight box leaves out (the
    extents of text in 3D axes are not projected): each label's projected anchor, padded by its font size."""
    boxes = []
    for fig in figures:
        renderer = fig.canvas.get_renderer()
        fig.draw(renderer)
        boxes.append(fig.get_tightbbox(renderer))
        for ax in fig.axes:
            for text in ax.texts:
                x, y, _ = proj3d.proj_transform(*text.get_position_3d(), ax.get_proj())
                x, y = fig.dpi_scale_trans.inverted().transform(ax.transData.transform((x, y)))
                pad = text.get_fontsize() / 72
                boxes.append(Bbox.from_extents(x - pad, y - pad, x + pad, y + pad))
    return Bbox.union(boxes).padded(0.03)


def main():
    plt.rcParams.update(STYLE)
    d = np.load(DATA)
    info = json.loads(str(d["metadata"]))
    t = d["time"]
    steps = [len(t) - 1 if when is None else int(np.argmin(np.abs(t - when))) for _, when in PANELS]
    X, triangles = d["surface_vertices"], d["surface_triangles"]
    displacement = [d["surface_displacement"][n].astype(float) for n in steps]
    clim = (0.0, max(float(np.linalg.norm(w, axis=1).max()) for w in displacement))
    figures = [draw(X, triangles, w, clim, k == len(steps) - 1, k == 0) for k, w in enumerate(displacement)]
    box = crop_box(figures)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for (name, _), fig, n in zip(PANELS, figures, steps):
        for extension in FORMATS:
            fig.savefig(OUTPUT / f"figure_7_cell_{name}.{extension}", bbox_inches=box, dpi=DPI)
        plt.close(fig)
        print(f"  {OUTPUT / f'figure_7_cell_{name}'}.{{{','.join(FORMATS)}}}: t = {t[n]:.6g} s (step {n}), "
              f"F_33 = {d['inputs'][n, 8]:.5f}, p = {d['inputs'][n, 9]:.4g} Pa")
    print(f"  Z = {info['Z']:.4f} m (point {info['point']} of {info['run']}); |u|/l from 0 to {clim[1]:.3g}; replay "
          f"mismatch {float(d['replay_mismatch']):.1e}")
    latex_figure(d, info, steps)


def latex_figure(d, info, steps):
    with np.load(ROOT / info["run"] / "fe2.npz") as run:
        traction = json.loads(str(run["metadata"]))["macro"]["traction"]
    t, inputs = d["time"], d["inputs"]
    values = {}
    for (name, _), n in zip(PANELS, steps):
        values[f"t_{name}"] = rf"$t\approx{tex.number(t[n], 2)}\,\mathrm s$" if name != "drained" else \
            rf"$t={tex.number(round(t[n]), 4)}\,\mathrm s$"
        values[f"F_{name}"] = f"{inputs[n, 8]:.3f}"
    undrained = steps[[name for name, _ in PANELS].index("undrained")]
    caption = tex.fill(CAPTION, Z=tex.number(info["Z"], 2), p_undrained=tex.percent(inputs[undrained, 9] / traction, 2),
                       mismatch=tex.number(float(d["replay_mismatch"]), 1), **values)
    panels = [(f"figure_7_cell_{name}", f"{SUBCAPTIONS[name]}, {values[f't_{name}']}, $F_{{33}}={values[f'F_{name}']}$.",
               f"fig:consolidation-cells-{name}") for name, _ in PANELS]
    source = f"examples/multiscale/plot_paper_cells.py from {tex.relative(DATA, ROOT)}"
    tex.write(OUTPUT / f"{LATEX_NAME}.tex", tex.figure("fig:consolidation-cells", caption, panels, placement="tb",
                                                         width=PANEL_WIDTH, per_row=len(panels)), source)


def command_line():
    global DATA, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="?", type=Path, help=f"replayed cell (default {DATA})")
    parser.add_argument("--output", type=Path, help=f"figure folder (default {OUTPUT})")
    args = parser.parse_args()
    DATA = args.data or DATA
    OUTPUT = args.output or OUTPUT


if __name__ == "__main__":
    command_line()
    main()
