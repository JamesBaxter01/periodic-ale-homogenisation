"""Paper Figures 1 (the schematic of the scales), 2 (the reference cell) and 5 (the consolidation set-up); no solver runs.

Press Run (fem environment: NGSolve reads the resolved mesh, PyVista renders the cell). Writes to OUTPUT, in each of
FORMATS:

- figure_1_scales, in 2D and needing no run: the transition between the scales as a zoom, from one point of the macroscale
  continuum to its cell with what passes each way between the zoom's lines, then the ALE map to the cell's current
  configuration (illustrative shapes, SCALES_STRETCH and SCALES_FLUCTUATION);
- figure_2_cell: the reference cell in 3D, one cell of the resolved column's mesh in RESOLVED, seen from VIEW (the
  view of Figure 7, plot_paper_cells.py) and rendered like the earlier cell figure: flat greys (solid) and blues (pore
  fluid) by face orientation, black edges, parallel projection, rasterised at DPI; with the regions named
  (PHASE_LABELS), the channel width and cell side dimensioned (DIMENSION_ROOM) and, optionally, the cell axes (TRIAD);
- figure_5_setup, in 2D, every cell drawn as its section through the channel centres (solid grey, pores
  blue; the channel width of the run): (a) the resolved column with its boundary conditions, the load through a
  rigid, frictionless platen whose pore openings are free of traction, periodic sides and the sealed mirror-plane
  base; (b) the FE² column of the run FE2 on the same height scale, with its graded elements, integration points,
  the resolved cells' boundaries (ticks on the left) and its boundary conditions, and the cell at one integration
  point (the one nearest HIGHLIGHT_Z) with what the scales exchange; the step and cyclic load histories, top right.

Also writes the LaTeX of each (OUTPUT/<name>.tex): CAPTION_SCALES, and CAPTION_CELL and CAPTION_SETUP filled from the
runs' metadata
(paper_latex.py).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import paper_latex as tex  # noqa: E402
from plot_paper_figures import STYLE  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
RESOLVED = ROOT / "results" / "consolidation" / "full_elastic_step_20261026_201037"   # resolved mesh (either solid)
FE2 = RESOLVED                     # FE² run: macroscale mesh, integration points and load (the same for both solids)
OUTPUT = ROOT / "results" / "paper" / "figures"
FORMATS = ["pdf"]
TEXT_WIDTH = 6.48                  # inches, the cas-sc text width (468.3 pt): the figures are drawn at twice their
                                   # printed size, so the paper figures' font sizes apply
LOAD = r"\overline{t}"             # symbol of the applied nominal traction (mathtext and LaTeX)
TEXT_SIZE = 15                     # annotations

# Figure 2, the cell in 3D.
CELL = 5                           # the column's cell drawn (0 = at the base)
VIEW = (22.0, -55.0)               # elevation, azimuth (degrees) as matplotlib's view_init: the view of Figure 7
GREYS = {"front": "#8f8f8f", "side": "#333333", "top": "#777777"}  # face colours by orientation (earlier cell figure)
BLUES = {"front": "#c6dbef", "side": "#4f8fc0", "top": "#9ecae1"}  # the pore fluid's faces; None draws the solid only
PHASE_LABELS = True                # name the solid and fluid regions, with leaders from the left
EDGE_WIDTH = 0.5                   # feature edges, printed points
RENDER_HEIGHT = 3000               # pixels
DPI = 600                          # of the rendered image in the PDF
TRIAD = None                       # labels of the cell axes, e.g. (r"$x$", r"$y$", r"$z$"); None draws none
CELL_SIZE = (3.6, 3.15)            # inches (twice printed); the width is trimmed to the drawing
DIMENSION_ROOM = 0.75              # inches below the cell for the channel-width and side dimensions; 0 draws none
CELL_WIDTH = None                  # printed width, fraction of the line width; None prints at half the drawn size
CAPTION_CELL = (r"Reference cell of side $\ell$: a solid matrix (grey) around three intersecting square channels "
                r"filled with pore fluid (blue); opposite faces are periodic.")

# The set-up, in 2D.
SOLID_FILL, FLUID_FILL = "#a6a6a6", "#c6dbef"
FLUID = "#2166AC"                  # outflow arrow
ACCENT = "#21918C"                 # the highlighted point, its cell and the exchange (viridis teal, as the other figures)
HIGHLIGHT_Z = 0.3                  # the integration point nearest this fraction of the height is shown with its cell
SETUP_SIZE = (2 * TEXT_WIDTH, 4.8)  # inches (twice printed)
SETUP_WIDTH = 1.0                  # printed width, fraction of the line width
CAPTION_SETUP = (r"Consolidation set-up, with each cell drawn as its section through the channel centres (solid grey, "
                 r"pores blue); top right: the step and cyclic loads. (a)~Resolved FSI of the <<cells>>-cell column. "
                 r"(b)~FE$^2$: <<elements>> $P_<<k>>$-$P_<<km>>$ elements refined towards the drained top, with "
                 r"<<points>> integration points (dots), each carrying its own cell; ticks mark the resolved cells.")

# The schematic of the scales, in 2D.
SCALES_SIZE = (2 * TEXT_WIDTH, 3.72)  # inches (twice printed)
SCALES_LATEX = {"text.usetex": True,  # the manuscript's fonts and \\bm, so the symbols match the equations
                "text.latex.preamble": r"\usepackage{stix}\usepackage{bm}"}
SCALES_POINT = 0.23                # inches: the marker at the point of the macroscale body that the zoom starts from
SCALES_WIDTH = 1.0                 # printed width, fraction of the line width
SCALES_CHANNEL = 0.33              # channel width, in cell lengths (the paper's cell)
SCALES_STRETCH = 0.78              # the homogeneous compression drawn: height of the deformed cell over the reference
SCALES_FLUCTUATION = dict(A=0.07, B=0.025, D=0.035, E=0.035)  # sizes of the periodic fluctuation (see current_map)
MACRO_FILL = "#d5dbe2"             # the continuum: a light mix of the solid's grey and the pores' blue
CAPTION_SCALES = (r"Transition between the scales. The macroscale body $\overline\Omega$ is a continuum with fields "
                  r"$\overline{\bm u}$ and $\overline p$. At each integration point, a periodic cell (section: solid "
                  r"$\Omega_s$ grey, pore fluid $\Omega_f$ blue) receives the deformation gradient $\overline{\bm F}$, "
                  r"pressure $\overline p$ and pressure gradient $\bm g_p$, and returns its averaged stress "
                  r"$\overline{\bm P}$, fluid content $m_f$ and fluid flux $\bm Q_0$. It keeps its state between increments, so "
                  r"its response depends on the loading history. The ALE map $\bm\chi_A$ takes the "
                  r"cell to its current configuration: the homogeneous deformation $\overline{\bm F}\bm X$ (dashed) plus "
                  r"a periodic fluctuation $\widetilde{\bm d}_A$, drawn exaggerated.")
# ---------------------------------------------------------------------------


def screen():
    """Unit vectors towards the camera, screen right and screen up, for VIEW as matplotlib's view_init."""
    e, a = np.radians(VIEW)
    towards = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross(-towards, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return towards, right, np.cross(right, -towards)


def render(corners, height, edge_pixels, fluid=None):
    """Flat-shaded render of the triangles corners (n, 3, 3) in parallel projection from VIEW, on white, coloured by
    the axis each face is normal to (GREYS, or BLUES for the triangles marked fluid; every face of the cell is
    axis-aligned), with its feature edges and the outlines of the fluid's faces drawn edge_pixels wide (as tubes:
    plain wide lines are one pixel on some OpenGL drivers). Returns the RGB image cropped to its contents."""
    import pyvista as pv

    points = corners.reshape(-1, 3)
    faces = np.c_[np.full(len(corners), 3), np.arange(len(points)).reshape(-1, 3)].ravel()
    normal = np.abs(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])).argmax(axis=1)
    towards, right, up = screen()
    front = int(np.abs(towards[:2]).argmax())
    fluid = np.zeros(len(corners), bool) if fluid is None else np.asarray(fluid, bool)
    role = {2: "top", front: "front", 1 - front: "side"}
    rgb = np.array([[[int(shades[role[k]][i:i + 2], 16) for i in (1, 3, 5)] for k in range(3)]
                    for shades in (GREYS, BLUES or GREYS)], dtype=np.uint8)   # [solid or fluid, normal axis]
    surface = pv.PolyData(points, faces)
    surface.cell_data["rgb"] = rgb[fluid.astype(int), normal]
    low, high = points.min(axis=0), points.max(axis=0)
    surface = surface.clean(tolerance=1e-9 * float(np.linalg.norm(high - low)))
    edges = surface.extract_feature_edges(feature_angle=30, boundary_edges=False, non_manifold_edges=False,
                                          manifold_edges=False)
    if fluid.any():                                 # the fluid's faces meet the solid's in plane: outline them
        pores = pv.PolyData(points, faces.reshape(-1, 4)[fluid].ravel())
        pores = pores.clean(tolerance=1e-9 * float(np.linalg.norm(high - low)))
        edges = edges.merge(pores.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                        non_manifold_edges=False, manifold_edges=False))

    box = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    extent = np.ptp(box @ right), np.ptp(box @ up)
    plotter = pv.Plotter(window_size=(int(height * extent[0] / extent[1] * 1.1) + 40, height), off_screen=True,
                         lighting="none")
    plotter.set_background("white")
    plotter.enable_parallel_projection()
    plotter.add_mesh(surface, scalars="rgb", rgb=True, lighting=False)
    plotter.add_mesh(edges, color="black", line_width=edge_pixels, render_lines_as_tubes=True, lighting=False)
    centre = (low + high) / 2
    plotter.camera.focal_point = centre
    plotter.camera.position = centre + 4 * float(np.linalg.norm(high - low)) * towards
    plotter.camera.up = tuple(up)
    plotter.camera.parallel_scale = 0.55 * extent[1]
    image = plotter.screenshot(return_img=True)
    plotter.close()
    filled = np.nonzero(image[..., :3].min(axis=2) < 250)
    return image[max(filled[0].min() - 4, 0):filled[0].max() + 5, max(filled[1].min() - 4, 0):filled[1].max() + 5]


def resolved_mesh():
    """The resolved column's mesh (in cell lengths) and its run settings."""
    from ngsolve import Mesh

    with np.load(RESOLVED / "resolved.npz", allow_pickle=False) as d:
        settings = json.loads(str(d["metadata"]))["settings"]
    return Mesh(str(RESOLVED / "resolved_mesh.vol.gz")), settings


def cell_surface(mesh, cell, fluid=True):
    """Surface of the column's cell (counted from the base), in cell lengths, as triangle corners (n, 3, 3) and a
    mask of the fluid's triangles: the outer faces of the solid and, with fluid, of the pore fluid, closed by the
    faces between the cell and its neighbours; without fluid, the pore walls instead."""
    from ngsolve import BND

    phases = ("solid", "fluid") if fluid else ("solid",)
    X = np.array([v.point for v in mesh.vertices])
    outer, between = [], []
    for e in mesh.Elements(BND):
        kind = e.mat.rsplit("_", 1)[-1]
        if e.mat.startswith("cells_") and kind in phases:
            between.append(([v.nr for v in e.vertices], kind == "fluid"))
        elif (e.mat == "interface" and not fluid) or (e.mat != "interface" and kind in phases):
            outer.append(([v.nr for v in e.vertices], kind == "fluid"))
    triangles = []
    for nodes, wet in outer:
        z = X[nodes, 2].mean()
        if cell < z < cell + 1:
            triangles.append((nodes, wet))
    for nodes, wet in between:
        z = X[nodes, 2].mean()
        if abs(z - cell) < 1e-9 or abs(z - cell - 1) < 1e-9:
            triangles.append((nodes, wet))
    nodes = np.array([t[0] for t in triangles])
    return X[nodes] - [0.0, 0.0, cell], np.array([t[1] for t in triangles])


def arrow(ax, start, end, colour="black", width=1.6, head=12):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=f"-|>,head_length={head / 18:.2f},head_width={head / 36:.2f}",
                                 mutation_scale=18, linewidth=width, color=colour, shrinkA=0, shrinkB=0, zorder=5))


def canvas(size):
    """A figure with one invisible axes in inches over all of it."""
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.set_axis_off()
    return fig, ax


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=DPI)
    plt.close(fig)
    print(f"  {OUTPUT / name}.{{{','.join(FORMATS)}}}")


def write_float(name, caption, label, width, placement="tb", run=True):
    lines = [rf"\begin{{figure}}[pos={placement}]", r"    \centering",
             rf"    \includegraphics[width={width}\linewidth]{{{tex.FIGURE_FOLDER}/{name}.pdf}}",
             rf"    \caption{{{caption}}}", rf"    \label{{{label}}}", r"\end{figure}"]
    source = "examples/multiscale/plot_paper_geometry.py" + (f" from {tex.relative(RESOLVED, ROOT)}" if run else "")
    tex.write(OUTPUT / f"{name}.tex", "\n".join(lines) + "\n", source)


def figure_cell(mesh, settings):
    """Figure 2: the reference cell in 3D, solid and pore fluid, with the regions named on the left (PHASE_LABELS),
    the channel width and the cell side below its front face (DIMENSION_ROOM) and its axes at the lower left
    (TRIAD)."""
    x0 = 1.45 if PHASE_LABELS else 0.75 if TRIAD else 0.05   # room for the labels or the axes
    y0 = DIMENSION_ROOM or 0.05                     # and for the dimensions
    corners, fluid = cell_surface(mesh, CELL, fluid=BLUES is not None)
    h = CELL_SIZE[1] - y0 - 0.05                    # the image's height (inches); the canvas is trimmed to its width
    probe = render(corners, 400, 1)                 # its aspect ratio, before the edge width is known
    w = h * probe.shape[1] / probe.shape[0]
    size = (2 * x0 + w, CELL_SIZE[1])               # as wide on the right as the labels on the left: the cell centred
    width = CELL_WIDTH or size[0] / (2 * TEXT_WIDTH)
    printed = h * width * TEXT_WIDTH / size[0]      # printed image height (inches)
    image = render(corners, RENDER_HEIGHT, EDGE_WIDTH / 72 * RENDER_HEIGHT / printed, fluid)
    w = h * image.shape[1] / image.shape[0]
    fig, ax = canvas(size)
    ax.imshow(image, extent=(x0, x0 + w, y0, y0 + h), interpolation="lanczos")
    at = projector(corners, image.shape, (x0, y0, w, h))
    low, high = corners.reshape(-1, 3).min(axis=0), corners.reshape(-1, 3).max(axis=0)
    channel = float(settings["channel_width"])
    if DIMENSION_ROOM:
        dimensions(ax, at, low, high, channel)
    if PHASE_LABELS:
        side = high[0] - low[0]
        for text, (x, z), height in ((("pore fluid" if BLUES else "pore"), (0.5, 0.5), 0.46),
                                     ("solid", (0.13, 0.75), 0.68)):   # tip on the front face; label height
            tip = at([low[0] + x * side, low[1], low[2] + z * side])
            label = np.array([0.05, y0 + height * h])
            ax.text(*label, text, ha="left", va="bottom", size=TEXT_SIZE)
            ax.plot([label[0], x0 - 0.1, tip[0]], [label[1], label[1], tip[1]], color="black", linewidth=0.6)
            ax.plot(*tip, "o", color="black", markersize=2)
    if TRIAD:
        _, right, up = screen()
        origin, length = np.array([0.2, 0.2]), 0.38   # below the cube's lower-left edge
        for axis, label in enumerate(TRIAD):
            d = np.array([right[axis], up[axis]])
            arrow(ax, origin, origin + length * d, width=1.1, head=7)
            ax.text(*(origin + length * d + 0.13 * d / np.linalg.norm(d)), label, ha="center", va="center",
                    size=TEXT_SIZE)
    save(fig, "figure_2_cell")
    write_float("figure_2_cell", CAPTION_CELL, "fig:rve-geometry", f"{width:.2f}", placement="tb")


def projector(corners, shape, placed):
    """The map from cell coordinates to canvas inches for the image of corners placed at (left, bottom, width,
    height): the image's contents span the projected bounding box, inside the 4-pixel margin render() leaves."""
    _, right, up = screen()
    low, high = corners.reshape(-1, 3).min(axis=0), corners.reshape(-1, 3).max(axis=0)
    box = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    s, t = box @ right, box @ up
    left, bottom, w, h = placed
    rows, columns = shape[:2]

    def at(point):
        u = 4 + (np.dot(point, right) - s.min()) / np.ptp(s) * (columns - 9)
        v = 4 + (np.dot(point, up) - t.min()) / np.ptp(t) * (rows - 9)
        return np.array([left + u * w / columns, bottom + v * h / rows])

    return at


def dimensions(ax, at, low, high, channel):
    """Dimension lines below the cell's front face (the face y = low, towards the camera): the opening of the
    horizontal channel through it (channel width, in cell lengths) and, below that, the cell side."""
    side = high[0] - low[0]
    front = lambda x, z: at([x, low[1], z])
    a, b = low[0] + (1 - channel) / 2 * side, low[0] + (1 + channel) / 2 * side
    z_bottom, z_opening = low[2], low[2] + (1 - channel) / 2 * side
    line = dict(color="black", linewidth=0.6)
    for level, (x1, x2), text, start in ((0.14, (a, b), rf"${tex.number(channel)}\ell$", z_opening),
                                         (0.42, (low[0], high[0]), r"$\ell$", z_bottom)):
        z = z_bottom - level * side
        for x in (x1, x2):                          # extension lines, dashed over the face
            ax.plot(*np.c_[front(x, start), front(x, z_bottom)], linestyle=(0, (2, 2)), **line)
            ax.plot(*np.c_[front(x, z_bottom), front(x, z - 0.05 * side)], **line)
        middle = (front(x1, z) + front(x2, z)) / 2
        for x in (x1, x2):
            arrow(ax, middle, front(x, z), width=0.9, head=5)
        ax.text(*(middle + [0.0, -0.1]), text, ha="center", va="top", size=TEXT_SIZE)


def section(ax, x0, y0, size, cells, w, zorder=2):
    """The section through the channel centres of a stack of cells of side size (inches), lower-left corner at
    (x0, y0): the pores (FLUID_FILL) behind the solid (SOLID_FILL, outlined), which runs in two columns either side of
    the vertical channel, broken by each cell's horizontal channel (width w in cell lengths)."""
    a, b = (1 - w) / 2, (1 + w) / 2
    ax.add_patch(Rectangle((x0, y0), size, cells * size, facecolor=FLUID_FILL, edgecolor="none", zorder=zorder))
    bounds = [0.0, *np.ravel([[i + a, i + b] for i in range(cells)]), float(cells)]   # solid between channels
    for left, right in ((0.0, a), (b, 1.0)):
        for low, high in zip(bounds[::2], bounds[1::2]):
            ax.add_patch(Rectangle((x0 + left * size, y0 + low * size), (right - left) * size, (high - low) * size,
                                   facecolor=SOLID_FILL, edgecolor="black", linewidth=0.8, zorder=zorder))


def cell_section(ax, origin, side, mapping=lambda X, Y: (X, Y)):
    """A cell's section through the channel centres under mapping (of the unit cell), with its lower-left corner at
    origin and reference side length side (inches): the pores behind four corner blocks of solid."""
    place = lambda p: np.column_stack(mapping(p[:, 0], p[:, 1])) * side + origin

    def loop(x0, x1, y0, y1):
        t = np.linspace(0, 1, 60)
        return np.r_[np.c_[x0 + (x1 - x0) * t, 0 * t + y0], np.c_[0 * t + x1, y0 + (y1 - y0) * t],
                     np.c_[x1 + (x0 - x1) * t, 0 * t + y1], np.c_[0 * t + x0, y1 + (y0 - y1) * t]]

    ax.add_patch(Polygon(place(loop(0, 1, 0, 1)), closed=True, facecolor=FLUID_FILL, edgecolor="#555555",
                         linewidth=0.7, zorder=2))
    a, b = (1 - SCALES_CHANNEL) / 2, (1 + SCALES_CHANNEL) / 2
    for x0, x1 in ((0, a), (b, 1)):
        for y0, y1 in ((0, a), (b, 1)):
            ax.add_patch(Polygon(place(loop(x0, x1, y0, y1)), closed=True, facecolor=SOLID_FILL, edgecolor="black",
                                 linewidth=0.8, zorder=3))


def homogeneous_map(X, Y):
    """The homogeneous part drawn: uniform compression, so solid and pores shorten alike."""
    return X, SCALES_STRETCH * Y


def current_map(X, Y):
    """The homogeneous part plus an illustrative periodic fluctuation: zero at the corners and the same on opposite
    faces. It keeps the solid's height, as a nearly incompressible solid would, and flattens the horizontal channel."""
    size = SCALES_FLUCTUATION
    x, y = homogeneous_map(X, Y)
    return (x + size["D"] * np.sin(2 * np.pi * Y) - size["B"] * np.sin(2 * np.pi * X),
            y + size["A"] * np.sin(2 * np.pi * Y) + size["E"] * np.sin(2 * np.pi * X))


def figure_1_scales():
    """The transition between the scales, as a zoom: from one point of the macroscale continuum to its cell, with what
    passes each way between the zoom's two lines; then the ALE map of that cell. Typeset by LaTeX with the manuscript's
    fonts when LaTeX is available (SCALES_LATEX), so the symbols are those of the equations."""
    latex = shutil.which("latex") is not None
    with plt.rc_context(SCALES_LATEX if latex else {}):
        fig, ax = canvas(SCALES_SIZE)
        bold = (lambda name: rf"\bm{{{name}}}") if latex else (lambda name: rf"\boldsymbol{{{name}}}")
        F_bar, P_bar, X_bar, u_bar = (rf"\overline{{{bold(n)}}}" for n in "FPXu")
        X, chi = bold("X"), bold(r"\chi") + "_A"

        def label(at, text, ha="center", va="center", size=TEXT_SIZE, **options):
            ax.text(*at, text, ha=ha, va=va, size=size, zorder=8, **options)

        # The row of images, all the same height, and the text above and below it.
        low, high = 0.62, 3.02
        middle, side = (low + high) / 2, high - low
        x_body, x_ref, x_cur = (0.2, 3.1), 5.4, 10.2         # the body's extent; the cells' left edges
        below, rule = 0.17, high + 0.28
        for extent, text in ((x_body, "Macroscale"), ((x_ref, x_cur + side), "Microscale")):
            ax.plot(extent, (rule, rule), color="#555555", linewidth=0.8, zorder=2)
            label((sum(extent) / 2, rule + 0.09), text, va="baseline", size=TEXT_SIZE + 1)

        # (a) The body: a continuum, with no pores, and the point that the zoom starts from.
        centre = np.array([sum(x_body) / 2, middle])
        angle = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        radius = 1 + 0.07 * np.cos(2 * angle - 0.5) + 0.05 * np.cos(3 * angle + 1.0)
        outline = np.c_[np.cos(angle), np.sin(angle)] * radius[:, None]
        outline = centre + (outline - outline.mean(axis=0)) * np.array([x_body[1] - x_body[0], side]) / np.ptp(outline, axis=0)
        ax.add_patch(Polygon(outline, closed=True, facecolor=MACRO_FILL, edgecolor="black", linewidth=1.1, zorder=2))
        half = SCALES_POINT / 2
        point = np.array([centre[0] + 0.8, middle])
        ax.add_patch(Rectangle(point - half, 2 * half, 2 * half, facecolor="white", edgecolor="black", linewidth=1.1,
                               zorder=6))
        ax.plot(*point, "o", color="black", markersize=2.6, zorder=7)
        label(point + np.array([-0.33, 0.0]), rf"${X_bar}$")
        label(centre + np.array([-0.62, 0.0]), rf"${u_bar},\ \overline{{p}}$", size=TEXT_SIZE + 1)
        label((centre[0], below), r"(a) Continuum body $\overline{\Omega}$", va="baseline", size=TEXT_SIZE)

        # The zoom: two thin lines from the point to the cell, with what passes each way between them.
        near = point[0] + half
        cone = np.array([[near, middle + half], [x_ref, high], [x_ref, low], [near, middle - half]])
        for corner in (1, 2):
            ax.plot(*np.c_[cone[0 if corner == 1 else 3], cone[corner]], color="black", linewidth=0.9, zorder=4)
        start, end = near + 0.1, x_ref - 0.07
        for y, sense, text, va in ((middle + 0.09, 1, rf"${F_bar},\ \overline{{p}},\ {bold('g')}_p$", "bottom"),
                                   (middle - 0.09, -1, rf"${P_bar},\ m_f,\ {bold('Q')}_0$", "top")):
            arrow(ax, (start, y) if sense > 0 else (end, y), (end, y) if sense > 0 else (start, y), colour="black",
                  width=1.1, head=9)
            label((x_ref - 0.92, y + 0.08 * sense), text, va=va)

        # (b) The reference cell.
        cell_section(ax, np.array([x_ref, low]), side)
        ax.add_patch(Rectangle((x_ref, low), side, side, facecolor="none", edgecolor="black", linewidth=1.1, zorder=6))
        block = (1 - SCALES_CHANNEL) / 2 * side
        label((x_ref + block / 2, low + block / 2), r"$\Omega_s$")
        label((x_ref + side / 2, middle), r"$\Omega_f$")
        label((x_ref + side / 2, below), rf"(b) Reference cell $\Omega$, coordinates ${X}$", va="baseline")

        # The ALE map, and (c) the current cell, whose dashed outline is the imposed deformation alone.
        start, end = x_ref + side + 0.16, x_cur - 0.16
        arrow(ax, (start, middle), (end, middle), width=1.1, head=10)
        label(((start + end) / 2, middle + 0.09), rf"ALE map ${chi}$", va="bottom")
        label(((start + end) / 2, middle - 0.09), rf"${chi}={F_bar}{X}+\widetilde{{{bold('d')}}}_A$", va="top")
        height = SCALES_STRETCH * side
        y_cur = middle - height / 2
        cell_section(ax, np.array([x_cur, y_cur]), side, current_map)
        ax.add_patch(Rectangle((x_cur, y_cur), side, height, facecolor="none", edgecolor="#333333", linewidth=0.9,
                               linestyle=(0, (4, 2.5)), zorder=6))
        label((x_cur + side, y_cur + height + 0.1), rf"${F_bar}{X}$", ha="right", va="bottom")
        label((x_cur + side / 2, below), rf"(c) Current cell, coordinates ${bold('x')}$", va="baseline")

        save(fig, "figure_1_scales")
    write_float("figure_1_scales", CAPTION_SCALES, "fig:scales", SCALES_WIDTH, placement="tb", run=False)


def figure_setup(settings):
    """(a) The resolved column beside (b) the FE² column, the cell at one point and the load histories."""
    with np.load(FE2 / "fe2.npz", allow_pickle=False) as d:
        macro = json.loads(str(d["metadata"]))["macro"]
        Z_points = np.asarray(d["Z_points"], dtype=float)
    H, ell, w = float(macro["height"]), float(settings["cell_length"]), float(settings["channel_width"])
    cells = int(round(H / ell))
    ratio = macro["grading"] ** (-1.0 / (macro["elements"] - 1))
    lengths = ratio ** np.arange(macro["elements"])
    edges = np.r_[0.0, np.cumsum(lengths)] / lengths.sum() * H
    heights = ell * np.arange(cells + 1)
    load = f"${LOAD}(t)$"

    fig, ax = canvas(SETUP_SIZE)
    BASE, HEIGHT = 0.75, 3.15                      # inches: the columns' base and height
    size = HEIGHT / cells                          # one cell (inches)
    Z = lambda z: BASE + z / H * HEIGHT
    label_right = dict(ha="left", va="center", size=TEXT_SIZE, linespacing=1.25)
    label_left = dict(ha="right", va="center", size=TEXT_SIZE, linespacing=1.25)

    # (a) The resolved column: cell numbers and height on the left.
    XA = 1.75
    section(ax, XA, BASE, size, cells, w)
    rule = XA - 0.16
    ax.plot([rule, rule], [BASE, Z(H)], color="black", linewidth=0.8)
    for z in heights:
        ax.plot([rule - 0.07, rule + 0.07], [Z(z), Z(z)], color="black", linewidth=0.8)
    for i in range(cells):
        ax.text(rule - 0.12, Z((i + 0.5) * ell), f"{i + 1}", ha="right", va="center", size=TEXT_SIZE - 2)
    dim, middle = rule - 0.62, Z(H / 2)
    for end in (BASE, Z(H)):
        arrow(ax, (dim, middle), (dim, end), width=1.0, head=9)
    ax.text(dim - 0.08, middle, rf"$H={tex.number(H)}\,$m", rotation=90, ha="right", va="center", size=TEXT_SIZE)

    # Platen with a gap over the vertical channel; the load on it and the outflow through the gap.
    top, gap = Z(H), (XA + (1 - w) / 2 * size, XA + (1 + w) / 2 * size)
    for a, b in ((XA - 0.08, gap[0]), (gap[1], XA + size + 0.08)):
        ax.add_patch(Rectangle((a, top), b - a, 0.08, facecolor="#555555", edgecolor="black", linewidth=0.6,
                               zorder=4))
    arrow(ax, (np.mean(gap), top - 0.3), (np.mean(gap), top + 0.42), colour=FLUID, width=1.4, head=8)
    for x in (XA + 0.04, XA + size - 0.04):
        arrow(ax, (x, top + 0.5), (x, top + 0.08))
    ax.text(XA + size / 2, top + 0.55, load, ha="center", va="bottom", size=TEXT_SIZE + 1)

    # Base: the mirror plane, dash-dotted.
    ax.plot([XA - 0.12, XA + size + 0.12], [BASE - 0.05, BASE - 0.05], color="black", linewidth=1.4,
            linestyle=(0, (6, 2, 1, 2)))

    # Boundary conditions on the right, with leaders.
    tx = XA + size + 0.5
    ax.text(tx, top + 0.1, "rigid, frictionless platen;\npore openings free of traction", **label_right)
    ax.plot([XA + size + 0.1, tx - 0.06], [top + 0.04, top + 0.1], color="black", linewidth=0.7)
    y_side = Z((cells // 2 + 0.5) * ell)           # a horizontal channel's opening
    ax.text(tx, y_side, "periodic sides", **label_right)
    ax.plot([XA + size + 0.02, tx - 0.06], [y_side, y_side], color="black", linewidth=0.7)
    ax.text(tx, BASE - 0.05, r"mirror plane, $v_Z=0$", **label_right)
    ax.plot([XA + size + 0.14, tx - 0.06], [BASE - 0.05, BASE - 0.05], color="black", linewidth=0.7)
    ax.text(XA + size / 2, 0.08, "(a) Resolved FSI", ha="center", va="bottom", size=TEXT_SIZE + 1)

    # (b) The FE² column on the same height scale: elements, integration points, resolved cells (ticks, left).
    XB = 7.1
    ax.add_patch(Rectangle((XB, BASE), size, HEIGHT, facecolor="#ededed", edgecolor="black", linewidth=1.2))
    for z in edges[1:-1]:
        ax.plot([XB, XB + size], [Z(z), Z(z)], color="black", linewidth=0.7)
    ax.plot(np.full_like(Z_points, XB + size / 2), Z(Z_points), "o", color="black", markersize=2.3, zorder=3)
    for z in heights:
        ax.plot([XB - 0.13, XB], [Z(z), Z(z)], color="black", linewidth=0.8)
    ax.plot([XB - 0.15, XB + size + 0.15], [BASE, BASE], color="black", linewidth=2.2, solid_capstyle="butt")
    for x in np.arange(XB - 0.03, XB + size + 0.15 + 1e-9, 0.12):
        ax.plot([x, x - 0.12], [BASE, BASE - 0.12], color="black", linewidth=1.0)
    for x in (XB + 0.04, XB + size - 0.04):
        arrow(ax, (x, top + 0.42), (x, top))
    ax.text(XB + size / 2, top + 0.47, load, ha="center", va="bottom", size=TEXT_SIZE + 1)
    ax.text(XB - 0.28, top - 0.05, r"drained, $\overline{p}=0$", **label_left)
    ax.text(XB - 0.28, Z(0.6 * H), "uniaxial strain", **label_left)
    ax.text(XB - 0.28, BASE - 0.06, r"$\overline{u}_Z=0$, no flux", **label_left)
    ax.text(XB + size / 2, 0.08, r"(b) FE$^2$", ha="center", va="bottom", size=TEXT_SIZE + 1)

    # One integration point, its cell and what the scales exchange.
    k = int(np.abs(Z_points - HIGHLIGHT_Z * H).argmin())
    point, ring = np.array([XB + size / 2, Z(Z_points[k])]), 0.075
    ax.add_patch(Circle(point, ring, facecolor="none", edgecolor=ACCENT, linewidth=1.6, zorder=6))
    side, pad = 1.75, 0.12                         # the cell (inches) and its frame's margin
    cx, cy = 9.1, point[1] - side / 2              # lower-left corner of the cell
    section(ax, cx, cy, side, 1, w)
    ax.add_patch(Rectangle((cx - pad, cy - pad), side + 2 * pad, side + 2 * pad, facecolor="none", edgecolor=ACCENT,
                           linewidth=1.4))
    ax.text(cx + side / 2, cy + side + pad + 0.1, rf"cell at every point, side $\ell={tex.number(ell)}\,$m",
            ha="center", va="bottom", size=TEXT_SIZE)
    # The exchange links the point and its cell: inputs along the upper arrow, outputs back along the lower one.
    start, end = point[0] + ring + 0.12, cx - pad - 0.08
    for y, sense, text, va in ((point[1] + 0.14, 1, r"$\overline{\mathbf{F}},\ \overline{p},\ \mathbf{g}_p$", "bottom"),
                               (point[1] - 0.14, -1, r"$\overline{\mathbf{P}},\ m_f,\ \mathbf{Q}_0$", "top")):
        arrow(ax, (start, y) if sense > 0 else (end, y), (end, y) if sense > 0 else (start, y), colour=ACCENT,
              width=1.3, head=10)
        ax.text((start + end) / 2 + 0.1, y + 0.08 * sense, text, ha="center", va=va, size=TEXT_SIZE)

    # The load histories, step and cyclic (period T), top right.
    cycles = 5
    s = np.linspace(0, cycles, 500)
    for j, (name, t, value, ticks, labels) in enumerate((
            ("step", np.array([-0.25, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0]), [0], ["0"]),
            ("cyclic", s, 0.5 * (1 - np.cos(2 * np.pi * s)), [0, 1, cycles], ["0", "$T$", f"{cycles}$T$"]))):
        inset = fig.add_axes([(8.95 + 2.05 * j) / SETUP_SIZE[0], 3.5 / SETUP_SIZE[1], 1.7 / SETUP_SIZE[0],
                              0.95 / SETUP_SIZE[1]])
        inset.plot(t, value, color="black", linewidth=1.8)
        inset.set_ylim(-0.06, 1.15)
        inset.set_yticks([0, 1])
        inset.set_yticklabels(["0", rf"${LOAD}_\mathrm{{max}}$"] if j == 0 else [])
        inset.set_xticks(ticks)
        inset.set_xticklabels(labels)
        inset.set_xlabel("$t$", labelpad=0, size=TEXT_SIZE)
        inset.set_title(f"{name} load", size=TEXT_SIZE, pad=5)
        inset.tick_params(direction="in", length=4, width=0.8, labelsize=TEXT_SIZE - 1)
        for spine in ("top", "right"):
            inset.spines[spine].set_visible(False)

    save(fig, "figure_5_setup")
    per_cell = np.histogram(Z_points, bins=heights)[0].tolist()
    print(f"    {cells} resolved cells; {len(Z_points)} FE² points, {per_cell} per resolved cell (base to top); "
          f"point {k} (Z = {Z_points[k]:.4f} m) drawn with its cell")
    caption = tex.fill(CAPTION_SETUP, cells=tex.words(cells), height=tex.number(H), ell=tex.number(ell),
                       elements=tex.words(int(macro["elements"])), k=int(macro["displacement_order"]),
                       km=int(macro["pressure_order"]), points=tex.words(len(Z_points)))
    write_float("figure_5_setup", caption, "fig:consolidation-setup", SETUP_WIDTH,
                placement="b")  # bottom: at the top it would float above the Section 5.3 heading


def main():
    plt.rcParams.update(STYLE)
    figure_1_scales()
    mesh, settings = resolved_mesh()
    figure_cell(mesh, settings)
    figure_setup(settings)


def command_line():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help=f"figure folder (default {OUTPUT})")
    args = parser.parse_args()
    OUTPUT = args.output or OUTPUT


if __name__ == "__main__":
    command_line()
    main()
