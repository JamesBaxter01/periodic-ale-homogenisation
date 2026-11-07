"""Paper table: equilibrated effective parameters of the cell at states from an FE² consolidation run.

Press Run (or pass the history on the command line). Reads an FE² consolidation history and, for
each of STATES (a name, a time and a height in the column), takes the nearest material point's
deformation F and pore pressure p at the nearest accepted time. A cell with the run's own settings
(material, mesh, solver) is brought to each state by PATH_STEPS equilibrated steps of PROBE_DT from
rest, with no pressure gradient, and the tangent of the last step gives the cell's equilibrated,
rate-independent parameters there, in the cell's axes (3 = the column's axis):

    permeability  k_axial = -dQ_3/dG_3, k_lateral = -dQ_1/dG_1     (permeability / fluid viscosity)
    Biot          alpha_axial = -dP_33/dp, alpha_lateral = -dP_11/dp, 1/M = dm/dp
    stiffness     C_3333, C_1313 = dP/dF

These do not depend on the skeleton's viscosity (each state is equilibrated), so one table serves
the elastic and viscoelastic runs. The run's own algorithmic tangents are not used: they describe
one step of the run and depend on its step size.

Writes OUTPUT/table_5_consolidation_parameters.csv (every value) and .tex (booktabs: one row per state, the
parameters in columns grouped by permeability, Biot coupling and stiffness; the rest values, then
the change from rest; caption CAPTION, label LABEL), and prints both. --table-only rebuilds the .tex
from the saved .csv without solving anything, for changes to the presentation.
"""

import argparse
import json
import re
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale", ROOT / "examples" / "multiscale"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
# Ramp load (10 s): full_elastic_20261026_201037 (elastic). The paper's Table 5 uses the step-load runs.
DATA = ROOT / "results" / "consolidation" / "full_viscoelastic_step_20261026_201037" / "fe2.npz"
OUTPUT = ROOT / "results" / "paper" / "tables"
# (state, cell, time (s), height Z (m)); time None = the end of the run. The first state is the reference.
STATES = [
    ("Rest", "", 0.0, 0.5),
    ("t = 10 s", "bottom", 10.0, 0.05),  # nearly undrained: high pore pressure, little compression
    ("t = 10 s", "top", 10.0, 0.95),     # draining: compressed, pore pressure falling
    ("Drained", "all cells", None, 0.95),   # fully drained: every cell is in the same state (F_33 within 1e-4)
]
GROUPS = [("Permeability", ["k_axial", "k_lateral"]),
          ("Biot coupling", ["alpha_axial", "alpha_lateral", "inverse_M"]),
          ("Stiffness", ["C_3333", "C_1313"])]
ROWS = [row for _, rows in GROUPS for row in rows]
FLUID_VISCOSITY = 1.0             # the run's mu_f (Pa s): the cell's k is permeability / viscosity
# Table columns: (symbol, factor the stored value is multiplied by, unit); unit None = shares the previous
# column's unit. Indices are the cell's axes (3 = the column's axis, 1 = across it).
COLUMNS = dict(k_axial=(r"$\kappa_{33}$", FLUID_VISCOSITY / 1e-6, r"$10^{-6}\,\mathrm{m^2}$"),
               k_lateral=(r"$\kappa_{11}$", FLUID_VISCOSITY / 1e-6, None),
               alpha_axial=(r"$\alpha_{33}$", 1, ""), alpha_lateral=(r"$\alpha_{11}$", 1, ""),
               inverse_M=(r"$1/M$", 1e3, r"$10^{-3}\,\mathrm{Pa^{-1}}$"),
               C_3333=(r"$C^{\mathrm{dr}}_{3333}$", 1, "Pa"), C_1313=(r"$C^{\mathrm{dr}}_{1313}$", 1, "Pa"))
DECIMALS = dict(rest=2, F=3, p=2, change=1)  # decimal places: rest values, F_33, p (Pa), change from rest (%)
CAPTION = (r"Equilibrated response derivatives of the cell at states of the FE$^2$ run of "
           r"Figure~\ref{fig:consolidation<<label>>} (<<run>>). Each entry is a derivative over the last of <<steps>> "
           r"<<dt>> increments that bring a fresh cell from rest to that $F_{33}$ and $\overline p$, with "
           r"$\kappa_{ii}=-\mu_f\,\partial Q_{0,i}/\partial G_{p,i}$ (referential), "
           r"$\alpha_{ii}=-\partial\overline P_{ii}/\partial\overline p$, $1/M=\partial m_f/\partial\overline p$ and "
           r"$C^{\mathrm{dr}}=\partial\overline{\bm P}/\partial\overline{\bm F}$. Index 3 is along the column and 1 "
           r"across it.")
LABEL = "tab:consolidation_parameters"
SUFFIX = ""                       # appended to the file names and labels (e.g. "_ramp"); --suffix
PROBE_DT = 1000.0                 # equilibrated steps (s): converged to 0.2 % between 100 and 1000 s
PATH_STEPS = 5                    # equilibrated steps from rest to each state (the last lands on it)
WORKERS = 4                       # one cell process per state, up to this many
# ---------------------------------------------------------------------------


QUANTITIES = {  # name: (tangent row, column, sign)
    "k_axial": (12, 12, -1),
    "k_lateral": (10, 10, -1),
    "alpha_axial": (8, 9, -1),
    "alpha_lateral": (0, 9, -1),
    "inverse_M": (9, 9, 1),
    "C_3333": (8, 8, 1),
    "C_1111": (0, 0, 1),
    "C_1133": (0, 8, 1),
    "C_1313": (2, 2, 1),
}


def referential(tangents, inputs):
    """The tangents with the flux columns taken with respect to the referential gradient G = F^T g: the cell's
    input is the spatial g, so dQ/dG = dQ/dg F^-T at fixed F. tangents (..., 13, 13), inputs (..., 13)."""
    F = np.asarray(inputs)[..., :9].reshape(*np.shape(inputs)[:-1], 3, 3)
    tangents = np.array(tangents, dtype=float)
    tangents[..., 10:] = tangents[..., 10:] @ np.swapaxes(np.linalg.inv(F), -1, -2)
    return tangents


def quantities(tangents, inputs):
    """Every quantity of QUANTITIES for tangents of shape (..., 13, 13) at inputs (..., 13); the permeabilities are
    referential (see referential)."""
    tangents = referential(tangents, inputs)
    return {name: sign * tangents[..., row, column] for name, (row, column, sign) in QUANTITIES.items()}


def states(path):
    """[(name, time, Z, inputs)] for STATES, from the history: F and p of the nearest point, G = 0."""
    with np.load(path, allow_pickle=False) as data:
        t, X, Z = data["time"], data["inputs"], data["Z_points"]
        metadata = json.loads(str(data["metadata"]))
    chosen = []
    for state, cell, when, height in STATES:
        n = len(t) - 1 if when is None else int(np.argmin(np.abs(t - when)))
        q = int(np.argmin(np.abs(Z - height)))
        x = X[n, q].copy()
        x[10:] = 0.0  # equilibrated: no driving gradient
        chosen.append(((state, cell), float(t[n]), float(Z[q]), x))
    return chosen, metadata


def equilibrated_parameters(chosen, metadata):
    from geometry import square_geo
    from multiscale.interface import build_model

    settings = dict(metadata["cell"])
    settings["geometry"] = partial(square_geo, metadata.get("channel_width", 0.33))
    settings.pop("compile_tangent", None)  # same results; interpreted keeps the memory down
    targets = np.array([x for *_, x in chosen])
    rest = np.r_[np.eye(3).ravel(), 0.0, np.zeros(3)]
    with build_model("fe", len(chosen), workers=min(WORKERS, len(chosen)), **settings) as model:
        model.evaluate(np.tile(rest, (len(chosen), 1)), PROBE_DT)
        model.commit()
        for s in np.linspace(0.0, 1.0, PATH_STEPS + 1)[1:]:
            response = model.evaluate(rest + s * (targets - rest), PROBE_DT)
            model.commit()
    return quantities(response.tangent, targets)  # the last step: at each state (re-evaluating there is redundant)


def write_csv(chosen, values, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"table_5_consolidation_parameters{SUFFIX}.csv").open("w") as stream:
        stream.write("quantity," + ",".join(f"{state}|{cell}" for (state, cell), *_ in chosen) + "\n")
        stream.write("time_s," + ",".join(f"{t:.6g}" for _, t, _, _ in chosen) + "\n")
        stream.write("Z_m," + ",".join(f"{z:.6g}" for _, _, z, _ in chosen) + "\n")
        stream.write("F_zz," + ",".join(f"{x[8]:.6g}" for *_, x in chosen) + "\n")
        stream.write("p_Pa," + ",".join(f"{x[9]:.6g}" for *_, x in chosen) + "\n")
        for row in ROWS:
            stream.write(f"{row}," + ",".join(f"{v:.6g}" for v in values[row]) + "\n")


def read_csv(output):
    """The table saved by write_csv: column headers (state, cell) and {quantity: values}."""
    lines = (output / f"table_5_consolidation_parameters{SUFFIX}.csv").read_text().splitlines()
    headers = [tuple(column.split("|")) for column in lines[0].split(",")[1:]]
    table = {line.split(",")[0]: np.array([float(v) for v in line.split(",")[1:]]) for line in lines[1:]}
    return headers, table


def latex(headers, table):
    """The paper table (booktabs), one row per state with the parameters in columns grouped by
    permeability, Biot coupling and stiffness: the rest values, then the change from rest."""
    n = len(ROWS)
    spans, units, column = [], [], 4
    for group, rows in GROUPS:
        spans.append((column, column + len(rows) - 1, group))
        column += len(rows)
        for k, row in enumerate(rows):
            unit = COLUMNS[row][2]
            if unit is None:
                continue
            width = 2 if k + 1 < len(rows) and COLUMNS[rows[k + 1]][2] is None else 1
            units.append(rf"\multicolumn{{2}}{{c}}{{{unit}}}" if width == 2 else unit)
    d = DECIMALS
    pressure = [rf"${abs(p) if abs(p) < 0.5 * 10 ** -d['p'] else p:.{d['p']}f}$" for p in table["p_Pa"]]

    def line(i, cells):
        state, cell = headers[i]
        state = re.sub(r"^t = (\S+) s$", r"$t=\1\\,\\mathrm s$", state)  # the CSV's "t = 10 s" typeset
        return (state + (f", {cell}" if cell else "") + rf" & ${table['F_zz'][i]:.{d['F']}f}$ & {pressure[i]} & "
                + " & ".join(cells) + r" \\")

    lines = [r"\begin{table}[pos=htb]", r"\centering", rf"\caption{{{caption()}}}", rf"\label{{{LABEL}{SUFFIX}}}",
             r"\small", r"\renewcommand{\arraystretch}{1.2}", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{lcc" + "c" * n + "}", r"\toprule",
             " & & & " + " & ".join(rf"\multicolumn{{{b - a + 1}}}{{c}}{{{group}}}" for a, b, group in spans) + r" \\",
             " ".join(rf"\cmidrule(lr){{{a}-{b}}}" for a, b, _ in spans),
             r"State & $F_{33}$ & $\overline p$ & " + " & ".join(COLUMNS[row][0] for row in ROWS) + r" \\",
             r" & & (Pa) & " + " & ".join(units) + r" \\", r"\midrule",
             line(0, [rf"${table[row][0] * COLUMNS[row][1]:.{d['rest']}f}$" for row in ROWS]),
             r"\addlinespace", rf"\multicolumn{{{n + 3}}}{{l}}{{\textit{{Change from rest}}}} \\"]
    for i in range(1, len(headers)):
        lines.append(line(i, [rf"${100 * (table[row][i] / table[row][0] - 1):+.{d['change']}f}\%$" for row in ROWS]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return (f"% Equilibrated cell parameters (consolidation_parameter_table.py): the rest values, then the change\n"
            f"% from rest at each state. Needs \\usepackage{{booktabs}}.\n" + "\n".join(lines) + "\n")


def main():
    started = time.perf_counter()
    chosen, metadata = states(DATA)
    for (state, cell), t, z, x in chosen:
        print(f"{state + ' ' + cell:20s} t = {t:8.4g} s, Z = {z:.3f} m: F_zz = {x[8]:.4f}, p = {x[9]:.3f} Pa", flush=True)
    values = equilibrated_parameters(chosen, metadata)
    print(f"\n{'':14s}" + "".join(f"{state + ' ' + cell:>22s}" for (state, cell), *_ in chosen))
    for row in ROWS:
        print(f"{row:14s}" + "".join(f"{v:11.4g} ({100 * (v / values[row][0] - 1):+6.1f} %)" for v in values[row]))
    write_csv(chosen, values, Path(OUTPUT))
    write_table()
    print(f"({(time.perf_counter() - started) / 60:.1f} min)")


def caption():
    """CAPTION for DATA's run: its skeleton and load (step, or ramp over RAMP_TIME)."""
    with np.load(DATA, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    ramp = metadata["macro"]["ramp_time"]
    chosen, _ = states(DATA)
    z = {cell: height for (_, cell), _, height, _ in chosen}
    heights = [f"{z['bottom']:.3f}", f"{z['top']:.3f}"]
    skeleton = metadata.get("skeleton", "elastic")
    if ramp > 0:
        run, early = f"{skeleton} skeleton, traction ramped over ${ramp:g}\\,\\mathrm{{s}}$", "at the end of the ramp ($t=10\\,\\mathrm{s}$)"
    else:
        run, early = f"{skeleton} skeleton, step load", "$10\\,\\mathrm{s}$ after loading"
    return (CAPTION.replace("<<run>>", run).replace("<<early>>", early).replace("<<label>>", SUFFIX.replace("_", "-"))
            .replace("<<z_bottom>>", heights[0]).replace("<<z_top>>", heights[1])
            .replace("<<steps>>", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][PATH_STEPS] if PATH_STEPS <= 10 else str(PATH_STEPS)).replace("<<dt>>", f"${PROBE_DT:g}\\,\\mathrm{{s}}$"))


def write_table():
    """table_5_consolidation_parameters.tex from the saved CSV (no cell solves): for presentation changes."""
    text = latex(*read_csv(Path(OUTPUT)))
    (Path(OUTPUT) / f"table_5_consolidation_parameters{SUFFIX}.tex").write_text(text)
    print("\n" + text + f"Saved {Path(OUTPUT) / f'table_5_consolidation_parameters{SUFFIX}.tex'} and .csv")


def command_line():
    global DATA, OUTPUT, SUFFIX
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="?", type=Path, help="FE² consolidation history (default DATA)")
    parser.add_argument("--output", type=Path, help=f"table folder (default {OUTPUT})")
    parser.add_argument("--table-only", action="store_true", help="rebuild the .tex from the saved .csv only")
    parser.add_argument("--suffix", help="appended to the file names and labels, e.g. _ramp (default SUFFIX)")
    args = parser.parse_args()
    SUFFIX = SUFFIX if args.suffix is None else args.suffix
    DATA = args.data or DATA
    OUTPUT = args.output or OUTPUT
    return args.table_only


if __name__ == "__main__":
    if command_line():
        write_table()
    else:
        main()
