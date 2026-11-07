"""Every number quoted in the text of Sections 4, 5.1, 5.2 and 5.3 of the paper, recomputed from the saved results.

Press Run (or python run.py paper). No cell solves: it reads the verification analysis, the Figure 3-4 data, the FE² and
resolved histories in results/, the Table 5 values and the cell replays, replays the step-load FE² runs from their stored
responses (Section 5.3.3), and solves the closed-form Biot law on the two macroscale meshes of Section 5.3.1.

Writes OUTPUT/paper_numbers.md with one row per number: what it is, the value as printed in the paper, the value
recomputed here, and whether they agree at the printed precision. Statements that are not numbers from a run (the
hardware, for example) are listed as such. Exits with status 1 if a number disagrees, so run.py paper reports it.
"""

import argparse
import contextlib
import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "examples" / "microscale", ROOT / "examples" / "multiscale", ROOT / "verification"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_consolidation import fe2_cell_averages  # noqa: E402
from compare_cyclic import cycle_metrics, differences  # noqa: E402
from compare_cyclic import read as read_cyclic  # noqa: E402
from consolidation_error_table import CELL_START, statistics  # noqa: E402
from cyclic_dissipation_split import split  # noqa: E402
from multiscale.column import Column1D  # noqa: E402
from multiscale.driver import run as run_macro  # noqa: E402
from multiscale.interface import INPUT_NAMES, build_model  # noqa: E402
from multiscale.stationary_cell import STEPS  # noqa: E402
from plot_consolidation_multiscale import load  # noqa: E402
from plot_paper_cells import PANELS  # noqa: E402
from plot_paper_consolidation import read  # noqa: E402
from plot_paper_figures import SANDSTROM_ALPHA, SANDSTROM_K, departure, load_rates_4, load_stationary_4, stationary_at  # noqa: E402
from replay_fe2 import problem_and_options, replay  # noqa: E402
from resolved_refinement import change, load_resolved, truncated  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
RESULTS = ROOT / "results"
CONFIG = ROOT / "configs" / "verification.json"
ANALYSIS = RESULTS / "verification" / "combined_20261026_201037" / "analysis"
PAPER = RESULTS / "paper"
STEP = {"elastic": RESULTS / "consolidation" / "full_elastic_step_20261026_201037",
        "viscoelastic": RESULTS / "consolidation" / "full_viscoelastic_step_20261026_201037"}
RAMP = RESULTS / "consolidation" / "full_elastic_20261026_201037"
REFINEMENT = RESULTS / "consolidation" / "refinement_20261026_201037"
BASELINES = RESULTS / "consolidation" / "baselines_20261028"
CYCLIC = RESULTS / "cyclic" / "20261026_201037"
TABLE_5 = RESULTS / "paper" / "tables" / "table_5_consolidation_parameters.csv"
OUTPUT = RESULTS / "paper" / "tables"
FINE_ELEMENTS = 160               # the macroscale mesh that the baseline mesh is checked against, Section 5.3.1
COINCIDE = 0.005                  # the two skeletons' settlements "coincide": relative difference below this
PERIODS = (0.2, 2.0)
SKELETONS = ("elastic", "viscoelastic")
# ---------------------------------------------------------------------------


class Report:
    """The rows of the table; each number is compared with its printed text at the printed precision."""

    def __init__(self):
        self.rows, self.failed = [], []

    def section(self, title):
        self.rows.append(title)

    def _add(self, what, paper, computed, agrees):
        self.rows.append((what, paper, computed, agrees))
        if agrees is False:
            self.failed.append(what)

    def number(self, what, paper, value, sig=None, within=None):
        """paper is the text as printed. value agrees if it rounds to it: to the printed decimals; to sig significant
        figures, for "about" values; or, with within, if it is inside that relative distance."""
        value = float(value)
        if within is not None:
            agrees = abs(value - float(paper)) <= within * abs(float(paper))
        elif sig is not None:
            agrees = float(f"{value:.{sig}g}") == float(f"{float(paper):.{sig}g}")
        else:
            decimals = len(paper.split(".")[1]) if "." in paper else 0
            agrees = f"{value:.{decimals}f}" == paper
        self._add(what, paper, f"{value:.0f}" if value == round(value) and abs(value) >= 1 else f"{value:.5g}", agrees)

    def holds(self, what, paper, computed, agrees):
        """An inequality or a qualitative statement, with the value that decides it."""
        self._add(what, paper, computed, bool(agrees))

    def noted(self, what, paper, computed="not from a saved run"):
        """Listed but not compared: a description, not a value in a saved run, or a time that depends on the machine."""
        self._add(what, paper, computed, None)

    def markdown(self):
        lines = ["# Numbers in the text of the paper, recomputed from the saved results", "",
                 "Written by `examples/paper_numbers.py`.", ""]
        for row in self.rows:
            if isinstance(row, str):
                lines += ["", f"## {row}", "", "| Quantity | Paper | Recomputed | Agrees |", "|---|---|---|---|"]
            else:
                mark = {True: "yes", False: "**NO**", None: "–"}[row[3]]
                lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {mark} |")
        return "\n".join(lines) + "\n"


def percent_error(run, key):
    """(time, |FE² - resolved| relative to the resolved final value), as Table 4."""
    fe2, res = run["FE²"], run["resolved"]
    t = res["time"][(res["time"] > 0) & (res["time"] <= fe2["time"][-1])]
    return t, np.abs(np.interp(t, fe2["time"], fe2[key]) - np.interp(t, res["time"], res[key])) / abs(res[key][-1])


def top_cell_error(run):
    """(time, |FE² - resolved| of the top cell's pressure over the traction)."""
    fe2, res = run["FE²"], run["resolved"]
    t = res["time"][(res["time"] > 0) & (res["time"] <= fe2["time"][-1])]
    k = run["cells"] - 1
    return t, np.abs(np.interp(t, fe2["time"], fe2["pressure"][:, k]) - np.interp(t, res["time"], res["pressure"][:, k])) \
        / run["traction"]


def cell_pressures(problem, history, ell, cells, width):
    return np.array([fe2_cell_averages(problem, y, o, ell, cells, width)[0]
                     for y, o in zip(history["y"], history["outputs"])])


def macro_mesh_check(folder, elements):
    """The closed-form Biot law of the run in folder, resolved again on elements macroscale elements (same grading,
    orders and time steps): the statistics of Table 4 of the saved run against the finer one."""
    saved = np.load(folder / "fe2.npz", allow_pickle=False)
    coarse, options, metadata = problem_and_options(saved)
    macro = metadata["macro"]
    fine = Column1D(height=macro["height"], area=macro["area"], elements=elements, traction=coarse.traction,
                    stress_scale=macro["traction"], modulus=metadata["cell"]["G"],
                    displacement_order=macro["displacement_order"], pressure_order=macro["pressure_order"],
                    gauss_points=macro["gauss_points"], grading=macro["grading"])
    options.save_tangents = False
    t_end = float(saved["time"][-1])
    stop = lambda history: history["time"][-1] >= t_end * (1 - 1e-12)  # noqa: E731
    with build_model(metadata["model"], fine.n_points, **metadata["model_settings"]) as material, \
            contextlib.redirect_stdout(io.StringIO()):
        history = run_macro(fine, material, options, stop=stop)
    history = {key: np.array(value) for key, value in history.items()}
    ell, width = metadata["cell"]["rve_length"], metadata.get("channel_width", 0.33)
    run = read(folder)
    resolved = dict(time=history["time"], settlement=history["settlement"],
                    drainage=history["cumulative_outflow"] / macro["area"],
                    pressure=cell_pressures(fine, history, ell, run["cells"], width))
    return statistics(dict(run, resolved=resolved))


def table_5():
    rows = {}
    for line in TABLE_5.read_text().splitlines()[1:]:
        name, *values = line.split(",")
        rows[name] = [float(v) for v in values]
    return rows


def consolidation(report):
    runs = {name: read(folder) for name, folder in STEP.items()}
    stats = {name: statistics(run) for name, run in runs.items()}
    meta = {name: load(folder / "fe2.npz")[1] for name, folder in STEP.items()}
    histories = {name: np.load(folder / "fe2.npz", allow_pickle=False) for name, folder in STEP.items()}
    rests = table_5()
    elastic, viscoelastic = meta["elastic"], meta["viscoelastic"]
    H, ell = elastic["macro"]["height"], elastic["cell"]["rve_length"]
    mu_f, G = elastic["cell"]["mu_f"], elastic["cell"]["G"]
    traction = elastic["macro"]["traction"]
    c_v = rests["k_axial"][0] / mu_f / (rests["inverse_M"][0] + rests["alpha_axial"][0] ** 2 / rests["C_3333"][0])

    report.section("Section 5.3.1: set-up")
    report.number("Column height H (m)", "1", H)
    report.number("Cell length ℓ (m)", "0.1", ell)
    report.number("Cells in the column", "10", runs["elastic"]["cells"])
    report.number("Step traction (Pa)", "5", traction)
    report.number("Run stops when the pore pressure is below this fraction of its peak", "1e-3",
                  elastic["macro"]["pressure_tolerance"], sig=1)
    report.number("Consolidation coefficient at rest c_v (m²/s), from Table 5", "6.3e-4", c_v, sig=2)
    report.number("Drainage time H²/c_v (s)", "1600", H ** 2 / c_v, sig=2)
    report.number("Pressure time across a cell ℓ²/c_v (s)", "16", ell ** 2 / c_v, sig=2)
    report.number("Final compression of the cells (%)", "7.5", 100 * (1 - rests["F_zz"][3]))
    report.number("Fluid viscous time μ_f/G (s)", "0.02", mu_f / G)
    report.number("Skeleton viscous time η_s/G (s)", "0.04", viscoelastic["cell"]["eta_s"] / G)
    for name, folder in STEP.items():
        with np.load(folder / "resolved.npz", allow_pickle=False) as d:
            resolved_meta = json.loads(str(d["metadata"]))
        report.number(f"Resolved mesh, elements ({name} run)", "48537", resolved_meta["elements"])
        report.number(f"Resolved mesh, unknowns ({name} run)", "236206", resolved_meta["unknowns"])
    report.number("FE² macroscale elements", "10", elastic["macro"]["elements"])
    report.number("FE² integration points", "40", elastic["macro"]["elements"] * elastic["macro"]["gauss_points"])
    report.number("FE² displacement order", "3", elastic["macro"]["displacement_order"])
    report.number("FE² pressure order", "2", elastic["macro"]["pressure_order"])
    mesh = macro_mesh_check(BASELINES / "consolidation_linear", FINE_ELEMENTS)
    for label, value in (("settlement", mesh["settlement"][0]), ("drained volume", mesh["drainage"][0]),
                         ("pressure, cells 1 to 9", mesh["lower"][0]), ("pressure, top cell", mesh["top"][0])):
        report.holds(f"Linear Biot law, 10 elements against {FINE_ELEMENTS}: {label}", "within about 1.5 %",
                     f"{100 * value:.2f} %", value <= 0.015)
    report.number("First time step (s)", "1e-6", elastic["time_steps"]["dt_initial"], sig=1)
    report.number("Growth per step (%)", "5", 100 * (elastic["time_steps"]["growth"] - 1))
    for name, steps, end in (("elastic", "415", "3933"), ("viscoelastic", "424", "4361")):
        h = histories[name]
        report.number(f"Steps ({name})", steps, len(h["time"]) - 1)
        report.number(f"End of the run (s, {name})", end, h["time"][-1])
        halved = int(np.sum(h["dt"][2:] / h["dt"][1:-1] < 0.75))
        report.holds(f"Steps halved late in the drainage ({name})", "at most 4", str(halved), halved <= 4)

    report.section("Section 5.3.1: step-load response")
    res_e, res_v = runs["elastic"]["resolved"], runs["viscoelastic"]["resolved"]
    s_10ms = np.interp(0.01, res_e["time"], res_e["settlement"])
    report.number("Elastic settlement at 10 ms, resolved (mm)", "32.5", 1e3 * s_10ms)
    early = res_v["time"] <= 1.0
    report.number("Viscoelastic time to settle as far (ms)", "60",
                  1e3 * np.interp(s_10ms, res_v["settlement"][early], res_v["time"][early]), sig=1)
    a, b = np.interp(0.3, res_e["time"], res_e["settlement"]), np.interp(0.3, res_v["time"], res_v["settlement"])
    report.holds(f"Skeletons' settlements at t = 0.3 s, relative difference (below {100 * COINCIDE:g} %)", "coincide",
                 f"{100 * abs(a - b) / a:.2f} %", abs(a - b) <= COINCIDE * a)
    for name, res in (("elastic", res_e), ("viscoelastic", res_v)):
        report.number(f"Settlement at t = 1 s (mm, {name})", "41", 1e3 * np.interp(1.0, res["time"], res["settlement"]),
                      sig=2)
        report.number(f"Drained at t = 1 s, of the final volume (%, {name})", "1.5",
                      100 * np.interp(1.0, res["time"], res["drainage"]) / res["drainage"][-1])
    report.holds("Viscoelastic pore pressure does not overshoot the traction (resolved, largest cell over traction)",
                 "no overshoot", f"{res_v['pressure'].max() / traction:.4f}", res_v["pressure"].max() <= traction)
    report.number("Settlement difference, largest (%)", "0.86", 100 * max(s["settlement"][0] for s in stats.values()))
    report.number("Drained volume difference, largest (%)", "1.63", 100 * max(s["drainage"][0] for s in stats.values()))
    report.number("Pressure difference, cells 1 to 9, largest", "0.027", max(s["lower"][0] for s in stats.values()))
    report.number("Pressure statistics start (ms)", "2", 1e3 * CELL_START)
    report.number("Cell thickness over the drainage layer at 2 ms", "100", ell / np.sqrt(c_v * CELL_START), within=0.15)
    report.number("Undrained value 1/α, from Table 5", "2.1", 1 / rests["alpha_axial"][0], sig=2)
    report.number("Elastic pressure peak over traction, FE²", "1.97", runs["elastic"]["FE²"]["pressure"].max() / traction)
    report.number("Elastic pressure peak over traction, resolved", "1.78", res_e["pressure"].max() / traction)

    ramp = statistics(read(RAMP))
    report.section("Section 5.3.1: load ramped over 10 s (elastic)")
    report.number("Ramp time (s)", "10", load(RAMP / "fe2.npz")[1]["macro"]["ramp_time"])
    report.number("Settlement difference (%)", "0.72", 100 * ramp["settlement"][0])
    report.number("Drained volume difference (%)", "1.37", 100 * ramp["drainage"][0])
    report.number("Pressure difference, cells 1 to 9", "0.021", ramp["lower"][0])
    report.number("Pressure difference, top cell", "0.075", ramp["top"][0])

    report.section("Section 5.3.1: Figure 7 (cell at the base, viscoelastic)")
    cell = np.load(STEP["viscoelastic"] / "cell_replay_bottom.npz", allow_pickle=False)
    time, inputs = cell["time"], cell["inputs"]
    steps = {name: len(time) - 1 if when is None else int(np.argmin(np.abs(time - when))) for name, when in PANELS}
    report.number("Early, time (s)", "0.0098", time[steps["early"]], sig=2)
    report.number("Early, F_33", "0.990", inputs[steps["early"], 8])
    report.number("Undrained, time (s)", "10", time[steps["undrained"]], sig=2)
    report.number("Undrained, F_33", "0.959", inputs[steps["undrained"], 8])
    report.number("Undrained, pore pressure over traction (%)", "96", 100 * inputs[steps["undrained"], 9] / traction)
    report.number("Drained, time (s)", "4361", time[steps["drained"]])
    report.number("Drained, F_33", "0.925", inputs[steps["drained"], 8])

    report.section("Section 5.3.1: top cell, storage moment and refined reference")
    for name, when in (("elastic", "2.7"), ("viscoelastic", "2.8")):
        report.number(f"Top-cell pressure difference, largest ({name})", "0.10", stats[name]["top"][0])
        report.number(f"Time of that difference (s, {name})", when, stats[name]["top"][1])
        for key, label in (("settlement", "Settlement"), ("drainage", "Drained-volume")):
            t, error = percent_error(runs[name], key)
            report.number(f"{label} difference peaks at (s, {name})", "2", t[np.argmax(error)], sig=1)
        t, error = top_cell_error(runs[name])
        report.number(f"Top-cell difference below 1e-3 after (s, {name})", "60", t[np.nonzero(error > 1e-3)[0][-1]], sig=1)
    top = np.load(STEP["elastic"] / "cell_replay_top.npz", allow_pickle=False)
    ratio = np.linalg.norm(top["storage_moment_rate"], axis=1) / np.linalg.norm(top["diagnostic_flux"], axis=1)
    report.number("Storage moment rate over flux at 10 ms", "0.56", ratio[np.argmin(np.abs(top["time"] - 0.01))])
    window = (top["time"] >= 0.1) & (top["time"] <= 3.0)
    report.number("Storage moment rate over flux, largest from 0.1 to 3 s", "0.10", ratio[window].max())

    run = runs["elastic"]
    area = elastic["macro"]["area"]
    fine, fine_meta = load_resolved(REFINEMENT / "mesh" / "resolved.npz", area)
    end = min(run["resolved"]["time"][-1], run["FE²"]["time"][-1], fine["time"][-1])
    moved = change(run, fine, run["resolved"], end)
    against = statistics(dict(run, resolved=truncated(fine, end), **{"FE²": truncated(run["FE²"], end)}))
    report.number("Finer resolved mesh, h_max/ℓ", "0.075", fine_meta["settings"]["maxh"])
    report.number("Finer resolved mesh, elements", "114520", fine_meta["elements"])
    report.number("Change of the reference, settlement (%)", "0.50", 100 * moved["settlement"])
    report.number("Change of the reference, drained volume (%)", "1.27", 100 * moved["drainage"])
    report.number("FE² against the finer reference, settlement (%)", "0.68", 100 * against["settlement"][0])
    report.number("FE² against the finer reference, drained volume (%)", "1.55", 100 * against["drainage"][0])
    report.number("FE² against the finer reference, top-cell pressure", "0.093", against["top"][0])

    report.section("Section 5.3.1: history-free Biot law and drained state")
    law = load(BASELINES / "consolidation_finite_strain" / "fe2.npz")[1]["model_settings"]
    report.number("Fitted cross term C_1122 (Pa)", "7.78", float(law["K"]) - 2 * float(law["G"]) / 3)
    report.number("Fitted shear modulus (Pa)", "28.0", float(law["G"]))
    report.number("Cell shear stiffness C_1313, Table 5 (Pa)", "19.94", rests["C_1313"][0])
    baseline = statistics(read(BASELINES / "consolidation_finite_strain" / "against_elastic"))
    report.number("Biot law, drained volume difference (%)", "1.52", 100 * baseline["drainage"][0])
    report.number("FE², drained volume difference (%)", "1.63", 100 * stats["elastic"]["drainage"][0])
    report.number("Biot law, largest settlement difference (%)", "57", 100 * baseline["settlement"][0])
    report.number("Drained lateral permeability below rest (%)", "35",
                  100 * (1 - rests["k_lateral"][3] / rests["k_lateral"][0]))
    axial = abs(rests["k_axial"][3] / rests["k_axial"][0] - 1)
    report.holds("Drained axial permeability against rest", "within 1 %", f"{100 * axial:.2f} %", axial <= 0.01)
    return c_v


def cyclic(report, c_v):
    cases = {(skeleton, period): read_cyclic(CYCLIC / f"{skeleton}_P{period:g}")
             for skeleton in SKELETONS for period in PERIODS}
    last = {key: {model: cycle_metrics(case, history, case["cycles"] - 1) for model, history in case["runs"].items()}
            for key, case in cases.items()}
    against = {key: {model: differences(case, model) for model in ("history", "stationary")}
               for key, case in cases.items()}
    meta = {key: load(CYCLIC / f"{key[0]}_P{key[1]:g}" / "fe2.npz")[1] for key in cases}
    stationary_meta = {period: load(CYCLIC / f"stationary_P{period:g}" / "fe2_stationary.npz")[1] for period in PERIODS}

    report.section("Section 5.3.2: set-up")
    first = meta["elastic", 2.0]
    report.number("Cycles", "5", first["cyclic"]["cycles"])
    report.number("Increments per cycle", "40", first["cyclic"]["steps_per_cycle"])
    growth = max(cycle_metrics(case, history, case["cycles"] - 1)["mean"] - cycle_metrics(case, history, 1)["mean"]
                 for case in cases.values() for history in case["runs"].values())
    report.holds("Growth of the mean settlement after the first cycle, largest (mm)", "less than 0.7",
                 f"{1e3 * growth:.3f}", 1e3 * growth < 0.7)
    report.number("Macroscopic residual, stationary run at T = 0.2 s", "1e-6",
                  stationary_meta[0.2]["newton"]["tolerance"], sig=1)
    others = {m["newton"]["tolerance"] for m in meta.values()} | {stationary_meta[2.0]["newton"]["tolerance"]}
    report.holds("Macroscopic residual, the other runs", "1e-8", ", ".join(f"{x:g}" for x in sorted(others)), others == {1e-8})
    tolerances = {x["newton"]["tolerance"] for x in stationary_meta[0.2].get("resumed", [])}
    report.holds("Stationary run at T = 0.2 s: tolerance of its resumed segments", "1e-6",
                 ", ".join(f"{x:g}" for x in sorted(tolerances)), tolerances == {1e-6})
    with np.load(CYCLIC / "stationary_P0.2" / "fe2_stationary.npz", allow_pickle=False) as d:
        residual = float(d["scaled_residual"][1:].max())
    report.holds("Stationary run at T = 0.2 s: largest residual of an accepted step", "at most 1e-6", f"{residual:.2g}",
                 residual <= 1e-6)
    departure_size = min(against[skeleton, 0.2]["stationary"]["settlement"] for skeleton in SKELETONS)
    report.holds("That residual against the departure from the transient cells (smallest settlement difference)",
                 "far below", f"{residual:.2g} against {departure_size:.3f}", residual < 1e-3 * departure_size)

    report.section("Section 5.3.2: transient cells against the resolved column")
    report.number("Settlement difference over all increments, of the resolved amplitude (%)", "1.6",
                  100 * max(a["history"]["settlement"] for a in against.values()))
    report.number("Pressure difference, cells 1 to 9, largest", "0.017", max(a["history"]["lower"] for a in against.values()))
    report.number("Pressure difference, top cell, largest", "0.069", max(a["history"]["top"] for a in against.values()))
    report.number("Last-cycle amplitude difference, largest (%)", "1.1",
                  100 * max(abs(m["history"]["amplitude"] / m["resolved"]["amplitude"] - 1) for m in last.values()))
    lag = max(abs(m["history"]["lag"] - m["resolved"]["lag"]) for m in last.values())
    report.holds("Last-cycle phase lag difference, largest (degrees)", "within 0.25", f"{lag:.3f}", lag <= 0.25)
    excess = {key: 1e3 * (m["history"]["energy"] - m["resolved"]["energy"]) for key, m in last.items()}
    report.number("Energy excess, smallest (mJ/m²)", "0.6", min(excess.values()))
    report.number("Energy excess, largest (mJ/m²)", "0.9", max(excess.values()))
    relative = {key: (m["history"]["energy"] / m["resolved"]["energy"] - 1) for key, m in last.items()}
    report.number("Relative energy excess, largest (%)", "10", 100 * max(relative.values()), sig=1)
    report.holds("The largest relative excess is at T = 2 s", "T = 2 s", f"T = {max(relative, key=relative.get)[1]:g} s",
                 max(relative, key=relative.get)[1] == 2.0)
    report.number("Layer √(c_v T) at T = 0.2 s (mm)", "11", 1e3 * np.sqrt(c_v * 0.2), sig=2)
    report.number("Layer √(c_v T) at T = 2 s (mm)", "35", 1e3 * np.sqrt(c_v * 2.0), sig=2)
    rows = split(CYCLIC)
    share = [100 * (seg_fe2 - seg_res)[-1] / (seg_fe2 - seg_res).sum() for *_, seg_res, seg_fe2, _ in rows]
    report.number("Share of the excess above the top cell's centre, smallest (%)", "65", min(share))
    report.number("Share of the excess above the top cell's centre, largest (%)", "79", max(share))
    report.number("Dissipation of λ_p, smallest (mJ/m²)", "0.07", 1e3 * min(row[-1] for row in rows))
    report.number("Dissipation of λ_p, largest (mJ/m²)", "0.18", 1e3 * max(row[-1] for row in rows))

    report.section("Section 5.3.2: effect of skeleton viscosity (resolved)")
    e, v = last["elastic", 0.2]["resolved"], last["viscoelastic", 0.2]["resolved"]
    report.number("T = 0.2 s, amplitude reduction (%)", "29", 100 * (1 - v["amplitude"] / e["amplitude"]))
    report.number("T = 0.2 s, lag, elastic (degrees)", "10", e["lag"])
    report.number("T = 0.2 s, lag, viscoelastic (degrees)", "45", v["lag"])
    report.holds("T = 0.2 s, energy nearly triples", "nearly triples", f"× {v['energy'] / e['energy']:.2f}",
                 2.7 <= v["energy"] / e["energy"] < 3.0)
    e, v = last["elastic", 2.0]["resolved"], last["viscoelastic", 2.0]["resolved"]
    report.number("T = 2 s, amplitude difference (%)", "1", 100 * (1 - v["amplitude"] / e["amplitude"]))
    report.holds("T = 2 s, lag and energy more than double", "more than double",
                 f"× {v['lag'] / e['lag']:.2f}, × {v['energy'] / e['energy']:.2f}",
                 v["lag"] / e["lag"] > 2 and v["energy"] / e["energy"] > 2)

    report.section("Section 5.3.2: stationary cells against the resolved column")
    for period, values in ((0.2, dict(amplitude=("17", "63"), settlement=("15", "51"), energy=("1.3", "0.5"))),
                           (2.0, dict(amplitude=("4.8", "5.7"), energy=("20", "8")))):
        for skeleton, k in zip(SKELETONS, (0, 1)):
            m = last[skeleton, period]
            report.number(f"T = {period:g} s, {skeleton}: amplitude above resolved (%)", values["amplitude"][k],
                          100 * (m["stationary"]["amplitude"] / m["resolved"]["amplitude"] - 1))
            if "settlement" in values:
                report.number(f"T = {period:g} s, {skeleton}: settlement difference, of the resolved amplitude (%)",
                              values["settlement"][k], 100 * against[skeleton, period]["stationary"]["settlement"])
            report.number(f"T = {period:g} s, {skeleton}: energy, of the resolved (%)", values["energy"][k],
                          100 * m["stationary"]["energy"] / m["resolved"]["energy"])
    lags = [last[skeleton, 2.0]["stationary"]["lag"] / last[skeleton, 2.0]["resolved"]["lag"] for skeleton in SKELETONS]
    report.holds("T = 2 s, lag a fifth or less of the resolved", "a fifth or less", ", ".join(f"{x:.3f}" for x in lags),
                 max(lags) <= 0.2)
    missing = {period: 100 * (1 - last["elastic", period]["stationary"]["energy"] / last["elastic", period]["resolved"]["energy"])
               for period in PERIODS}
    report.number("Elastic, energy missed at T = 2 s (%)", "80", missing[2.0])
    report.number("Elastic, energy missed at T = 0.2 s (%)", "99", missing[0.2])
    return meta, stationary_meta


def analysis(name):
    with (ANALYSIS / f"{name}.csv").open() as stream:
        return list(csv.DictReader(stream))


def verification(report):
    config = json.loads(CONFIG.read_text())
    material, loading, base, studies = config["material"], config["loading"], config["production"], config["studies"]
    inventory, budget, errors = analysis("inventory"), analysis("error_budget"), analysis("errors")
    job = lambda **k: next(r for r in inventory if r["cross_check"] == "False"  # noqa: E731
                           and all(abs(float(r[key]) - value) < 1e-12 for key, value in k.items()) and r["study"] == "A")
    baseline = {T: next(r for r in inventory if r["study"] == "C" and float(r["T"]) == T
                        and float(r["newton_tolerance"]) == base["newton_tolerance"]) for T in loading["durations"]}

    report.section("Section 4: set-up")
    report.number("Cell side ℓ (m)", "1", material["rve_length"])
    report.number("Channel width, of ℓ", "0.33", material["channel_width"])
    with np.load(PAPER / "figure4_stationary.npz", allow_pickle=False) as d:
        report.number("Initial porosity", "0.255", d["fluid_content"][0])
    report.number("Bulk modulus K (Pa)", "50", material["K"])
    report.number("Shear modulus G (Pa)", "50", material["G"])
    report.number("Fluid viscosity μ_f (Pa s)", "1", material["mu_f"])
    report.number("Skeleton viscosity η_s (Pa s)", "2", material["eta_s"])
    report.number("Skeleton viscosity ζ_s (Pa s)", "1", material["zeta_s"])
    report.number("Discretisation order", "2", config["discretisation"]["order"])
    report.holds("Durations T (s)", "0.2, 2", ", ".join(f"{x:g}" for x in loading["durations"]), loading["durations"] == [0.2, 2.0])
    report.number("Shear in the loading", "0.02", loading["shear"])
    report.number("Compression in the loading", "0.2", loading["compression"])
    report.number("Pressure amplitude (Pa)", "20", loading["pressure"])
    report.number("Gradient amplitude (Pa/m)", "10", loading["gradient"])
    report.holds("Gradient direction", "(1, 0.5, 0.25)", str(loading["gradient_direction"]), loading["gradient_direction"] == [1.0, 0.5, 0.25])
    report.holds("Outputs sampled at t/T", "1/4, 1/2, 3/4, 1", str(config["report_fractions"]), config["report_fractions"] == [0.25, 0.5, 0.75, 1.0])
    report.holds("Tangents sampled at t/T", "1/2, 1", str(config["tangent_fractions"]), config["tangent_fractions"] == [0.5, 1.0])
    report.number("Baseline h_max/ℓ", "0.09", base["maxh"])
    report.number("Baseline elements", "5912", baseline[0.2]["n_elements"])
    report.number("Baseline unknowns (total degrees of freedom)", "43382", baseline[0.2]["total_dofs"])
    report.number("Baseline steps N", "100", base["steps"])
    report.number("Baseline Newton tolerance", "1e-8", base["newton_tolerance"], sig=1)
    report.number("Mesh-law ratio λ_m/μ_m", "5", config["discretisation"]["mesh_lame_lambda"] / config["discretisation"]["mesh_mu"])
    order = config["discretisation"]["order"]
    report.number("Quadrature degree (the solver's default rule)", "5", max(2 * order, 4 * order - 3))

    report.section("Section 4.1: discretisation and algebraic errors")
    A, B, C = studies["A"], studies["B"], studies["C"]
    report.number("Mesh levels", "6", len(A["levels"]))
    report.number("Coarsest mesh h_max/ℓ", "0.15", max(A["levels"]))
    report.number("Finest mesh h_max/ℓ", "0.03", min(A["levels"]))
    report.number("Steps in the mesh study", "100", A["steps"])
    report.number("Time-step levels", "6", len(B["levels"]))
    report.number("Fewest steps", "25", min(B["levels"]))
    report.number("Most steps", "800", max(B["levels"]))
    report.number("Mesh of the time-step study h_max/ℓ", "0.09", B["maxh"])
    report.holds("Newton tolerance of both studies", "1e-10", f"{A['newton_tolerance']:g}, {B['newton_tolerance']:g}",
                 A["newton_tolerance"] == B["newton_tolerance"] == 1e-10)
    report.number("Loosest tolerance compared", "1e-6", max(C["levels"]), sig=1)
    report.number("Tightest tolerance compared", "1e-12", min(C["levels"]), sig=1)
    report.number("Algebraic error at tolerance 1e-8, largest", "2.6e-8", max(float(r["algebraic_error"]) for r in budget), sig=2)
    orders = [float(r["fitted_order"]) for r in analysis("orders") if r["study"] == "B" and r["fitted_order"] not in ("", "nan")]
    report.number("Temporal order, smallest", "0.84", min(orders))
    report.number("Temporal order, largest", "1.07", max(orders))
    outputs = [r for r in budget if r["quantity"] in ("stress", "content")]
    flux = [r for r in budget if r["quantity"] == "flux"]
    report.number("Stress and content error, largest total (%)", "1.01", 100 * max(float(r["total"]) for r in outputs))
    worst = max(flux, key=lambda r: float(r["total"]))
    report.number("Flux error, largest total (%)", "3.72", 100 * float(worst["total"]))
    report.holds("That flux error is mainly from the mesh (mesh, time step, %)", "mainly the mesh",
                 f"{100 * float(worst['mesh_error']):.2f}, {100 * float(worst['time_step_error']):.2f}",
                 float(worst["mesh_error"]) > 2 * float(worst["time_step_error"]))
    for T, total in ((0.2, "1.3"), (2.0, "1.9")):
        combined = []
        for row in (r for r in flux if float(r["T"]) == T):
            mesh = next(float(e["relative_error"]) for e in errors if e["study"] == "A" and float(e["T"]) == T
                        and e["quantity"] == "flux" and float(e["time"]) == float(row["time"]) and float(e["maxh"]) == 0.06)
            combined.append(mesh + float(row["time_step_error"]) + float(row["algebraic_error"]))
        report.number(f"Flux total on the 0.06ℓ mesh (%, T = {T:g} s)", total, 100 * max(combined))
    for T, times in ((0.2, "5"), (2.0, "6")):
        ratio = float(job(T=T, maxh=0.06)["seconds_per_step"]) / float(job(T=T, maxh=0.09)["seconds_per_step"])
        report.number(f"Cost per step, 0.06ℓ against 0.09ℓ (T = {T:g} s)", times, ratio, sig=1)
    cross = [r for r in analysis("cross_check") if float(r["change_at_fine_mesh"]) > 1e-4]
    agreement = max(abs(float(r["ratio"]) - 1) for r in cross)
    report.holds("N = 100 to 200: change on the 0.045ℓ and 0.09ℓ meshes, for changes above 1e-4", "within 10 %",
                 f"{100 * agreement:.1f} %", agreement <= 0.10)
    report.number("Smallest ALE Jacobian, baseline (T = 0.2 s)", "0.59", baseline[0.2]["ale_jacobian_min"])
    report.number("Smallest ALE Jacobian, baseline (T = 2 s)", "0.37", baseline[2.0]["ale_jacobian_min"])
    finest = job(T=2.0, maxh=0.03)
    report.holds("Finest mesh at T = 2 s inverts in the final increment", "final increment",
                 f"{finest['state']} after {finest['reached_step']} of {A['steps']} steps",
                 finest["state"] == "failed" and int(finest["reached_step"]) == A["steps"] - 1)
    levels = {r["reference_levels"] for r in analysis("orders") if r["study"] == "A" and float(r["T"]) == 2.0 and float(r["time"]) == 2.0}
    report.holds("Final-time reference levels at T = 2 s", "0.09, 0.06, 0.045", ", ".join(levels), levels == {"0.09 0.06 0.045"})
    below = [(max(float(r["mesh_error"]) for r in budget if float(r["T"]) == 2.0 and r["quantity"] == q),
              next(float(r["mesh_error"]) for r in budget if float(r["T"]) == 2.0 and float(r["time"]) == 2.0 and r["quantity"] == q))
             for q in ("stress", "content", "flux")]
    report.holds("Its mesh errors against the tabulated maxima (%)", "below",
                 ", ".join(f"{100 * b:.2f} < {100 * a:.2f}" for a, b in below), all(b < a for a, b in below))
    law = [r for r in analysis("mesh_law") if float(r["mesh_lame_ratio"]) != 5.0]
    report.number("Newton tolerance of the mesh-law histories", "1e-10", studies["F"]["newton_tolerance"], sig=1)
    report.number("Mesh law: stress, content and full tangent, largest change", "6e-5",
                  max(float(r["change"]) for r in law if r["quantity"] in ("stress", "content", "tangent")), sig=1)
    report.number("Mesh law: flux, largest change (%)", "0.6", 100 * max(float(r["change"]) for r in law if r["quantity"] == "flux"))
    share = max(float(r["change_over_discretisation"]) for r in law if r["quantity"] in ("stress", "content", "tangent", "flux"))
    report.holds("Mesh law: change over the discretisation error, largest", "below a quarter", f"{share:.3f}", share < 0.25)

    report.section("Section 4.2: algorithmic tangent")
    fd = analysis("tangent_fd_summary")
    D = studies["D"]["levels"]
    report.number("Inputs perturbed", "13", len(INPUT_NAMES))
    report.number("Perturbations ρ", "10", len(D))
    report.number("Largest ρ", "1e-1", max(D), sig=1)
    report.number("Smallest ρ", "1e-6", min(D), sig=1)
    slopes = [float(r["slope_largest_steps"]) for r in fd]
    report.holds("Slope of e(ρ) at the largest perturbations", "ρ²", f"{min(slopes):.3f} to {max(slopes):.3f}",
                 all(abs(x - 2) < 0.05 for x in slopes))
    minima = [float(r["min_error"]) for r in fd]
    report.number("Minimum of e(ρ), smallest", "2.2e-10", min(minima), sig=2)
    report.number("Minimum of e(ρ), largest", "1.2e-9", max(minima), sig=2)
    blocks = [r for r in errors if r["study"] == "A" and r["quantity"].startswith("tangent ") and float(r["maxh"]) == base["maxh"]]
    report.number("Scaled tangent blocks", "9", len({r["quantity"] for r in blocks}))
    for T, value, name in ((0.2, "0.30", "stress/p"), (2.0, "1.22", "content/p")):
        top = max((r for r in blocks if float(r["T"]) == T), key=lambda r: float(r["relative_error"]))
        report.number(f"Largest block error (%, T = {T:g} s)", value, 100 * float(top["relative_error"]))
        report.holds(f"That block (T = {T:g} s)", name, top["quantity"].replace("tangent ", ""), top["quantity"] == f"tangent {name}")

    report.section("Section 4.3: power balance and constraints")
    runs = analysis("balance_runs")
    raw = {T: {int(float(r["steps"])): float(r["raw_residual"]) for r in runs if r["study"] == "B" and float(r["T"]) == T}
           for T in loading["durations"]}
    for N, value in ((25, "1.4e-3"), (800, "4.6e-5"), (100, "3.6e-4")):
        report.number(f"Power defect e_W, largest at N = {N}", value, max(raw[T][N] for T in raw), sig=2)
    for T in raw:
        steps = sorted(raw[T])
        report.number(f"Order of e_W in the time step (T = {T:g} s)", "1",
                      -np.polyfit(np.log(steps), np.log([raw[T][n] for n in steps]), 1)[0], sig=1)
    corrected = max(runs, key=lambda r: float(r["corrected_residual"]))
    report.number("Corrected residual, worst", "6.1e-8", corrected["corrected_residual"], sig=2)
    report.holds("It is set by the histories at Newton tolerance", "1e-6", corrected["newton_tolerance"], float(corrected["newton_tolerance"]) == 1e-6)
    report.number("Corrected residual, worst at Newton tolerance 1e-8", "5.9e-11",
                  max(float(r["corrected_residual"]) for r in runs if float(r["newton_tolerance"]) == 1e-8), sig=2)
    for T, storage, flow in ((0.2, "29", "0.2"), (2.0, "70", "1")):
        report.number(f"Peak storage power, of the stress power (%, T = {T:g} s)", storage,
                      100 * max(float(r["storage_over_stress"]) for r in runs if float(r["T"]) == T))
        report.number(f"Peak flux power, of the stress power (%, T = {T:g} s)", flow,
                      100 * max(float(r["flux_over_stress"]) for r in runs if float(r["T"]) == T))


def biot(report):
    report.section("Section 5.1: Biot coefficient")
    with np.load(PAPER / "figure3_biot.npz", allow_pickle=False) as d:
        alpha, moduli, pressures = d["alpha_secant"], d["bulk_moduli"], d["pressures"]
        settings = json.loads(str(d["metadata"]))["settings"]
    fixed, periodic = alpha[0], alpha[1]
    report.holds("Pore pressures (Pa)", "1, 10, 20, 30", ", ".join(f"{x:g}" for x in pressures), list(pressures) == [1.0, 10.0, 20.0, 30.0])
    differences_ = []
    for j, p in enumerate(pressures):
        computed = np.interp(np.log(SANDSTROM_K), np.log(moduli), fixed[j])
        differences_.append(np.abs(computed - np.array(SANDSTROM_ALPHA[float(p)])))
    closest = float(np.nanmax(differences_))
    report.holds("Fixed cell against the published values, largest difference", "within 0.015", f"{closest:.4f}", closest <= 0.015)
    report.noted("Uncertainty in reading the published figure", "about 0.01")
    gap = periodic - fixed
    share = float(np.mean(gap[~np.isnan(gap)] > 0))
    report.holds("Periodic α above fixed, share of the computed cases", "much of the range", f"{100 * share:.0f} %", share > 0.5)
    k_peak = moduli[int(np.nanargmax(gap[0]))]
    report.holds("The difference shrinks as K grows (p = 1 Pa: largest, at K, and at the largest K)", "shrinks",
                 f"{np.nanmax(gap[0]):.3f} at K = {k_peak:.0f} Pa, {gap[0, -1]:.3f} at K = {moduli[-1]:.0f} Pa", gap[0, -1] < 0.2 * np.nanmax(gap[0]))
    K, G = float(moduli[0]), float(settings["G"])
    report.number("Softest bulk modulus K (Pa)", "10", K)
    report.number("Its Poisson ratio", "-0.44", (3 * K - 2 * G) / (2 * (3 * K + G)))
    J = np.linspace(0.2, 0.9, 7001)
    lame = K - 2 * G / 3
    hydrostatic = (lame * np.log(J) + G * (J ** (2 / 3) - 1)) / J
    report.number("J_s where the law's hydrostatic stress stops increasing", "0.5", J[np.argmin(hydrostatic)], sig=1)
    report.holds("Fixed cell at K = 10 Pa, p = 30 Pa", "does not equilibrate", f"α = {fixed[3, 0]}", np.isnan(fixed[3, 0]))
    measured = json.loads((PAPER / "figure3_solid_compression.json").read_text())
    report.number("Threshold of the fraction below, J_s", "0.5", measured["threshold"], sig=1)
    settled = measured["boundaries"]["periodic"]["records"][-1]
    ramp_end = measured["boundaries"]["fixed"]["records"][0]
    failing = measured["boundaries"]["fixed"]["records"][-1]
    report.holds("The compression run's periodic cell against Figure 3 (α)", "the same",
                 f"{settled['alpha']:.4f}, {periodic[3, 0]:.4f}", abs(settled["alpha"] - periodic[3, 0]) < 1e-3)
    for label, record in (("periodic cell, settled", settled), ("fixed cell, end of the ramp", ramp_end),
                          ("fixed cell, last step before it fails", failing)):
        report.holds(f"Solid below J_s = 0.5, {label}", "about 1–2 %", f"{100 * record['fraction_below']:.2f} %",
                     0.005 <= record["fraction_below"] < 0.025)


def pulse(report):
    report.section("Section 5.2: compression pulse")
    with np.load(PAPER / "figure4_cycles.npz", allow_pickle=False) as d:
        a = {key: d[key] for key in d.files}
    settings = json.loads(str(a["metadata"]))["settings"]
    col = {str(name): i for i, name in enumerate(a["columns"])}
    stationary = load_stationary_4()
    report.number("Peak compression", "0.20", settings["compression"])
    report.holds("Durations T (s)", "2, 0.2", ", ".join(f"{x:g}" for x in settings["durations"]), settings["durations"] == [2.0, 0.2])
    report.number("Steps per pulse", "100", settings["steps"])
    report.number("Steps in the hold", "50", settings["steps"] * settings["recovery_fraction"])
    report.number("Pressure gradient (Pa/m)", "-1", settings["pressure_gradient"][0])
    report.number("Pressure difference across the cell (Pa)", "1", abs(settings["pressure_gradient"][0]) * settings["rve_length"])
    viscous = settings["mu_f"] / settings["G"]
    report.number("Short pulse over the fluid viscous time", "10", 0.2 / viscous)
    report.number("Short pulse over the skeleton viscous time", "5", 0.2 / (settings["eta_s"] / settings["G"]))

    with np.load(PAPER / "figure4_stationary.npz", allow_pickle=False) as d:
        steps, equilibrium = json.loads(str(d["metadata"]))["settings"]["steps"], d["equilibrium"]
    report.number("Increments of the stationary cell", "3", len(steps))
    report.number("Length of each increment (s)", "1000", steps[0])
    report.holds("Its outputs' change over the last increment, largest", "below 1e-3", f"{equilibrium.max():.2g}", equilibrium.max() < 1e-3)

    q_s, s_s = stationary["flux"][:, 0] / stationary["flux"][0, 0], -stationary["stress"][:, 2, 2]
    cases = {}
    for j, T in enumerate(a["durations"]):
        for m, solid in enumerate(str(n) for n in a["materials"]):
            r = a["rows"][(a["rows"][:, col["material"]] == m) & (a["rows"][:, col["duration_index"]] == j)]
            initial = a["initial_flux"][(a["initial_flux"][:, 0] == m) & (a["initial_flux"][:, 1] == j), 2].item()
            q, stress = r[:, col["Q1"]] / initial, -r[:, col["P33"]]
            gap = q - stationary_at(stationary, r[:, col["compression"]])[0]
            cases[solid, float(T)] = dict(rows=r, q=q, stress=stress, time=r[:, col["time"]], q_min=q.min(), q_max=q.max(),
                                          peak=stress.max(), peak_time=r[np.argmax(stress), col["time"]] / T,
                                          tension=max(-stress.min(), 0.0), largest=abs(gap[np.argmax(np.abs(gap))]))
    e2, v2, e02, v02 = (cases[key] for key in (("elastic", 2.0), ("viscoelastic", 2.0), ("elastic", 0.2), ("viscoelastic", 0.2)))
    report.holds("T = 2 s: flux decreases less than the stationary cell (minima)", "less",
                 f"{e2['q_min']:.2f}, {v2['q_min']:.2f} against {q_s.min():.2f}", min(e2["q_min"], v2["q_min"]) > q_s.min())
    report.holds("T = 2 s: stress peaks higher than the stationary cell (Pa)", "higher",
                 f"{e2['peak']:.1f}, {v2['peak']:.1f} against {s_s.max():.1f}", min(e2["peak"], v2["peak"]) > s_s.max())
    report.holds("T = 2 s: the peak comes slightly before mid-pulse (t/T)", "slightly before",
                 f"{e2['peak_time']:.2f}, {v2['peak_time']:.2f}", 0.4 < min(e2["peak_time"], v2["peak_time"]) and max(e2["peak_time"], v2["peak_time"]) < 0.5)
    report.holds("T = 2 s: tensile near the end of unloading (Pa)", "tensile", f"{e2['tension']:.2f}, {v2['tension']:.2f}", min(e2["tension"], v2["tension"]) > 0)
    report.holds("T = 0.2 s, elastic: the flux falls further than the stationary cell (minima)", "further",
                 f"{e02['q_min']:.2f} against {q_s.min():.2f}", e02["q_min"] < q_s.min())
    report.holds("T = 0.2 s, elastic: the flux overshoots its initial value", "overshoots", f"{e02['q_max']:.2f}", e02["q_max"] > 1)
    report.holds("T = 0.2 s: skeleton viscosity raises the peak stress (Pa)", "raises", f"{v02['peak']:.1f} against {e02['peak']:.1f}", v02["peak"] > e02["peak"])
    report.holds("T = 0.2 s: skeleton viscosity brings the peak earlier (t/T)", "earlier", f"{v02['peak_time']:.2f} against {e02['peak_time']:.2f}", v02["peak_time"] < e02["peak_time"])
    report.holds("T = 0.2 s: skeleton viscosity more than doubles the tension", "more than doubles", f"× {v02['tension'] / e02['tension']:.2f}", v02["tension"] > 2 * e02["tension"])
    report.holds("T = 0.2 s: flux far closer to the stationary cell with skeleton viscosity (largest departure)", "far closer",
                 f"{v02['largest']:.3f} against {e02['largest']:.3f}", v02["largest"] < 0.5 * e02["largest"])
    report.holds("T = 2 s: departure nearly the same for both skeletons", "nearly the same", f"{e2['largest']:.3f}, {v2['largest']:.3f}", abs(e2["largest"] - v2["largest"]) < 0.01)
    for name, case in (("elastic", e2), ("viscoelastic", v2)):
        hold = case["time"] >= 2.0 - 1e-9
        tau = -1 / np.polyfit(case["time"][hold], np.log(np.abs(case["q"][hold] - 1)), 1)[0]
        report.number(f"Relaxation time of the flux departure in the hold (s, T = 2 s, {name})", "0.7", tau, within=0.1)
        report.number(f"That time over the fluid viscous time ({name})", "35", tau / viscous, within=0.1)
    rates = load_rates_4()["paths"]
    report.number("Pulse durations of the sweep", "9", len(rates))
    report.number("Shortest duration (s)", "0.2", min(p["period"] for p in rates))
    report.number("Longest duration (s)", "20", max(p["period"] for p in rates))
    flux = {p["period"]: departure(p["compression"], p["flux"], p["fluid_content"], stationary)[0] for p in rates}
    up_to_2 = [value for period, value in flux.items() if period <= 2.0 + 1e-9]
    report.number("Largest flux departure up to T = 2 s, smallest (%)", "6", 100 * min(up_to_2), sig=1)
    report.number("Largest flux departure up to T = 2 s, largest (%)", "9", 100 * max(up_to_2), sig=1)
    report.number("Largest flux departure at T = 20 s (%)", "1.3", 100 * flux[max(flux)])
    hold = e02["time"] >= 0.2 - 1e-9
    content, stress = e02["rows"][hold, col["fluid_content"]], e02["stress"][hold]
    report.number("Elastic, T = 0.2 s, hold: fluid content at its start", "0.238", content[0])
    report.number("Elastic, T = 0.2 s, hold: fluid content at its end", "0.252", content[-1])
    report.number("Elastic, T = 0.2 s, hold: axial tension at its start (Pa)", "2.6", -stress[0])
    report.number("Elastic, T = 0.2 s, hold: axial tension at its end (Pa)", "0.42", -stress[-1])


def log_text(folder, pattern):
    return "\n".join(path.read_text() for path in sorted(Path(folder, "slurm").glob(pattern)))


def resolved_time(folder):
    """Wall time of a resolved run (s) and its threads, from its Slurm log."""
    text = log_text(folder, "resolved-*.out")
    return float(re.findall(r"finished \(complete\) at .*, (\d+) s$", text, re.M)[-1]), int(re.search(r"(\d+) threads", text)[1])


def cost(report, cyclic_meta, stationary_meta):
    report.section("Section 5.3.3: computational cost")
    for name, folder, steps in (("elastic", STEP["elastic"], "415"), ("viscoelastic", STEP["viscoelastic"], "424")):
        _, comparison = replay(folder / "fe2.npz", quiet=True)
        report.number(f"Replayed steps ({name})", steps, comparison["steps"])
        report.noted(f"Replay time on one core, on this machine ({name})", "less than 1 s", f"{comparison['seconds']:.2f} s")
        difference = max(comparison[key]["relative"] for key in ("settlement", "base_pressure", "cumulative_outflow"))
        report.holds(f"Replay against the saved solution, largest relative difference ({name})",
                     "to the macroscopic Newton tolerance", f"{difference:.1e}", difference < 1e-4)
    report.noted("The Aire node: two 84-core AMD EPYC 9634 processors", "as stated")
    for name, per_call, calls, core_hours, hours in (("elastic", "46", "49280", "680", "17.1"),
                                                     ("viscoelastic", "51", "50000", "790", "19.8")):
        folder = STEP[name]
        meta = load(folder / "fe2.npz")[1]
        report.number(f"FE² cell increment on one core (s, {name})", per_call,
                      float(re.findall(r"cell solves, ([\d.]+) s each", log_text(folder, "fe2-*.out"))[-1]))
        report.number(f"FE² cell calls ({name})", calls, int(np.load(folder / "fe2.npz")["cell_calls"][-1]))
        core_count = meta["workers"] * meta["threads_per_worker"]
        report.number(f"FE² cores ({name})", "40", core_count)
        report.number(f"FE² wall time (h, {name})", hours, meta["wall_seconds"] / 3600)
        report.number(f"FE² core-hours ({name})", core_hours, core_count * meta["wall_seconds"] / 3600, sig=2)
    for name, core_hours, hours in (("elastic", "76", "9.5"), ("viscoelastic", "92", "11.5")):
        seconds, threads = resolved_time(STEP[name])
        report.number(f"Resolved cores ({name})", "8", threads)
        report.number(f"Resolved wall time (h, {name})", hours, seconds / 3600)
        report.number(f"Resolved core-hours ({name})", core_hours, threads * seconds / 3600, sig=2)
    report.number("Increments of a cyclic run", "200", cyclic_meta["elastic", 2.0]["cyclic"]["cycles"]
                  * cyclic_meta["elastic", 2.0]["cyclic"]["steps_per_cycle"])
    fe2_hours = {key: meta["wall_seconds"] / 3600 for key, meta in cyclic_meta.items()}
    resolved_hours = {key: resolved_time(CYCLIC / f"{key[0]}_P{key[1]:g}")[0] / 3600 for key in cyclic_meta}
    report.number("Cyclic FE², shortest (h)", "8.1", min(fe2_hours.values()))
    report.number("Cyclic FE², longest (h)", "9.6", max(fe2_hours.values()))
    report.number("Cyclic resolved, shortest (h)", "5.4", min(resolved_hours.values()))
    report.number("Cyclic resolved, longest (h)", "8.1", max(resolved_hours.values()))
    report.number("Long increments of a stationary cell evaluation", "3", len(STEPS))
    for period, times in ((2.0, "4"), (0.2, "6")):
        ratio = stationary_meta[period]["wall_seconds"] / 3600 / np.mean([fe2_hours[s, period] for s in SKELETONS])
        report.number(f"Stationary against transient FE², wall time (T = {period:g} s)", times, ratio, sig=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT, help=f"folder (default {OUTPUT})")
    args = parser.parse_args()
    report = Report()
    verification(report)
    biot(report)
    with contextlib.redirect_stdout(io.StringIO()):  # the plotting helpers say what they read
        pulse(report)
    c_v = consolidation(report)
    cost(report, *cyclic(report, c_v))
    text = report.markdown()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "paper_numbers.md").write_text(text)
    total = sum(1 for row in report.rows if not isinstance(row, str) and row[3] is not None)
    print(text)
    print(f"Wrote {args.output / 'paper_numbers.md'}: {total - len(report.failed)} of {total} agree with the paper.")
    for what in report.failed:
        print(f"  DISAGREES: {what}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
