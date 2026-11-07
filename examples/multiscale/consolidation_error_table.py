"""Paper table: FE² against resolved consolidation, differences for each solid.

Press Run. Reads the run folders RUNS (fe2.npz and resolved.npz, one per solid) and writes
OUTPUT/table_4_consolidation_errors.tex (booktabs, one column per solid; caption CAPTION, label LABEL).

Statistics, on the resolved run's times (the accepted FE² times) up to the common end:

- final settlement of each model;
- settlement and drained volume: |FE² - resolved| relative to the resolved final value, the maximum
  over time and the mean over logarithmic time (weights d ln t) from LOG_START;
- cell pressures: |FE² - resolved| over the applied traction, for cells 1 to n-1 (numbered from the
  base) and for the top cell, from CELL_START (the elastic top cell lags the overshoot until ~2 ms).

FE² cell pressures are the fluid-weighted cell averages (compare_consolidation.fe2_cell_averages),
compared with the resolved average over each cell's current fluid volume.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale", ROOT / "examples" / "multiscale"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plot_paper_consolidation import read  # noqa: E402
import paper_latex as tex  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
RUNS = [ROOT / "results" / "consolidation" / "full_elastic_step_20261026_201037",
        ROOT / "results" / "consolidation" / "full_viscoelastic_step_20261026_201037"]
OUTPUT = ROOT / "results" / "paper" / "tables"
LOG_START = 1e-6                  # (s) start of the logarithmic-time means (the first step)
CELL_START = 2e-3                 # (s) start of the cell-pressure statistics
CAPTION = (r"Differences between FE$^2$ and resolved FSI under the step load of Figure~\ref{fig:consolidation}. "
           r"Settlement and drained volume: relative to the resolved final value. Cell pressures: fluid-weighted "
           r"cell averages, relative to the traction, from $t=<<cell_start>>$; cells are numbered from the base, "
           r"with the largest of cells 1--<<lower>> at each time. Maxima over time; means over logarithmic time.")
LABEL = "tab:consolidation_errors"
# ---------------------------------------------------------------------------


def log_mean(t, values, start):
    keep = t >= start
    lt, v = np.log(t[keep]), values[keep]
    return float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(lt)) / (lt[-1] - lt[0]))


def statistics(run):
    fe2, res = run["FE²"], run["resolved"]
    t = res["time"][(res["time"] > 0) & (res["time"] <= fe2["time"][-1])]
    on = lambda h, key: np.interp(t, h["time"], h[key])  # noqa: E731
    s = dict(final_fe2=fe2["settlement"][-1], final_resolved=res["settlement"][-1])
    for key in ("settlement", "drainage"):
        error = np.abs(on(fe2, key) - on(res, key)) / abs(res[key][-1])
        s[key] = (error.max(), log_mean(t, error, LOG_START))
    p = lambda h: np.array([np.interp(t, h["time"], h["pressure"][:, k])  # noqa: E731
                            for k in range(run["cells"])]).T
    error = np.abs(p(fe2) - p(res)) / run["traction"]
    late = t >= CELL_START
    lower, top = error[:, :-1].max(axis=1), error[:, -1]
    s["lower"] = (lower[late].max(), log_mean(t, lower, CELL_START))
    s["top"] = (top[late].max(), float(t[late][np.argmax(top[late])]), log_mean(t, top, CELL_START))
    return s


def latex(runs, stats):
    n = runs[0]["cells"]
    pct = lambda x: rf"${100 * x:.2f}$"  # noqa: E731
    val = lambda x: rf"${x:.4f}$"  # noqa: E731
    rows = [(r"Final settlement, FE$^2$ (mm)", [rf"${1e3 * s['final_fe2']:.2f}$" for s in stats]),
            (r"Final settlement, resolved (mm)", [rf"${1e3 * s['final_resolved']:.2f}$" for s in stats]),
            None,
            (r"Settlement, max (\%)", [pct(s["settlement"][0]) for s in stats]),
            (r"Settlement, log-time mean (\%)", [pct(s["settlement"][1]) for s in stats]),
            (r"Drained volume, max (\%)", [pct(s["drainage"][0]) for s in stats]),
            (r"Drained volume, log-time mean (\%)", [pct(s["drainage"][1]) for s in stats]),
            None,
            (f"Pressure, cells 1--{n - 1}, max", [val(s["lower"][0]) for s in stats]),
            (f"Pressure, cells 1--{n - 1}, log-time mean", [val(s["lower"][1]) for s in stats]),
            (f"Pressure, cell {n}, max", [val(s["top"][0]) for s in stats]),
            (f"Pressure, cell {n}, log-time mean", [val(s["top"][2]) for s in stats])]
    caption = tex.fill(CAPTION, lower=n - 1, cell_start=rf"{tex.number(CELL_START * 1e3)}\,\mathrm{{ms}}",
                       log_start=rf"{tex.number(LOG_START * 1e6)}\,\mu\mathrm{{s}}")
    lines = [r"\begin{table}[pos=htb]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{LABEL}}}", r"\small",
             r"\begin{tabular}{lcc}", r"\toprule",
             " & " + " & ".join(f"{run['label'].capitalize()} skeleton" for run in runs) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(r"\addlinespace" if row is None else f"{row[0]} & " + " & ".join(row[1]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "% FE² against resolved consolidation (consolidation_error_table.py). Needs \\usepackage{booktabs}.\n" \
           + "\n".join(lines) + "\n"


def main():
    runs = [read(Path(folder)) for folder in RUNS]
    stats = [statistics(run) for run in runs]
    for run, s in zip(runs, stats):
        print(f"{run['label']}: final {1e3 * s['final_fe2']:.2f} / {1e3 * s['final_resolved']:.2f} mm; settlement "
              f"max {100 * s['settlement'][0]:.2f} % mean {100 * s['settlement'][1]:.2f} %; drained max "
              f"{100 * s['drainage'][0]:.2f} % mean {100 * s['drainage'][1]:.2f} %; cells 1-{run['cells'] - 1} max "
              f"{s['lower'][0]:.4f} mean {s['lower'][1]:.4f}; top max {s['top'][0]:.4f} at {s['top'][1]:.3g} s, "
              f"mean {s['top'][2]:.4f}")
    tex.write(OUTPUT / "table_4_consolidation_errors.tex", latex(runs, stats), tex.relative(__file__, ROOT))


def command_line():
    global RUNS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run folders (default RUNS)")
    args = parser.parse_args()
    RUNS = args.runs or RUNS


if __name__ == "__main__":
    command_line()
    main()
