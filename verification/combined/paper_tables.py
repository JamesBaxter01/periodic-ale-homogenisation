#!/usr/bin/env python3
"""Paper Section 4 tables from an analysed verification campaign (no solver runs, no NGSolve).

  python verification/combined/paper_tables.py [results/verification/<campaign>] [--output DIR]

Press Run (or pass the campaign). Reads <campaign>/analysis/ (written by analyse.py) and the
campaign settings CONFIG, and writes to OUTPUT, each a complete table environment with caption
and label (paper_latex.py):

- table_1_verification_errors.tex (tab:verification-errors): estimated relative errors (%) at production
  settings per output group and duration T, from error_budget.csv. Mesh and time-step columns are
  each the maximum over the sampled times; the total is the largest per-time sum of mesh,
  time-step and algebraic errors. The tangent has no time-step error ("--").
- table_2_verification_tangent_balance.tex (tab:verification-tangent-balance): smallest central-difference
  tangent discrepancy at t = T/2 and T (tangent_fd_summary.csv), the power defect e_W of the
  production histories (balance_runs.csv; the coarser-step histories only show its first-order
  decrease, quoted in the text), and the worst power-balance and constraint diagnostics over every
  study's histories (balance_summary.csv): corrected residual, corrected residual / Newton target, the largest of the mean constraints /
  Newton target and the largest identity residual (power accounting, cell volume, solid frame).

Also prints every tabulated value. The captions are CAPTION_* below; <<name>> is filled from CONFIG.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "microscale"))

import paper_latex as tex  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
CAMPAIGN = ROOT / "results" / "verification" / "combined_20261026_201037"
CONFIG = ROOT / "configs" / "verification.json"
OUTPUT = ROOT / "results" / "paper" / "tables"

OUTPUTS = [("stress", r"$\overline{\bm P}$"), ("content", r"$m_f$"), ("flux", r"$\bm Q_0$"),
           ("tangent", r"$\widehat{\bm C}$")]
CAPTION_ERRORS = (r"Estimated relative errors (\%) of the homogenised outputs and the scaled tangent at baseline "
                  r"settings, for the viscoelastic verification histories of duration $T$. Each column gives its own "
                  r"maximum over the sampled times; the total is the largest per-time sum of mesh, time-step and "
                  r"algebraic errors. The tangent has no time-step error (Section~\ref{sec:verification-tangent}).")
CAPTION_BALANCE = (r"Tangent, power-balance and constraint checks for the histories of duration $T$. Tangent: smallest "
                   r"discrepancy $e(\rho)$. Balances and constraints: maxima over all completed steps of the "
                   r"verification histories, except the power defect $e_W$, which is for the baseline histories "
                   r"($N=<<steps>>$).")
CONSTRAINTS = ["mean fluctuation pressure / Newton target", "solid frame (mean velocity) / Newton target",
               "mean continuity / Newton target"]
STUDY_NAMES = {"A": "mesh", "B": "time-step", "C": "tolerance", "F": "mesh-law"}  # caption wording per study
IDENTITIES = ["power accounting", "cell volume", "solid frame (assembled vs integrated)"]
# ---------------------------------------------------------------------------


def read(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def sci(value):
    """Three significant figures in scientific notation, as the paper's tables: $1.06\\times10^{-9}$."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def error_table(campaign, durations, config):
    budget = read(campaign / "analysis" / "error_budget.csv")
    rows, printed = [], []
    for quantity, symbol in OUTPUTS:
        cells = [symbol]
        for T in durations:
            entries = [r for r in budget if float(r["T"]) == T and r["quantity"] == quantity]
            if not entries:
                raise ValueError(f"error_budget.csv has no {quantity} at T = {T:g} s")
            mesh = max(float(r["mesh_error"]) for r in entries)
            steps = [float(r["time_step_error"]) for r in entries if r["time_step_error"] != "n/a"]
            total = max(float(r["total"]) for r in entries)
            cells += [f"{100 * mesh:.2f}", f"{100 * max(steps):.2f}" if steps else "--", f"{100 * total:.2f}"]
            printed.append(f"  T = {T:g} s {quantity:8s} mesh {100 * mesh:.3f} %, time step "
                           f"{f'{100 * max(steps):.3f} %' if steps else '--'}, total {100 * total:.3f} %")
        rows.append("        " + " & ".join(cells) + r"\\")
    n = len(durations)
    header = " & ".join(rf"\multicolumn{{3}}{{c}}{{$T={tex.number(T)}\,\mathrm s$}}" for T in durations)
    rules = "".join(rf"\cmidrule({'lr' if j < n - 1 else 'l'}){{{2 + 3 * j}-{4 + 3 * j}}}" for j in range(n))
    production = config["production"]
    caption = tex.fill(CAPTION_ERRORS, maxh=tex.number(production["maxh"]), steps=production["steps"],
                       tolerance=f"${tex.number(production['newton_tolerance'])}$")
    lines = [r"\begin{table}[pos=htb]", r"    \centering", r"    \small", rf"    \caption{{{caption}}}",
             r"    \label{tab:verification-errors}",
             r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l" + "rrr" * n + "@{}}", r"        \toprule",
             f"        & {header}\\\\", f"        {rules}",
             "        Output" + " & Mesh & $\\Delta t$ & Total" * n + r"\\", r"        \midrule", *rows,
             r"        \bottomrule", r"    \end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n", printed


def balance_table(campaign, durations, config):
    fd = read(campaign / "analysis" / "tangent_fd_summary.csv")
    balance = read(campaign / "analysis" / "balance_summary.csv")
    runs = read(campaign / "analysis" / "balance_runs.csv")
    production = config["production"]
    printed = []

    def fd_min(T, fraction):
        rows = [r for r in fd if float(r["T"]) == T and abs(float(r["time"]) - fraction * T) < 1e-9 * T]
        if len(rows) != 1:
            raise ValueError(f"tangent_fd_summary.csv: expected one row at T = {T:g} s, t = {fraction:g} T")
        return float(rows[0]["min_error"])

    def worst(T, columns):
        return max(float(r[c]) for r in balance if float(r["T"]) == T for c in columns)

    def production_worst(T, column):
        rows = [r for r in runs if float(r["T"]) == T and float(r["maxh"]) == production["maxh"]
                and int(r["steps"]) == production["steps"]]
        if not rows:
            raise ValueError(f"balance_runs.csv has no production history at T = {T:g} s")
        return max(float(r[column]) for r in rows)

    entries = [
        ("Minimum $e(\\rho)$ at $t=T/2$", lambda T: fd_min(T, 0.5)),
        ("Minimum $e(\\rho)$ at $t=T$", lambda T: fd_min(T, 1.0)),
        None,
        ("Power defect $e_W$", lambda T: production_worst(T, "raw_residual")),
        ("Corrected residual $e_W^{\\rm corr}$", lambda T: worst(T, ["corrected_residual"])),
        ("$e_W^{\\rm corr}$ / Newton target", lambda T: worst(T, ["corrected_over_newton_target"])),
        ("Mean constraints / Newton target", lambda T: worst(T, CONSTRAINTS)),
        ("Maximum identity residual", lambda T: worst(T, IDENTITIES)),
    ]
    width = max(len(e[0]) for e in entries if e)
    rows = [r"        \multicolumn{" + str(1 + len(durations)) + r"}{@{}l}{\itshape Algorithmic tangent}\\[2pt]"]
    for entry in entries:
        if entry is None:
            rows += [r"        \midrule",
                     r"        \multicolumn{" + str(1 + len(durations)) + r"}{@{}l}{\itshape Power balance and constraints}\\[2pt]"]
            continue
        name, value = entry
        values = [value(T) for T in durations]
        rows.append(f"        {name:{width}s} & " + " & ".join(sci(v) for v in values) + r"\\")
        printed.append(f"  {name:{width}s} " + "  ".join(f"T = {T:g} s: {v:.3e}" for T, v in zip(durations, values)))
    studies = [STUDY_NAMES.get(s, config["studies"][s]["name"]) for s in sorted({r["study"] for r in balance})]
    caption = tex.fill(CAPTION_BALANCE, steps=production["steps"], fd_tolerance=f"${tex.number(config['studies']['D']['newton_tolerance'])}$",
                       studies=", ".join(studies[:-1]) + " and " + studies[-1] if len(studies) > 1 else studies[0])
    lines = [r"\begin{table}[pos=htb]", r"    \centering", r"    \small", rf"    \caption{{{caption}}}",
             r"    \label{tab:verification-tangent-balance}", r"    \setlength{\tabcolsep}{9pt}",
             r"    \renewcommand{\arraystretch}{1.12}", r"    \begin{tabular}{@{}l" + "r" * len(durations) + "@{}}",
             r"        \toprule",
             "        Diagnostic & " + " & ".join(rf"$T={tex.number(T)}\,\mathrm s$" for T in durations) + r"\\",
             r"        \midrule", *rows, r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n", printed


def main():
    config = json.loads(Path(CONFIG).read_text())
    durations = [float(T) for T in config["loading"]["durations"]]
    source = f"verification/combined/paper_tables.py from {tex.relative(CAMPAIGN, ROOT)}/analysis"
    for name, build in (("table_1_verification_errors", error_table), ("table_2_verification_tangent_balance", balance_table)):
        text, printed = build(Path(CAMPAIGN), durations, config)
        print(f"{name}:")
        print("\n".join(printed))
        tex.write(Path(OUTPUT) / f"{name}.tex", text, source)


def command_line():
    global CAMPAIGN, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign", nargs="?", type=Path, help=f"analysed campaign folder (default {CAMPAIGN})")
    parser.add_argument("--output", type=Path, help=f"table folder (default {OUTPUT})")
    args = parser.parse_args()
    CAMPAIGN = args.campaign or CAMPAIGN
    OUTPUT = args.output or OUTPUT


if __name__ == "__main__":
    command_line()
    main()
