"""Regenerate every paper figure and table (and its LaTeX float), recompute the numbers quoted in the text of
Sections 4 and 5 and compare them with the paper.

Press Run (or python run.py paper). Runs, in order, each paper script on its default data; no cell solves unless
--solve. A step that fails is reported and the rest still run.

1. plot_paper_geometry.py         Figure 1 (scales), Figure 2 (reference cell), Figure 5 (consolidation set-up)
                                  (figure_1_scales.tex, figure_2_cell.tex, figure_5_setup.tex); needs NGSolve and PyVista (fem environment)
2. plot_paper_figures.py          Figures 3-4 and Table 3, single cell (figure_3_biot.tex, figure_4_cycles.tex,
                                  table_3_criterion.tex, from the 0.2-200 s pulse sweep)
3. plot_paper_consolidation.py    Figure 6, step-load consolidation (figure_6_consolidation.tex)
4. consolidation_parameter_table  Table 5, response derivatives (table_5_consolidation_parameters.tex), rebuilt
                                  from the saved .csv; --solve recomputes the cell states (about 8 min), reruns the
                                  history-free Biot laws (baseline_models.py, about 2 min) and the solid compression
                                  at K = 10 Pa, p = 30 Pa (figure3_solid_compression.py, about 5 min)
   consolidation_error_table      Table 4, FE² against resolved differences (table_4_consolidation_errors.tex)
5. plot_paper_cells.py            Figure 7, a consolidation cell at three times (figure_7_cells.tex), from the
                                  replayed cell (consolidation_cell_replay.py, about an hour, not rerun here)
   plot_paper_cyclic.py           Figure 8 and Table 6, cyclic consolidation (figure_8_cyclic.tex, table_6_cyclic.tex;
                                  Table 6 also from the T = 20 and 200 s runs)
6. paper_tables.py                Tables 1-2, verification (table_1_verification_errors.tex,
                                  table_2_verification_tangent_balance.tex)
7. paper_numbers.py               every number in the text of Sections 4 and 5, against the value printed in the paper
                                  (paper_numbers.md); fails if one disagrees
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def steps(solve):
    table = [] if solve else ["--table-only"]
    multiscale = ROOT / "examples" / "multiscale"
    baselines = ["--output", ROOT / "results" / "consolidation" / "baselines_20261028"]
    biot = [("History-free Biot laws", [multiscale / "baseline_models.py", *baselines]),
            ("Solid compression at K = 10 Pa", [ROOT / "examples" / "microscale" / "figure3_solid_compression.py"])] if solve else []
    return [
        ("Figures 1, 2 and 5: scales, cell and consolidation set-up", [multiscale / "plot_paper_geometry.py"]),
        ("Figures 3-4 and Table 3", [ROOT / "examples" / "microscale" / "plot_paper_figures.py"]),
        ("Figure 6, consolidation", [multiscale / "plot_paper_consolidation.py"]),
        ("Table 5, response derivatives", [multiscale / "consolidation_parameter_table.py", *table]),
        *biot,
        ("Table 4, consolidation differences", [multiscale / "consolidation_error_table.py"]),
        ("Figure 7, consolidation cells", [multiscale / "plot_paper_cells.py"]),
        ("Figure 8 and Table 6, cyclic consolidation", [multiscale / "plot_paper_cyclic.py"]),
        ("Tables 1-2, verification", [ROOT / "verification" / "combined" / "paper_tables.py"]),
        ("Numbers in the text against the paper", [ROOT / "examples" / "paper_numbers.py"]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solve", action="store_true",
                        help="also recompute the Table 5 cell states, the history-free Biot laws and the solid compression")
    args = parser.parse_args()
    failed = []
    for name, command in steps(args.solve):
        print(f"\n=== {name}", flush=True)
        if subprocess.run([sys.executable, *map(str, command)], cwd=ROOT).returncode:
            failed.append(name)
    print("\nAll paper figures and tables regenerated; the numbers in the text agree with the paper." if not failed else f"\nFailed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
