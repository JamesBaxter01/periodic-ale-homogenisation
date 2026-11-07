"""Refinement check of the resolved consolidation column: how much the reference moves.

Press Run (or pass folders on the command line). Reads the production run folder BASE (fe2.npz and
resolved.npz) and the refined resolved runs in STUDY/<variant>/resolved.npz (written by
hpc/submit_resolved_refinement.sh: "mesh", a finer mesh on the same times, and "time", every interval
of the FE² time grid split in two). Statistics use the conventions of consolidation_error_table.py, on
each refined run's times up to the end common to all runs:

- change of the resolved reference: |refined - production| for settlement and drainage (relative to
  the production final value) and cell pressures (over the traction; cells 1 to n-1 and the top cell,
  from CELL_START);
- the FE² differences of Table 4 recomputed against each refined reference.

A run that has not finished is compared up to its last saved step (its status is printed), so the
script can be run while the jobs are still going. Writes STUDY/refinement_summary.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale", ROOT / "examples" / "multiscale"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from consolidation_error_table import CELL_START, statistics  # noqa: E402
from plot_paper_consolidation import read  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
BASE = ROOT / "results" / "consolidation" / "full_elastic_step_20261026_201037"   # production FE² + resolved
STUDY = None                      # refinement folder (submit_resolved_refinement.sh output); None: newest
VARIANTS = ["mesh", "time"]       # subfolders of STUDY with a resolved.npz
# ---------------------------------------------------------------------------


def newest_study():
    found = sorted((ROOT / "results" / "consolidation").glob("refinement_*"))
    if not found:
        raise SystemExit("no results/consolidation/refinement_* folder; pass one on the command line")
    return found[-1]


def load_resolved(path, area):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        return dict(time=data["time"], settlement=data["settlement"], drainage=data["cumulative_outflow"] / area,
                    pressure=data["cell_pressure"]), metadata


def truncated(history, end):
    keep = history["time"] <= end * (1 + 1e-12)
    return {key: (value[keep] if isinstance(value, np.ndarray) and len(value) == len(keep) else value)
            for key, value in history.items()}


def change(run, refined, production, end):
    """Max change of the resolved reference, on the refined run's times up to end."""
    t = refined["time"][(refined["time"] > 0) & (refined["time"] <= end * (1 + 1e-12))]
    on = lambda h, key: np.interp(t, h["time"], h[key])  # noqa: E731
    out = {}
    for key in ("settlement", "drainage"):
        out[key] = float(np.max(np.abs(on(refined, key) - on(production, key))) / abs(production[key][-1]))
    p = lambda h: np.array([np.interp(t, h["time"], h["pressure"][:, k])  # noqa: E731
                            for k in range(run["cells"])]).T
    error = np.abs(p(refined) - p(production)) / run["traction"]
    late = t >= CELL_START
    out["lower"] = float(error[late, :-1].max()) if late.any() else float("nan")
    out["top"] = float(error[late, -1].max()) if late.any() else float("nan")
    i = int(np.argmin(np.abs(production["time"] - t[-1])))
    out["final_settlement"] = float(np.interp(t[-1], refined["time"], refined["settlement"])
                                    / production["settlement"][i] - 1)
    return out


def main(study):
    run = read(BASE)
    area = json.loads(str(np.load(BASE / "fe2.npz", allow_pickle=False)["metadata"]))["macro"]["area"]
    production = run["resolved"]
    variants = {}
    for name in VARIANTS:
        path = study / name / "resolved.npz"
        if path.exists():
            variants[name] = load_resolved(path, area)
        else:
            print(f"{name}: no {path} yet")
    if not variants:
        raise SystemExit("no refined runs found")
    end = min([production["time"][-1], run["FE²"]["time"][-1]] + [h["time"][-1] for h, _ in variants.values()])

    def table3(resolved):
        if end <= CELL_START:  # too short for the pressure statistics (a smoke run)
            return None
        return statistics(dict(run, resolved=truncated(resolved, end), **{"FE²": truncated(run["FE²"], end)}))

    rows = []
    reference = table3(production)
    rows.append(("production", "complete", None, reference))
    for name, (history, metadata) in variants.items():
        settings = metadata.get("settings", {})
        label = f"maxh {settings.get('maxh')}, time grid / {settings.get('time_refine', 1)}"
        moved = change(run, history, production, end)
        fe2 = table3(history)
        rows.append((name, f"{metadata.get('status')} at t = {history['time'][-1]:.4g} s, "
                           f"{metadata.get('elements')} elements, {label}", moved, fe2))

    pct = lambda x: f"{100 * x:.3f} %"  # noqa: E731
    lines = [f"# Resolved refinement check (elastic step load)", "",
             f"Production: `{BASE.relative_to(ROOT)}`; study: `{study.relative_to(ROOT) if study.is_relative_to(ROOT) else study}`.",
             f"Compared up to t = {end:.4g} s (the end common to all runs).", "",
             "## Change of the resolved reference (refined minus production)", "",
             "| Run | Status | Settlement | Drained volume | Cells 1-9 pressure | Top-cell pressure | Settlement at end |",
             "|---|---|---|---|---|---|---|"]
    for name, status, moved, _ in rows[1:]:
        lines.append(f"| {name} | {status} | {pct(moved['settlement'])} | {pct(moved['drainage'])} | "
                     f"{moved['lower']:.4f} | {moved['top']:.4f} | {pct(moved['final_settlement'])} |")
    lines += ["", "Settlement and drained volume: max |change| over the production final value. Pressures: max "
              f"|change| over the traction from t = {CELL_START:g} s.", "",
              "## Table 4 statistics of FE² against each reference", "",
              "| Reference | Settlement max | Drained volume max | Cells 1-9 max | Top cell max (at t) |",
              "|---|---|---|---|---|"]
    for name, _, _, s in rows:
        if s is None:
            lines.append(f"| {name} | n/a (runs end before t = {CELL_START:g} s) | | | |")
            continue
        lines.append(f"| {name} | {pct(s['settlement'][0])} | {pct(s['drainage'][0])} | {s['lower'][0]:.4f} | "
                     f"{s['top'][0]:.4f} ({s['top'][1]:.3g} s) |")
    text = "\n".join(lines) + "\n"
    print(text)
    (study / "refinement_summary.md").write_text(text)
    print(f"Wrote {study / 'refinement_summary.md'}")


def command_line():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", nargs="?", type=Path, help="refinement folder (default STUDY, else the newest)")
    parser.add_argument("--base", type=Path, help="production run folder (default BASE)")
    args = parser.parse_args()
    BASE = args.base or BASE
    return Path(args.study or STUDY or newest_study()).resolve()


if __name__ == "__main__":
    main(command_line())
