"""History-free Biot baselines for the FE² consolidation column.

Press Run (or python examples/multiscale/baseline_models.py [--smoke]). The equilibrated, history-free
alternative to the FE² cell is a single-scale Biot law fitted to the cell's equilibrated response at rest:
the Table 5 rest state (a fresh cell, a long step DT_FIT from rest, elastic solid), with an isotropic
skeleton matching C_1111 and C_1122 (lam = C_1122, mu = (C_1111 - C_1122) / 2), Biot coefficient alpha,
Biot modulus M, permeability kappa = -dQ_1/dG_1 and porosity. Both closed-form models of multiscale/ are
run with it, in the same macro problems and solvers as FE²:

    linear         LinearPoroelastic (small strain)
    finite_strain  FiniteStrainPoroelastic (neo-Hookean skeleton, pulled-back Darcy flow)

Problem: the step-load consolidation column (consolidation_multiscale.py --step). The baselines are
compared with the resolved runs of both skeletons, alongside FE², with the Table 4 statistics (the
history-free Biot law of Section 5.3.1).

Saves OUTPUT/consolidation_<model>/fe2.npz (the driver's history format) and OUTPUT/baselines.md.
"""

import argparse
import contextlib
import importlib
import io
import json
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "examples" / "microscale", ROOT / "examples" / "multiscale", ROOT / "verification"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
CONSOLIDATION = {"elastic": ROOT / "results" / "consolidation" / "full_elastic_step_20261026_201037",
                 "viscoelastic": ROOT / "results" / "consolidation" / "full_viscoelastic_step_20261026_201037"}
MODELS = ["finite_strain", "linear"]
DT_FIT = 1000.0                   # long step for the equilibrated rest response (s), as Table 5
THREADS = 8
OUTPUT = ROOT / "results" / "consolidation" / "baselines"
SMOKE = False
# ---------------------------------------------------------------------------


def fit(cell_settings):
    """Biot parameters of the cell's equilibrated rest response (elastic solid)."""
    from ngsolve import SetNumThreads, TaskManager
    from geometry import square_geo
    from homogenisation_microscale import FSI

    SetNumThreads(THREADS)
    settings = {k: v for k, v in cell_settings.items()
                if k not in ("compile_tangent", "print_iterations", "eta_s", "zeta_s", "newton_options")}
    if SMOKE:
        settings["maxh"] = 0.5
    with TaskManager(), contextlib.redirect_stdout(io.StringIO()):
        r = FSI(**settings, geometry=partial(square_geo, 0.33), dt=DT_FIT, t_final=DT_FIT, F_target=np.eye(3),
                P_target=0.0, G_target=np.zeros(3), homogenise=True, print_iterations=False)
    h = r["homogenised_response"]
    J, out = np.asarray(h["jacobian"]), np.asarray(h["outputs"])
    C1111, C1122 = J[0, 0], J[0, 4]
    lam, mu = C1122, (C1111 - C1122) / 2
    parameters = dict(K=lam + 2 * mu / 3, G=mu, kappa=-J[10, 10], alpha=-J[0, 9], biot_modulus=1.0 / J[9, 9],
                      porosity=out[9])
    checks = dict(C1111=C1111, C3333=J[8, 8], C1122=C1122, C1212=J[1, 1], kappa_33=-J[12, 12], alpha_33=-J[8, 9])
    return {k: float(v) for k, v in parameters.items()}, {k: float(v) for k, v in checks.items()}


def run_driver(module_name, model, parameters, output, overrides):
    """Run a macro driver's main() with a closed-form model (its settings blocks otherwise unchanged)."""
    driver = importlib.import_module(module_name)
    driver.MODEL, driver.MODEL_SETTINGS, driver.OUTPUT = model, dict(parameters), output
    driver.SMOKE_OUTPUT = output
    for key, value in overrides.items():
        setattr(driver, key, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    driver.main()
    return time.perf_counter() - started


def consolidation_rows(folder, model):
    from consolidation_error_table import statistics
    from plot_paper_consolidation import read

    rows = []
    for skeleton, production in CONSOLIDATION.items():
        compare = folder / f"against_{skeleton}"
        compare.mkdir(exist_ok=True)
        shutil.copy(folder / "fe2.npz", compare / "fe2.npz")
        shutil.copy(production / "resolved.npz", compare / "resolved.npz")
        base, fe2 = statistics(read(compare)), statistics(read(production))
        rows.append((skeleton, model, base, fe2))
    return rows


def main():
    folder = Path(OUTPUT)
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(CONSOLIDATION["elastic"] / "fe2.npz", allow_pickle=False) as d:
        consolidation_cell = json.loads(str(d["metadata"]))["cell"]
    lines = ["# History-free baselines", ""]
    results = {}
    for problem, cell, module, overrides in (
            ("consolidation", consolidation_cell, "consolidation_multiscale", dict(STEP_LOAD=True, SKELETON="elastic")),):
        parameters, checks = fit(cell)
        lines += [f"## {problem}: fitted parameters", "", "`" + json.dumps({k: float(f"{v:.6g}") for k, v in parameters.items()}) + "`",
                  "", "Cell rest tangent (cubic): `" + json.dumps({k: float(f"{v:.6g}") for k, v in checks.items()}) + "`", ""]
        for model in MODELS:
            out = folder / f"{problem}_{model}" / "fe2.npz"
            if SMOKE:
                overrides = dict(overrides, SMOKE_TEST=True)
            seconds = run_driver(module, model, parameters, out, overrides)
            results[(problem, model)] = (out, seconds)
            print(f"{problem} {model}: {seconds:.0f} s", flush=True)

    if SMOKE:
        print("Smoke run: comparison skipped (the baselines are not on the production meshes).")
        return
    lines += ["## Consolidation (step load): Table 4 statistics against the resolved runs", "",
              "| Resolved skeleton | Model | Settlement max | Drained volume max | Cells 1-9 max | Top cell max (at t) | Final settlement (mm) |",
              "|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for skeleton, _, base, fe2 in consolidation_rows(results[("consolidation", model)][0].parent, model):
            for label, s in ((model, base), ("FE²", fe2)):
                if label == "FE²" and model != MODELS[0]:
                    continue
                lines.append(f"| {skeleton} | {label} | {100 * s['settlement'][0]:.3f} % | {100 * s['drainage'][0]:.3f} % | "
                             f"{s['lower'][0]:.4f} | {s['top'][0]:.4f} ({s['top'][1]:.3g} s) | {1e3 * s['final_fe2']:.3f} |")
    lines += ["", "Each row compares the named model with the resolved run of that skeleton (Table 4 conventions)."]
    text = "\n".join(lines) + "\n"
    print(text)
    (folder / "baselines.md").write_text(text)
    print(f"Wrote {folder / 'baselines.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="coarse fit and the drivers' smoke settings")
    parser.add_argument("--output", type=Path, help=f"folder (default {OUTPUT})")
    parser.add_argument("--threads", type=int, help=f"NGSolve threads for the fit (default {THREADS})")
    args = parser.parse_args()
    SMOKE = args.smoke
    OUTPUT = args.output or OUTPUT
    THREADS = args.threads or THREADS
    main()
