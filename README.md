# Periodic ALE–FSI computational homogenisation

Code and data for **Transient pore-resolved computational homogenisation of visco-hyperelastic porous materials**, by James Baxter Chapman, Mark Walkley, David Head, Shufan Yang and Gregory de Boer.

A periodic cell of a visco-hyperelastic solid skeleton and a Stokes pore fluid is solved with a monolithic arbitrary Lagrangian–Eulerian (ALE) fluid–structure interaction formulation. The cell returns the homogenised stress, fluid content and flux, with their consistent 13×13 tangent, for use as a material point in FE² macroscale problems. A fully resolved solver, built from the same equations, provides the reference for the FE² results.

## Quick start

From nothing to the paper's figures and tables in five steps, on macOS or Linux (on Windows, use [WSL](https://learn.microsoft.com/windows/wsl/install) and follow the Linux steps). Expect about ten minutes, mostly downloads. Type the commands in a terminal.

**1. Install conda**, if you don't have it (`conda --version` prints a version if you do). Miniforge is the smallest:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Accept the defaults, then close and reopen the terminal.

**2. Download the repository** (or, without git, use *Code → Download ZIP* on GitHub and unzip it):

```bash
git clone https://github.com/JamesBaxter01/periodic-ale-homogenisation.git
cd periodic-ale-homogenisation
```

**3. Create the environment** (Python 3.14, NGSolve, PETSc with MUMPS, Jupyter), once:

```bash
conda env create -f environment.yml
```

**4. Activate it**, in every new terminal:

```bash
conda activate ngsolve-petsc
```

**5. Reproduce the paper's figures and tables.** From the repository folder:

```bash
python run.py paper
```

In about a minute this redraws every figure and rebuilds every table from the saved data in `results/`, recomputes every number quoted in the text of Sections 4 and 5, compares each with the value printed in the paper and reports any that disagree. No cell is solved. Figure 1 is typeset with LaTeX, so that step needs a LaTeX installation with the `stix` and `bm` packages. The redrawn PDFs differ from the committed ones only in their date stamps.

**Troubleshooting**

- `conda: command not found` after step 1: open a new terminal, or run `~/miniforge3/bin/conda init` and then open one.
- `ModuleNotFoundError: No module named 'ngsolve'`: the environment is not active; run step 4.

The pinned versions are those that computed the paper's results: Python 3.14, NGSolve/Netgen OCC 6.2.2606 and serial petsc4py/PETSc 3.25.6 with MUMPS 5.8.2. Run commands from the repository folder; `python run.py --help` lists the paper commands. To use the solver from another project, install the repository into that project's environment with `python -m pip install --no-deps -e /path/to/repository`; `homogenisation_microscale`, `periodic_fsi`, `multiscale` and `resolved` then import directly.

## Repository layout

```text
homogenisation_microscale.py                 Periodic cell solver (entry point: FSI, FSISession)
periodic_fsi/                                Its implementation: forms, Newton, correctors, homogenisation
geometry.py                                  Periodic square-channel cell
multiscale/                                  Macro–micro coupling (FE²) for any material-point model
resolved/                                    Fully resolved column from the cell solver's equations
examples/microscale/                         Paper Figures 3–4 and Table 3 (data and plotting)
examples/multiscale/                         FE² and resolved consolidation (step and cyclic loads), paper Figures 5–8 and Tables 4–6, Slurm scripts
verification/                                Verification campaign (paper Section 4): drivers, analysis, Slurm scripts
configs/                                     Verification settings
results/                                     Data and figures of the paper (see Data)
run.py                                       Command-line entry point
```

## Using the cell solver

```python
import numpy as np
from geometry import square_geo
from homogenisation_microscale import FSI

result = FSI(K=50.0, G=50.0, mu_f=1.0, eta_s=2.0, zeta_s=1.0, dt=0.01, t_final=0.1,
             F_target=np.diag([1.0, 1.0, 0.98]), P_target=0.0, G_target=np.zeros(3),
             geometry=square_geo, homogenise=True)
stress = result["homogenised_response"]["P_total"]    # (3, 3) nominal stress
tangent = result["homogenised_response"]["jacobian"]  # (13, 13)
state = result["restart_state"]                       # pass as initial_state to continue
```

`FSISession` keeps the cell, its forms and compiled code across calls, for sequences of steps such as an FE² run. Parameters and returned entries are described in the docstrings of `FSI` and `FSISession` in `periodic_fsi/api.py`.

## Reproducing the paper

Every figure and table of the paper is drawn from the data in `results/` by the scripts below. `python run.py paper` regenerates all of them with their LaTeX floats and recomputes every number quoted in the text of Sections 4 and 5, comparing each with the value printed in the paper (about a minute, no cell solves); the comparison is in `results/paper/tables/paper_numbers.md`. With `--solve` it also recomputes the Table 5 cell states, the history-free Biot laws and the solid compression of Section 5.1. Each script keeps its settings at the top and runs without arguments.

| Paper | Data | Solve | Plot or tabulate |
|---|---|---|---|
| Figure 1: transition between the scales | – (a schematic) | – | `plot_paper_geometry.py` |
| Figure 2: cell geometry | `geometry.py` (channel width 0.33) | – | `plot_paper_geometry.py` |
| Tables 1–2: verification | `results/verification/combined_20261026_201037/analysis/` | `verification/combined/hpc/submit.sh` (Slurm) | `python run.py verify-tables` |
| Figure 3: secant Biot coefficient | `results/paper/figure3_biot.npz` | `python run.py figure3` | `python run.py plot` |
| Figure 4: compression pulse, transient and stationary cell | `results/paper/figure4_cycles.npz`, `figure4_stationary.npz`; `figure4_rates.npz` (an earlier sweep of the pulse at nine durations, viscoelastic) | `python run.py figure4`, `python run.py figure4-stationary` (Slurm: `examples/microscale/hpc/submit_figure4_stationary.sh`); the nine durations: `python run.py figure4 --rates` (`figure4_rates.npz`) | `python run.py plot` |
| Figure 5: consolidation set-up | `results/consolidation/full_elastic_step_20261026_201037/` (mesh and macroscale settings) | – | `plot_paper_geometry.py` |
| Table 3: departure of the cell from the stationary cell at pulse durations of 0.2 to 200 s | `results/paper/criterion_20261108_153003/`: `figure4_criterion_{elastic,viscoelastic}.npz` (the pulse of Figure 4 at thirteen durations from 0.2 to 200 s; the first nine reproduce `figure4_rates.npz`) and `figure4_criterion_steps40.npz` (T = 2 and 20 s at 40 steps per pulse), with `figure4_stationary.npz` | `python run.py figure4 --criterion --solids elastic` (and `viscoelastic`), `--criterion-check` (Slurm: `examples/microscale/hpc/submit_figure4_criterion.sh`) | `python run.py plot` |
| Figure 6, Table 4: step-load consolidation, FE² against resolved | `results/consolidation/full_*_step_20261026_201037/` | `examples/multiscale/hpc/submit_consolidation.sh` (Slurm) | `plot_paper_consolidation.py`, `consolidation_error_table.py` |
| Table 5: response derivatives | `results/paper/tables/table_5_consolidation_parameters.csv` | `consolidation_parameter_table.py` (replays cells of the viscoelastic run) | `consolidation_parameter_table.py --table-only` |
| Section 5.3.1: ramp load, refined resolved reference, history-free Biot law | `results/consolidation/full_elastic_20261026_201037/`, `refinement_20261026_201037/`, `baselines_20261028/` | `submit_consolidation.sh`, `submit_resolved_refinement.sh` (Slurm), `baseline_models.py` | `compare_consolidation.py`, `resolved_refinement.py`, `baseline_models.py` |
| Figure 7: a consolidation cell at three times | `results/consolidation/full_viscoelastic_step_20261026_201037/cell_replay_bottom.npz` (the run's cell nearest the base, replayed) | `consolidation_cell_replay.py` (about an hour) | `plot_paper_cells.py` |
| Figure 8, Table 6: cyclic consolidation | `results/cyclic/20261026_201037/` (T = 0.2 and 2 s) and `results/cyclic/20261108_220130/` (T = 20 and 200 s, Table 6) | `examples/multiscale/hpc/submit_cyclic.sh` (Slurm; `--periods "20 200"` for the longer periods) | `plot_paper_cyclic.py` (Figure 8, Table 6), `compare_cyclic.py` |
| Section 5.3.3: macroscopic cost (the FE² runs replayed from their stored cell responses, no cell solves) | `results/consolidation/full_*_step_20261026_201037/` | – | `replay_fe2.py <run>/fe2.npz` |
| Sections 4 and 5: every number in the text | the folders above, the Slurm logs for the timings, and `results/paper/figure3_solid_compression.json` (`examples/microscale/figure3_solid_compression.py`) | – | `examples/paper_numbers.py` |

The multiscale scripts are in `examples/multiscale/`. Figures go to `results/paper/figures/` and tables to `results/paper/tables/`. The FE² and resolved runs were made on an HPC cluster; each script has a `SMOKE_TEST` setting (or `--smoke`) for a coarse run on a laptop.

On a Slurm cluster, `bash examples/hpc/submit_paper_runs.sh` submits every solver run behind the paper at once: the cell figures, the verification campaign, the consolidation and cyclic columns, and the runs that read the step-load FE² runs (Table 5, Figure 7, the storage moment and the refined resolved reference), which wait for them in the queue. `SMOKE=1` in front submits the same jobs in smoke mode, and `--check` shows them without submitting. The pulse sweep of Table 3 and the cyclic column at T = 20 and 200 s were submitted on their own: `bash examples/microscale/hpc/submit_figure4_criterion.sh` and `bash examples/multiscale/hpc/submit_cyclic.sh --periods "20 200"`.

### The resolved reference

The FE² consolidation results are compared with a fully resolved column: ten periodic cells stacked under a porous platen, every pore resolved, solved with the cell solver's own equations (`resolved/`). The baseline mesh has 48 537 elements (236 206 unknowns).

### Verification

The verification campaign uses one loading that activates every input (20 % compression, a small shear, a pressure that rises and falls, and a pressure gradient with three nonzero components) for pulse durations of 0.2 s and 2 s. Each study varies one numerical parameter: the mesh (A), the time step (B), the Newton tolerance (C), the finite-difference step of a tangent check (D) and the mesh law (F); the Hill–Mandel balance (E) is recorded in every history. The settings are in `configs/verification.json` and [verification/combined/hpc/README.md](verification/combined/hpc/README.md) describes the Slurm runs. A coarse check of the whole pipeline:

```sh
python run.py verify --all --smoke
```

Jobs stop at a checkpoint before their wall-time limit and resume from it; a resumed history matches an uninterrupted one to round-off. Each job records the solver digest and refuses to resume on a different solver.

### Data

| Folder | Contents |
|---|---|
| `results/paper/` | Figures 3–4 data (`.npz`), every paper figure (PDF), Table 5 values, and `tables/paper_numbers.md`, the numbers in the text against the paper |
| `results/verification/combined_20261026_201037/analysis/` | Verification errors, orders, tangent and balance diagnostics, figures |
| `results/consolidation/full_{elastic,viscoelastic}_step_20261026_201037/` | FE² (`fe2.npz`) and resolved (`resolved.npz`, mesh) step-load consolidation, comparison metrics, Slurm logs |
| `results/consolidation/full_elastic_20261026_201037/` | The same for a 10 s ramp load (elastic skeleton) |
| `results/consolidation/refinement_20261026_201037/` | Resolved step-load reference on a finer mesh (`mesh/`) and with halved time steps (`time/`), with `refinement_summary.md` |
| `results/consolidation/baselines_20261028/` | History-free Biot laws (linear and finite-strain) on the step-load column, compared in `baselines.md` |
| `results/cyclic/20261026_201037/` | Cyclic consolidation: FE² (`fe2.npz`) and resolved (`resolved.npz`) for each skeleton and period, comparison summaries, Slurm logs |

FE² and resolved restart checkpoints (about 1 GB per run) and the raw verification histories are not included. The crosses in the Biot figure are approximate readings from Sandström et al. (2016), Figure 5(b), [doi:10.1016/j.cma.2016.05.021](https://doi.org/10.1016/j.cma.2016.05.021).

## Licence

- **Code** (all source code, scripts and configuration files): [MIT licence](LICENSE).
- **Data, figures and documentation** (everything under `results/`, and the Markdown documentation): [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA.md).

## Citation

Please cite the paper when using this code or data; `CITATION.cff` holds the citation metadata.
