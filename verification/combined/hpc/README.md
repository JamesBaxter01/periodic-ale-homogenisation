# Reproducing the paper's verification on a Slurm cluster

These are the scripts used for the verification numbers of the paper (Section 4).
Every setting is in `configs/verification.json`. One loading activates every input of the
cell (K = G = 50 Pa, mu_f = 1 Pa s, eta_s = 2 Pa s, zeta_s = 1 Pa s, channel width 0.33,
order 2), with s(t) = (1 - cos(pi t / T)) / 2:

| input | history | role |
|---|---|---|
| F33 | 1 - 0.2 s(t) | 20 % compression, as the pulse of Figure 4 |
| F12 | 0.02 s(t) | breaks the cell's symmetry: off-diagonal stress and tangent blocks nonzero |
| p | 20 sin(pi t / T) Pa | rises and falls: the storage power changes sign |
| grad p | 10 s(t) (1, 0.5, 0.25)/norm Pa/m | all three flux components nonzero |

Each study varies one numerical parameter, with the others fixed (production settings:
maxh 0.09, N = 100 steps, Newton tolerance 1e-8):

| study | varies | jobs | fixed |
|---|---|---|---|
| A, mesh | maxh 0.15, 0.12, 0.09, 0.06, 0.045, 0.03 (+ 0.045 at N = 200) | 14 | N = 100, Newton 1e-10 |
| B, time step | N = 25, 50, 100, 200, 400, 800 | 12 | maxh 0.09, Newton 1e-10 |
| C, Newton tolerance | 1e-6, 1e-8, 1e-10, 1e-12 | 8 | maxh 0.09, N = 100 |
| D, tangent | relative FD step 1e-1 ... 1e-6 (10 values) | 20 | states of C at 1e-12, Newton 1e-12 |
| F, mesh law | lambda_m / mu_m = 1, 5, 25 | 6 | maxh 0.09, N = 100, Newton 1e-10 |
| E, Hill-Mandel balance | recorded in every history of A-C and F | – | – |

Every study runs for both pulse durations, T = 0.2 s and T = 2 s. One job is one
history (or, in D, one FD step at T/2 and T) in its own process, on one node.

## 1. Set up (once)

```bash
git clone <repository> periodic-ale-homogenisation
cd periodic-ale-homogenisation
git checkout <the commit to verify>
module load calder/cpu miniforge3
conda env create -f environment.yml      # creates ngsolve-petsc; or use an existing NGSolve + PETSc environment
```

The jobs run `module load calder/cpu miniforge3` (a batch job starts without calder/cpu, which miniforge3 needs) and `conda activate ngsolve-petsc` (after
`eval "$(conda shell.bash hook)"`, as Calder's module has no `source activate`); edit
`run_study.sh` if your cluster names these differently. Check the environment with
a coarse run on a compute node (a few minutes):

```bash
srun --cpus-per-task=4 --mem=8G --time=01:00:00 --pty bash
module load calder/cpu miniforge3 && eval "$(conda shell.bash hook)" && conda activate ngsolve-petsc
python verification/combined/run.py --all --smoke --threads 4
```

This runs every study at maxh 0.7 (A: 0.7, 0.2, 0.15) with 4-16 steps and writes
`results/verification/combined_smoke/analysis/summary.md`. Smoke results are not
verification data.

## 2. Submit

Edit the settings block at the top of `verification/combined/hpc/submit.sh` if needed
(memory, wall time and segments per study, `MAX_WALLTIME`, `PARTITION`, `ACCOUNT`,
`MAX_PARALLEL`), then, from a login node:

```bash
bash verification/combined/hpc/submit.sh --check     # preview: job list and sbatch commands
bash verification/combined/hpc/submit.sh             # submit
```

`submit.sh` creates `results/verification/combined_<date>_<time>/`, writes its
`config.json` (the frozen settings), and submits one array job per study, plus a
collect job that runs `analyse.py` when everything has finished. Each task runs on
one node with 32 CPUs (`CPUS`): NGSolve's TaskManager uses all of them, and BLAS, OpenMP and
Numba are pinned to one thread.

Jobs are limited to 48 hours (`MAX_WALLTIME`; `submit.sh` refuses a longer `TIME_*`).
A history that would run longer stops at a checkpoint 20 minutes (`TIME_MARGIN`)
before its limit, judged from its longest step so far, and exits with code 76
(Slurm shows the task as FAILED with exit code 76; its `status.json` says `partial`).
Each study is therefore submitted as several segments (`SEGMENTS_A=3`, `SEGMENTS_B=2`,
`SEGMENTS_C=1`, `SEGMENTS_D=1`): the same array again, each segment starting when the
previous one has ended. A segment resumes the unfinished histories from their last
checkpoint and skips the complete ones within seconds. D starts after C's last
segment, the collect job after every study's last segment.

Commit any local change before submitting: each job records the git commit, a
SHA-256 digest of the solver (entry file plus `periodic_fsi/`), package versions,
the Slurm job id, threads, wall times and resident memory per step in its
`metadata.json` and `result.npz`.

To add a study to a campaign that is already running (e.g. F to one started with
A-D), update the repository first (`git pull`: the queued segments must run the new
`run.py`, which accepts a config that only adds studies), then:

```bash
bash verification/combined/hpc/submit.sh --studies F --resume results/verification/combined_YYYYmmdd_HHMMSS
```

This records F in the campaign's `config.json`, submits F's array and a new collect
job (the analysis can be rerun at any time).

## 3. Monitor, resume, collect

```bash
squeue -u $USER
tail -f results/verification/combined_*/slurm/A-*_11.out   # e.g. A, task 11
cat results/verification/combined_*/A/T2/05/status.json     # per-job state
```

Jobs write a restart checkpoint every 10 steps (`checkpoint_every`), and when they
stop before the time limit. If jobs are still unfinished after the last segment, or
one failed (fix the cause first), resubmit into the same folder:

```bash
bash verification/combined/hpc/submit.sh --resume results/verification/combined_YYYYmmdd_HHMMSS
```

Complete jobs are skipped; the others resume from their last checkpoint (a resumed
history matches an uninterrupted one to round-off). To bound any memory leak, jobs
with maxh <= 0.045 restart their solver process every 25 steps from the checkpoint
(`RESTART_EVERY`, `RESTART_MAXH` in `submit.sh`). Resident memory per step is in
each `result.npz` (`rss_gib`) and plotted in `analysis/fig_memory.pdf`.

The analysis can be rerun at any time, also on partial results, without NGSolve:

```bash
python verification/combined/analyse.py results/verification/combined_YYYYmmdd_HHMMSS
```

## 4. Outputs

`<campaign>/analysis/`:

- `summary.md`: jobs, observed orders, the error budget at production settings
  (maxh 0.09, N = 100, Newton 1e-8), the independence cross-check, the tolerance
  study, the mesh-law study, the FD tangent comparison and the balance tables;
- CSV tables: `inventory`, `orders`, `errors`, `error_budget`, `cross_check`,
  `tolerance`, `tangent_fd`, `tangent_fd_summary`, `mesh_law`, `balance_runs`, `balance_summary`;
- figures (PNG and PDF): `fig_A_mesh`, `fig_B_time_step`, `fig_C_tolerance`,
  `fig_D_tangent`, `fig_E_balance`, `fig_memory`.

Per job, `<campaign>/<study>/T<duration>/<index>/`: `result.npz` (per-step inputs,
outputs, Hill-Mandel diagnostics, Newton record, mesh quality, timings and memory;
outputs at T/4, T/2, 3T/4, T; tangents at T/2 and T), `metadata.json`, `run.log`,
`status.json`, per-step records in `steps/` and the restart checkpoints in `states/`
(the two most recent, and in study C the states before each tangent time, which D uses).

## Resources

The defaults are starting allocations; adjust them after the first jobs, whose
`metadata.json` records wall time and peak memory (and `run.log` the time per step).
An FD job (D) is not split: its 54 solves at maxh 0.09 must fit in one segment. The finest mesh (A, maxh 0.03,
about 190k elements and 1.2M unknowns) dominates memory, mainly through the corrector
factorisation at T/2 and T; B's N = 800 history dominates time. On a laptop
(4 threads), maxh 0.09 took about 26 s per step with correctors at every step and
1.6 GB.
