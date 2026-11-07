#!/usr/bin/env python3
"""Run jobs of the verification campaign (paper Section 4; hpc/README.md).

One job is one history (studies A, B, C, F) or one finite-difference tangent check (D):

  run.py --study A --duration 0 --index 2      one job: study, duration index, level index
  run.py --study A --task 7                    a Slurm array task: duration = task // n, index = task % n
  run.py --list                                the jobs of every study (hpc/submit.sh reads the counts)
  run.py --all [--smoke]                       every job in turn, then analyse.py
  run.py --prepare --output DIR                write DIR/config.json (hpc/submit.sh, before the arrays start)

Options: --output DIR (campaign folder, default results/verification/combined, or
results/verification/combined_smoke with --smoke), --config PATH (default
configs/verification.json), --threads N (default $SLURM_CPUS_PER_TASK, else the
config's local_threads), --restart-every K with optional --restart-maxh H (restart
the process every K steps, from its checkpoint, for jobs with maxh <= H), --time-limit
[D-]HH:MM:SS with --time-margin MIN (stop at a checkpoint MIN minutes, default 20,
before the job's wall-time limit; $SLURM_JOB_END_TIME is used when Slurm sets it),
--smoke (the config's coarse smoke settings: the whole pipeline in minutes on a laptop).

A job stopped before its time limit exits with code 76 and resumes from its
checkpoint when run again (hpc/submit.sh queues follow-on segments for this).

Each job runs in a child process; nothing accumulates across jobs. A job writes
<output>/<study>/T<duration>/<index>/: result.npz and metadata.json when complete,
status.json, steps/ (per-step records), states/ (restart checkpoints) and run.log.
A rerun skips complete jobs and resumes the others from their last checkpoint.
D reads the restart states that study C saves before each tangent time, so D runs
after C. The first job writes <output>/config.json; later jobs must match it, except
that a config adding a study (e.g. F to a campaign started with A-D) extends it:
--prepare (hpc/submit.sh) and --all record the added study in config.json; a single
job of an added study needs that record first.
"""

from pathlib import Path
import argparse
import copy
import datetime
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDIES = ("A", "B", "C", "D", "F")
RESTART_EXIT = 75  # a worker that stops early so the driver restarts it from the checkpoint
TIME_EXIT = 76  # a job that stopped at a checkpoint before its wall-time limit; a later run resumes it


# ---------------------------------------------------------------------------
# Settings and jobs (standard library only: submit.sh runs --list on the login node)
# ---------------------------------------------------------------------------

def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_settings(config, smoke):
    cfg = json.loads(Path(config).read_text())
    smoke_settings = cfg.pop("smoke", {})
    if smoke:
        cfg = merge(cfg, {k: v for k, v in smoke_settings.items() if k != "description"})
        cfg["smoke"] = True
    return cfg


def study_jobs(cfg, study):
    """The jobs of one study for duration index 0, in index order (same for every duration)."""
    s = cfg["studies"][study]
    p = cfg["production"]
    if study == "A":
        jobs = [dict(maxh=h, steps=s["steps"], newton_tolerance=s["newton_tolerance"]) for h in s["levels"]]
        jobs += [dict(maxh=c["maxh"], steps=c["steps"], newton_tolerance=s["newton_tolerance"], cross_check=True)
                 for c in s.get("cross_check", [])]
    elif study == "B":
        jobs = [dict(maxh=s.get("maxh", p["maxh"]), steps=n, newton_tolerance=s["newton_tolerance"]) for n in s["levels"]]
    elif study == "C":
        jobs = [dict(maxh=s.get("maxh", p["maxh"]), steps=s.get("steps", p["steps"]), newton_tolerance=t)
                for t in s["levels"]]
    elif study == "D":
        source = source_job(cfg, 0)
        jobs = [dict(maxh=source["maxh"], steps=source["steps"], newton_tolerance=s["newton_tolerance"], fd_step=e)
                for e in s["levels"]]
    elif study == "F":
        jobs = [dict(maxh=s.get("maxh", p["maxh"]), steps=s.get("steps", p["steps"]), newton_tolerance=s["newton_tolerance"],
                     mesh_lame_ratio=r) for r in s["levels"]]
    else:
        raise ValueError(study)
    return jobs


def job(cfg, study, duration, index):
    jobs = study_jobs(cfg, study)
    durations = cfg["loading"]["durations"]
    if not 0 <= duration < len(durations) or not 0 <= index < len(jobs):
        raise ValueError(f"study {study} has durations 0..{len(durations) - 1} and indices 0..{len(jobs) - 1}")
    T = durations[duration]
    out = dict(jobs[index], study=study, duration_index=duration, index=index, T=T,
               folder=f"{study}/T{T:g}/{index:02d}")
    out.setdefault("cross_check", False)
    out.setdefault("fd_step", None)
    return out


def source_job(cfg, duration):
    """The study-C history whose saved states D differentiates."""
    source = cfg["studies"]["D"]["source"]
    levels = study_jobs(cfg, source["study"])
    matches = [i for i, j in enumerate(levels) if math.isclose(j["newton_tolerance"], source["newton_tolerance"])]
    if len(matches) != 1:
        raise ValueError("D's source must match exactly one study-C level")
    return job(cfg, source["study"], duration, matches[0])


def all_jobs(cfg, study):
    n = len(study_jobs(cfg, study))
    return [job(cfg, study, d, i) for d in range(len(cfg["loading"]["durations"])) for i in range(n)]


def report_steps(steps, fractions):
    """Step indices of the times fractions * T that fall on a step of a steps-step history.

    A time between steps is skipped (e.g. T/4 for N = 25 or 50): that history then
    reports, and enters the convergence studies, only at the times it does reach.
    """
    return [int(round(f * steps)) for f in fractions if abs(f * steps - round(f * steps)) <= 1e-9 and round(f * steps) >= 1]


def skipped_fractions(steps, fractions):
    """The fractions of T that are not steps of a steps-step history."""
    return [f for f in fractions if abs(f * steps - round(f * steps)) > 1e-9 or round(f * steps) < 1]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def jsonable(x):
    if hasattr(x, "tolist"):
        return jsonable(x.tolist())
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False))
    tmp.replace(path)


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def memory_gib():
    """(current, peak) resident memory of this process in GiB; current is NaN if unavailable."""
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2)
    current = float("nan")
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    current = int(line.split()[1]) / 1024**2
    except OSError:
        try:
            import psutil
            current = psutil.Process().memory_info().rss / 1024**3
        except ImportError:
            pass
    return current, peak


def git_state():
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return dict(commit=commit, modified_tracked_files=dirty.splitlines())
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, modified_tracked_files=None)


def versions():
    out = dict(python=sys.version, platform=platform.platform())
    for name in ("ngsolve", "netgen", "numpy", "scipy", "petsc4py", "mpi4py"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            out[name] = None
    try:
        from petsc4py import PETSc
        out["petsc"] = ".".join(map(str, PETSc.Sys.getVersion()))
    except Exception:
        pass
    return out


def slurm_environment():
    keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION",
            "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_CLUSTER_NAME")
    return {k: os.environ[k] for k in keys if k in os.environ}


def thread_environment(threads):
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    return dict(ngsolve_threads=threads, os_cpu_count=os.cpu_count(), **{k: os.environ.get(k) for k in keys})


class Log:
    """Print to stdout (the Slurm log) and append to the job's run.log."""

    def __init__(self, folder):
        self.path = Path(folder) / "run.log"

    def __call__(self, message):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------

def solver_options(cfg, j):
    from functools import partial
    sys.path.insert(0, str(ROOT))
    from geometry import square_geo
    m, d = cfg["material"], cfg["discretisation"]
    tol = float(j["newton_tolerance"])
    # Study F varies the mesh law's lambda_m / mu_m (the only parameter of the law the solver uses).
    mesh_lambda = d["mesh_lame_lambda"] if j.get("mesh_lame_ratio") is None else j["mesh_lame_ratio"] * d["mesh_mu"]
    return dict(K=m["K"], G=m["G"], mu_f=m["mu_f"], eta_s=m["eta_s"], zeta_s=m["zeta_s"], rve_length=m["rve_length"],
                maxh=j["maxh"], order=d["order"], quadrature_order=d["quadrature_order"], mesh_mu=d["mesh_mu"],
                mesh_lame_lambda=mesh_lambda, geometry=partial(square_geo, m["channel_width"]),
                newton_options=dict(tolerance=tol, relative_tolerance=tol), homogenise=True,
                print_iterations=False, Profiling=False)


def newton_summary(r):
    histories = r["newton_histories"]
    if not histories or not all(h and h[-1].get("converged", False) for h in histories):
        raise RuntimeError("Newton did not converge")
    final = histories[-1][-1]
    return dict(newton_iterations=sum(len(h) for h in histories), newton_increments=len(histories),
                newton_residual=float(final["residual_after_step"]), newton_target=float(final["convergence_target"]),
                newton_initial_residual=float(final["initial_residual"]),
                gmres_iterations=sum(int(it.get("gmres_iterations", 0) or 0) for h in histories for it in h))


def numeric_items(prefix, d):
    import numpy as np
    out = {}
    for key, value in d.items():
        if isinstance(value, (dict, str, bytes)) or value is None:
            continue
        a = np.asarray(value)
        if a.dtype.kind in "fiub":
            out[prefix + key] = a.astype(float)
    return out


# ---------------------------------------------------------------------------
# Workers (run in their own process)
# ---------------------------------------------------------------------------

def history_worker(cfg, j, folder, threads, restart_every, deadline=None):
    import gc
    import numpy as np
    import ngsolve
    from ngsolve import SetNumThreads, TaskManager
    sys.path[:0] = [str(ROOT), str(ROOT / "verification"), str(HERE)]
    from homogenisation_microscale import FSISession
    from state_io import save_snapshot, load_snapshot, mark_retained_snapshots, remove_retired_snapshots, solver_sha256
    import loading as L

    log = Log(folder)
    SetNumThreads(threads)
    T, N = float(j["T"]), int(j["steps"])
    dt = T / N
    path = L.loading_path(T, N, cfg["loading"])
    report = report_steps(N, cfg["report_fractions"])
    tangent = report_steps(N, cfg["tangent_fractions"])
    skipped = skipped_fractions(N, cfg["report_fractions"] + cfg["tangent_fractions"])
    if skipped:
        log(f"{j['folder']}: times {', '.join(f'{f:g} T' for f in sorted(set(skipped)))} are not steps of a {N}-step "
            f"history and are not reported (outputs at steps {report}, tangents at steps {tangent})")
    # Study C keeps the states before each tangent time: D's frozen histories.
    bases = [k - 1 for k in tangent] if j["study"] == cfg["studies"]["D"]["source"]["study"] else []
    digest = solver_sha256(ROOT / "homogenisation_microscale.py")
    status_path = folder / "status.json"
    # A job that failed before its first step left only its state and error: fill in the rest. The saved
    # status overrides the defaults (a resumed job's has every key, last_step included).
    status = {**dict(last_step=0, snapshots=[], bases={}, processes=[], solve_seconds=0.0),
              **(json.loads(status_path.read_text()) if status_path.exists() else {})}
    if status.get("solver_sha256", digest) != digest:
        raise RuntimeError("The solver changed since this job started; start it in a new campaign folder")
    status.update(state="running", solver_sha256=digest, job=j)
    status.pop("error", None)
    process = dict(pid=os.getpid(), host=socket.gethostname(), started=time.time(), first_step=status["last_step"] + 1,
                   slurm=slurm_environment())
    status["processes"].append(process)
    save_json(status_path, status)
    (folder / "steps").mkdir(exist_ok=True)

    state = None
    if status["last_step"]:
        item = status["snapshots"][-1]
        snapshot = load_snapshot(folder / "states" / item["path"], expected_sha256=item["sha256"])
        state = snapshot["result"]["restart_state"]
        del snapshot
        log(f"resumed from the checkpoint at step {item['increment']}")
    first = status["last_step"] + 1
    case = dict(id=j["folder"], maxh=j["maxh"], dt=dt, quadrature_order=cfg["discretisation"]["quadrature_order"])

    def snapshot(r, k, name):
        return save_snapshot(folder / "states" / name, r, case=case, settings=dict(cfg=cfg, job=j), increment=k,
                             time=k * dt, compression=1 - float(path[k, 8]), solver_sha256=digest,
                             previous_restart_state=None)

    started = time.perf_counter()
    if deadline is not None:
        log(f"stops at a checkpoint before {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))} "
            f"({(deadline - time.time()) / 3600:.2f} h from now)")
        if time.time() + 120 > deadline:
            log("too close to the wall-time limit to start a step; a later run resumes")
            return TIME_EXIT
    longest = {False: 0.0, True: 0.0}  # longest outputs-only and corrector step of this process (s)
    checkpoint_seconds = 60.0  # until a checkpoint has been timed
    out_of_time = False
    with TaskManager(), FSISession(**solver_options(cfg, j), diagnostics=True, tangent_method=None) as run:
        for k in range(first, N + 1):
            z = path[k]
            with_tangent = k in tangent
            t0 = time.perf_counter()
            r = run(dt=dt, t_final=dt, F_target=z[:9].reshape(3, 3), P_target=float(z[9]), G_target=z[10:].copy(),
                    initial_state=state, tangent_method="linear correctors" if with_tangent else None)
            seconds = time.perf_counter() - t0
            a = r["homogenised_response"]
            outputs = np.asarray(a["outputs"], dtype=float)
            if not np.isfinite(outputs).all():
                raise RuntimeError(f"non-finite outputs at step {k}")
            np.testing.assert_allclose(a["inputs"], z, rtol=1e-12, atol=1e-13)
            hm = a["hill_mandel_diagnostics"]
            F, F_old = z[:9].reshape(3, 3), path[k - 1, :9].reshape(3, 3)
            jac = np.asarray(a["jacobian"], dtype=float) if with_tangent else np.full((13, 13), np.nan)
            if with_tangent and not np.isfinite(jac).all():
                raise RuntimeError(f"non-finite tangent at step {k}")
            quality = r["mesh_quality_summary"] or {}
            current, peak = memory_gib()
            record = dict(step=k, time=k * dt, seconds=seconds, rss_gib=current, rss_peak_gib=peak, pid=os.getpid(),
                          inputs=z, outputs=outputs, tangent=jac, has_tangent=with_tangent,
                          input_scales=a["physical_scales"]["input"], output_scales=a["physical_scales"]["output"],
                          stress_power=float(np.sum(np.asarray(hm["P_endpoint"]) * (F - F_old) / dt)),
                          storage_power=float(z[9] * hm["m_rate_be"]),
                          flux_power=float(-(F.T @ z[10:]) @ np.asarray(hm["Q0"])),
                          ale_jacobian_min=quality.get("ale_jacobian_min", np.nan),
                          ale_jacobian_max=quality.get("ale_jacobian_max", np.nan),
                          mesh_quality_min=quality.get("min", np.nan),
                          inverted_elements=quality.get("inverted_element_count", -1),
                          n_elements=r["mesh"].ne, total_dofs=r["total_dofs"],
                          **newton_summary(r),
                          **numeric_items("hm_", hm), **numeric_items("hmhat_", a["hill_mandel_diagnostics_hat"]))
            save_npz(folder / "steps" / f"step_{k:05d}.npz", **record)
            if k in bases:
                status["bases"][str(k)] = snapshot(r, k, f"base_{k:05d}.pkl")
            state = r["restart_state"]
            status["solve_seconds"] += seconds
            longest[with_tangent] = max(longest[with_tangent], seconds)
            if deadline is not None and k < N:
                # Will the next step and a checkpoint fit? A first corrector step is taken as 3x an outputs-only one.
                following = k + 1 in tangent
                estimate = longest[following] or 3 * longest[False] or longest[True]
                out_of_time = time.time() + 1.5 * estimate + 2 * checkpoint_seconds > deadline
            stop = (restart_every and k - first + 1 >= restart_every and k < N) or out_of_time
            if k % cfg["checkpoint_every"] == 0 or k == N or stop:
                t1 = time.perf_counter()
                status["snapshots"].append(snapshot(r, k, f"step_{k:05d}.pkl"))
                checkpoint_seconds = time.perf_counter() - t1
                status["last_step"] = k
                mark_retained_snapshots(status, keep=2)
                save_json(status_path, status)
                remove_retired_snapshots(folder, status)
            log(f"{j['folder']}: step {k}/{N}, t = {k * dt:.4g} s, {seconds:.2f} s, Newton {record['newton_iterations']} "
                f"(residual {record['newton_residual']:.2e}), ALE J min {record['ale_jacobian_min']:.3f}, "
                f"RSS {current:.3f} GiB (peak {peak:.3f})" + (", tangent" if with_tangent else ""))
            del r, a, hm
            gc.collect()
            if stop:
                break
    process.update(seconds=time.perf_counter() - started, last_step=status["last_step"], peak_rss_gib=memory_gib()[1])
    if status["last_step"] < N:
        status["state"] = "partial"
        save_json(status_path, status)
        if out_of_time:
            log(f"stopping after step {status['last_step']} at a checkpoint, before the wall-time limit; "
                "a later run (a follow-on segment or submit.sh --resume) resumes")
            return TIME_EXIT
        log(f"stopping after step {status['last_step']} (--restart-every {restart_every}); the driver restarts")
        return RESTART_EXIT

    # Complete: stack the per-step records into result.npz.
    records = []
    for k in range(1, N + 1):
        with np.load(folder / "steps" / f"step_{k:05d}.npz") as f:
            records.append({name: f[name] for name in f.files})
    keys = sorted(set().union(*records))
    stacked = {key: np.stack([np.asarray(rec.get(key, np.nan), dtype=float) for rec in records]) for key in keys}
    tangent_index = [k - 1 for k in tangent]
    report_index = [k - 1 for k in report]
    result = dict(stacked, report_steps=np.asarray(report), tangent_steps=np.asarray(tangent),
                  tangents=stacked["tangent"][tangent_index], report_outputs=stacked["outputs"][report_index],
                  report_times=stacked["time"][report_index], tangent_times=stacked["time"][tangent_index])
    del result["tangent"]
    save_npz(folder / "result.npz", **result)
    n_el = int(stacked["n_elements"][0])
    volume = cfg["material"]["rve_length"] ** 3
    meta = dict(kind="history", job=j, config=cfg, git=git_state(), solver_sha256=digest, versions=versions(),
                threads=thread_environment(threads), slurm=slurm_environment(), processes=status["processes"],
                command=sys.argv, n_elements=n_el, total_dofs=int(stacked["total_dofs"][0]), cell_volume=volume,
                h_eff=(volume / n_el) ** (1 / 3), steps=N, dt=dt, report_steps=report, tangent_steps=tangent,
                solve_seconds=status["solve_seconds"], wall_seconds=sum(p.get("seconds", 0) for p in status["processes"]),
                peak_rss_gib=max(p.get("peak_rss_gib", 0) for p in status["processes"]),
                ngsolve=ngsolve.__version__, finished=time.time())
    meta["rss_drift_gib_per_step"], meta["rss_corrector_jumps_gib"] = rss_growth(stacked)
    save_json(folder / "metadata.json", meta)
    status["state"] = "complete"
    save_json(status_path, status)
    log(f"{j['folder']}: complete; {meta['wall_seconds']:.0f} s, peak RSS {meta['peak_rss_gib']:.2f} GiB, "
        f"RSS drift {meta['rss_drift_gib_per_step'] * 1024:.3f} MiB/step between corrector steps, "
        f"jumps at corrector steps {', '.join(f'{x * 1024:.0f}' for x in meta['rss_corrector_jumps_gib'])} MiB")
    return 0


def rss_growth(stacked):
    """Drift of resident memory over outputs-only steps (GiB per step), and the jumps at corrector steps.

    The steps are split into segments at each corrector step and process restart; the
    drift is the slope of current RSS against step pooled over segments of at least
    three steps, each with its own offset, so the one-off growth of a corrector
    factorisation is reported separately (jumps, GiB) instead of as a leak.
    """
    import numpy as np
    step, rss, pid, tangent = (stacked[k] for k in ("step", "rss_gib", "pid", "has_tangent"))
    segment = np.cumsum((tangent > 0) | np.r_[True, pid[1:] != pid[:-1]])
    num = den = 0.0
    for k in np.unique(segment):
        mask = (segment == k) & (tangent == 0) & np.isfinite(rss)
        if mask.sum() >= 3:
            x, y = step[mask] - step[mask].mean(), rss[mask] - rss[mask].mean()
            num, den = num + float(x @ y), den + float(x @ x)
    jumps = [float(rss[i] - rss[i - 1]) for i in np.flatnonzero(tangent > 0) if i > 0 and pid[i] == pid[i - 1]]
    return (num / den if den else float("nan")), jumps

def tangent_worker(cfg, j, folder, threads, campaign):
    import gc
    import numpy as np
    from ngsolve import SetNumThreads, TaskManager
    sys.path[:0] = [str(ROOT), str(ROOT / "verification"), str(HERE)]
    from homogenisation_microscale import FSI
    from state_io import load_snapshot, solver_sha256
    import loading as L

    log = Log(folder)
    SetNumThreads(threads)
    src = source_job(cfg, j["duration_index"])
    src_folder = campaign / src["folder"]
    src_status = json.loads((src_folder / "status.json").read_text()) if (src_folder / "status.json").exists() else {}
    if src_status.get("state") != "complete":
        raise RuntimeError(f"source history {src['folder']} is not complete; run study C first")
    digest = solver_sha256(ROOT / "homogenisation_microscale.py")
    if src_status["solver_sha256"] != digest:
        raise RuntimeError("the solver changed since the source history ran")
    T, N = float(j["T"]), int(j["steps"])
    dt = T / N
    path = L.loading_path(T, N, cfg["loading"])
    tangent = report_steps(N, cfg["tangent_fractions"])
    options = solver_options(cfg, j)
    eps = float(j["fd_step"])
    with np.load(src_folder / "result.npz") as f:
        src_tangents, src_steps = f["tangents"].copy(), list(f["tangent_steps"])
        src_outputs = f["outputs"].copy()
    rows = {name: [] for name in ("fd", "lc", "fd_outputs", "lc_outputs", "steps", "fd_seconds", "lc_seconds",
                                   "fd_newton_max_iterations", "rss_gib", "rss_peak_gib")}
    started = time.perf_counter()

    def frozen(k):
        item = src_status["bases"][str(k - 1)]
        return load_snapshot(src_folder / "states" / item["path"], expected_sha256=item["sha256"])["result"]["restart_state"]

    for k in tangent:
        z = path[k]
        loads = dict(dt=dt, t_final=dt, F_target=z[:9].reshape(3, 3), P_target=float(z[9]), G_target=z[10:].copy())
        t0 = time.perf_counter()
        with TaskManager():
            r = FSI(**options, **loads, initial_state=frozen(k), tangent_method="central difference",
                    central_difference_step=eps)
        a = r["homogenised_response"]
        rows["fd"].append(np.asarray(a["jacobian"], dtype=float))
        rows["fd_outputs"].append(np.asarray(a["outputs"], dtype=float))
        rows["steps"].append(np.asarray(a["central_difference_steps"], dtype=float))
        rows["fd_seconds"].append(time.perf_counter() - t0)
        rows["fd_newton_max_iterations"].append(newton_summary(r)["newton_iterations"])
        input_scales, output_scales = a["physical_scales"]["input"], a["physical_scales"]["output"]
        del r, a
        gc.collect()
        t0 = time.perf_counter()
        with TaskManager():
            r = FSI(**options, **loads, initial_state=frozen(k), tangent_method="linear correctors")
        newton_summary(r)
        a = r["homogenised_response"]
        rows["lc"].append(np.asarray(a["jacobian"], dtype=float))
        rows["lc_outputs"].append(np.asarray(a["outputs"], dtype=float))
        rows["lc_seconds"].append(time.perf_counter() - t0)
        del r, a
        gc.collect()
        current, peak = memory_gib()
        rows["rss_gib"].append(current)
        rows["rss_peak_gib"].append(peak)
        scale = np.asarray(input_scales)[None, :] / np.asarray(output_scales)[:, None]
        error = np.max(abs(rows["fd"][-1] - rows["lc"][-1]) * scale)
        log(f"{j['folder']}: FD step {eps:g} at t = {k * dt:.4g} s: max scaled |FD - corrector| {error:.3e}; "
            f"FD {rows['fd_seconds'][-1]:.1f} s, corrector {rows['lc_seconds'][-1]:.1f} s, RSS {current:.3f} GiB")
    if not all(np.isfinite(x).all() for x in rows["fd"] + rows["lc"]):
        raise RuntimeError("non-finite tangent")
    src_index = [src_steps.index(k) for k in tangent]
    result = {name: np.asarray(v, dtype=float) for name, v in rows.items()}
    result.update(fd_step=eps, tangent_steps=np.asarray(tangent), tangent_times=np.asarray(tangent) * dt,
                  inputs=path[tangent], input_scales=np.asarray(input_scales), output_scales=np.asarray(output_scales),
                  source_tangents=src_tangents[src_index], source_outputs=src_outputs[[k - 1 for k in tangent]])
    save_npz(folder / "result.npz", **result)
    current, peak = memory_gib()
    meta = dict(kind="tangent", job=j, source=src, config=cfg, git=git_state(), solver_sha256=digest,
                versions=versions(), threads=thread_environment(threads), slurm=slurm_environment(), command=sys.argv,
                host=socket.gethostname(), wall_seconds=time.perf_counter() - started, peak_rss_gib=peak,
                finished=time.time())
    save_json(folder / "metadata.json", meta)
    save_json(folder / "status.json", dict(state="complete", job=j, solver_sha256=digest))
    log(f"{j['folder']}: complete; {meta['wall_seconds']:.0f} s, peak RSS {peak:.2f} GiB")
    return 0


# ---------------------------------------------------------------------------
# Driver: one child process per job (and per --restart-every segment)
# ---------------------------------------------------------------------------

def check_campaign_config(campaign, cfg, extend=False, study=None):
    """Write <campaign>/config.json, or check the requested settings against it.

    Settings that only add studies to the recorded ones are compatible: with extend
    (--prepare, --all) the added studies are recorded; a single job (study given) must
    belong to a recorded study, so that analyse.py, which reads config.json, sees it.
    """
    campaign.mkdir(parents=True, exist_ok=True)
    path = campaign / "config.json"
    if not path.exists():
        save_json(path, cfg)
        return
    stored, requested = json.loads(path.read_text()), json.loads(json.dumps(jsonable(cfg)))
    if stored == requested:
        return
    added = set(requested.get("studies", {})) - set(stored.get("studies", {}))
    restricted = dict(requested, studies={k: v for k, v in requested["studies"].items() if k not in added})
    if not added or restricted != stored:
        raise SystemExit(f"{path} differs from the requested settings; use a new --output folder")
    if extend:
        save_json(path, cfg)
        print(f"{path}: added stud{'ies' if len(added) > 1 else 'y'} {', '.join(sorted(added))}", flush=True)
    elif study in added:
        raise SystemExit(f"study {study} is not recorded in {path}; run hpc/submit.sh (or run.py --prepare) first")


def parse_time_limit(text):
    """Slurm-style [D-]HH:MM[:SS] or MM to seconds."""
    days, _, clock = text.rpartition("-")
    parts = [int(x) for x in clock.split(":")]
    if len(parts) == 1:
        parts = [0, parts[0], 0]
    elif len(parts) == 2:
        parts = parts + [0]
    h, m, sec = parts
    return (int(days or 0) * 24 + h) * 3600 + m * 60 + sec


def deadline_from(args, started):
    """Epoch seconds by which a job must have stopped at a checkpoint, or None."""
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end and end.isdigit():
        limit_end = float(end)
    elif args.time_limit:
        limit_end = started + parse_time_limit(args.time_limit)
    else:
        return None
    return limit_end - 60 * args.time_margin


def run_job(cfg, j, campaign, args):
    import fcntl
    started = time.time()
    folder = campaign / j["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{j['folder']} is running in another process; skipped", flush=True)
            return 1
        status_path = folder / "status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("state") == "complete":
            print(f"{j['folder']}: complete; skipped", flush=True)
            return 0
        restart = args.restart_every if args.restart_every and (args.restart_maxh is None or j["maxh"] <= args.restart_maxh) else 0
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--study", j["study"],
                   "--duration", str(j["duration_index"]), "--index", str(j["index"]), "--output", str(campaign),
                   "--config", str(args.config), "--threads", str(args.threads), "--restart-every", str(restart)]
        if args.smoke:
            command.append("--smoke")
        deadline = deadline_from(args, started)
        if deadline is not None:
            command += ["--deadline", f"{deadline:.0f}"]
        print(f"{j['folder']}: T = {j['T']:g} s, maxh {j['maxh']:g}, {j['steps']} steps, Newton {j['newton_tolerance']:g}"
              + (f", FD step {j['fd_step']:g}" if j["fd_step"] else "")
              + (f", lambda_m/mu_m {j['mesh_lame_ratio']:g}" if j.get("mesh_lame_ratio") is not None else "")
              + f"; {args.threads} threads", flush=True)
        for _ in range(int(j["steps"]) // max(restart, 1) + 3):
            code = subprocess.run(command).returncode
            if code != RESTART_EXIT:
                break
        if code == TIME_EXIT:
            print(f"{j['folder']}: stopped at a checkpoint before the wall-time limit (exit {TIME_EXIT}); "
                  "a follow-on segment or submit.sh --resume continues it", flush=True)
        elif code:
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(state="failed", exit_code=code)
            save_json(status_path, status)
            print(f"{j['folder']}: FAILED (exit {code}); see {folder / 'run.log'}", flush=True)
        return code


def worker(cfg, j, campaign, args):
    import traceback
    folder = campaign / j["folder"]
    try:
        if j["study"] == "D":
            return tangent_worker(cfg, j, folder, args.threads, campaign)
        return history_worker(cfg, j, folder, args.threads, args.restart_every, args.deadline)
    except BaseException:
        Log(folder)(traceback.format_exc())
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update(state="failed", error=traceback.format_exc())
        save_json(status_path, status)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", choices=["worker"], help=argparse.SUPPRESS)
    p.add_argument("--study", choices=STUDIES)
    p.add_argument("--duration", type=int)
    p.add_argument("--index", type=int)
    p.add_argument("--task", type=int, help="Slurm array task id: duration = task // n, index = task %% n")
    p.add_argument("--list", action="store_true", help="print the jobs of every study and exit")
    p.add_argument("--all", action="store_true", help="run every job in turn (D after C), then analyse")
    p.add_argument("--prepare", action="store_true", help="write the campaign's config.json and exit")
    p.add_argument("--smoke", action="store_true", help="coarse settings: the whole pipeline in minutes")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "verification.json")
    p.add_argument("--output", type=Path, help="campaign folder")
    p.add_argument("--threads", type=int, help="NGSolve threads")
    p.add_argument("--restart-every", type=int, default=0, help="restart the process every K steps (0: never)")
    p.add_argument("--restart-maxh", type=float, help="apply --restart-every only to jobs with maxh <= H")
    p.add_argument("--time-limit", help="the job's wall-time limit, [D-]HH:MM:SS: stop at a checkpoint before it")
    p.add_argument("--time-margin", type=float, default=20.0, help="minutes before the limit to stop by (default 20)")
    p.add_argument("--deadline", type=float, help=argparse.SUPPRESS)  # epoch seconds, set by the driver for its worker
    a = p.parse_args()
    a.config = a.config.resolve()
    cfg = load_settings(a.config, a.smoke)
    if a.list:
        for study in STUDIES:
            n = len(study_jobs(cfg, study))
            print(f"{study} {n * len(cfg['loading']['durations'])} {n}")
            for j in all_jobs(cfg, study):
                extra = (f"  FD step {j['fd_step']:g}" if j["fd_step"] else "  cross-check" if j["cross_check"] else
                         f"  lambda_m/mu_m {j['mesh_lame_ratio']:g}" if j.get("mesh_lame_ratio") is not None else "")
                skipped = sorted(set(skipped_fractions(j["steps"], cfg["report_fractions"] + cfg["tangent_fractions"])))
                if skipped and j["study"] != "D":
                    extra += f"  (not reported at {', '.join(f'{f:g} T' for f in skipped)}: not a step)"
                print(f"#   {j['folder']}: maxh {j['maxh']:g}, {j['steps']} steps, Newton {j['newton_tolerance']:g}{extra}")
        return 0
    a.output = (a.output or ROOT / "results" / "verification" / ("combined_smoke" if a.smoke else "combined")).resolve()
    a.threads = a.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", cfg["local_threads"]))
    if a.restart_every < 0:
        p.error("--restart-every must be >= 0")
    if a.prepare:
        check_campaign_config(a.output, cfg, extend=True)
        print(a.output / "config.json", flush=True)
        return 0
    if a.all:
        check_campaign_config(a.output, cfg, extend=True)
        failed = [j["folder"] for study in STUDIES for j in all_jobs(cfg, study) if run_job(cfg, j, a.output, a)]
        sys.path.insert(0, str(HERE))
        import analyse
        analyse.analyse(a.output)
        if failed:
            print("Failed jobs: " + ", ".join(failed), flush=True)
        return 1 if failed else 0
    if a.study is None:
        p.error("give --study with --duration and --index (or --task), or --all, or --list")
    if a.task is not None:
        n = len(study_jobs(cfg, a.study))
        a.duration, a.index = divmod(a.task, n)
    if a.duration is None or a.index is None:
        p.error("give --duration and --index, or --task")
    j = job(cfg, a.study, a.duration, a.index)
    if a.mode == "worker":
        return worker(cfg, j, a.output, a)
    check_campaign_config(a.output, cfg, study=a.study)
    return run_job(cfg, j, a.output, a)


if __name__ == "__main__":
    raise SystemExit(main())
