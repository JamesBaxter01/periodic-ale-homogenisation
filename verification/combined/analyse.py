#!/usr/bin/env python3
"""Analyse a verification campaign written by run.py (no NGSolve needed).

  python verification/combined/analyse.py results/verification/combined

Writes <campaign>/analysis/: CSV tables, summary.md and figures (PNG and PDF).

Quantities: the outputs at the report times, in groups (stress P, fluid content m,
flux Q), and the tangent at the tangent times, in its nine blocks
(stress/content/flux x F/p/grad p). Tangents are compared scaled
(J_hat = J * input_scale / output_scale, the solver's physical_scales). A group's
error is the relative Euclidean norm ||q - q_ref|| / ||q_ref||; a tangent block
is normalised by max(||block_ref||, 1e-3 ||J_hat_ref||), so near-zero blocks do
not divide by ~0.

A (mesh) and B (time step): observed orders from successive level triples in
x = h_eff = (V / n_el)^(1/3) or dt, solving (x1^p - x2^p) / (x2^p - x3^p) = e12 / e23
(e_ij = ||q_i - q_j||); Richardson reference from the finest three levels that
reached the quantity's time, with their observed order (the expected order if the
observed one falls outside [expected / 2, 2 expected + 1], flagged); each level's
error against that reference. A history that stopped early (e.g. an inverted ALE
element) counts at the report times it reached, read from its per-step records.
Reported with each quantity: the levels used, the minimum ALE Jacobian of the three
reference levels at that time, and whether the last two local orders agree (to 0.5).
The tangent is left out of B: it is the derivative of one backward-Euler step and
scales with dt by construction (viscous stress ~ eta/dt, one-step compliance ~ dt/eta),
so it has no dt -> 0 limit; it is verified at fixed dt by A and D. Its time-step
error in the budget is "n/a" and its total is mesh + algebraic.
C (Newton tolerance): change against the tightest tolerance.
Error budget at production settings: mesh error of the production maxh (A), time
error of the production N (B), algebraic error of the production tolerance (C).
Cross-check: maxh 0.045 at N = 200 vs N = 100 (A) against N = 200 vs 100 at maxh
0.09 (B); a ratio near 1 supports the independence of the studies.
D (tangent): max scaled |J_FD - J_corrector| against the relative FD step eps,
fitted by e = a eps^2 + b / eps + c (non-negative least squares, relative
weights): a eps^2 is FD truncation, b / eps the noise of the Newton tolerance, c a
step-independent difference, i.e. a bound on the corrector's own error.
F (mesh law): the change of each quantity when lambda_m / mu_m of the auxiliary mesh
law differs from the production value (the reference level), against the production
discretisation error (mesh + time step of the error budget); the reference level is
also compared with the study-C history at the same settings (reproducibility).
E (balance): per step, the power residual, Eq. (42) predicted defect, corrected
residual and constraint power (scaled by the run's largest power), the corrected
residual against the Newton residual, the exact identities (round-off on any mesh,
step and tolerance) and the mean constraints, which the multipliers enforce to the
Newton tolerance (nondimensional, and divided by the Newton target).
"""

from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import STUDIES, all_jobs, report_steps  # noqa: E402

GROUPS = {"stress": slice(0, 9), "content": slice(9, 10), "flux": slice(10, 13)}
BLOCKS = {f"{r}/{c}": (rs, cs) for r, rs in GROUPS.items()
          for c, cs in {"F": slice(0, 9), "p": slice(9, 10), "g": slice(10, 13)}.items()}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
IDENTITIES = {  # nondimensional (hmhat_) diagnostics that vanish at round-off on any mesh, step, tolerance
    "power accounting": "power_accounting_residual",
    "cell volume": "cell_volume_residual",
    "solid frame (assembled vs integrated)": "solid_frame_quadrature_difference",
}
CONSTRAINTS = {  # multiplier-enforced mean constraints: satisfied to the Newton tolerance, not round-off
    "mean fluctuation pressure": "pressure_mean",
    "solid frame (mean velocity)": "solid_frame_mean",
    "mean continuity": "continuity_mean_residual",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_partial(folder, cfg, j):
    """A history that stopped early, from its per-step records 1, 2, ... up to the last one written."""
    records, k = [], 1
    while (folder / "steps" / f"step_{k:05d}.npz").exists():
        with np.load(folder / "steps" / f"step_{k:05d}.npz") as f:
            records.append({name: f[name] for name in f.files})
        k += 1
    if not records:
        return None
    d = {key: np.stack([np.asarray(r.get(key, np.nan), dtype=float) for r in records])
         for key in sorted(set().union(*records))}
    last = len(records)
    report = [s for s in report_steps(j["steps"], cfg["report_fractions"]) if s <= last]
    tangent = [s for s in report_steps(j["steps"], cfg["tangent_fractions"]) if s <= last]
    d.update(report_steps=np.asarray(report, dtype=int), tangent_steps=np.asarray(tangent, dtype=int),
             report_outputs=d["outputs"][[s - 1 for s in report]], report_times=d["time"][[s - 1 for s in report]],
             tangents=d["tangent"][[s - 1 for s in tangent]], tangent_times=d["time"][[s - 1 for s in tangent]])
    n_el = int(d["n_elements"][0])
    volume = cfg["material"]["rve_length"] ** 3
    meta = dict(kind="history", partial=True, last_step=last, n_elements=n_el, total_dofs=int(d["total_dofs"][0]),
                h_eff=(volume / n_el) ** (1 / 3))
    return dict(job=j, meta=meta, data=d)


def load_campaign(campaign):
    """Complete runs; histories that stopped early (partial: their steps so far); inventory; solver digests."""
    cfg = json.loads((campaign / "config.json").read_text())
    runs, partial, inventory = [], [], []
    for study in [s for s in STUDIES if s in cfg["studies"]]:
        for j in all_jobs(cfg, study):
            folder = campaign / j["folder"]
            status = json.loads((folder / "status.json").read_text()) if (folder / "status.json").exists() else {}
            row = dict(job=j["folder"], study=study, T=j["T"], maxh=j["maxh"], steps=j["steps"],
                       newton_tolerance=j["newton_tolerance"], fd_step=j["fd_step"], cross_check=j["cross_check"],
                       mesh_lame_ratio=j.get("mesh_lame_ratio"),
                       state=status.get("state", "not run"))
            if row["state"] == "complete" and (folder / "result.npz").exists():
                meta = json.loads((folder / "metadata.json").read_text())
                with np.load(folder / "result.npz") as f:
                    data = {k: f[k].copy() for k in f.files}
                runs.append(dict(job=j, meta=meta, data=data))
                row.update(wall_hours=meta["wall_seconds"] / 3600, peak_rss_gib=meta["peak_rss_gib"],
                           solver_sha256=meta["solver_sha256"][:12], commit=(meta["git"]["commit"] or "")[:10],
                           threads=meta["threads"]["ngsolve_threads"], slurm_job=meta["slurm"].get("SLURM_JOB_ID"))
                if meta["kind"] == "history":
                    row.update(n_elements=meta["n_elements"], total_dofs=meta["total_dofs"], h_eff=meta["h_eff"],
                               rss_drift_mib_per_step=(float("nan") if meta["rss_drift_gib_per_step"] is None
                                                       else meta["rss_drift_gib_per_step"] * 1024),
                               rss_corrector_jumps_mib=" ".join(f"{x * 1024:.0f}" for x in meta["rss_corrector_jumps_gib"]),
                               newton_mean=float(np.mean(data["newton_iterations"])),
                               newton_max=int(np.max(data["newton_iterations"])),
                               ale_jacobian_min=float(np.min(data["ale_jacobian_min"])),
                               seconds_per_step=float(np.mean(data["seconds"])))
            elif study != "D" and (folder / "steps").is_dir():
                r = load_partial(folder, cfg, j)
                if r is not None:
                    partial.append(r)
                    d = r["data"]
                    row.update(reached_step=r["meta"]["last_step"], n_elements=r["meta"]["n_elements"],
                               total_dofs=r["meta"]["total_dofs"], h_eff=r["meta"]["h_eff"],
                               newton_mean=float(np.mean(d["newton_iterations"])),
                               newton_max=int(np.max(d["newton_iterations"])),
                               ale_jacobian_min=float(np.min(d["ale_jacobian_min"])),
                               seconds_per_step=float(np.mean(d["seconds"])),
                               failure=(status.get("error") or "").strip().splitlines()[-1:] or [""])
                    row["failure"] = row["failure"][0]
            inventory.append(row)
    digests = {r["meta"]["solver_sha256"] for r in runs}
    return cfg, runs, partial, inventory, digests


def pick(runs, study, T, **match):
    out = []
    for r in runs:
        j = r["job"]
        if j["study"] == study and math.isclose(j["T"], T) and all(
                (j.get(k) is None and v is None) or (j.get(k) is not None and v is not None and math.isclose(j[k], v))
                if not isinstance(v, bool) else j.get(k) == v for k, v in match.items()):
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Quantities and errors
# ---------------------------------------------------------------------------

def quantities(r):
    """Dict name -> array: output groups at each report time, tangent blocks at each tangent time."""
    d = r["data"]
    q = {}
    for t, out in zip(d["report_times"], d["report_outputs"]):
        for g, s in GROUPS.items():
            q[(g, float(t))] = out[s]
    scale = d["input_scales"][0][None, :] / d["output_scales"][0][:, None]
    for t, J in zip(d["tangent_times"], d["tangents"]):
        Jh = J * scale
        q[("tangent", float(t))] = Jh
        for b, (rs, cs) in BLOCKS.items():
            q[("tangent " + b, float(t))] = Jh[rs, cs]
    return q


def rel(a, ref, key, q_ref):
    """||a - ref|| / ||ref||; a tangent block's norm is floored at 1e-3 ||J_hat|| of q_ref (a quantity dict)."""
    a, ref = np.asarray(a), np.asarray(ref)
    norm = np.linalg.norm(ref)
    if key[0].startswith("tangent "):
        norm = max(norm, 1e-3 * np.linalg.norm(q_ref[("tangent", key[1])]))
    return float(np.linalg.norm(a - ref) / norm) if norm > 0 else float("nan")


def order_from_triple(x, e12, e23):
    """p with (x1^p - x2^p) / (x2^p - x3^p) = e12 / e23, x1 > x2 > x3; NaN if none in (0.05, 12)."""
    if not (e12 > 0 and e23 > 0 and x[0] > x[1] > x[2]):
        return float("nan")
    R = e12 / e23
    f = lambda p: (x[0] ** p - x[1] ** p) / (x[1] ** p - x[2] ** p) - R
    lo, hi = 0.05, 12.0
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def ale_at(r, t):
    """Minimum sampled ALE Jacobian of a history at the step at time t."""
    d = r["data"]
    return float(d["ale_jacobian_min"][int(np.argmin(abs(d["time"] - t)))])


def convergence(levels, x, expected):
    """levels: quantity dicts, coarse to fine (x decreasing); each key over the levels that have it (at least 3)."""
    out = {}
    for key in dict.fromkeys(k for lv in levels for k in lv):
        use = [i for i, lv in enumerate(levels) if key in lv]
        if len(use) >= 3:
            out[key] = dict(convergence_one(key, [levels[i] for i in use], [x[i] for i in use], expected), used=use)
    return out


def convergence_one(key, levels, x, expected):
    """One quantity over its levels (coarse to fine): local orders, Richardson reference, errors."""
    q = [lv[key] for lv in levels]
    local = []
    for i in range(len(q) - 2):
        e12 = np.linalg.norm(q[i] - q[i + 1])
        e23 = np.linalg.norm(q[i + 1] - q[i + 2])
        local.append(order_from_triple(x[i:i + 3], e12, e23))
    p = local[-1] if local else float("nan")
    used_expected = not (expected / 2 <= p <= 2 * expected + 1)
    p_ref = expected if used_expected else p
    ref = q[-1] + (q[-1] - q[-2]) / ((x[-2] / x[-1]) ** p_ref - 1)
    errors = [rel(v, ref, key, levels[-1]) for v in q]
    fit = float("nan")
    good = [(xi, e) for xi, e in zip(x[:-1], errors[:-1]) if e > 0]
    if len(good) >= 2:
        fit = float(np.polyfit(np.log([g[0] for g in good]), np.log([g[1] for g in good]), 1)[0])
    spread = abs(local[-1] - local[-2]) if len(local) >= 2 else float("nan")
    return dict(local_orders=local, finest_order=p, reference_order=p_ref, used_expected_order=used_expected,
                reference=ref, errors=errors, fitted_order=fit, x=list(x),
                finest_difference=rel(q[-2], q[-1], key, levels[-1]),
                order_spread=spread, orders_consistent=bool(spread <= 0.5) if math.isfinite(spread) else None)


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

def study_convergence(cfg, runs, study, T):
    """runs: complete and partial histories. Each result carries the runs, x and ALE Jacobians it used."""
    s = cfg["studies"][study]
    if study == "A":
        chosen = [pick(runs, "A", T, maxh=h, cross_check=False) for h in s["levels"]]
    else:
        chosen = [pick(runs, "B", T, steps=n) for n in s["levels"]]
    chosen = [c[0] for c in chosen if len(c) == 1]
    if len(chosen) < 3:
        return None
    x = [c["meta"]["h_eff"] if study == "A" else T / c["job"]["steps"] for c in chosen]
    levels = [quantities(c) for c in chosen]
    if study == "B":  # one-step tangent: depends on dt by construction (module docstring)
        levels = [{k: v for k, v in q.items() if not k[0].startswith("tangent")} for q in levels]
    results = convergence(levels, x, s["expected_order"])
    for key, v in results.items():
        v["runs"] = [chosen[i] for i in v.pop("used")]
        v["ale_jacobian_min"] = [ale_at(r, key[1]) for r in v["runs"]]
    return dict(runs=chosen, results=results)


def tolerance_study(cfg, runs, T):
    levels = cfg["studies"]["C"]["levels"]
    chosen = [pick(runs, "C", T, newton_tolerance=t) for t in levels]
    if any(len(c) != 1 for c in chosen):
        return None
    chosen = [c[0] for c in chosen]
    tightest = int(np.argmin(levels))
    q = [quantities(c) for c in chosen]
    return dict(tolerances=levels, runs=chosen,
                results={key: [rel(qi[key], q[tightest][key], key, q[tightest]) for qi in q] for key in q[0]})


def mesh_law_study(cfg, runs, T):
    """Change of every quantity against the reference lambda_m / mu_m, and the reference against study C."""
    s = cfg["studies"].get("F")
    if not s:
        return None
    chosen = [pick(runs, "F", T, mesh_lame_ratio=r) for r in s["levels"]]
    ref = pick(runs, "F", T, mesh_lame_ratio=s["reference"])
    if len(ref) != 1 or not any(len(c) == 1 for c in chosen):
        return None
    q_ref = quantities(ref[0])
    out = dict(levels=[], reproducibility=float("nan"))
    for r, c in zip(s["levels"], chosen):
        if len(c) == 1 and not math.isclose(r, s["reference"]):
            q = quantities(c[0])
            out["levels"].append((r, c[0], {key: rel(q[key], q_ref[key], key, q_ref) for key in q_ref}))
    same = pick(runs, "C", T, maxh=ref[0]["job"]["maxh"], steps=ref[0]["job"]["steps"],
                newton_tolerance=ref[0]["job"]["newton_tolerance"])
    if len(same) == 1:
        q_c = quantities(same[0])
        out["reproducibility"] = max(rel(q_ref[key], q_c[key], key, q_c) for key in q_ref)
    return out


def nnls_fit(eps, err):
    """e = a eps^2 + b / eps + c, a, b, c >= 0, weighted relative to e."""
    from scipy.optimize import nnls
    eps, err = np.asarray(eps), np.asarray(err)
    A = np.column_stack([eps ** 2, 1 / eps, np.ones_like(eps)]) / err[:, None]
    coef, _ = nnls(A, np.ones_like(err))
    return coef


def fd_study(cfg, runs, T):
    rows, curves = [], {}
    chosen = sorted(pick(runs, "D", T), key=lambda r: -r["job"]["fd_step"])
    if not chosen:
        return rows, curves
    tol = cfg["studies"]["D"]["newton_tolerance"]
    for ti, t in enumerate(chosen[0]["data"]["tangent_times"]):
        eps, err, frob = [], [], []
        for r in chosen:
            d = r["data"]
            scale = d["input_scales"][None, :] / d["output_scales"][:, None]
            diff = (d["fd"][ti] - d["lc"][ti]) * scale
            lc = d["lc"][ti] * scale
            eps.append(float(d["fd_step"]))
            err.append(float(np.max(abs(diff))))
            frob.append(float(np.linalg.norm(diff) / np.linalg.norm(lc)))
            blocks = {b: float(np.max(abs(diff[rs, cs]))) for b, (rs, cs) in BLOCKS.items()}
            rows.append(dict(T=T, time=float(t), fd_step=eps[-1], max_scaled_error=err[-1], relative_frobenius=frob[-1],
                             corrector_vs_source_history=float(np.max(abs((d["lc"][ti] - d["source_tangents"][ti]) * scale))),
                             fd_seconds=float(d["fd_seconds"][ti]), **{"block " + b: v for b, v in blocks.items()}))
        coef = nnls_fit(eps, err) if len(eps) >= 3 else np.full(3, np.nan)
        i = int(np.argmin(err))
        trunc = list(np.argsort(eps)[-3:])  # the three largest steps: FD truncation regime
        slope = (float(np.polyfit(np.log(np.asarray(eps)[trunc]), np.log(np.asarray(err)[trunc]), 1)[0])
                 if len(trunc) >= 2 else float("nan"))
        curves[float(t)] = dict(eps=eps, err=err, frob=frob, coef=coef, min_error=err[i], argmin=eps[i],
                                noise_floor_at_min=tol / eps[i], slope_largest_steps=slope)
    return rows, curves


def balance(cfg, runs):
    per_run = []
    for r in runs:
        if r["meta"]["kind"] != "history":
            continue
        d, j = r["data"], r["job"]
        W = d["hm_physical_power"]
        bracket = d["stress_power"] + d["storage_power"] + d["flux_power"]
        scale = float(np.max(abs(W) + abs(d["stress_power"]) + abs(d["storage_power"]) + abs(d["flux_power"])))
        row = dict(job=j["folder"], study=j["study"], T=j["T"], maxh=j["maxh"], steps=j["steps"],
                   newton_tolerance=j["newton_tolerance"], power_scale=scale,
                   raw_residual=float(np.max(abs(d["hm_power_residual"]))) / scale,
                   predicted_defect=float(np.max(abs(d["hm_predicted_discrete_power_defect"]))) / scale,
                   corrected_residual=float(np.max(abs(d["hm_power_identity_residual"]))) / scale,
                   constraint_power=float(np.max(abs(d["hm_constraint_power_residual"]))) / scale,
                   reconstruction_check=float(np.max(abs(W - bracket - d["hm_power_residual"]))) / scale,
                   corrected_over_newton_residual=float(np.max(abs(d["hmhat_power_identity_residual"]) / d["newton_residual"])),
                   corrected_over_newton_target=float(np.max(abs(d["hmhat_power_identity_residual"]) / d["newton_target"])),
                   storage_over_stress=float(np.max(abs(d["storage_power"])) / np.max(abs(d["stress_power"]))),
                   flux_over_stress=float(np.max(abs(d["flux_power"])) / np.max(abs(d["stress_power"]))),
                   storage_sign_changes=int(np.sum(np.diff(np.sign(d["storage_power"])) != 0)))
        for name, key in IDENTITIES.items():
            row[name] = float(np.max(abs(d["hmhat_" + key])))
        for name, key in CONSTRAINTS.items():
            value = abs(d["hmhat_" + key]).reshape(len(d["step"]), -1).max(axis=1)
            row[name] = float(np.max(value))
            row[name + " / Newton target"] = float(np.max(value / d["newton_target"]))
        per_run.append(row)
    return per_run


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    import csv
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, (int, np.integer)) or isinstance(x, str):
        return str(x)
    return f"{x:.{digits}e}" if (x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e4)) else f"{x:.{digits + 1}g}"


def md_table(rows, keys, headers=None):
    headers = headers or keys
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r.get(k)) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def style(ax, xlabel, ylabel, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def slope_line(ax, x, y_anchor, order, label):
    x = np.asarray([min(x), max(x)])
    ax.plot(x, y_anchor * (x / x[-1]) ** order, color="#888888", linestyle="--", linewidth=1, label=label)


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("pdf",):
        fig.savefig(out / f"{name}.{ext}", dpi=160)


def analyse(campaign):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    campaign = Path(campaign).resolve()
    out = campaign / "analysis"
    out.mkdir(exist_ok=True)
    cfg, runs, partial, inventory, digests = load_campaign(campaign)
    durations = cfg["loading"]["durations"]
    prod = cfg["production"]
    tf = cfg["tangent_fractions"]
    md = [f"# Verification campaign analysis\n\nCampaign: `{campaign}`" + ("  \n**Smoke settings: not verification data.**" if cfg.get("smoke") else ""),
          f"\nJobs complete: {sum(r['state'] == 'complete' for r in inventory)} of {len(inventory)}. "
          f"Solver digests: {', '.join(d[:12] for d in digests) or 'none'}"
          + (" **(more than one: results from different solver versions)**" if len(digests) > 1 else "") + "."]
    write_csv(out / "inventory.csv", inventory)

    order_rows, error_rows, budget_rows, tol_rows, cross_rows, fd_rows, law_rows, law_repro = [], [], [], [], [], [], [], []
    conv, tolr, fdc = {}, {}, {}
    for T in durations:
        for study in "AB":
            c = study_convergence(cfg, runs + partial, study, T)
            if c is None:
                continue
            conv[(study, T)] = c
            field = "maxh" if study == "A" else "steps"
            for key, v in c["results"].items():
                order_rows.append(dict(study=study, T=T, quantity=key[0], time=key[1], fitted_order=v["fitted_order"],
                                       finest_order=v["finest_order"], reference_order=v["reference_order"],
                                       used_expected_order=v["used_expected_order"],
                                       finest_difference=v["finest_difference"],
                                       levels=" ".join(f"{r['job'][field]:g}" for r in v["runs"]),
                                       reference_levels=" ".join(f"{r['job'][field]:g}" for r in v["runs"][-3:]),
                                       reference_min_ale_jacobian=min(v["ale_jacobian_min"][-3:]),
                                       order_spread=v["order_spread"], orders_consistent=v["orders_consistent"],
                                       **{f"local_order_{i}": p for i, p in enumerate(v["local_orders"])}))
                for x, r, e, J in zip(v["x"], v["runs"], v["errors"], v["ale_jacobian_min"]):
                    error_rows.append(dict(study=study, T=T, quantity=key[0], time=key[1], x=x, maxh=r["job"]["maxh"],
                                           steps=r["job"]["steps"], relative_error=e, ale_jacobian_min=J,
                                           complete=not r["meta"].get("partial", False)))
        t = tolerance_study(cfg, runs, T)
        if t is not None:
            tolr[T] = t
            for key, errs in t["results"].items():
                for tol, e in zip(t["tolerances"], errs):
                    tol_rows.append(dict(T=T, quantity=key[0], time=key[1], newton_tolerance=tol, change=e))
        rows, curves = fd_study(cfg, runs, T)
        fd_rows += rows
        fdc[T] = curves
        # Error budget at production settings.
        keys = set()
        for c in (conv.get(("A", T)), conv.get(("B", T))):
            if c:
                keys |= set(c["results"])
        for key in sorted(keys, key=lambda k: (k[1], k[0])):
            row = dict(T=T, quantity=key[0], time=key[1])
            for study, name, field in (("A", "mesh_error", "maxh"), ("B", "time_step_error", "steps")):
                c = conv.get((study, T))
                if c and key in c["results"]:
                    match = [i for i, r in enumerate(c["results"][key]["runs"]) if math.isclose(r["job"][field], prod[field])]
                    row[name] = c["results"][key]["errors"][match[0]] if match else float("nan")
            if T in tolr and key in tolr[T]["results"]:
                i = [k for k, tol in enumerate(tolr[T]["tolerances"]) if math.isclose(tol, prod["newton_tolerance"])]
                row["algebraic_error"] = tolr[T]["results"][key][i[0]] if i else float("nan")
            if key[0].startswith("tangent"):
                row["time_step_error"] = "n/a"  # one-step tangent: scales with dt, no limit (module docstring)
            parts = [row.get(k, float("nan")) for k in ("mesh_error", "time_step_error", "algebraic_error") if row.get(k) != "n/a"]
            row["total"] = sum(parts) if all(math.isfinite(v) for v in parts) else float("nan")  # missing: not a total
            budget_rows.append(row)
        # Cross-check of independence.
        for cc in cfg["studies"]["A"].get("cross_check", []):
            a2 = pick(runs, "A", T, maxh=cc["maxh"], steps=cc["steps"], cross_check=True)
            a1 = pick(runs, "A", T, maxh=cc["maxh"], cross_check=False)
            b2 = pick(runs, "B", T, steps=cc["steps"])
            b1 = pick(runs, "B", T, steps=cfg["studies"]["A"]["steps"])
            if all(len(x) == 1 for x in (a1, a2, b1, b2)):
                qa1, qa2, qb1, qb2 = (quantities(x[0]) for x in (a1, a2, b1, b2))
                for key in qa1:
                    dA, dB = rel(qa2[key], qa1[key], key, qa1), rel(qb2[key], qb1[key], key, qb1)
                    cross_rows.append(dict(T=T, quantity=key[0], time=key[1], maxh_fine=cc["maxh"],
                                           change_at_fine_mesh=dA, change_at_B_mesh=dB,
                                           ratio=dA / dB if dB > 0 else float("nan")))
        # F: mesh-law sensitivity against the production discretisation error (mesh + time step).
        law = mesh_law_study(cfg, runs, T)
        if law is not None:
            law_repro.append(dict(T=T, reference_vs_C=law["reproducibility"]))
            budget = {(b["quantity"], b["time"]): b for b in budget_rows if math.isclose(b["T"], T)}
            for ratio, r, changes in law["levels"]:
                for key, change in changes.items():
                    b = budget.get(key, {})
                    disc = sum(b[k] for k in ("mesh_error", "time_step_error")
                               if isinstance(b.get(k), float) and math.isfinite(b[k]))
                    law_rows.append(dict(T=T, quantity=key[0], time=key[1], mesh_lame_ratio=ratio, change=change,
                                         discretisation_error=disc if b else float("nan"),
                                         change_over_discretisation=change / disc if b and disc > 0 else float("nan"),
                                         ale_jacobian_min=float(np.min(r["data"]["ale_jacobian_min"]))))
    balance_rows = balance(cfg, runs + partial)
    for name, rows in (("orders", order_rows), ("errors", error_rows), ("error_budget", budget_rows),
                       ("tolerance", tol_rows), ("cross_check", cross_rows), ("tangent_fd", fd_rows),
                       ("balance_runs", balance_rows), ("mesh_law", law_rows)):
        write_csv(out / f"{name}.csv", rows)
    fd_summary = [dict(T=T, time=t, **{k: v for k, v in c.items() if k not in ("eps", "err", "frob", "coef")},
                       fit_a=c["coef"][0], fit_b=c["coef"][1], fit_c_corrector_bound=c["coef"][2])
                  for T, curves in fdc.items() for t, c in curves.items()]
    write_csv(out / "tangent_fd_summary.csv", fd_summary)
    balance_summary = []
    for study in [s for s in STUDIES if s != "D" and s in cfg["studies"]]:
        for T in durations:
            rows = [r for r in balance_rows if r["study"] == study and math.isclose(r["T"], T)]
            if rows:
                worst = {k: max(r[k] for r in rows) for k in rows[0] if k not in ("job", "study", "T", "maxh", "steps", "newton_tolerance", "storage_sign_changes")}
                balance_summary.append(dict(study=study, T=T, runs=len(rows), **worst))
    write_csv(out / "balance_summary.csv", balance_summary)

    # ----------------------------------------------------------------- summary.md
    md.append("\n## Jobs\n\n" + md_table(inventory, ["job", "state", "n_elements", "total_dofs", "newton_mean", "newton_max",
                                                      "ale_jacobian_min", "seconds_per_step", "wall_hours", "peak_rss_gib",
                                                      "rss_drift_mib_per_step", "rss_corrector_jumps_mib"]))
    if partial:
        md.append("\n## Histories that stopped early\n\nUsed in A and B at the report times they reached "
                  "(read from their per-step records), and in E for the steps they completed.\n\n"
                  + md_table([r for r in inventory if r.get("reached_step")],
                             ["job", "state", "reached_step", "steps", "ale_jacobian_min", "failure"]))
    final = lambda rows: [r for r in rows if r["quantity"] in ("stress", "content", "flux", "tangent")]
    if order_rows:
        md.append("\n## Observed orders (A: in h_eff, expected 2; B: in dt, expected 1)\n\n"
                  "Groups at every report time, the full scaled tangent at the tangent times; all blocks in orders.csv. "
                  "reference: the three finest levels that reached the time (Richardson); min ALE J: their smallest "
                  "sampled ALE Jacobian at that time; consistent: the last two local orders agree to 0.5.\n\n"
                  + md_table(final(order_rows), ["study", "T", "quantity", "time", "fitted_order", "finest_order",
                                                 "used_expected_order", "finest_difference", "reference_levels",
                                                 "reference_min_ale_jacobian", "orders_consistent"],
                             ["study", "T", "quantity", "time", "fitted order", "finest order", "expected order used",
                              "finest difference", "reference", "min ALE J", "consistent"]))
    if budget_rows:
        md.append(f"\n## Error budget at production settings (maxh {prod['maxh']}, N = {prod['steps']}, Newton {prod['newton_tolerance']:g})\n\n"
                  "Relative errors: mesh (A, vs Richardson reference), time step (B, vs Richardson reference), "
                  "algebraic (C, vs the tightest tolerance). Tangent time step n/a: the one-step tangent scales "
                  "with dt by construction (total = mesh + algebraic).\n\n"
                  + md_table(final(budget_rows), ["T", "quantity", "time", "mesh_error", "time_step_error", "algebraic_error", "total"],
                             ["T", "quantity", "t", "mesh", "time step", "algebraic", "total"]))
    if cross_rows:
        cc = cfg["studies"]["A"]["cross_check"][0]
        md.append(f"\n## Independence cross-check: change N = {cfg['studies']['A']['steps']} -> {cc['steps']} "
                  f"at maxh {cc['maxh']} (A) and at maxh {cfg['studies']['B'].get('maxh', prod['maxh'])} (B)\n\n"
                  + md_table(final(cross_rows), ["T", "quantity", "time", "change_at_fine_mesh", "change_at_B_mesh", "ratio"]))
    if tol_rows:
        md.append("\n## C: change of the outputs against the tightest Newton tolerance\n\n"
                  + md_table(final(tol_rows), ["T", "quantity", "time", "newton_tolerance", "change"]))
    if law_rows:
        F = cfg["studies"]["F"]
        md.append(f"\n## F: mesh law, change against lambda_m/mu_m = {F['reference']:g} "
                  f"(maxh {F.get('maxh', prod['maxh'])}, N = {F.get('steps', prod['steps'])}, Newton {F['newton_tolerance']:g})\n\n"
                  "change: relative change of the quantity; discretisation: mesh + time-step error of the error budget "
                  "(production settings); ratio < 1: the mesh law changes the outputs by less than the discretisation error.\n\n"
                  + md_table(final(law_rows), ["T", "quantity", "time", "mesh_lame_ratio", "change", "discretisation_error",
                                               "change_over_discretisation", "ale_jacobian_min"],
                             ["T", "quantity", "t", "lambda_m/mu_m", "change", "discretisation", "change / discretisation",
                              "min ALE J"])
                  + "\n\nReference level against the study-C history at the same settings (max relative difference, "
                  "all quantities; round-off expected):\n\n" + md_table(law_repro, ["T", "reference_vs_C"]))
    if fd_summary:
        md.append("\n## D: FD tangent against the corrector tangent\n\n"
                  "max scaled |J_FD - J_corrector| over the steps; fit e = a eps^2 + b/eps + c "
                  "(c bounds a step-independent difference, i.e. a corrector error).\n\n"
                  + md_table(fd_summary, ["T", "time", "min_error", "argmin", "noise_floor_at_min", "slope_largest_steps",
                                          "fit_a", "fit_b", "fit_c_corrector_bound"]))
    if balance_summary:
        md.append("\n## E: Hill-Mandel balance, worst per study (powers scaled by the run's largest power)\n\n"
                  + md_table(balance_summary, ["study", "T", "runs", "raw_residual", "predicted_defect", "corrected_residual",
                                               "constraint_power", "reconstruction_check", "corrected_over_newton_target",
                                               "storage_over_stress", "flux_over_stress"])
                  + "\n\nExact identities (nondimensional, worst per study; round-off on any mesh, step and tolerance):\n\n"
                  + md_table(balance_summary, ["study", "T", *IDENTITIES])
                  + "\n\nMean constraints (nondimensional, worst per study; enforced by multipliers to the Newton tolerance):\n\n"
                  + md_table(balance_summary, ["study", "T", *[k for name in CONSTRAINTS for k in (name, name + " / Newton target")]]))
    (out / "summary.md").write_text("\n".join(md) + "\n")

    # ----------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    groups = ["stress", "content", "flux", "tangent"]
    for study, xlabel, expected, name in (("A", "h_eff = (V / n_el)^(1/3)", 2, "fig_A_mesh"),
                                          ("B", "time step dt (s)", 1, "fig_B_time_step")):
        present = [T for T in durations if (study, T) in conv]
        if not present:
            continue
        fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 3.6), squeeze=False)
        for ax, T in zip(axes[0], present):
            c = conv[(study, T)]
            for gi, g in enumerate(groups):
                key = (g, float(T))
                if key not in c["results"]:
                    continue
                v = c["results"][key]
                ax.plot(v["x"][:-1], v["errors"][:-1], color=COLORS[gi], marker=MARKERS[gi], markersize=5, linewidth=2,
                        label=f"{g} at T (p = {v['fitted_order']:.2f})")
            curves = [c["results"][(g, float(T))] for g in groups if (g, float(T)) in c["results"]]
            finite = [e for v in curves for e in v["errors"][:-1] if e > 0]
            if finite:
                slope_line(ax, [x for v in curves for x in v["x"][:-1]], min(finite), expected, f"order {expected}")
            style(ax, xlabel, "relative error vs Richardson reference", f"{cfg['studies'][study]['name']}, T = {T:g} s")
            if ax.lines:
                ax.legend(fontsize=7, frameon=False, loc="best")
            else:
                ax.text(0.5, 0.5, "fewer than three levels reached t = T", transform=ax.transAxes, ha="center")
        save(fig, out, name)
        plt.close(fig)
    if tolr:
        fig, axes = plt.subplots(1, len(tolr), figsize=(4.6 * len(tolr), 3.6), squeeze=False)
        for ax, (T, t) in zip(axes[0], tolr.items()):
            tight = int(np.argmin(t["tolerances"]))
            xs = [x for i, x in enumerate(t["tolerances"]) if i != tight]
            for gi, g in enumerate(groups):
                key = (g, float(T))
                ys = [y for i, y in enumerate(t["results"][key]) if i != tight]
                ax.plot(xs, ys, color=COLORS[gi], marker=MARKERS[gi], markersize=5, linewidth=2, label=f"{g} at T")
            slope_line(ax, xs, max(xs), 1, "change = tolerance")
            style(ax, "Newton tolerance", "relative change vs tightest tolerance", f"Newton tolerance, T = {T:g} s")
            ax.legend(fontsize=7, frameon=False, loc="best")
        save(fig, out, "fig_C_tolerance")
        plt.close(fig)
    panels = [(T, t, c) for T, curves in fdc.items() for t, c in curves.items()]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.6), squeeze=False)
        tol = cfg["studies"]["D"]["newton_tolerance"]
        for ax, (T, t, c) in zip(axes[0], panels):
            eps = np.asarray(c["eps"])
            ax.plot(eps, c["err"], color=COLORS[0], marker="o", markersize=5, linewidth=2, label="max scaled |FD - corrector|")
            if np.isfinite(c["coef"]).all():
                grid = np.geomspace(eps.min(), eps.max(), 100)
                a, b, k = c["coef"]
                ax.plot(grid, a * grid ** 2 + b / grid + k, color=COLORS[1], linewidth=1.2, label="fit a eps² + b/eps + c")
            ax.plot(eps, tol / eps, color="#888888", linestyle=":", linewidth=1, label="tolerance / eps")
            slope_line(ax, eps, c["err"][0], 2, "order 2")
            style(ax, "relative FD step eps (decreasing to the right)", "error", f"T = {T:g} s, t = {t:g} s")
            ax.invert_xaxis()  # the step shrinks left to right, as in a refinement study
            ax.legend(fontsize=6.5, frameon=False)
        save(fig, out, "fig_D_tangent")
        plt.close(fig)
    prod_runs = [r for r in runs if r["job"]["study"] == "C" and math.isclose(r["job"]["newton_tolerance"], prod["newton_tolerance"])]
    if prod_runs:
        fig, axes = plt.subplots(2, len(prod_runs), figsize=(4.6 * len(prod_runs), 6.4), squeeze=False)
        for col, r in enumerate(prod_runs):
            d = r["data"]
            scale = float(np.max(abs(d["hm_physical_power"]) + abs(d["stress_power"]) + abs(d["storage_power"]) + abs(d["flux_power"])))
            ax = axes[0][col]
            for i, (label, key) in enumerate((("stress P:dF/dt", "stress_power"), ("storage p dm/dt", "storage_power"),
                                              ("flux -G.Q", "flux_power"), ("micro power W", "hm_physical_power"))):
                ax.plot(d["time"], d[key], color=COLORS[i], linewidth=2, label=label)
            ax.set_xlabel("t (s)")
            ax.set_ylabel("power per volume (W/m³)")
            ax.set_title(f"power terms, T = {r['job']['T']:g} s", fontsize=10)
            ax.grid(True, color="#dddddd", linewidth=0.6)
            ax.legend(fontsize=7, frameon=False, loc="best")
            ax = axes[1][col]
            for i, (label, key) in enumerate((("power residual", "hm_power_residual"),
                                              ("Eq. (42) predicted defect", "hm_predicted_discrete_power_defect"),
                                              ("corrected residual", "hm_power_identity_residual"),
                                              ("accounting residual", "hm_power_accounting_residual"))):
                ax.plot(d["time"], np.maximum(abs(d[key]) / scale, 1e-18), color=COLORS[i], marker=MARKERS[i],
                        markersize=3, linewidth=1.5, label=label)
            ax.set_yscale("log")
            ax.set_xlabel("t (s)")
            ax.set_ylabel("|value| / largest power")
            ax.set_title(f"Hill-Mandel balance, Newton {prod['newton_tolerance']:g}", fontsize=10)
            ax.grid(True, color="#dddddd", linewidth=0.6)
            ax.legend(fontsize=7, frameon=False, loc="best")
        save(fig, out, "fig_E_balance")
        plt.close(fig)
    histories = [r for r in runs + partial if r["meta"]["kind"] == "history"]
    if histories:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), squeeze=False)
        for ax, study in zip(axes[0], "ABC"):
            for i, r in enumerate([r for r in histories if r["job"]["study"] == study]):
                d = r["data"]
                ax.plot(d["step"], d["rss_gib"], color=COLORS[i % 8], linewidth=1.5,
                        label=f"T{r['job']['T']:g} #{r['job']['index']}")
            ax.set_xlabel("step")
            ax.set_ylabel("resident memory (GiB)")
            ax.set_title(f"study {study}: memory per step", fontsize=10)
            ax.grid(True, color="#dddddd", linewidth=0.6)
            if ax.lines:
                ax.legend(fontsize=6, frameon=False, ncol=2)
        save(fig, out, "fig_memory")
        plt.close(fig)
    print(f"Analysis written to {out}", flush=True)
    print((out / "summary.md").read_text(), flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign", type=Path)
    analyse(p.parse_args().campaign)
