"""Check of the Section 5.2 mechanism: what inside the cell carries the departure from the stationary cell.

The pulse of Figure 4 (figure4_cycles.py, same settings), every step sampled, with these extras per step:
- the axial nominal stress P33 split into solid elastic (with the solid pressure), solid Kelvin-Voigt,
  fluid pressure (macroscopic plus fluctuation), fluid viscous deviatoric (flow within the pores) and
  fluid viscous volumetric (the uniform dilatation lambda_p);
- the dissipation rate split the same way: fluid deviatoric, fluid volumetric (lambda_p), skeleton;
- lambda_p, the range of the pore-pressure fluctuation p_f and the fluid-averaged |dev D_f|;
- VTK snapshots of p_f, the fluctuation velocity and |dev D_f| on the deformed cell at t/T = SNAPSHOTS.

Stresses and rates are integrated in the solver's internal units; the stress parts are rescaled so that
they sum to the dimensional P33 the solver returns. Dissipation rates stay in internal units (compare them
with each other and with the settled background flow, step 0 of the hold). The Jf-weighted mean of p_f is
constrained to zero, so the fluid-pressure part carries only the macroscopic pressure (zero here): the pore
pressure reaches the homogenised stress through the solid walls, in the solid elastic part.

Press Run, or: python run.py pore-fields [--smoke] [--solids elastic] [--output DIR] (about 50 min on 4 threads).
Saves <output>/figure4_pore_fields[_<solid>].npz and <output>/vtk/<solid>_T<T>_<t/T>.vtu (not tracked);
pore_field_summary.py <output> writes the summary pore_fields.md. The paper's run: results/paper/pore_fields_20261112_180711.
Calder: examples/microscale/hpc/submit_figure4_pore_fields.sh.
"""

import time
from pathlib import Path

import numpy as np

import figure4_cycles as f4
import paper_common
from paper_common import command_line, save_results
from ngsolve import CF, Grad, Id, InnerProduct, Integrate, Inv, Norm, Trace, VTKOutput, sqrt
from periodic_fsi import homogenisation as hom
from periodic_fsi.formulation import solid_viscous_stress
from periodic_fsi.layout import ALE_FIELD, LAMBDA_PF_FIELD, PF_FIELD, U_FIELD

# ---------------------------------------------------------------------------
# Settings: edit here and press Run. Material, mesh and pulse come from figure4_cycles.py.
# ---------------------------------------------------------------------------
DURATIONS = [0.2, 2.0]            # pulse durations T (s), those of Figure 4
SOLIDS = ["elastic", "viscoelastic"]
SNAPSHOTS = [0.25, 0.5, 0.75, 1.0]  # VTK snapshots at these t/T
NAME = "figure4_pore_fields"
THREADS = 4
SMOKE = dict(f4_settings=f4.SMOKE, DURATIONS=[0.2], SNAPSHOTS=[0.5, 1.0])
# ---------------------------------------------------------------------------

PARTS = ["solid_elastic", "solid_kv", "fluid_pressure", "fluid_viscous_dev", "fluid_viscous_vol"]
DISSIPATION = ["fluid_dev", "fluid_vol", "skeleton"]
COLUMNS = (["material", "duration_index", "step", "time", "compression", "Q1", "fluid_content", "P33"]
           + [f"P33_{p}" for p in PARTS] + [f"D_{d}" for d in DISSIPATION]
           + ["lambda_p", "pf_min", "pf_max", "devD_fluid_mean"])

CAPTURE = {"active": False, "last": None, "snapshot": None}
_original_output = hom.Homogenisation.homogenisation_output


def _diagnosed_output(self, state, E_dot_trial, P0_trial, F_macro, order, timings=None):
    """The solver's outputs, plus the stress and dissipation split of this step when a pulse is running."""
    result = _original_output(self, state, E_dot_trial, P0_trial, F_macro, order, timings)
    if not CAPTURE["active"]:
        return result
    mesh, geometry, material = self.discretisation.mesh, self.geometry, self.material
    mesh.UnsetDeformation()
    rate = CF(tuple(np.asarray(E_dot_trial).reshape(9)), dims=(3, 3))
    macro_F = CF(tuple(np.asarray(F_macro).reshape(9)), dims=(3, 3))
    c = state.components
    (Ps, Pf, Jf, _), g = self.hill_mandel_integrands(c, Grad(c[U_FIELD]), Grad(c[ALE_FIELD]), rate, P0_trial, macro_F)
    minv_T = g["macro_inverse"].trans
    Js = g["Js"]
    Ds, Df = 0.5 * (g["Ls"] + g["Ls"].trans), 0.5 * (g["Lf"] + g["Lf"].trans)
    tr_f = Trace(Df)
    dev_f = Df - tr_f / 3.0 * Id(3)
    kv = solid_viscous_stress(Ds, eta=material.eta_s, zeta=material.zeta_s)

    def av(cf, region):
        return float(Integrate(cf, mesh, definedon=region, order=order)) / geometry.reference_volume

    S, F = geometry.solid_region, geometry.fluid_region
    kv33 = av((Js * kv * minv_T)[2, 2], S)
    parts = {
        "solid_kv": kv33,
        "solid_elastic": av(Ps[2, 2], S) - kv33,
        "fluid_pressure": av((-Jf * (P0_trial + c[PF_FIELD]) * minv_T)[2, 2], F),
        "fluid_viscous_dev": av((Jf * 2 * material.mu_f * dev_f * minv_T)[2, 2], F),
        "fluid_viscous_vol": av((Jf * 2 * material.mu_f * tr_f / 3.0 * minv_T)[2, 2], F),
    }
    dev_s = Ds - Trace(Ds) / 3.0 * Id(3)
    dissipation = {
        "fluid_dev": av(Jf * 2 * material.mu_f * InnerProduct(dev_f, dev_f), F),
        "fluid_vol": av(Jf * 2 * material.mu_f * tr_f * tr_f / 3.0, F),
        "skeleton": av(Js * (2 * material.eta_s * InnerProduct(dev_s, dev_s)
                             + material.zeta_s * Trace(Ds) * Trace(Ds)), S),
    }
    fluid_volume = av(Jf, F)
    pf = c[PF_FIELD]
    pf_values = np.asarray(pf.vec.FV().NumPy()) if hasattr(pf, "vec") else np.array([np.nan])
    CAPTURE["last"] = dict(
        parts=parts, dissipation=dissipation,
        lambda_p=av(Jf * c[LAMBDA_PF_FIELD], F) / fluid_volume,
        pf_min=float(np.min(pf_values)), pf_max=float(np.max(pf_values)),
        devD=av(Jf * sqrt(InnerProduct(dev_f, dev_f)), F) / fluid_volume,
    )
    if CAPTURE["snapshot"]:
        try:
            deformation = getattr(self.cell, "mesh_displacement", None)
            if deformation is not None:
                mesh.SetDeformation(deformation)
            VTKOutput(ma=mesh, coefs=[pf, c[U_FIELD], sqrt(InnerProduct(dev_f, dev_f))],
                      names=["pore_pressure_fluctuation", "velocity_fluctuation", "fluid_dev_rate"],
                      filename=str(CAPTURE["snapshot"]), subdivision=1).Do()
        except Exception as error:  # a snapshot must never stop the run
            print(f"    VTK snapshot failed: {error}", flush=True)
        finally:
            mesh.UnsetDeformation()
    return result


hom.Homogenisation.homogenisation_output = _diagnosed_output


def row(m, j, step, t, compression, flux, content, P, d):
    """One output row; the stress parts rescaled to sum to the dimensional P33."""
    internal = sum(d["parts"].values())
    scale = P[2, 2] / internal if abs(internal) > 1e-14 else np.nan
    return [m, j, step, t, compression, flux[0], content, P[2, 2], *[scale * d["parts"][p] for p in PARTS],
            *[d["dissipation"][k] for k in DISSIPATION], d["lambda_p"], d["pf_min"], d["pf_max"], d["devD"]]


def pulse(run, settled, m, j, duration, material, rows, vtk):
    """The Figure 4 pulse and hold, every step, with the extras of _diagnosed_output."""
    dt = duration / f4.STEPS
    state = settled[0]
    total = f4.STEPS + int(round(f4.RECOVERY_FRACTION * f4.STEPS))
    snapshots = {int(round(s * f4.STEPS)): s for s in SNAPSHOTS}
    started = time.perf_counter()
    for step in range(1, total + 1):
        compression = 0.5 * f4.COMPRESSION * (1 - np.cos(2 * np.pi * step / f4.STEPS)) if step <= f4.STEPS else 0.0
        CAPTURE["active"] = True
        CAPTURE["snapshot"] = vtk / f"{material}_T{duration:g}_{snapshots[step]:.2f}" if step in snapshots else None
        r = f4.advance(run, dt, np.diag([1.0, 1.0, 1.0 - compression]), state, True, {})
        CAPTURE["active"] = False
        state = r["restart_state"]
        flux, content, P = f4.values(r)
        rows.append(row(m, j, step, step * dt, compression, flux, content, P, CAPTURE["last"]))
        if step % 10 == 0 or step == total:
            per_step = (time.perf_counter() - started) / step
            print(f"    step {step}/{total}, t/T = {step / f4.STEPS:.2f}, {per_step:.2f} s/step", flush=True)


def main():
    if f4.COMPILE_TANGENT:
        f4.compiler_environment()
    f4.SetNumThreads(THREADS)
    vtk = paper_common.DATA / "vtk"
    vtk.mkdir(parents=True, exist_ok=True)
    rows, started = [], time.perf_counter()
    options = dict(K=f4.K, G=f4.G, mu_f=f4.MU_F, rve_length=f4.RVE_LENGTH, maxh=f4.MAXH, order=f4.ORDER,
                   quadrature_order=f4.QUADRATURE_ORDER, mesh_mu=f4.MESH_MU, mesh_lame_lambda=f4.MESH_LAME_LAMBDA,
                   geometry=f4.partial(f4.square_geo, f4.CHANNEL_WIDTH), print_iterations=False, Profiling=True,
                   tangent_method=None, compile_tangent=f4.COMPILE_TANGENT)
    for m, material in enumerate(SOLIDS):
        eta, zeta = f4.MATERIALS[material]
        with f4.TaskManager(), f4.FSISession(**options, eta_s=eta, zeta_s=zeta) as run:
            with f4.contextlib.redirect_stdout(f4.io.StringIO()):
                rest = run(dt=f4.SETTLE_DT, t_final=f4.SETTLE_DT, F_target=np.eye(3), P_target=0.0,
                           G_target=np.zeros(3))["restart_state"]
            print(f"{material}: settling", flush=True)
            CAPTURE["active"] = True  # the last settling check is the steady background flow: step 0
            settled = f4.settle(run, rest, {})
            CAPTURE["active"] = False
            background = CAPTURE["last"]
            for j, duration in enumerate(DURATIONS):
                print(f"{material}, T = {duration:g} s:", flush=True)
                rows.append(row(m, j, 0, 0.0, 0.0, settled[1], settled[2], settled[3], background))
                pulse(run, settled, m, j, duration, material, rows, vtk)
    settings = dict(durations=DURATIONS, solids=SOLIDS, snapshots=SNAPSHOTS, steps=f4.STEPS,
                    compression=f4.COMPRESSION, maxh=f4.MAXH, eta_s=f4.ETA_S, zeta_s=f4.ZETA_S, threads=THREADS)
    save_results(NAME, settings, time.perf_counter() - started, rows=np.asarray(rows, dtype=float),
                 columns=np.array(COLUMNS), materials=np.array(SOLIDS), durations=np.asarray(DURATIONS))


if __name__ == "__main__":
    smoke = dict(SMOKE)
    f4_smoke = smoke.pop("f4_settings")
    import sys
    if "--smoke" in sys.argv:
        for key, value in f4_smoke.items():
            if key not in ("DURATIONS", "SAMPLE_EVERY"):
                setattr(f4, key, value)
    command_line(globals(), smoke)
    main()
