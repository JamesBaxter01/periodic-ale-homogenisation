"""Physical scaling, restarts, elastic constants and the fixed-step driver."""

from __future__ import annotations
import gc
from time import perf_counter
import numpy as np
from ngsolve import GridFunction, VectorH1

from .core import _scaled_core_FSI
from .inputs import _central_difference_steps, _resolve_elastic_parameters


# State containers, restart helpers and reusable model sessions.
def _restart_for_timestep(state, *, dt, rve_length, stress_scale):
    """Return a scale-compatible restart without mutating the accepted arrays.

    The six unknowns are u, p_f, p_s, lambda_u, lambda_pf, delta_w.
    Velocity and dilatation scale with the increment duration; pressure and
    momentum multiplier scales do not. Material/mesh compatibility is still
    checked by the core after converting the stored dimensionless signature.
    """
    if state is None:
        return None
    required = ("time_scale", "rve_length", "stress_scale", "scaling_schema")
    if any((key not in state for key in required)):
        raise ValueError(
            "Restart lacks physical scaling metadata; recreate it with the dimensional FSI interface"
        )
    if state["scaling_schema"] != "unit_geometry_increment_v1":
        raise ValueError("Unsupported restart scaling schema")
    old_dt = float(state["time_scale"])
    if not np.isfinite(old_dt) or old_dt <= 0:
        raise ValueError("Invalid restart time scale")
    for name, current in (("rve_length", rve_length), ("stress_scale", stress_scale)):
        if not np.isclose(float(state[name]), current, rtol=1e-12, atol=0):
            raise ValueError(f"Restart requires unchanged physical {name}")
    signature = tuple(state["signature"])
    if len(signature) != 9 or len(state["unknowns"]) != 6:
        raise ValueError("Incompatible restart signature or field layout")
    if dt == old_dt:
        return state
    converted = dict(state)
    ratio = dt / old_dt
    converted["signature"] = tuple(
        (value / ratio if index in (2, 3, 4) else value for index, value in enumerate(signature))
    )
    converted["unknowns"] = tuple(
        (
            np.array(values, dtype=float, copy=True) * (ratio if index in (0, 4, 5) else 1.0)
            for index, values in enumerate(state["unknowns"])
        )
    )
    converted.update(time_scale=dt, velocity_scale=rve_length / dt)
    return converted


def restore_accepted_mesh(state):
    """Restore the shared mesh display after a rejected trial (RAM only).

    Persistent accepted arrays remain unchanged. Callers must discard live trial
    fields and reload accepted arrays on the next solve, as FSI already does.
    """
    if state is None:
        return
    mesh = state["mesh"]
    mesh.UnsetDeformation()
    field = GridFunction(VectorH1(mesh, order=int(state["signature"][5])))
    values = field.vec.FV().NumPy()
    stored = np.asarray(state["mesh_displacement"])
    if values.shape != stored.shape:
        raise ValueError("Incompatible accepted mesh displacement")
    values[:] = stored
    mesh.SetDeformation(field)


# Physical units and the fixed-step public interface.
def _dimensional_FSI(
    E,
    nu,
    dt,
    t_final,
    mu_f,
    F_target,
    P_target,
    G_target,
    *,
    eta_s=0.0,
    zeta_s=0.0,
    F_rate=None,
    rve_length=1.0,
    **kwargs,
):
    """Nondimensional RVE interface with dimensional homogenised outputs."""
    E = float(E)
    nu = float(nu)
    dt = float(dt)
    t_final = float(t_final)
    mu_f = float(mu_f)
    eta_s = float(eta_s)
    zeta_s = float(zeta_s)
    rve_length = float(rve_length)
    if F_target is not None:
        F_target = np.asarray(F_target, dtype=float)
    G_target = np.asarray(G_target, dtype=float)
    P_target = float(P_target)
    session = kwargs.get("_solver_session")
    if session is not None:
        if session._scales is not None and np.isclose(dt, session._scales[-1], rtol=1e-13, atol=0):
            dt = session._scales[-1]
        scales = (rve_length, E, nu, mu_f, eta_s, zeta_s, dt)
        if session._scales is not None and scales != session._scales:
            if rve_length != session._scales[0]:
                raise ValueError("FSISession cell length changed; start a new session")
            if scales[1:-1] != session._scales[1:-1] and kwargs.get("initial_state") is not None:
                raise ValueError("FSISession material changed; start a new history by omitting initial_state")
            # The cell is kept: the core passes the new (scaled) material values to it.
        session._scales = scales
    mu_s = E / (2.0 * (1.0 + nu))
    velocity_scale = rve_length / dt
    E_hat = E / mu_s
    mu_f_hat = mu_f / (mu_s * dt)
    eta_s_hat = eta_s / (mu_s * dt)
    zeta_s_hat = zeta_s / (mu_s * dt)
    F_rate_hat = (
        None
        if F_rate is None
        else (
            (lambda time: dt * np.asarray(F_rate(time * dt), dtype=float))
            if callable(F_rate)
            else dt * np.asarray(F_rate, dtype=float)
        )
    )
    P_hat = P_target / mu_s
    G_hat = rve_length * G_target / mu_s
    core_kwargs = dict(kwargs)
    if core_kwargs.get("loading_path") is not None:
        scaled_path = np.array(core_kwargs["loading_path"], dtype=float, copy=True)
        scaled_path[:, 9] /= mu_s
        scaled_path[:, 10:] *= rve_length / mu_s
        core_kwargs["loading_path"] = scaled_path
    mesh_mu_physical = float(core_kwargs.pop("mesh_mu", 1.0))
    mesh_lambda_physical = float(core_kwargs.pop("mesh_lame_lambda", 5.0))
    core_kwargs["mesh_mu"] = 1.0
    core_kwargs["mesh_lame_lambda"] = mesh_lambda_physical / mesh_mu_physical
    preconditioner_options = core_kwargs.get("preconditioner_options")
    if preconditioner_options is not None:
        preconditioner_options = dict(preconditioner_options)
        if "pressure_mass_inverse_scale" in preconditioner_options:
            preconditioner_options["pressure_mass_inverse_scale"] /= mu_s * dt
        core_kwargs["preconditioner_options"] = preconditioner_options
    core_kwargs["_time_scale"] = dt  # printed times in physical units
    core_kwargs["initial_state"] = _restart_for_timestep(
        kwargs.get("initial_state"), dt=dt, rve_length=rve_length, stress_scale=mu_s
    )
    try:
        result = _scaled_core_FSI(
            E_hat,
            nu,
            1.0,
            t_final / dt,
            mu_f_hat,
            F_target,
            P_hat,
            G_hat,
            eta_s=eta_s_hat,
            zeta_s=zeta_s_hat,
            F_rate=F_rate_hat,
            **core_kwargs,
        )
    except Exception:
        restore_accepted_mesh(kwargs.get("initial_state"))
        raise
    result["E"] = E
    result["mu_s"] = mu_s
    result["mu_f"] = mu_f
    result["eta_s"] = eta_s
    result["zeta_s"] = zeta_s
    result["dt"] = dt
    result["t_final"] = t_final
    result["P_target"] = P_target
    result["G_target"] = G_target.copy()
    result["rve_length"] = rve_length
    result["prescribed_times"] = np.asarray(result["prescribed_times"], dtype=float) * dt
    result["prescribed_P_history"] = np.asarray(result["prescribed_P_history"]) * mu_s
    result["prescribed_G_history"] = np.asarray(result["prescribed_G_history"]) * mu_s / rve_length
    result["loading_path"] = np.column_stack(
        (
            np.asarray(result["prescribed_F_history"]).reshape(-1, 9),
            result["prescribed_P_history"],
            result["prescribed_G_history"],
        )
    )
    result["prescribed_F_rate_history"] = np.asarray(result["prescribed_F_rate_history"], dtype=float) / dt
    result["max_displacements"] = [rve_length * float(value) for value in result["max_displacements"]]
    if "interface_correction_norm" in result:
        result["interface_correction_norm"] *= velocity_scale * rve_length
    result["nondimensional_scales"] = {
        "length": rve_length,
        "stress": mu_s,
        "time": dt,
        "velocity": velocity_scale,
        "E_hat": E_hat,
        "mu_f_hat": mu_f_hat,
        "eta_s_hat": eta_s_hat,
        "zeta_s_hat": zeta_s_hat,
        "P_hat": P_hat,
        "G_hat": G_hat.copy(),
    }
    if "history" in result:
        result["history"]["times"] = np.asarray(result["history"]["times"], dtype=float) * dt
        result["history"]["time_units"] = "physical (same units as dt)"
        result["history"]["scales"] = dict(result["nondimensional_scales"])
    response = result.get("homogenised_response")
    if response is not None:
        inputs_hat = np.asarray(response["inputs"], dtype=float)
        outputs_hat = np.asarray(response["outputs"], dtype=float)
        jacobian_hat = None if response["jacobian"] is None else np.asarray(response["jacobian"], dtype=float)
        input_scales = np.concatenate((np.ones(9), np.asarray([mu_s]), np.full(3, mu_s / rve_length)))
        output_scales = np.concatenate((np.full(9, mu_s), np.asarray([1.0]), np.full(3, velocity_scale)))
        response["inputs_hat"] = inputs_hat.copy()
        response["outputs_hat"] = outputs_hat.copy()
        response["jacobian_hat"] = None if jacobian_hat is None else jacobian_hat.copy()
        response["inputs"] = inputs_hat * input_scales
        response["outputs"] = outputs_hat * output_scales
        response["jacobian"] = (
            None if jacobian_hat is None else output_scales[:, None] * jacobian_hat / input_scales[None, :]
        )
        response["P_total"] = response["outputs"][:9].reshape((3, 3))
        if "solid_nominal_stress" in response:
            response["solid_nominal_stress_hat"] = np.asarray(
                response["solid_nominal_stress"], dtype=float
            ).copy()
            response["solid_nominal_stress"] = mu_s * response["solid_nominal_stress_hat"]
        if "fluid_nominal_stress" in response:
            response["fluid_nominal_stress_hat"] = np.asarray(
                response["fluid_nominal_stress"], dtype=float
            ).copy()
            response["fluid_nominal_stress"] = mu_s * response["fluid_nominal_stress_hat"]
        response["fluid_content"] = float(response["outputs"][9])
        response["flux"] = response["outputs"][10:].copy()
        diagnostics = response.get("hill_mandel_diagnostics")
        if diagnostics is not None:
            response["hill_mandel_diagnostics_hat"] = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in diagnostics.items()
            }
            scales = {}
            for key in (
                "Q0",
                "Q_ALE",
                "Q_kinematic",
                "Q_lift_difference",
                "K_rate",
                "K_rate_be",
                "moment_correction",
                "moment_projection_residual",
                "solid_frame_mean",
                "solid_frame_assembled",
                "solid_frame_quadrature_difference",
                "Q_boundary",
                "moment_identity_residual",
            ):
                scales[key] = velocity_scale
            for key in ("K", "fluid_centroid"):
                scales[key] = rve_length
            for key in (
                "m_rate_be",
                "lambda_p",
                "fluid_divergence_integral",
                "continuity_mean_residual",
                "storage_be_defect",
                "geometric_chain_defect",
                "relative_net_outflow",
            ):
                scales[key] = 1.0 / dt
            for key in ("P_endpoint", "P_incremental", "pressure_mean"):
                scales[key] = mu_s
            for key in (
                "physical_power",
                "power_residual",
                "predicted_discrete_power_defect",
                "power_identity_residual",
                "pressure_fluctuation_power",
                "momentum_fluctuation_residual",
                "solid_frame_power",
                "constraint_power_residual",
                "power_accounting_residual",
            ):
                scales[key] = mu_s / dt
            for key, scale in scales.items():
                if diagnostics.get(key) is not None:
                    diagnostics[key] = diagnostics[key] * scale
            diagnostics["units"] = (
                "physical; moments in length, flux in length/time, power per reference volume"
            )
            diagnostics["conversion_scales"] = scales
        response["physical_scales"] = {"input": input_scales, "output": output_scales}
    restart = result.get("restart_state")
    if restart is not None:
        restart["rve_length"] = rve_length
        restart["stress_scale"] = mu_s
        restart["time_scale"] = dt
        restart["velocity_scale"] = velocity_scale
        restart["scaling_schema"] = "unit_geometry_increment_v1"
        if response is not None:
            restart["homogenised_outputs"] = np.asarray(response["outputs"], dtype=float).copy()
    return result


def _FSI_fixed(
    E=None,
    nu=None,
    dt=None,
    t_final=None,
    mu_f=None,
    F_target=None,
    P_target=None,
    G_target=None,
    *,
    K=None,
    G=None,
    mu=None,
    lam=None,
    eta_s=0.0,
    zeta_s=0.0,
    F_rate=None,
    rve_length=1.0,
    tangent_method="linear correctors",
    central_difference_step=0.0001,
    save_history=False,
    loading_path=None,
    **kwargs,
):
    """Advance the cell; optionally choose its homogenised tangent method.

    ``loading_path`` is a real array of shape (N + 1, 13): row-major F (9),
    physical mean pressure (1), physical spatial pressure gradient (3).
    Row i applies at i*dt; row zero must match the initial/restart state.
    Repeated rows prescribe holds, not skipped timesteps. Omit t_final to infer
    N*dt, or provide that matching value. Do not also pass *_target or F_rate.
    The existing target/rate interface remains available when no path is given.
    Returned loading_path and prescribed_P/G_history use physical input units.


    ``save_history=True`` additionally returns ``result["history"]`` with
    physical ``times`` (relative to this call), ``field_names``, one multidim
    GridFunction per entry of ``components``, and a matching multidim
    ``mesh_displacement``. Frame zero is the supplied initial/restart state;
    later frames are accepted physical timesteps only. Coefficients are copied
    and remain in the same dimensionless units as the final fields. History
    storage is off by default and grows linearly with frames and field DOFs.
    Clear the returned mesh's final deformation before WebGUI playback to avoid
    applying it twice. Histories are visualization snapshots, not restart files.


    ``mesh_quality_summary`` is always returned. With diagnostics=True::

        Draw(result["mesh_quality"], result["mesh"], "mesh_quality")
        print(result["mesh_quality_summary"])

    ``mesh_quality`` is a scalar L2(0) snapshot of sampled signed mean-ratio
    shape quality on the final deformed tetrahedral mesh: one is ideal, zero
    is collapsed/nonfinite, negative indicates inversion relative to the
    reference mesh. ``mesh_ale_jacobian_min`` is the sampled minimum ALE
    volume ratio per element. Both include curved geometry and the total
    accumulated displacement. The mesh already has that deformation applied;
    do not add the displacement again when drawing. Sampled extrema do not
    certify validity between sample points. The summary records extrema,
    unweighted mean element quality, worst element (zero-based), and counts.
    These dimensionless diagnostics are independent of ``rve_length``.

    Supply exactly two elastic constants among E, nu, K, G (alias mu),
    and lam (first Lame parameter). Moduli use consistent physical units.
    Examples: FSI(E=112.5, nu=0.125, ...), FSI(K=50, G=50, ...),
    FSI(K=50, nu=0.125, ...), FSI(mu=50, lam=50/3, ...).
    Existing positional E, nu calls remain supported. G is solid shear
    modulus, distinct from the pressure-gradient input G_target and mu_f.
    Physical conversion requires E, K, G > 0 and -1 < nu < 0.5.
    Zero and negative Poisson ratios are supported using a nonsingular
    scaled solid-pressure equation; the elastic energy is unchanged.

    ``linear correctors`` (default) uses the existing symbolic correctors;
    ``None`` returns the homogenised outputs without a tangent (no correctors).
    ``central difference`` resolves 26 perturbed cells and the base cell,
    skipping correctors. It requires homogenise=True, F_target, and one
    increment (t_final=dt). All solves start from the SAME committed state
    and reference mesh; perturbed states are never committed. Fresh calls
    first construct an unloaded seed on that mesh. The returned state and
    diagnostics belong to the unperturbed solve, performed last.

    A scalar central_difference_step is relative to max(abs(z_i), s_i),
    with z=(vec(F), p, spatial grad(p)) and dimensional characteristic
    scales s=(1, mu_s, mu_s/rve_length). A length-13 array instead specifies
    positive ABSOLUTE steps in physical input units, in row-major order.
    Compare several step sizes and nonlinear tolerances for verification.
    Central differences measure the same frozen-previous-state incremental
    derivative as the correctors, not the derivative of an entire history.
    """
    E, nu = _resolve_elastic_parameters(E=E, nu=nu, K=K, G=G, mu=mu, lam=lam)
    common = dict(
        E=E,
        nu=nu,
        dt=dt,
        t_final=t_final,
        mu_f=mu_f,
        eta_s=eta_s,
        zeta_s=zeta_s,
        rve_length=rve_length,
        save_history=bool(save_history),
        loading_path=loading_path,
        **kwargs,
    )
    if tangent_method in ("linear correctors", None):
        if tangent_method is None:
            # Homogenised outputs without a tangent: no correctors are solved.
            common["_outputs_only"] = True
        result = _dimensional_FSI(
            F_target=F_target, P_target=P_target, G_target=G_target, F_rate=F_rate, **common
        )
        if result.get("homogenised_response") is not None:
            result["homogenised_response"]["tangent_method"] = tangent_method
        return result
    common.pop("loading_path", None)
    # Per increment: the sequence driver supplies new targets on every call.
    steps = _central_difference_steps(
        F_target,
        P_target,
        G_target,
        F_rate,
        E,
        nu,
        dt,
        t_final,
        rve_length,
        central_difference_step,
        kwargs.get("homogenise", False),
    )
    F = np.asarray(F_target, dtype=float)
    g = np.asarray(G_target, dtype=float)
    z = np.r_[F.reshape(9), float(P_target), g]
    previous = kwargs.get("initial_state")
    if previous is None:
        seed_options = dict(common, homogenise=False, save_history=False)
        seed_options.pop("initial_state", None)
        seed = _dimensional_FSI(F_target=np.eye(3), P_target=0.0, G_target=np.zeros(3), **seed_options)
        previous = seed["restart_state"]
        del seed
        gc.collect()
    trial_options = dict(common, initial_state=previous, _outputs_only=True, save_history=False)
    jacobian = np.empty((13, 13))
    started = perf_counter()
    for j in range(13):
        values = []
        for sign in (1, -1):
            trial = z.copy()
            trial[j] += sign * steps[j]
            result = _dimensional_FSI(
                F_target=trial[:9].reshape(3, 3), P_target=trial[9], G_target=trial[10:], **trial_options
            )
            value = result["homogenised_response"]["outputs"].copy()
            if not np.isfinite(value).all():
                raise RuntimeError(f"Non-finite central-difference response in column {j}")
            values.append(value)
            del result
            gc.collect()
        jacobian[:, j] = (values[0] - values[1]) / (2 * steps[j])
    trial_options["save_history"] = bool(save_history)
    result = _dimensional_FSI(F_target=F, P_target=P_target, G_target=g, **trial_options)
    response = result["homogenised_response"]
    input_scales = response["physical_scales"]["input"]
    output_scales = response["physical_scales"]["output"]
    response.update(
        jacobian=jacobian,
        jacobian_hat=jacobian * input_scales[None, :] / output_scales[:, None],
        tangent_method=tangent_method,
        sensitivity_method="central_difference_resolved",
        central_difference_steps=steps,
        central_difference_trial_solves=26,
        central_difference_seconds=float(perf_counter() - started),
        corrector_solver=None,
        coupling_factorization=None,
        seconds=float(perf_counter() - started),
    )
    return result
