"""Input checks shared by the entry point, the sequence driver and the core."""

from __future__ import annotations
import numpy as np


def _validate_loading_path(loading_path, dt, t_final=None, expected_initial=None):
    """Copy physical endpoint targets: row-major F (9), pressure, gradient (3)."""
    if np.iscomplexobj(loading_path):
        raise ValueError("loading_path must contain real values")
    path = np.array(loading_path, dtype=float, copy=True)
    if path.ndim != 2 or path.shape[1] != 13 or path.shape[0] < 2:
        raise ValueError("loading_path must have shape (N + 1, 13), N >= 1")
    if not np.isfinite(path).all():
        raise ValueError("loading_path must contain finite values")
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError("loading_path requires finite dt > 0")
    duration = (len(path) - 1) * float(dt)
    if t_final is not None and (
        not np.isfinite(t_final)
        or not np.isclose(t_final, duration, rtol=1e-10, atol=1e-12 * max(1.0, duration))
    ):
        raise ValueError("t_final must equal (len(loading_path) - 1) * dt")
    determinants = np.linalg.det(path[:, :9].reshape(-1, 3, 3))
    if not np.isfinite(determinants).all() or np.any(determinants <= 0):
        raise ValueError("every loading_path row must have finite det(F) > 0")
    if expected_initial is not None and (not np.allclose(path[0], expected_initial, rtol=1e-10, atol=1e-12)):
        raise ValueError("loading_path row 0 must match the initial/restart macroscopic state")
    return (path, duration)


def _resolve_elastic_parameters(*, E=None, nu=None, K=None, G=None, mu=None, lam=None):
    """Convert exactly two independent isotropic elastic constants to E, nu.

    G and mu both denote solid shear modulus; lam is the first Lame
    parameter, not bulk modulus. This conversion checks the physical
    small-strain bounds K > 0, G > 0 and -1 < nu < 0.5.
    All moduli must use the same units.
    """
    supplied = {
        name: value for name, value in dict(E=E, nu=nu, K=K, G=G, mu=mu, lam=lam).items() if value is not None
    }
    if len(supplied) != 2:
        raise ValueError(
            f"Supply exactly two elastic parameters from E, nu, K, G (or mu), lam; received {len(supplied)}: {', '.join(supplied) or 'none'}"
        )
    if G is not None and mu is not None:
        raise ValueError("G and mu are aliases for shear modulus, not independent parameters")
    values = {}
    for name, value in supplied.items():
        if isinstance(value, (bool, np.bool_, str, bytes)) or np.ndim(value) != 0 or np.iscomplexobj(value):
            raise ValueError(f"{name} must be a finite real scalar")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a finite real scalar") from error
        if not np.isfinite(number):
            raise ValueError(f"{name} must be a finite real scalar")
        if name in ("E", "K", "G", "mu") and number <= 0:
            raise ValueError(f"{name} must be positive")
        if name == "nu" and (not -1.0 < number < 0.5):
            raise ValueError("Physical Poisson ratio must satisfy -1 < nu < 0.5")
        values["G" if name == "mu" else name] = number
    scale = max((abs(value) for name, value in values.items() if name != "nu"))
    if scale == 0:
        raise ValueError("nu and lam must determine a nonzero elastic stiffness")
    v = {name: value if name == "nu" else value / scale for name, value in values.items()}
    try:
        if "nu" in v:
            poisson = v["nu"]
            if "E" in v:
                young = v["E"]
            elif "G" in v:
                young = 2 * v["G"] * (1 + poisson)
            elif "K" in v:
                young = 3 * v["K"] * (1 - 2 * poisson)
            else:
                if poisson == 0:
                    raise ValueError("nu=0 and lam do not determine a unique physical stiffness")
                young = v["lam"] * (1 + poisson) * (1 - 2 * poisson) / poisson
        elif "E" in v:
            young = v["E"]
            if "G" in v:
                poisson = young / (2 * v["G"]) - 1
            elif "K" in v:
                poisson = (3 * v["K"] - young) / (6 * v["K"])
            else:
                lame = v["lam"]
                shear = (young - 3 * lame + np.sqrt((young + lame) ** 2 + 8 * lame**2)) / 4
                poisson = young / (2 * shear) - 1
        else:
            if "K" in v and "G" in v:
                bulk, shear = (v["K"], v["G"])
            elif "K" in v:
                bulk = v["K"]
                shear = 1.5 * (bulk - v["lam"])
            else:
                shear = v["G"]
                bulk = v["lam"] + 2 * shear / 3
            if bulk <= 0 or shear <= 0:
                raise ValueError("Elastic parameters must produce K > 0 and G > 0")
            young = 9 * bulk * shear / (3 * bulk + shear)
            poisson = (3 * bulk - 2 * shear) / (2 * (3 * bulk + shear))
        young = float(young * scale)
        poisson = float(poisson)
    except (ZeroDivisionError, OverflowError) as error:
        raise ValueError("Elastic pair is singular or outside the representable range") from error
    if not np.isfinite(young) or young <= 0 or (not np.isfinite(poisson)) or (not -1 < poisson < 0.5):
        raise ValueError("Elastic pair must produce finite E > 0 and -1 < nu < 0.5 (K > 0, G > 0)")
    return (young, poisson)


def newton_defaults():
    """Default Newton, line-search and admissibility options (nondimensional)."""
    return {
        "maximum_iterations": 40,
        "initial_guess": "previous",
        "tolerance": 1e-08,
        "relative_tolerance": 1e-08,
        "line_search": True,
        "dampfactor": 1.0,
        "armijo": 0.0001,
        "backtrack_factor": 0.5,
        "minimum_step": 1e-06,
        "max_backtracks": 20,
        "minimum_solid_jacobian": 1e-08,
        "minimum_mesh_jacobian": 1e-08,
        "minimum_porosity": 1e-08,
        "check_solid_jacobian": True,
        "check_mesh_jacobian": True,
        "check_porosity": True,
        "jacobian_check_order": 6,
        "linear_solver": "gmres",  # Newton directions: "gmres" (block-preconditioned) or "mumps" (direct LU)
    }


def preconditioner_defaults(mu_f):
    """Default linear-solver options; ``mu_f`` is the fluid viscosity in the units of the options."""
    return {
        "corrector_blas_threads": 13,
        "petsc_newton_pc": "amg",
        "pressure_mass_inverse_scale": 2.0 * mu_f,
        "velocity_mean_regularization": 1e-08,
        "homogenisation_gmres_tol": 1e-10,
        "homogenisation_order": 4,
    }


def _merge_options(defaults, overrides, family):
    """Return validated option overrides without mutating shared defaults."""
    result = dict(defaults)
    if overrides is None:
        return result
    unknown = set(overrides) - set(result)
    if unknown:
        raise ValueError(f"unknown {family} option(s): " + ", ".join(sorted(unknown)))
    result.update(overrides)
    return result


def _check_newton_options(newton_options):
    options = _merge_options(newton_defaults(), newton_options, "Newton")
    if options["initial_guess"] not in ("previous", "zero"):
        raise ValueError("initial_guess must be previous or zero")
    if options["maximum_iterations"] < 1:
        raise ValueError("maximum_iterations must be positive")
    if options["tolerance"] <= 0.0:
        raise ValueError("tolerance must be positive")
    if options["relative_tolerance"] < 0.0:
        raise ValueError("relative_tolerance must be non-negative")
    if not 0.0 < options["dampfactor"] <= 1.0:
        raise ValueError("dampfactor must lie in (0, 1]")
    if not 0.0 < options["armijo"] < 1.0:
        raise ValueError("armijo must lie in (0, 1)")
    if not 0.0 < options["backtrack_factor"] < 1.0:
        raise ValueError("backtrack_factor must lie in (0, 1)")
    if not 0.0 < options["minimum_step"] <= 1.0:
        raise ValueError("minimum_step must lie in (0, 1]")
    if options["max_backtracks"] < 0:
        raise ValueError("max_backtracks must be non-negative")
    if options["minimum_solid_jacobian"] <= 0.0:
        raise ValueError("minimum_solid_jacobian must be positive")
    if options["minimum_mesh_jacobian"] <= 0.0:
        raise ValueError("minimum_mesh_jacobian must be positive")
    if not 0.0 <= options["minimum_porosity"] < 1.0:
        raise ValueError("minimum_porosity must lie in [0, 1)")
    for option_name in ("check_solid_jacobian", "check_mesh_jacobian", "check_porosity"):
        if not isinstance(options[option_name], bool):
            raise TypeError(f"{option_name} must be boolean")
    if options["jacobian_check_order"] < 1:
        raise ValueError("jacobian_check_order must be positive")


def _check_preconditioner_options(preconditioner_options, quadrature_order, order, mu_f):
    pc_options = _merge_options(preconditioner_defaults(mu_f), preconditioner_options, "preconditioner")
    if pc_options["petsc_newton_pc"] != "amg":
        raise ValueError("petsc_newton_pc must be 'amg'")
    corrector_threads = pc_options["corrector_blas_threads"]
    if corrector_threads is not None and (
        isinstance(corrector_threads, (bool, np.bool_))
        or not isinstance(corrector_threads, (int, np.integer))
        or corrector_threads < 1
    ):
        raise ValueError("corrector_blas_threads must be a positive integer or None")
    if pc_options["velocity_mean_regularization"] <= 0.0:
        raise ValueError("velocity_mean_regularization must be positive")
    if (
        not np.isfinite(pc_options["pressure_mass_inverse_scale"])
        or pc_options["pressure_mass_inverse_scale"] <= 0.0
    ):
        raise ValueError("pressure_mass_inverse_scale must be positive")
    if pc_options["homogenisation_order"] < 1:
        raise ValueError("homogenisation_order must be positive")
    if quadrature_order is None:
        quadrature_order = max(2 * order, 4 * order - 3, int(pc_options["homogenisation_order"]))
    if (
        isinstance(quadrature_order, (bool, np.bool_))
        or not isinstance(quadrature_order, (int, np.integer))
        or quadrature_order < 1
    ):
        raise ValueError("quadrature_order must be a positive integer or None")


def _central_difference_steps(F_target, P_target, G_target, F_rate, E, nu, dt, t_final, rve_length, step, homogenise):
    """Check a central-difference request; return the 13 absolute input steps."""
    if not homogenise:
        raise ValueError("central difference requires homogenise=True")
    if F_rate is not None or F_target is None:
        raise ValueError("central difference requires F_target and no F_rate")
    if not np.isfinite(dt) or dt <= 0 or (not np.isclose(t_final, dt, rtol=1e-12, atol=0)):
        raise ValueError("central difference requires one increment: t_final=dt>0")
    F = np.asarray(F_target, dtype=float)
    g = np.asarray(G_target, dtype=float)
    if F.shape != (3, 3) or g.shape != (3,):
        raise ValueError("F_target must be (3,3) and G_target must be (3,)")
    z = np.r_[F.reshape(9), float(P_target), g]
    if not np.isfinite(z).all():
        raise ValueError("central-difference inputs must be finite")
    if (
        not np.isfinite(E)
        or E <= 0
        or (not -1 < nu < 0.5)
        or (not np.isfinite(rve_length))
        or (rve_length <= 0)
    ):
        raise ValueError("invalid material or length scale")
    stress_scale = E / (2 * (1 + nu))
    scales = np.r_[np.ones(9), stress_scale, np.full(3, stress_scale / rve_length)]
    step = np.asarray(step, dtype=float)
    if step.ndim == 0:
        steps = float(step) * np.maximum(np.abs(z), scales)
    elif step.shape == (13,):
        steps = step.copy()
    else:
        raise ValueError("central_difference_step must be scalar or length 13")
    if not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError("central difference steps must be finite and positive")
    for j in range(13):
        for sign in (-1, 1):
            trial = z.copy()
            trial[j] += sign * steps[j]
            if trial[j] == z[j]:
                raise ValueError("central difference step is below floating-point resolution")
            if np.linalg.det(trial[:9].reshape(3, 3)) <= 0:
                raise ValueError("central difference crosses det(F)<=0; reduce the step")
    return steps


def _check_viscosities(mu_f, eta_s, zeta_s):
    mu_f = float(mu_f)
    eta_s = float(eta_s)
    zeta_s = float(zeta_s)
    if not np.isfinite(mu_f) or mu_f <= 0.0:
        raise ValueError("mu_f must be finite and positive")
    if not np.isfinite(eta_s) or eta_s < 0.0:
        raise ValueError("eta_s must be finite and non-negative")
    if not np.isfinite(zeta_s) or zeta_s < 0.0:
        raise ValueError("zeta_s must be finite and non-negative")
    return mu_f


def check_inputs(
    *,
    sequence,
    E,
    nu,
    K,
    G,
    mu,
    lam,
    dt,
    t_final,
    mu_f,
    F_target,
    P_target,
    G_target,
    F_rate,
    eta_s,
    zeta_s,
    rve_length,
    tangent_method,
    central_difference_step,
    save_history,
    save_response_history,
    loading_path,
    kwargs,
):
    """Check FSI's arguments once, before any solve, in the order the solver used to.

    For a fixed-step call the loading path is validated here and replaces the
    targets, so this returns ``(t_final, F_target, P_target, G_target,
    loading_path)``; a sequence call (variable dt) returns None, and its time
    grid and path are checked in sequence.py. Checks that need the restart
    state or the session (restart signature, det(F) along the path from the
    restart, session settings) and the values of a callable F_rate stay where
    those are used. Central-difference steps are also recomputed per increment.
    """
    if not isinstance(save_history, (bool, np.bool_)):
        raise TypeError("save_history must be a bool")
    if save_response_history is not None and (not isinstance(save_response_history, (bool, np.bool_))):
        raise TypeError("save_response_history must be bool or None")
    if sequence:
        if tangent_method not in ("linear correctors", "central difference", None):
            raise ValueError("Unknown tangent_method")
        if mu_f is None:
            raise TypeError("mu_f is required")
        _resolve_elastic_parameters(E=E, nu=nu, K=K, G=G, mu=mu, lam=lam)
        if not np.isfinite(rve_length) or rve_length <= 0:
            raise ValueError("rve_length must be finite and positive")
        if "_outputs_only" in kwargs:
            raise ValueError("_outputs_only is reserved for internal use")
        mu_f = _check_viscosities(mu_f, eta_s, zeta_s)
    else:
        E, nu = _resolve_elastic_parameters(E=E, nu=nu, K=K, G=G, mu=mu, lam=lam)
        if loading_path is not None:
            if F_rate is not None or any((v is not None for v in (F_target, P_target, G_target))):
                raise ValueError("loading_path replaces F_target, P_target, G_target and F_rate")
            previous = kwargs.get("initial_state")
            stress_scale = E / (2 * (1 + nu))
            expected_initial = (
                np.r_[np.eye(3).ravel(), 0.0, np.zeros(3)]
                if previous is None
                else np.r_[
                    np.asarray(previous["F_macro"]).ravel(),
                    float(previous["P_macro"]) * stress_scale,
                    np.asarray(previous["G_macro"]) * stress_scale / rve_length,
                ]
            )
            loading_path, t_final = _validate_loading_path(loading_path, dt, t_final, expected_initial)
            F_target = loading_path[-1, :9].reshape(3, 3).copy()
            P_target = float(loading_path[-1, 9])
            G_target = loading_path[-1, 10:].copy()
        missing = [
            name
            for name, value in dict(
                dt=dt, t_final=t_final, mu_f=mu_f, P_target=P_target, G_target=G_target
            ).items()
            if value is None
        ]
        if missing:
            raise TypeError("Missing required FSI inputs: " + ", ".join(missing))
        methods = {"linear correctors", "central difference"}
        if tangent_method is not None and tangent_method not in methods:
            raise ValueError(f"tangent_method must be one of {sorted(methods)} or None")
        if "_outputs_only" in kwargs:
            raise ValueError("_outputs_only is reserved for internal use")
        if tangent_method == "central difference":
            _central_difference_steps(
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
        dt_value = float(dt)
        t_final_value = float(t_final)
        mu_f = _check_viscosities(mu_f, eta_s, zeta_s)
        rve_length_value = float(rve_length)
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(t_final_value) or t_final_value <= 0.0:
            raise ValueError("t_final must be finite and positive")
        if not np.isfinite(rve_length_value) or rve_length_value <= 0.0:
            raise ValueError("rve_length must be finite and positive")
        if F_rate is not None and F_target is not None:
            raise ValueError("specify either F_target or F_rate, not both")
        F_array = None if F_target is None else np.asarray(F_target, dtype=float)
        G_array = np.asarray(G_target, dtype=float)
        if F_array is not None and (F_array.shape != (3, 3) or not np.isfinite(F_array).all()):
            raise ValueError("F_target must be a finite (3, 3) array")
        if G_array.shape != (3,) or not np.isfinite(G_array).all():
            raise ValueError("G_target must be a finite (3,) array")
        if not np.isfinite(float(P_target)):
            raise ValueError("P_target must be finite")
    mesh_mu = float(kwargs.get("mesh_mu", 1.0))
    mesh_lame_lambda = float(kwargs.get("mesh_lame_lambda", 5.0))
    if not np.isfinite(mesh_mu) or mesh_mu <= 0.0:
        raise ValueError("mesh_mu must be finite and positive")
    if not np.isfinite(mesh_lame_lambda) or mesh_lame_lambda < 0.0:
        raise ValueError("mesh_lame_lambda must be finite and non-negative")
    maxh = float(kwargs.get("maxh", 0.1))
    if "geometry" in kwargs and not callable(kwargs["geometry"]):
        raise TypeError("geometry must be a zero-argument callable returning an OCC shape")
    if kwargs.get("exterior", "periodic") not in ("periodic", "fixed"):
        raise ValueError('exterior must be "periodic" or "fixed"')
    for flag in ("diagnostics", "Profiling", "homogenise", "compile_tangent"):
        if not isinstance(kwargs.get(flag, False), bool):
            raise TypeError(f"{flag} must be True or False")
    if maxh <= 0.0:
        raise ValueError("maxh must be positive")
    order = kwargs.get("order", 2)
    if order < 2:
        raise ValueError("order must be at least two")
    if not sequence:
        # The core sees t_final / dt with a unit step.
        t_hat = t_final_value / dt_value
        number_of_steps = int(round(t_hat))
        if number_of_steps < 1 or not np.isclose(
            number_of_steps * 1.0, t_hat, rtol=1e-10, atol=1e-12 * max(1.0, abs(t_hat))
        ):
            raise ValueError("t_final must be an integer multiple of dt")
        if F_target is None and F_rate is None:
            raise ValueError("one of F_target or F_rate must be supplied")
        if F_rate is not None and not callable(F_rate):
            rate_array = dt_value * np.asarray(F_rate, dtype=float)
            if rate_array.shape != (3, 3) or not np.all(np.isfinite(rate_array)):
                raise ValueError("F_rate must be a finite (3, 3) array")
    _check_newton_options(kwargs.get("newton_options"))
    _check_preconditioner_options(
        kwargs.get("preconditioner_options"), kwargs.get("quadrature_order"), order, mu_f
    )
    if sequence:
        return None
    return (t_final, F_target, P_target, G_target, loading_path)
