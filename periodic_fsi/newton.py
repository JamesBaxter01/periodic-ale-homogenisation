"""Newton iteration for the cell problem: the solve, admissibility checks and failure types."""

from __future__ import annotations
import copy
from time import perf_counter
import numpy as np
from ngsolve import Det, Grad, Id, IfPos, InnerProduct, Integrate, Inv, Norm

from .layout import (
    ALE_FIELD,
    FIELD_NAMES,
    LAMBDA_PF_FIELD,
    LAMBDA_U_FIELD,
    PF_FIELD,
    PS_FIELD,
    U_FIELD,
    BlockSolution,
)
from .linear import DirectNewtonSolver, PETScNewtonSolver


class NonlinearConvergenceFailure(RuntimeError):
    """A nonlinear Newton/line-search failure eligible for continuation."""


class NonlinearAdmissibilityFailure(RuntimeError):
    """A determinant/admissibility failure eligible for continuation."""


class AdmissibilityCheck:
    """Determinant checks on the reference mesh.

    A map is admissible when the measure where det(F) < minimum is at most
    ``measure_tolerance`` and both integrals are finite.
    """

    def __init__(self, mesh, check_order, measure_tolerance):
        self.mesh = mesh
        self.check_order = check_order
        self.measure_tolerance = measure_tolerance

    def jacobian(self, F, minimum, definedon=None):
        arguments = {"order": self.check_order}
        if definedon is not None:
            arguments["definedon"] = definedon
        determinant = Det(F)
        invalid_measure = Integrate(IfPos(minimum - determinant, 1.0, 0.0), self.mesh, **arguments)
        jacobian_integral = Integrate(determinant, self.mesh, **arguments)
        return (
            np.isfinite(float(invalid_measure))
            and np.isfinite(float(jacobian_integral))
            and (float(invalid_measure) <= self.measure_tolerance)
        )

    def total_mesh(self, candidate, minimum):
        """Check the total ALE map on the fixed reference mesh."""
        self.mesh.UnsetDeformation()
        return self.jacobian(Id(3) + Grad(candidate), minimum)


def solve_newton(
    gfu,
    system,
    preconditioner,
    F_n,
    E_dot,
    displacement_n,
    *,
    settings,
    discretisation,
    geometry,
    admissibility,
    dt,
    linear_cache,
):
    """Advance gfu to the converged state of one load level; return the Newton history.

    gfu, system, preconditioner and the step data (F_n, E_dot, displacement_n)
    are the cell's live objects; the groups and dt are fixed for the cell.
    linear_cache ({"backend", "pc", "key"}) belongs to the cell: its block PC
    keeps the CSR bridges between solves, while the KSP and the proxy PCs
    are rebuilt for each solve.
    """
    backend_diagnostics = {
        "ksp_creations": 0,
        "setup_seconds": 0.0,
        "reuse_scope": "KSP and proxy PCs: one Newton solve, including retries; CSR bridges: the cell",
    }

    state = gfu.vec
    if settings.options["initial_guess"] == "zero":
        discretisation.block_layout.zero(state)
    residual = gfu.vec.CreateVector()
    correction = gfu.vec.CreateVector()
    trial_residual = gfu.vec.CreateVector()
    trial = BlockSolution(FIELD_NAMES, discretisation.field_spaces, discretisation.total_state_dofs, name="monolithic_fsi_ale_trial")
    trial_state = trial.vec
    history = []
    initial_residual_norm = None
    norm_inverses = tuple(
        (
            pc.mat
            for pc in (
                preconditioner.velocity_preconditioner,
                preconditioner.pressure_preconditioner,
                preconditioner.solid_pressure_preconditioner,
                preconditioner.mesh_preconditioner,
            )
        )
    )
    norm_workspaces = {
        id(inverse): state[field].CreateVector()
        for field, inverse in zip((U_FIELD, PF_FIELD, PS_FIELD, ALE_FIELD), norm_inverses)
    }
    defect_workspace = state.CreateVector()
    increment_workspace = state.CreateVector()
    linear_image_workspace = state.CreateVector()
    linear_residual_workspace = state.CreateVector()

    def _dual_block_norm(block, inverse):
        """Apply a block Riesz approximation to a dual residual."""
        if float(Norm(block)) == 0.0:
            return 0.0
        image = norm_workspaces[id(inverse)]
        image.data = inverse * block
        value = float(InnerProduct(block, image))
        if not np.isfinite(value) or value <= 0.0:
            raise NonlinearConvergenceFailure("Fixed residual metric is not positive and finite")
        return float(np.sqrt(value))

    def scaled_block_norms(vector):
        """Return physically scaled dual norms for all six rows."""
        values = [
            _dual_block_norm(vector[U_FIELD], norm_inverses[0]),
            _dual_block_norm(vector[PF_FIELD], norm_inverses[1]),
            _dual_block_norm(vector[PS_FIELD], norm_inverses[2]),
            float(Norm(vector[LAMBDA_U_FIELD])) / np.sqrt(geometry.reference_solid_volume),
            float(Norm(vector[LAMBDA_PF_FIELD])) / np.sqrt(geometry.reference_fluid_volume),
            _dual_block_norm(vector[ALE_FIELD], norm_inverses[3]),
        ]
        values = tuple((value if np.isfinite(value) else np.inf for value in values))
        total = float(np.sqrt(sum((value * value for value in values))))
        return (total if np.isfinite(total) else np.inf, values)

    def residual_measure(state_vector, destination):
        system.ApplyResidual(state_vector, destination)
        discretisation.block_layout.mask(destination)
        return scaled_block_norms(destination)

    F_A_n = Id(3) + Grad(displacement_n)
    A_n = Inv(F_A_n)

    def admissibility_fields(state):
        """Build reusable determinant-check expressions for one state."""
        velocity = state.components[U_FIELD]
        relative_mesh_velocity = state.components[ALE_FIELD]
        F_A_s_trial = F_A_n + dt * (Grad(velocity) + E_dot * F_A_n)
        F_A_f_trial = F_A_n + dt * (Grad(velocity) + Grad(relative_mesh_velocity) + E_dot * F_A_n)
        F_solid_trial = F_A_s_trial * A_n * F_n
        return (F_solid_trial, F_A_s_trial, F_A_f_trial, F_A_f_trial * A_n)

    accepted_admissibility_fields = admissibility_fields(gfu)
    trial_admissibility_fields = admissibility_fields(trial)
    admissibility_failure = [None]

    def state_is_admissible(state):
        admissibility_failure[0] = None
        F_solid_trial, F_A_s_trial, F_A_f_trial, F_mesh_inc_trial = (
            accepted_admissibility_fields if state is gfu else trial_admissibility_fields
        )
        if settings.options["check_solid_jacobian"] and (
            not admissibility.jacobian(
                F_solid_trial, settings.options["minimum_solid_jacobian"], definedon=geometry.solid_region
            )
        ):
            admissibility_failure[0] = "solid deformation determinant"
            return False
        if settings.options["check_mesh_jacobian"] and (
            not admissibility.jacobian(
                F_A_s_trial, settings.options["minimum_mesh_jacobian"], definedon=geometry.solid_region
            )
        ):
            admissibility_failure[0] = "solid ALE determinant"
            return False
        if settings.options["check_mesh_jacobian"] and (
            not admissibility.jacobian(
                F_A_f_trial, settings.options["minimum_mesh_jacobian"], definedon=geometry.fluid_region
            )
        ):
            admissibility_failure[0] = "fluid ALE determinant"
            return False
        if settings.options["check_mesh_jacobian"] and (
            not admissibility.jacobian(
                F_mesh_inc_trial, settings.options["minimum_mesh_jacobian"], definedon=geometry.fluid_region
            )
        ):
            admissibility_failure[0] = "incremental fluid ALE determinant"
            return False
        return True

    initial_guess = {"used": settings.options["initial_guess"]}

    def probe_zero_state():
        """Residual norm of the zero state, as an alternative to a nonzero previous state.

        The previous state extrapolates the last increment: velocities are kept, or
        rescaled to the same physical velocity after a dt change. When the solution
        relaxes, e.g. under held loads after a dt increase, that guess overshoots, and
        the linearisation there (and the current-state proxies refreshed there) slows
        or stalls GMRES. The zero state is the natural alternative. Evaluated before
        the previous state's residual, which then resets the residual forms' state.
        """
        if initial_guess["used"] != "previous" or discretisation.block_layout.norm(state) == 0.0:
            return None
        discretisation.block_layout.zero(trial_state)
        if not state_is_admissible(trial):
            return None
        return residual_measure(trial_state, trial_residual)

    def use_zero_state(previous_residual_norm, zero_residual_norm):
        """Start from the zero state: its residual is at most half the previous state's."""
        initial_guess.update(
            used="zero", previous_residual=previous_residual_norm, zero_residual=zero_residual_norm
        )
        discretisation.block_layout.zero(state)
        if settings.print_iterations:
            print(
                f"  Initial guess: zero state (residual {zero_residual_norm:.3e}, previous {previous_residual_norm:.3e})",
                flush=True,
            )

    def newton_iteration(iteration):
        nonlocal initial_residual_norm
        iteration_started = settings.profiler.start()
        iteration_timings = {}
        stage_started = settings.profiler.start()
        if not state_is_admissible(gfu):
            if settings.print_iterations:
                print("  Initial guess reset:", admissibility_failure[0], flush=True)
            discretisation.block_layout.zero(gfu.vec)
            repaired = state_is_admissible(gfu)
            if not repaired:
                raise NonlinearAdmissibilityFailure(
                    "current monolithic Newton state has a non-admissible solid or mesh Jacobian"
                )
        zero_start = probe_zero_state() if iteration == 0 else None
        current_residual_norm, current_residual_blocks = residual_measure(state, residual)
        if zero_start is not None and zero_start[0] <= 0.5 * current_residual_norm:
            use_zero_state(current_residual_norm, zero_start[0])
            current_residual_norm, current_residual_blocks = residual_measure(state, residual)
        elif zero_start is not None:
            initial_guess.update(previous_residual=current_residual_norm, zero_residual=zero_start[0])
        if not np.isfinite(current_residual_norm):
            raise RuntimeError("nonlinear residual evaluation produced a non-finite scaled norm")
        if initial_residual_norm is None:
            initial_residual_norm = max(current_residual_norm, 1e-30)
        convergence_target = max(
            settings.options["tolerance"], settings.options["relative_tolerance"] * initial_residual_norm
        )
        iteration_timings["residual"] = settings.profiler.finish(
            "Newton: residual evaluation", stage_started
        )
        if current_residual_norm <= convergence_target:
            iteration_timings["total"] = settings.profiler.finish(
                "Newton: iteration total", iteration_started
            )
            history.append(
                {
                    "iteration": iteration,
                    "residual_norm": current_residual_norm,
                    "residual_before_step": current_residual_norm,
                    "residual_after_step": current_residual_norm,
                    "scaled_residual_blocks": current_residual_blocks,
                    "initial_residual": initial_residual_norm,
                    "convergence_target": convergence_target,
                    "converged": True,
                    "correction_norm": 0.0,
                    "step_length": 0.0,
                    "gmres_iterations": 0,
                    "residual_ratio": None,
                    "forcing_term": None,
                    "timings": iteration_timings,
                }
            )
            return True
        stage_started = settings.profiler.start()
        system.AssembleLinearization(state)
        iteration_timings["tangent"] = settings.profiler.finish("Newton: tangent assembly", stage_started)
        iteration_timings["adaptive preconditioner refresh"] = preconditioner.maybe_refresh_for_newton(
            current_residual_norm, iteration, history
        )
        metric_check, _ = scaled_block_norms(residual)
        metric_drift = abs(metric_check - current_residual_norm) / max(current_residual_norm, 1e-30)
        if metric_drift > 1e-10:
            raise RuntimeError("Residual metric changed during preconditioner refresh")
        correction[:] = 0.0
        gmres_iterations = [0]
        linear_attempts = []

        def solve_linear_direction(reltol):
            """Solve the Newton linearisation and verify its true residual.

            NGSolve's Krylov stopping test is based on its native vector
            norm, whereas Newton convergence uses the scaled block dual
            norm below.  If the first solve satisfies only the native
            criterion, retry from zero with progressively tighter
            absolute/relative tolerances until the requested scaled
            residual is met.
            """
            stage = settings.profiler.start()
            rhs_norm = current_residual_norm
            requested_reltol = reltol if reltol is not None else settings.gmres_tol
            linear_target = min(
                0.5 * rhs_norm, max(min(settings.gmres_tol, 0.1 * convergence_target), requested_reltol * rhs_norm)
            )
            native_fraction = 0.05 * linear_target / rhs_norm
            solver_tol = max(min(float(requested_reltol), native_fraction), 1e-14)
            solver_atol = 0.0
            gmres_iterations[0] = 0
            last_linear_norm = np.inf
            linear_attempts.clear()
            defect = defect_workspace
            increment = increment_workspace
            discretisation.block_layout.copy(defect, residual)
            correction[:] = 0.0
            maximum_retries = 4
            linear_solver = DirectNewtonSolver if settings.options["linear_solver"] == "mumps" else PETScNewtonSolver
            for retry in range(maximum_retries + 1):
                increment[:] = 0.0
                outer_solver = linear_solver(
                    mat=system.mat,
                    atol=solver_atol,
                    tol=solver_tol,
                    maxiter=settings.gmres_maxsteps,
                    system=system,
                    preconditioner=preconditioner,
                    cache=linear_cache,
                    diagnostics=backend_diagnostics,
                    free_dof_blocks=discretisation.free_dof_blocks,
                    template=gfu.vec,
                )
                try:
                    outer_solver.Solve(defect, increment)
                except NonlinearConvergenceFailure as failure:
                    linear_attempts.append({"retry": retry, "failure": str(failure)})
                    if retry < maximum_retries:
                        before_refresh = preconditioner.adaptive_refresh_count
                        preconditioner.refresh_for_linear_failure(iteration, gmres_iterations[0])
                        if preconditioner.adaptive_refresh_count > before_refresh:
                            continue
                    raise
                discretisation.block_layout.axpy(correction, 1.0, increment)
                gmres_iterations[0] += max(0, int(outer_solver.iterations) - 1)
                discretisation.block_layout.mask(correction)
                check_started = perf_counter()
                linear_image = discretisation.block_layout.apply(
                    system.mat, correction, destination=linear_image_workspace
                )
                linear_residual = linear_residual_workspace
                discretisation.block_layout.copy(linear_residual, residual)
                discretisation.block_layout.axpy(linear_residual, -1.0, linear_image)
                discretisation.block_layout.mask(linear_residual)
                last_linear_norm, block_norms = scaled_block_norms(linear_residual)
                check_seconds = perf_counter() - check_started
                linear_details = dict(getattr(outer_solver, "linear_metadata", {}))
                linear_details["physical_residual_check_seconds"] = check_seconds
                attempts = {
                    "linear_backend": linear_details,
                    "retry": retry,
                    "iterations": max(0, int(outer_solver.iterations) - 1),
                    "native_residual": (
                        float(outer_solver.residuals[-1]) if outer_solver.residuals else None
                    ),
                    "scaled_residual": float(last_linear_norm),
                    "target": float(linear_target),
                    "scaled_blocks": tuple((float(v) for v in block_norms)),
                    "solver": linear_solver.__name__,
                }
                linear_attempts.append(attempts)
                if np.isfinite(last_linear_norm) and last_linear_norm <= linear_target:
                    return settings.profiler.finish("Newton: GMRES solve", stage)
                if retry < maximum_retries:
                    refresh_elapsed = preconditioner.refresh_for_linear_failure(
                        iteration, gmres_iterations[0]
                    )
                    if refresh_elapsed:
                        iteration_timings["adaptive preconditioner refresh"] = (
                            iteration_timings.get("adaptive preconditioner refresh", 0.0)
                            + refresh_elapsed
                        )
                    discretisation.block_layout.copy(defect, linear_residual)
                    solver_tol *= 0.25
                    solver_atol *= 0.25
                    continue
                raise NonlinearConvergenceFailure(
                    f"Newton GMRES failed: scaled linear residual={last_linear_norm:.6e}, target={linear_target:.6e}, iterations={gmres_iterations[0]}, retries={maximum_retries}, solver={linear_solver.__name__}, attempts={linear_attempts!r}"
                )

        line_search_rejections = {}

        def try_line_search(direction, tau_start):
            """Apply one global Armijo/admissibility line search."""
            tau_local = tau_start
            accepted_local = False
            residual_local = np.inf
            stage = settings.profiler.start()
            residual_local_blocks = ()
            for _ in range(settings.options["max_backtracks"] + 1):
                discretisation.block_layout.copy(trial_state, state)
                discretisation.block_layout.axpy(trial_state, -tau_local, direction)
                discretisation.block_layout.mask(trial_state)
                residual_local = np.inf
                residual_local_blocks = ()
                admissible = state_is_admissible(trial)
                if admissible:
                    residual_local, residual_local_blocks = residual_measure(trial_state, trial_residual)
                if settings.options["line_search"]:
                    accepted_local = (
                        admissible
                        and np.isfinite(residual_local)
                        and (
                            residual_local
                            <= (1.0 - settings.options["armijo"] * tau_local) * current_residual_norm
                        )
                    )
                else:
                    accepted_local = admissible and np.isfinite(residual_local)
                if accepted_local:
                    discretisation.block_layout.copy(gfu.vec, trial.vec)
                    break
                rejection_reason = (
                    admissibility_failure[0]
                    if not admissible
                    else (
                        "non-finite residual"
                        if not np.isfinite(residual_local)
                        else "insufficient residual decrease"
                    )
                )
                line_search_rejections[rejection_reason] = (
                    line_search_rejections.get(rejection_reason, 0) + 1
                )
                tau_local *= settings.options["backtrack_factor"]
                if tau_local < settings.options["minimum_step"]:
                    break
            return (
                accepted_local,
                tau_local,
                residual_local,
                residual_local_blocks,
                settings.profiler.finish("Newton: line search", stage),
            )

        residual_ratio = None
        linear_reltol = 0.1
        if history:
            # Compare successive Newton iterates. The previous post-step residual
            # measures this same state and would make the ratio approximately one.
            residual_ratio = current_residual_norm / max(
                history[-1]["residual_before_step"], 1e-30
            )
            linear_reltol = min(0.1, max(0.0001, 0.9 * residual_ratio**2))
        if current_residual_norm <= 1.0:
            linear_reltol = min(linear_reltol, 0.001)
        elif current_residual_norm <= 10.0:
            linear_reltol = min(linear_reltol, 0.025)
        if current_residual_norm <= 100.0 * settings.options["tolerance"]:
            linear_reltol = min(linear_reltol, 0.001)
        elif current_residual_norm <= 10000.0 * settings.options["tolerance"]:
            linear_reltol = min(linear_reltol, 0.025)
        iteration_timings["gmres"] = solve_linear_direction(linear_reltol)
        discretisation.block_layout.mask(correction)
        correction_norm = discretisation.block_layout.norm(correction)
        if not np.isfinite(correction_norm):
            raise RuntimeError("the monolithic Newton correction is not finite")
        tau = min(1.0, (iteration + 1) * settings.options["dampfactor"])
        accepted = False
        trial_residual_norm = np.inf
        accepted, tau, trial_residual_norm, trial_residual_blocks, iteration_timings["line_search"] = (
            try_line_search(correction, tau)
        )
        history.append(
            {
                "iteration": iteration,
                "residual_norm": current_residual_norm,
                "residual_before_step": current_residual_norm,
                "residual_after_step": trial_residual_norm,
                "scaled_residual_blocks": trial_residual_blocks,
                "initial_residual": initial_residual_norm,
                "convergence_target": convergence_target,
                "converged": bool(accepted and trial_residual_norm <= convergence_target),
                "trial_residual_norm": trial_residual_norm,
                "line_search_rejections": dict(line_search_rejections),
                "correction_norm": correction_norm,
                "step_length": tau,
                "gmres_iterations": int(gmres_iterations[0]),
                "residual_ratio": residual_ratio,
                "forcing_term": float(linear_reltol),
                "linear_attempts": list(linear_attempts),
                "residual_metric_relative_drift": float(metric_drift),
                "timings": iteration_timings,
            }
        )
        if settings.print_iterations:
            print(
                f"Newton {iteration:2d}: ||R||={current_residual_norm:.6e}, ||R_trial||={trial_residual_norm:.6e}, ||dx||={correction_norm:.6e}, tau={tau:.3e}, GMRES={int(gmres_iterations[0])}"
            )
        if line_search_rejections and (settings.print_iterations or settings.profiling):
            print(
                "  line-search rejections: "
                + "; ".join((f"{reason} ({count})" for reason, count in line_search_rejections.items()))
            )
        if not accepted:
            raise NonlinearConvergenceFailure(
                "the monolithic Newton line search could not find an admissible residual-reducing step"
            )
        if trial_residual_norm <= convergence_target:
            iteration_timings["total"] = settings.profiler.finish(
                "Newton: iteration total", iteration_started
            )
            return True
        iteration_timings["total"] = settings.profiler.finish("Newton: iteration total", iteration_started)
        return False

    try:
        for iteration in range(settings.options["maximum_iterations"]):
            if newton_iteration(iteration):
                return history
        raise NonlinearConvergenceFailure(
            f"the monolithic FSI--ALE solve did not converge in {settings.options['maximum_iterations']} Newton iterations"
        )
    finally:
        backend = linear_cache["backend"]
        pc = linear_cache["pc"]
        linear_cache["backend"] = linear_cache["key"] = None
        if backend is not None:
            backend_diagnostics["krylov"] = dict(backend.metadata)
            backend.close()
        if pc is not None:
            backend_diagnostics["petsc_pc"] = copy.deepcopy(pc.metadata)
            pc.release_proxy_pcs()
        for row in history:
            row["newton_backend"] = backend_diagnostics
        if history:
            history[0]["initial_guess"] = dict(initial_guess)
