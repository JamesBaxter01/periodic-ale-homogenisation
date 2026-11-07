"""Homogenised outputs, the 13x13 tangent by linear correctors, and Hill--Mandel diagnostics.

Implements eqs. (40)--(44) of the paper on the accepted state of a CellProblem.
"""

from __future__ import annotations
from time import perf_counter
import numpy as np
from ngsolve import (
    CF,
    BlockVector,
    Det,
    Grad,
    GridFunction,
    Id,
    InnerProduct,
    Integrate,
    Inv,
    Norm,
    Parameter,
    Sym,
    Trace,
    dx,
    specialcf,
)

from .formulation import _hm_fluid_map, _hm_solid_map, solid_cauchy_stress, solid_viscous_stress
from .layout import ALE_FIELD, LAMBDA_PF_FIELD, LAMBDA_U_FIELD, PF_FIELD, PS_FIELD, U_FIELD
from .linear import PETScMUMPSSolver
from .memory import _collect_solver_memory, _detach_finished_preconditioner
from .residual_metric import ResidualMetric

INPUT_NAMES = tuple((f"F_{i + 1}{j + 1}" for i in range(3) for j in range(3))) + (
    "P0",
    "G_1",
    "G_2",
    "G_3",
)
OUTPUT_NAMES = tuple((f"P_{i + 1}{j + 1}" for i in range(3) for j in range(3))) + (
    "fluid_content",
    "flux_1",
    "flux_2",
    "flux_3",
)


class Homogenisation:
    """Homogenisation at the converged state of one CellProblem.

    Keeps the cell's fixed groups and reads its live state (gfu, system,
    preconditioner, F_n, displacement_n, mesh_displacement, load_parameters)
    through ``cell`` when called, so one instance serves a whole loading path.
    ``metric`` holds the residual metric the correctors are accepted in after
    compute_homogenised_response.
    """

    def __init__(self, cell):
        self.cell = cell
        self.material = cell.material
        self.settings = cell.settings
        self.discretisation = cell.discretisation
        self.geometry = cell.geometry
        self.dt = cell.dt
        self.keeps_newton_workspace = cell._solver_session is not None
        self.metric = None

    def homogenisation_load_from_macro(self, F_macro, F_macro_previous):
        """Convert a macro deformation input to this step's strain rate."""
        return (F_macro @ np.linalg.inv(F_macro_previous) - np.eye(3)) / self.dt

    def hill_mandel_integrands(
        self, state, velocity_gradient, relative_gradient, rate, pressure, macro_F, material=None
    ):
        """Physical endpoint outputs on the exact residual geometry.

        Explicit gradients also support symbolic directions: Grad cannot be
        applied to every composite vector CoefficientFunction in NGSolve.
        The mixed compressible solid pressure law is unchanged.
        ``material`` defaults to the float Material; the compiled output
        derivatives pass the cell's MaterialParameters.
        """
        cell = self.cell
        material = self.material if material is None else material
        old_map = Id(3) + Grad(cell.displacement_n)
        solid_map = _hm_solid_map(old_map, velocity_gradient, rate, self.dt)
        fluid_map = _hm_fluid_map(old_map, velocity_gradient, relative_gradient, rate, self.dt)
        As, Af = (Inv(solid_map), Inv(fluid_map))
        Js, Jf = (Det(solid_map), Det(fluid_map))
        H = velocity_gradient + rate * old_map
        Ls, Lf = (H * As, H * Af)
        Fs = solid_map * Inv(old_map) * cell.F_n
        # With the solid pressure given, the stress does not use lam.
        sigma_s = solid_cauchy_stress(Fs, state[PS_FIELD], mu=material.mu_s, lam=None) + solid_viscous_stress(
            Sym(Ls), eta=material.eta_s, zeta=material.zeta_s
        )
        sigma_f = 2 * material.mu_f * Sym(Lf) - (pressure + state[PF_FIELD]) * Id(3)
        macro_inverse = Inv(macro_F)
        Ps = Js * sigma_s * macro_inverse.trans
        Pf = Jf * sigma_f * macro_inverse.trans
        flux = macro_inverse * (Jf * state[U_FIELD])
        return (
            (Ps, Pf, Jf, flux),
            {
                "old_map": old_map,
                "solid_map": solid_map,
                "fluid_map": fluid_map,
                "Js": Js,
                "Jf": Jf,
                "As": As,
                "Af": Af,
                "Ls": Ls,
                "Lf": Lf,
                "sigma_s": sigma_s,
                "sigma_f": sigma_f,
                "H": H,
                "macro_inverse": macro_inverse,
            },
        )

    def integrate_hill_mandel_outputs(self, integrands, order, timings=None):
        Ps, Pf, content, flux = integrands

        def average(cf, region):
            return (
                np.asarray(Integrate(cf, self.discretisation.mesh, definedon=region, order=order), dtype=float)
                / self.geometry.reference_volume
            )

        started = self.settings.profiler.start()
        solid = average(Ps, self.geometry.solid_region).reshape(3, 3)
        fluid = average(Pf, self.geometry.fluid_region).reshape(3, 3)
        elapsed = self.settings.profiler.finish("Homogenisation: calculating stress", started)
        if timings is not None:
            timings["calculating stress"] += elapsed
        started = self.settings.profiler.start()
        Q = average(flux, self.geometry.fluid_region).reshape(3)
        elapsed = self.settings.profiler.finish("Homogenisation: calculating Q", started)
        if timings is not None:
            timings["calculating Q"] += elapsed
        started = self.settings.profiler.start()
        m = float(average(content, self.geometry.fluid_region))
        elapsed = self.settings.profiler.finish("Homogenisation: calculating m.f.", started)
        if timings is not None:
            timings["calculating m.f."] += elapsed
        return (np.r_[(solid + fluid).reshape(9), m, Q], solid, fluid)

    def homogenisation_output(self, state, E_dot_trial, P0_trial, F_macro, order, timings=None):
        """Return endpoint (P[9], m, Q[3]) for the projected HM closure.

        Outputs use the exact residual maps, without mesh interpolation.
        """
        self.discretisation.mesh.UnsetDeformation()
        try:
            rate = CF(tuple(np.asarray(E_dot_trial).reshape(9)), dims=(3, 3))
            macro_F = CF(tuple(np.asarray(F_macro).reshape(9)), dims=(3, 3))
            fields, _ = self.hill_mandel_integrands(
                state.components,
                Grad(state.components[U_FIELD]),
                Grad(state.components[ALE_FIELD]),
                rate,
                P0_trial,
                macro_F,
            )
            result = self.integrate_hill_mandel_outputs(fields, order, timings)
            return result
        finally:
            self.discretisation.mesh.UnsetDeformation()

    def output_derivative_workspace(self, base_E_dot, base_pressure, base_F_macro, *, timings=None):
        """Exact directional output derivatives, built once per cell.

        Differentiation precedes compilation. The base load, the direction
        and the material are live Parameters and the direction fields are
        updated in place for each corrector, so the cell (and an FSISession)
        keeps one workspace. With compile_tangent its two expressions are
        compiled natively in the background (code independent of loads,
        dt and material, so ccache reuses it); until then they are interpreted.
        """
        cell = self.cell
        workspace = getattr(cell, "output_derivative_workspace", None)
        if workspace is None:
            workspace = self._build_output_derivative_workspace()
            cell.output_derivative_workspace = workspace
            if timings is not None:
                timings["building output derivatives"] += workspace["metadata"]["build_seconds"]
                timings["compiling output derivatives"] += workspace["metadata"]["compile_seconds"]
        for parameter, value in zip(workspace["base_rate"], np.asarray(base_E_dot).reshape(9)):
            parameter.Set(float(value))
        for parameter, value in zip(workspace["base_macro"], np.asarray(base_F_macro).reshape(9)):
            parameter.Set(float(value))
        workspace["base_pressure"].Set(float(base_pressure))
        workspace["metadata"]["workspace_uses"] += 1
        return workspace

    def _build_output_derivative_workspace(self):
        cell = self.cell
        started = self.settings.profiler.start()
        parameter = Parameter(0.0)
        base_rate = tuple((Parameter(0.0) for _ in range(9)))
        base_macro = tuple((Parameter(0.0) for _ in range(9)))
        base_pressure = Parameter(0.0)
        rate_direction = tuple((Parameter(0.0) for _ in range(9)))
        macro_direction = tuple((Parameter(0.0) for _ in range(9)))
        pressure_direction = Parameter(0.0)
        rate = CF(base_rate, dims=(3, 3)) + parameter * CF(rate_direction, dims=(3, 3))
        macro_F = CF(base_macro, dims=(3, 3)) + parameter * CF(macro_direction, dims=(3, 3))
        pressure = base_pressure + parameter * pressure_direction
        components = tuple(
            (
                GridFunction(space, name=f"homogenisation_direction_{index}")
                for index, space in enumerate(self.discretisation.field_spaces)
            )
        )
        state = tuple(
            (field + parameter * direction for field, direction in zip(cell.gfu.components, components))
        )
        gv = Grad(cell.gfu.components[U_FIELD]) + parameter * Grad(components[U_FIELD])
        gd = Grad(cell.gfu.components[ALE_FIELD]) + parameter * Grad(components[ALE_FIELD])
        fields, _ = self.hill_mandel_integrands(
            state, gv, gd, rate, pressure, macro_F, material=cell.material_parameters
        )
        Ps, Pf, content, flux = tuple((field.Diff(parameter) for field in fields))
        solid = CF(tuple((Ps[i, j] for i in range(3) for j in range(3))))
        fluid = CF(
            tuple((Pf[i, j] for i in range(3) for j in range(3)))
            + (content,)
            + tuple((flux[i] for i in range(3)))
        )
        build_seconds = self.settings.profiler.finish(
            "Homogenisation: building output derivative expressions", started
        )
        started = self.settings.profiler.start()
        realcompile = bool(self.settings.compile_tangent)
        if realcompile:
            compiled_solid = solid.Compile(realcompile=True, wait=False)
            compiled_fluid = fluid.Compile(realcompile=True, wait=False)
        else:
            compiled_solid = solid.Compile()
            compiled_fluid = fluid.Compile()
        compile_seconds = self.settings.profiler.finish(
            "Homogenisation: compiling output derivative expressions", started
        )
        return {
            "components": components,
            "parameter": parameter,
            "base_rate": base_rate,
            "base_macro": base_macro,
            "base_pressure": base_pressure,
            "rate_direction": rate_direction,
            "macro_direction": macro_direction,
            "pressure_direction": pressure_direction,
            "solid": compiled_solid,
            "fluid": compiled_fluid,
            "metadata": {
                "mode": "ngsolve_compiled_output_derivatives",
                "realcompile": realcompile,
                "workspace_builds": 1,
                "workspace_uses": 0,
                "expression_compilations": 2,
                "evaluations": 0,
                "integration_calls_per_column": 2,
                "reuse_scope": "cell",
                "build_seconds": build_seconds,
                "compile_seconds": compile_seconds,
                "retained_corrector_solutions": 1,
            },
        }

    def homogenisation_output_directional_derivative(self, 
        corrector, E_dot_direction, pressure_direction, F_direction, order, *, workspace, timings=None
    ):
        """Evaluate the compiled exact derivative with a new live direction."""
        for field, block in zip(workspace["components"], corrector):
            field.vec.data = block
        for parameter, value in zip(workspace["rate_direction"], np.asarray(E_dot_direction).reshape(9)):
            parameter.Set(float(value))
        for parameter, value in zip(workspace["macro_direction"], np.asarray(F_direction).reshape(9)):
            parameter.Set(float(value))
        workspace["pressure_direction"].Set(float(pressure_direction))
        self.discretisation.mesh.UnsetDeformation()
        try:
            started = self.settings.profiler.start()
            solid = (
                np.asarray(
                    Integrate(workspace["solid"], self.discretisation.mesh, definedon=self.geometry.solid_region, order=order), dtype=float
                ).reshape(9)
                / self.geometry.reference_volume
            )
            solid_seconds = self.settings.profiler.finish(
                "Homogenisation: integrating compiled solid output derivatives", started
            )
            started = self.settings.profiler.start()
            fluid = (
                np.asarray(
                    Integrate(workspace["fluid"], self.discretisation.mesh, definedon=self.geometry.fluid_region, order=order), dtype=float
                ).reshape(13)
                / self.geometry.reference_volume
            )
            fluid_seconds = self.settings.profiler.finish(
                "Homogenisation: integrating compiled fluid output derivatives", started
            )
            if timings is not None:
                timings["integrating solid output derivatives"] += solid_seconds
                timings["integrating fluid output derivatives"] += fluid_seconds
            fluid[:9] += solid
            workspace["metadata"]["evaluations"] += 1
            return fluid
        finally:
            self.discretisation.mesh.UnsetDeformation()

    def hill_mandel_diagnostics(self, F_current, F_previous, rate_array, pressure, gradient, order):
        """Audit the final micro step, with reference quadrature throughout.

        Rates are instantaneous endpoint kinematic rates unless suffixed _be.
        Q_kinematic removes the endpoint affine velocity; Q0 removes the actual
        old-map BE lift. Keeping both avoids mislabelling lift error as storage
        redistribution. Boundary transport is independently integrated over
        fluid element boundaries; internal faces cancel for compatible maps.
        """
        cell = self.cell
        self.discretisation.mesh.UnsetDeformation()
        matrix = lambda value: CF(tuple(np.asarray(value).reshape(9)), dims=(3, 3))
        rate, macro_F = (matrix(rate_array), matrix(F_current))
        B_array = (F_current - F_previous) / self.dt
        B = matrix(B_array)
        state = cell.gfu.components
        gv, gd = (Grad(state[U_FIELD]), Grad(state[ALE_FIELD]))
        outputs, geom = self.hill_mandel_integrands(state, gv, gd, rate, pressure, macro_F)

        def av(cf, region=self.geometry.fluid_region):
            value = (
                np.asarray(Integrate(cf, self.discretisation.mesh, definedon=region, order=order), dtype=float)
                / self.geometry.reference_volume
            )
            return float(value) if value.ndim == 0 else value

        Js, Jf, Af = (geom["Js"], geom["Jf"], geom["Af"])
        old_x = self.geometry.pos + CF(tuple(cell.displacement_n.components)) - self.geometry.xbar
        velocity = state[U_FIELD] + rate * old_x
        w = velocity + state[ALE_FIELD]
        x_end = old_x + self.dt * w
        k = geom["macro_inverse"] * x_end
        kdot = geom["macro_inverse"] * (w - B * k)
        Jdot = Jf * Trace((geom["H"] + gd) * Af)
        c = Jf * Trace(geom["Lf"])
        K = av(Jf * k)
        Kdot = av(Jdot * k + Jf * kdot)
        m, m_old = (av(Jf), av(Det(geom["old_map"])))
        d, mdot_be = (av(c), (m - m_old) / self.dt)
        lam = av(Jf * state[LAMBDA_PF_FIELD]) / m
        Q0 = av(outputs[3])
        Qkin = av(geom["macro_inverse"] * (Jf * (velocity - B * k)))
        j = Jf * Af * -state[ALE_FIELD]
        P_endpoint = (av(outputs[0], self.geometry.solid_region) + av(outputs[1])).reshape(3, 3)
        old_macro_inverse_T = matrix(np.linalg.inv(F_previous).T)
        P_incremental = (
            av(
                Js * geom["sigma_s"] * geom["As"].trans * geom["old_map"].trans * old_macro_inverse_T,
                self.geometry.solid_region,
            )
            + av(Jf * geom["sigma_f"] * Af.trans * geom["old_map"].trans * old_macro_inverse_T)
        ).reshape(3, 3)
        W = av(Js * InnerProduct(geom["sigma_s"], Sym(geom["Ls"])), self.geometry.solid_region) + av(
            2 * Jf * self.material.mu_f * InnerProduct(Sym(geom["Lf"]), Sym(geom["Lf"]))
        )
        G_reference = F_current.T @ np.asarray(gradient)
        power_residual = W - (np.sum(P_endpoint * B_array) + pressure * mdot_be - G_reference @ Q0)
        storage_defect = mdot_be - lam * m
        # delta_W of Eq. (42): the measured divergence d replaces the storage rate. The
        # mean-continuity residual d - lam*m is reported separately, not absorbed here.
        expected_defect = np.sum((P_incremental - P_endpoint) * B_array) + pressure * (d - mdot_be)
        frame_power = av(Js * InnerProduct(state[LAMBDA_U_FIELD], state[U_FIELD]), self.geometry.solid_region)
        momentum_residual = (
            av(Js * InnerProduct(geom["sigma_s"] * geom["As"].trans, gv), self.geometry.solid_region)
            + av(Jf * InnerProduct(geom["sigma_f"] * Af.trans, gv))
            + np.asarray(gradient) @ av(Jf * state[U_FIELD])
            + frame_power
        )
        # R_v(v~) - R_f(p_f) - R_lambda_v(lambda_v) + R_lambda_p(lambda_p), the right side of Eq. (42).
        constraint_power = momentum_residual - frame_power + av(state[PF_FIELD] * c)
        stored_F = Id(3) + Grad(cell.mesh_displacement)
        transfer_norm = np.sqrt(
            max(
                0.0,
                av(InnerProduct(stored_F - geom["solid_map"], stored_F - geom["solid_map"]), self.geometry.solid_region)
                + av(InnerProduct(stored_F - geom["fluid_map"], stored_F - geom["fluid_map"])),
            )
        )
        cell.system._set_state(cell.gfu.vec)
        frame_form = cell.system.residual_forms[LAMBDA_U_FIELD]
        frame_form.Assemble()
        assembled_frame = frame_form.vec.FV().NumPy().copy() / self.geometry.reference_volume
        direct_frame = av(Js * state[U_FIELD], self.geometry.solid_region)
        data = {
            # v2: power_identity_residual and constraint_power_residual follow Eq. (42) (delta_W uses d).
            "schema": "projected_hill_mandel_endpoint_v2",
            "scope": "final_micro_step_frozen_previous_state",
            "volume_quadrature_order": int(order),
            "Q0": Q0,
            "Q_ALE": av(j),
            "Q_kinematic": Qkin,
            "Q_lift_difference": Q0 - Qkin,
            "K": K,
            "K_rate": Kdot,
            "K_rate_be": (K - av(Det(geom["old_map"]) * (matrix(np.linalg.inv(F_previous)) * old_x)))
            / self.dt,
            "fluid_centroid": K / m,
            "moment_correction": Kdot - lam * K,
            "moment_projection_residual": av(k * c) - lam * K,
            "m": m,
            "m_old": m_old,
            "m_rate_be": mdot_be,
            "lambda_p": lam,
            "fluid_divergence_integral": d,
            "continuity_mean_residual": d - lam * m,
            "storage_be_defect": storage_defect,
            "geometric_chain_defect": mdot_be - av(Jdot),
            "solid_volume": av(Js, self.geometry.solid_region),
            "cell_volume_residual": av(Js, self.geometry.solid_region) + m - np.linalg.det(F_current),
            "solid_frame_mean": direct_frame,
            "solid_frame_assembled": assembled_frame,
            "solid_frame_quadrature_difference": direct_frame - assembled_frame,
            "pressure_mean": av(Jf * state[PF_FIELD]),
            "pressure_fluctuation_power": av(state[PF_FIELD] * c),
            "P_endpoint": P_endpoint,
            "P_incremental": P_incremental,
            "physical_power": W,
            "power_residual": power_residual,
            "predicted_discrete_power_defect": expected_defect,
            "power_identity_residual": power_residual - expected_defect,
            "momentum_fluctuation_residual": momentum_residual,
            "solid_frame_power": frame_power,
            "constraint_power_residual": constraint_power,
            "power_accounting_residual": power_residual - expected_defect - constraint_power,
            "mesh_transfer_gradient_rms": float(transfer_norm),
            "mesh_transfer_fluid_content": av(Det(stored_F)) - m,
        }
        try:
            normal = specialcf.normal(self.discretisation.mesh.dim)
            boundary_measure = dx(element_boundary=True, definedon=self.geometry.fluid_region, bonus_intorder=0)
            boundary_moment = np.array(
                [
                    float(Integrate(k[i] * InnerProduct(j, normal) * boundary_measure, self.discretisation.mesh))
                    / self.geometry.reference_volume
                    for i in range(3)
                ]
            )
            net = float(Integrate(InnerProduct(j, normal) * boundary_measure, self.discretisation.mesh)) / self.geometry.reference_volume
            data.update(
                Q_boundary=boundary_moment,
                relative_net_outflow=net,
                moment_identity_residual=Qkin - boundary_moment - Kdot + av(k * c),
                boundary_transport_status="fluid_element_boundary_integral",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            data.update(
                Q_boundary=None,
                relative_net_outflow=None,
                moment_identity_residual=None,
                boundary_transport_status=f"unavailable: {type(exc).__name__}: {exc}",
            )
        return data

    def homogenisation_residual_derivative(self, 
        load_derivative_vectors, load_direction, *, destination, timings=None
    ):
        """Combine exact load-derivative vectors into one directional RHS.

        ``load_derivative_vectors[k]`` is the assembled derivative of the
        complete six-row residual with respect to the kth independent
        ``(E_dot, P0, G)`` parameter.  The supplied physical input direction
        is a 13-vector in that same basis, so this operation is only a cheap
        block-vector linear combination.
        """
        stage_started = self.settings.profiler.start()
        self.discretisation.block_layout.zero(destination)
        for coefficient, basis_vector in zip(
            np.asarray(load_direction, dtype=float).reshape(-1), load_derivative_vectors
        ):
            if coefficient != 0.0:
                self.discretisation.block_layout.axpy(destination, float(coefficient), basis_vector)
        self.discretisation.block_layout.mask(destination)
        if timings is not None:
            timings["linearising"] += self.settings.profiler.finish(
                "Homogenisation: combining exact forcing derivative", stage_started
            )
        return destination

    def _homogenisation_dual_block_norm(self, block, inverse):
        if float(Norm(block)) == 0.0:
            return 0.0
        image = block.CreateVector()
        image.data = inverse * block
        value = float(InnerProduct(block, image))
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"Corrector residual metric is not positive and finite: quadratic_value={value!r}, block_norm={float(Norm(block))!r}"
            )
        return float(np.sqrt(value))

    def homogenisation_scaled_block_norms(self, vector, metric):
        values = [
            self._homogenisation_dual_block_norm(vector[U_FIELD], metric.velocity),
            self._homogenisation_dual_block_norm(vector[PF_FIELD], metric.pressure),
            self._homogenisation_dual_block_norm(vector[PS_FIELD], metric.solid_pressure),
            float(Norm(vector[LAMBDA_U_FIELD])) / np.sqrt(self.geometry.reference_solid_volume),
            float(Norm(vector[LAMBDA_PF_FIELD])) / np.sqrt(self.geometry.reference_fluid_volume),
            self._homogenisation_dual_block_norm(vector[ALE_FIELD], metric.mesh),
        ]
        values = tuple((value if np.isfinite(value) else np.inf for value in values))
        total = float(np.sqrt(sum((value * value for value in values))))
        return (total if np.isfinite(total) else np.inf, values)

    def prepare_corrector_rhs(self, residual_derivative, workspace, metric):
        """Prepare the unchanged masked RHS, norms and acceptance targets."""
        preparation_started = perf_counter()
        details = workspace["corrector_timings"]
        rhs = workspace["rhs"]
        self.discretisation.block_layout.copy(rhs, residual_derivative)
        for block in range(self.discretisation.number_of_fields):
            rhs[block].data *= -1.0
        self.discretisation.block_layout.mask(rhs)
        rhs_euclidean_blocks = tuple((float(Norm(rhs[row])) for row in range(self.discretisation.number_of_fields)))
        euclidean_block_targets = tuple(
            (self.settings.pc_options["homogenisation_gmres_tol"] * max(value, 1.0) for value in rhs_euclidean_blocks)
        )
        details["rhs_preparation"] += perf_counter() - preparation_started
        metric_started = perf_counter()
        rhs_norm_value, _ = self.homogenisation_scaled_block_norms(rhs, metric)
        details["rhs_metric_checks"] += perf_counter() - metric_started
        if not np.isfinite(rhs_norm_value):
            raise RuntimeError(
                "homogenisation corrector cannot start: non-finite scaled right-hand-side norm"
            )
        corrector_target = self.settings.pc_options["homogenisation_gmres_tol"] * max(rhs_norm_value, 1.0)
        solver_tol = max(float(self.settings.pc_options["homogenisation_gmres_tol"]), 1e-14)
        return {
            "rhs_euclidean_blocks": rhs_euclidean_blocks,
            "euclidean_block_targets": euclidean_block_targets,
            "rhs_norm_value": rhs_norm_value,
            "corrector_target": corrector_target,
            "solver_tol": solver_tol,
        }

    def homogenisation_linear_corrector(self, 
        residual_derivative,
        *,
        workspace,
        metric,
        timings=None,
    ):
        """Solve with one reused LU factor and the preserved defect acceptance loop."""
        cell = self.cell
        stage_started = self.settings.profiler.start()
        details = workspace["corrector_timings"]
        prepared_rhs = self.prepare_corrector_rhs(residual_derivative, workspace, metric)
        rhs = workspace["rhs"]
        corrector = workspace["corrector"]
        rhs_euclidean_blocks = prepared_rhs["rhs_euclidean_blocks"]
        euclidean_block_targets = prepared_rhs["euclidean_block_targets"]
        corrector_target = prepared_rhs["corrector_target"]
        solver_tol = prepared_rhs["solver_tol"]
        residual_norm_value = np.inf
        residual_block_values = (np.inf,) * self.discretisation.number_of_fields
        total_iterations = 0
        maximum_retries = 4
        compact_solver = workspace["compact_solver"]
        solver_atol = 0.1 * min(euclidean_block_targets)
        defect_rhs = workspace["defect_rhs"]
        correction_increment = workspace["correction_increment"]
        corrector[:] = 0.0
        self.discretisation.block_layout.copy(defect_rhs, rhs)
        for retry in range(maximum_retries + 1):
            correction_increment[:] = 0.0
            solve_info = compact_solver.solve(
                defect_rhs,
                correction_increment,
                tol=solver_tol,
                atol=solver_atol,
                force_gmres=retry > 0,
            )
            total_iterations += int(solve_info["iterations"])
            physical_check_started = perf_counter()
            self.discretisation.block_layout.mask(correction_increment)
            self.discretisation.block_layout.axpy(corrector, 1.0, correction_increment)
            self.discretisation.block_layout.mask(corrector)
            linear_image = self.discretisation.block_layout.apply(cell.system.mat, corrector, workspace["linear_image"])
            linear_residual = workspace["linear_residual"]
            self.discretisation.block_layout.copy(linear_residual, rhs)
            self.discretisation.block_layout.axpy(linear_residual, -1.0, linear_image)
            self.discretisation.block_layout.mask(linear_residual)
            residual_norm_value, residual_block_values = self.homogenisation_scaled_block_norms(
                linear_residual, metric
            )
            residual_euclidean_blocks = tuple(
                (float(Norm(linear_residual[row])) for row in range(self.discretisation.number_of_fields))
            )
            common_residual_accepted = all(
                (
                    np.isfinite(value) and value <= target
                    for value, target in zip(residual_euclidean_blocks, euclidean_block_targets)
                )
            )
            details["physical_residual_checks"] += perf_counter() - physical_check_started
            if (
                np.isfinite(residual_norm_value)
                and residual_norm_value <= corrector_target
                and common_residual_accepted
            ):
                break
            if retry < maximum_retries:
                self.discretisation.block_layout.copy(defect_rhs, linear_residual)
                solver_tol *= 0.25
                solver_atol *= 0.25
        else:
            raise RuntimeError(
                f"homogenisation corrector solve failed: linear residual={residual_norm_value:.6e}, target={corrector_target:.6e}, iterations={total_iterations}, retries={maximum_retries}, solver=petsc_mumps, Euclidean blocks={residual_euclidean_blocks!r}, Euclidean targets={euclidean_block_targets!r}, scaled blocks="
                + repr(tuple((float(value) for value in residual_block_values)))
            )
        euclidean_rhs_norm = float(np.linalg.norm(rhs_euclidean_blocks))
        euclidean_residual_norm = float(np.linalg.norm(residual_euclidean_blocks))
        workspace.setdefault("residual_diagnostics", []).append(
            {
                "rhs_euclidean_blocks": rhs_euclidean_blocks,
                "residual_euclidean_blocks": residual_euclidean_blocks,
                "euclidean_block_targets": euclidean_block_targets,
                "relative_euclidean_residual": (
                    euclidean_residual_norm / max(euclidean_rhs_norm, 1e-300)
                    if euclidean_rhs_norm > 0.0
                    else euclidean_residual_norm
                ),
                "scaled_residual": float(residual_norm_value),
                "scaled_target": float(corrector_target),
                "retries": int(retry),
                "common_residual_accepted": bool(common_residual_accepted),
            }
        )
        if timings is not None:
            timings["solving correctors"] += self.settings.profiler.finish(
                "Homogenisation: solving corrector", stage_started
            )
        return (corrector, total_iterations)

    def compute_homogenised_response(self, prescribed_F_path, P_target, G_target_array, _outputs_only):
        """Return 13 homogenised outputs and their 13-by-13 Jacobian.

        Inputs are the nine entries of the current macroscopic deformation
        gradient, the scalar pressure P0, and the three entries of G.  Each
        column uses the frozen converged monolithic tangent and one exact
        directional corrector.  Both the residual forcing and the output
        functional are differentiated symbolically at the deformed base
        state; no central-difference trial states are constructed.
        """
        cell = self.cell
        self.discretisation.mesh.UnsetDeformation()
        homogenisation_started = self.settings.profiler.start()
        order = int(self.settings.pc_options["homogenisation_order"])
        homogenisation_timings = {
            "linearising": 0.0,
            "assembling residual sensitivities": 0.0,
            "linear solver setup": 0.0,
            "solving correctors": 0.0,
            "calculating stress": 0.0,
            "calculating m.f.": 0.0,
            "calculating Q": 0.0,
            "assembling Jacobian": 0.0,
            "building output derivatives": 0.0,
            "compiling output derivatives": 0.0,
            "integrating solid output derivatives": 0.0,
            "integrating fluid output derivatives": 0.0,
        }
        F_macro_current = prescribed_F_path[-1].copy()
        F_macro_previous = prescribed_F_path[-2].copy()
        homogenisation_base_E_dot = self.homogenisation_load_from_macro(F_macro_current, F_macro_previous)
        homogenisation_base_pressure = float(P_target)
        homogenisation_base_gradient = G_target_array.copy()
        cell.load_parameters.set(
            homogenisation_base_E_dot, homogenisation_base_pressure, homogenisation_base_gradient
        )
        if _outputs_only:
            base_output, solid, fluid = self.homogenisation_output(
                cell.gfu, homogenisation_base_E_dot, homogenisation_base_pressure, F_macro_current, order
            )
            audit = (
                self.hill_mandel_diagnostics(
                    F_macro_current,
                    F_macro_previous,
                    homogenisation_base_E_dot,
                    homogenisation_base_pressure,
                    homogenisation_base_gradient,
                    order,
                )
                if self.settings.diagnostics
                else None
            )
            cell.set_mesh_state(cell.mesh_displacement)
            return {
                "inputs": np.r_[
                    F_macro_current.reshape(9), homogenisation_base_pressure, homogenisation_base_gradient
                ],
                "outputs": base_output,
                "jacobian": None,
                "solid_nominal_stress": solid,
                "fluid_nominal_stress": fluid,
                **({"hill_mandel_diagnostics": audit} if audit is not None else {}),
                "closure": "projected_hill_mandel_endpoint_v1",
                "tangent_scope": "final_micro_step_frozen_previous_state",
                "input_names": INPUT_NAMES,
                "output_names": OUTPUT_NAMES,
                "corrector_gmres_iterations": (),
                "sensitivity_method": "outputs_only",
            }
        stage_started = self.settings.profiler.start()
        cell.system.AssembleLinearization(cell.gfu.vec)
        # Correctors are accepted in the Newton residual metric (reference configuration).
        self.metric = ResidualMetric.of(cell.preconditioner)
        if not self.keeps_newton_workspace:
            _detach_finished_preconditioner(cell.preconditioner)
            cell.preconditioner = None
            _collect_solver_memory(True)
        homogenisation_timings["linearising"] += self.settings.profiler.finish(
            "Homogenisation: linearising monolithic tangent", stage_started
        )
        stage_started = self.settings.profiler.start()
        compact_solver = None
        try:
            compact_solver = PETScMUMPSSolver(
                system=cell.system,
                free_dof_blocks=self.discretisation.free_dof_blocks,
                template=cell.gfu.vec,
                blas_threads=self.settings.pc_options["corrector_blas_threads"],
            )
            homogenisation_timings["linear solver setup"] += self.settings.profiler.finish(
                "Homogenisation: compact linear solver setup", stage_started
            )
            stage_started = self.settings.profiler.start()
            cell.system.AssembleLoadDerivatives()
            zero_derivative = {}
            for row, component in enumerate(cell.gfu.components):
                if any((forms[row] is None for forms in cell.system.load_derivative_forms)):
                    zero_derivative[row] = component.vec.CreateVector()
                    zero_derivative[row][:] = 0.0
            load_derivative_vectors = [
                BlockVector(
                    [
                        form.vec if form is not None else zero_derivative[row]
                        for row, form in enumerate(rows)
                    ]
                )
                for rows in cell.system.load_derivative_forms
            ]
            homogenisation_timings["assembling residual sensitivities"] += self.settings.profiler.finish(
                "Homogenisation: collecting exact residual sensitivities", stage_started
            )
            base_output, base_solid_stress, base_fluid_stress = self.homogenisation_output(
                cell.gfu,
                homogenisation_base_E_dot,
                homogenisation_base_pressure,
                F_macro_current,
                order,
                timings=homogenisation_timings,
            )
            jacobian = np.empty((13, 13), dtype=float)
            corrector_iterations = []
            residual_derivative = cell.gfu.vec.CreateVector()
            corrector_workspace = {
                "corrector_timings": dict(
                    rhs_preparation=0.0,
                    rhs_metric_checks=0.0,
                    physical_residual_checks=0.0,
                ),
                "compact_solver": compact_solver,
                "rhs": cell.gfu.vec.CreateVector(),
                "corrector": cell.gfu.vec.CreateVector(),
                "defect_rhs": cell.gfu.vec.CreateVector(),
                "correction_increment": cell.gfu.vec.CreateVector(),
                "linear_residual": cell.gfu.vec.CreateVector(),
                "linear_image": cell.gfu.vec.CreateVector(),
            }
            output_sensitivity_workspace = self.output_derivative_workspace(
                homogenisation_base_E_dot,
                homogenisation_base_pressure,
                F_macro_current,
                timings=homogenisation_timings,
            )
            F_macro_previous_inverse = np.linalg.inv(F_macro_previous)
            load_directions = []
            output_directions = []
            for input_index in range(13):
                E_dot_direction = np.zeros((3, 3), dtype=float)
                F_direction = np.zeros((3, 3), dtype=float)
                pressure_direction = 0.0
                gradient_direction = np.zeros(3, dtype=float)
                if input_index < 9:
                    F_direction = np.zeros((3, 3), dtype=float)
                    F_direction.flat[input_index] = 1.0
                    E_dot_direction = F_direction @ F_macro_previous_inverse / self.dt
                elif input_index == 9:
                    pressure_direction = 1.0
                else:
                    gradient_direction[input_index - 10] = 1.0
                load_directions.append(
                    np.concatenate(
                        (E_dot_direction.reshape(9), np.asarray([pressure_direction]), gradient_direction)
                    )
                )
                output_directions.append((E_dot_direction, pressure_direction, F_direction.copy()))
            for input_index in range(len(load_directions)):
                residual_derivative = self.homogenisation_residual_derivative(
                    load_derivative_vectors,
                    load_directions[input_index],
                    destination=residual_derivative,
                    timings=homogenisation_timings,
                )
                corrector, iterations = self.homogenisation_linear_corrector(
                    residual_derivative,
                    timings=homogenisation_timings,
                    workspace=corrector_workspace,
                    metric=self.metric,
                )
                corrector_iterations.append(iterations)
                stage_started = self.settings.profiler.start()
                jacobian[:, input_index] = self.homogenisation_output_directional_derivative(
                    corrector,
                    *output_directions[input_index],
                    order,
                    timings=homogenisation_timings,
                    workspace=output_sensitivity_workspace,
                )
                homogenisation_timings["assembling Jacobian"] += self.settings.profiler.finish(
                    "Homogenisation: assembling exact Jacobian column", stage_started
                )
            cell.load_parameters.set(
                homogenisation_base_E_dot, homogenisation_base_pressure, homogenisation_base_gradient
            )
            elapsed = self.settings.profiler.finish(
                "Homogenisation: 13 linear correctors", homogenisation_started
            )
            homogenisation_timings["total"] = float(elapsed)
            linear_metadata = compact_solver.metadata
            corrector_timings = dict(corrector_workspace["corrector_timings"])
            corrector_timings["factor_applications"] = linear_metadata["factor_apply_seconds"]
            corrector_timings["backend_native_residual_checks"] = linear_metadata[
                "native_residual_seconds"
            ]
            corrector_timings["packing"] = linear_metadata["packing_seconds"]
            corrector_timings["blas_context"] = linear_metadata["blas_context_seconds"]
            corrector_timings["remaining_overhead"] = (
                max(0.0, homogenisation_timings["solving correctors"] - sum(corrector_timings.values()))
                if self.settings.profiling
                else 0.0
            )
            solver_setup_timings = {
                "matrix_conversion": linear_metadata["conversion_seconds"],
                "factorization": linear_metadata["factor_setup_seconds"],
            }
            solver_setup_timings["remaining_overhead"] = (
                max(
                    0.0,
                    homogenisation_timings["linear solver setup"] - sum(solver_setup_timings.values()),
                )
                if self.settings.profiling
                else 0.0
            )
            if self.settings.profiling:
                self.settings.profiler.summary(
                    f"homogenisation: 13 exact linear correctors in {elapsed:.3f} s; linear work count {sum(corrector_iterations)}; solver petsc_mumps"
                )
                self.settings.profiler.summary("homogenisation stage timings:")
                for label in (
                    "linearising",
                    "assembling residual sensitivities",
                    "linear solver setup",
                    "solving correctors",
                    "calculating stress",
                    "calculating m.f.",
                    "calculating Q",
                    "assembling Jacobian",
                    "building output derivatives",
                    "compiling output derivatives",
                    "integrating solid output derivatives",
                    "integrating fluid output derivatives",
                ):
                    self.settings.profiler.summary(f"  {label}: {homogenisation_timings[label]:.3f} s")
            if self.settings.profiling:
                self.settings.profiler.summary("linear solver setup breakdown (included above):")
                for label, value in solver_setup_timings.items():
                    self.settings.profiler.summary(f"  {label}: {value:.3f} s")
                self.settings.profiler.summary("corrector solve breakdown (included above):")
                for label, value in corrector_timings.items():
                    self.settings.profiler.summary(f"  {label}: {value:.3f} s")
                self.settings.profiler.summary(
                    f"corrector strategy: BLAS threads {self.settings.pc_options['corrector_blas_threads']}; one reused MUMPS factorization"
                )
            audit = (
                self.hill_mandel_diagnostics(
                    F_macro_current,
                    F_macro_previous,
                    homogenisation_base_E_dot,
                    homogenisation_base_pressure,
                    homogenisation_base_gradient,
                    order,
                )
                if self.settings.diagnostics
                else None
            )
            cell.set_mesh_state(cell.mesh_displacement)
            return {
                **({"hill_mandel_diagnostics": audit} if audit is not None else {}),
                "closure": "projected_hill_mandel_endpoint_v1",
                "tangent_scope": "final_micro_step_frozen_previous_state",
                "inputs": np.concatenate(
                    (
                        F_macro_current.reshape(9),
                        np.asarray([homogenisation_base_pressure]),
                        homogenisation_base_gradient,
                    )
                ),
                "outputs": base_output,
                "solid_nominal_stress": base_solid_stress,
                "fluid_nominal_stress": base_fluid_stress,
                "jacobian": jacobian,
                "input_names": INPUT_NAMES,
                "output_names": OUTPUT_NAMES,
                "P_total": base_output[:9].reshape((3, 3)),
                "fluid_content": float(base_output[9]),
                "flux": base_output[10:].copy(),
                "corrector_gmres_iterations": tuple(corrector_iterations),
                "corrector_residual_diagnostics": tuple(
                    corrector_workspace.get("residual_diagnostics", ())
                ),
                "homogenisation_pc_design": "mumps",
                "petsc_preconditioner_metadata": dict(compact_solver.metadata),
                "residual_metric_design": "newton_reference_configuration_bddc",
                "homogenisation_pc_spec": {
                    "implementation": "mumps",
                    "acceptance_metric": "newton_reference_configuration_bddc",
                },
                "corrector_acceptance": "candidate_scaled_residual_and_common_per_block_euclidean",
                "corrector_solver": "petsc_mumps",
                "corrector_timings": corrector_timings,
                "solver_setup_timings": solver_setup_timings,
                "corrector_strategy": {
                    "blas_threads": self.settings.pc_options["corrector_blas_threads"],
                    "factorization_threads": (
                        "process_settings"
                        if self.settings.pc_options["corrector_blas_threads"] in (None, 1)
                        else self.settings.pc_options["corrector_blas_threads"]
                    ),
                    "timing_breakdowns_are_subsets": bool(self.settings.profiling),
                },
                "efficiency_linear_metadata": dict(compact_solver.metadata),
                "coupling_factorization": "mumps",
                "sensitivity_method": "exact_symbolic_directional",
                "output_derivative_metadata": dict(output_sensitivity_workspace["metadata"]),
                "timings": {label: float(value) for label, value in homogenisation_timings.items()},
                "seconds": float(elapsed),
            }
        finally:
            if compact_solver is not None:
                compact_solver.close()
