"""The nondimensional cell problem: input checks, the CellProblem workspace and time stepping.

_scaled_core_FSI validates and merges options, handles FSISession reuse and
builds a CellProblem. The forms are built by formulation.build_system, each
step is solved by newton.solve_newton, and homogenisation.Homogenisation
computes the outputs and tangent.
"""

from __future__ import annotations
import copy
from dataclasses import replace
from time import perf_counter
import numpy as np
from ngsolve import (
    BND,
    CF,
    ET,
    Grad,
    GridFunction,
    H1,
    Id,
    InnerProduct,
    Integrate,
    IntegrationRule,
    Inv,
    L2,
    MatrixValued,
    Mesh,
    NumberSpace,
    Periodic,
    Sym,
    VOL,
    VectorH1,
    VectorValued,
    x,
    y,
    z,
)
from ngsolve.fem import specialcf
from netgen.meshing import MeshingParameters
from netgen.occ import OCCGeometry

from .cell_data import CellGeometry, Discretisation, Material, SolverSettings
from .formulation import LoadParameters, MaterialParameters, build_system
from .homogenisation import Homogenisation
from .inputs import newton_defaults, preconditioner_defaults
from .layout import (
    BlockSolution,
    ALE_FIELD,
    FIELD_NAMES,
    PS_FIELD,
    U_FIELD,
    BlockLayout,
)
from .linear import _PETScBDDCCholesky
from .memory import _collect_solver_memory, _detach_finished_preconditioner
from .newton import (
    AdmissibilityCheck,
    NonlinearAdmissibilityFailure,
    NonlinearConvergenceFailure,
    solve_newton,
)
from .profiling import Profiler

# exterior="fixed": the faces of the cell (geometry.square_geo names them) on which the velocity and the
# relative mesh velocity vanish.
FIXED_EXTERIOR_FACES = tuple(
    f"{axis}{side}_{phase}" for axis in "xyz" for side in ("min", "max") for phase in ("solid", "fluid")
)


def generate_cell_mesh(rve, maxh, order):
    """Mesh a periodic cell geometry as the solver does: graded, volume-optimised, curved.

    The solver builds every new cell's reference mesh with this function; the
    time/quadrature verification driver calls it for its shared reference mesh.
    """
    ngmesh = OCCGeometry(rve).GenerateMesh(maxh=maxh, grading=0.3)
    ngmesh.OptimizeVolumeMesh(MeshingParameters(optimize3d="ms", optsteps3d=3))
    mesh = Mesh(ngmesh)
    mesh.Curve(order)
    return mesh


def _final_ale_mesh_quality(mesh, displacement, sample_order=2, *, store_fields=False):
    """Snapshot sampled quality of the total, curved ALE mesh (3D tetrahedra)."""
    sample_order = max(2, int(sample_order))
    quality = jacobian = None
    if store_fields:
        space = L2(mesh, order=0)
        quality = GridFunction(space, name="mesh_quality")
        jacobian = GridFunction(space, name="mesh_ale_jacobian_min")
        q_data = quality.vec.FV().NumPy()
        j_data = jacobian.vec.FV().NumPy()
    ideal_inverse = np.linalg.inv(
        np.array(
            [[1.0, 0.5, 0.5], [0.0, np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 6.0], [0.0, 0.0, np.sqrt(2.0 / 3.0)]]
        )
    )
    rule = IntegrationRule(ET.TET, order=sample_order)
    points = list(rule.points) + [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    samples = IntegrationRule(points, [1.0] * len(points))
    ale_gradient = Id(3) + Grad(displacement)
    mesh.UnsetDeformation()
    try:
        # One batched evaluation over all elements: rows are element-major,
        # samples-minor, with element i at rows i*len(points):(i+1)*len(points).
        mapped = mesh.MapToAllElements({ET.TET: samples}, VOL)
        element_count = mesh.GetNE(VOL)
        if len(mapped) != element_count * len(points):
            raise ValueError("ALE mesh quality requires tetrahedral volume elements")
        # Evaluated separately: a tuple CF changes F by round-off.
        F, reference_jacobians = (
            np.asarray(function(mapped), dtype=float).reshape(element_count, len(points), 3, 3)
            for function in (ale_gradient, specialcf.JacobianMatrix(3))
        )
    finally:
        mesh.SetDeformation(displacement)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        det_F = np.linalg.det(F)
        B = F @ reference_jacobians @ ideal_inverse
        scale = np.max(np.abs(B), axis=(2, 3))
        B = B / np.where(scale > 0.0, scale, 1.0)[:, :, None, None]
        norm2 = np.sum(B * B, axis=(2, 3))
        q = 3.0 * np.abs(np.linalg.det(B)) ** (2.0 / 3.0)
        q = np.divide(q, norm2, out=np.zeros_like(q), where=norm2 > 0.0)
        q = np.clip(q, 0.0, 1.0) * np.sign(det_F)
    finite = np.isfinite(det_F) & np.isfinite(q) & np.isfinite(scale)
    all_finite = np.all(finite, axis=1)
    nonfinite_elements = int(np.count_nonzero(~all_finite))
    inverted_elements = int(np.count_nonzero(np.any(finite & (det_F < 0.0), axis=1)))
    collapsed_elements = int(np.count_nonzero(np.any(finite & ((det_F == 0.0) | (q == 0.0)), axis=1)))
    q[~finite] = 0.0
    element_quality = np.min(q, axis=1)
    finite_j = det_F[finite]
    jacobian_min = float(np.min(finite_j)) if finite_j.size else np.inf
    jacobian_max = float(np.max(finite_j)) if finite_j.size else -np.inf
    if store_fields:
        # Lowest-order L2 numbers its single dof per element by element number.
        q_data[:] = element_quality
        j_data[:] = np.where(all_finite, np.min(det_F, axis=1), np.nan)
    values = element_quality
    summary = {
        "metric": "signed tetrahedral mean ratio; minimum over samples per element",
        "min": float(np.min(values)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
        "mean": float(np.mean(values)) if values.size else None,
        "worst_element": int(np.argmin(values)) if values.size else None,
        "element_count": int(values.size),
        "inverted_element_count": inverted_elements,
        "collapsed_element_count": collapsed_elements,
        "nonfinite_element_count": nonfinite_elements,
        "ale_jacobian_min": float(jacobian_min) if np.isfinite(jacobian_min) else None,
        "ale_jacobian_max": float(jacobian_max) if np.isfinite(jacobian_max) else None,
        "sample_order": sample_order,
        "samples_per_element": len(points),
        "sampling": "volume quadrature plus vertices; sampled extrema, not certified bounds",
    }
    return (quality, jacobian, summary)


def _reject_invalid_sampled_geometry(summary, minimum_jacobian):
    """Raise if the sampled total ALE map is inverted, degenerate or below the Newton threshold.

    The Newton and post-convergence checks integrate at quadrature points only;
    the sampled summary adds the element vertices, where curved elements invert first.
    """
    counts = {
        key: summary[key]
        for key in ("inverted_element_count", "collapsed_element_count", "nonfinite_element_count")
        if summary[key]
    }
    minimum = summary["ale_jacobian_min"]
    if counts or minimum is None or minimum < minimum_jacobian:
        raise NonlinearAdmissibilityFailure(
            f"converged state has an invalid sampled ALE map: {counts or 'no invalid elements'}, "
            f"minimum sampled det = {minimum}, threshold = {minimum_jacobian:g}"
        )


def _new_solution_history(field_names, components, displacement):
    """Allocate viewable snapshot series, sharing FE spaces but not coefficients."""
    history = {
        "times": [],
        "field_names": tuple(field_names),
        "components": tuple(
            (
                GridFunction(field.space, multidim=0, name=f"history_{name}")
                for name, field in zip(field_names, components)
            )
        ),
        "mesh_displacement": GridFunction(displacement.space, multidim=0, name="history_mesh_displacement"),
        "includes_initial_state": True,
        "time_origin": "start of this FSI call",
        "field_units": "internal dimensionless solver coefficients",
    }
    return history


def _append_solution_history(history, time, components, displacement):
    """AddMultiDimComponent copies coefficients; later solves cannot alias them."""
    for target, source in zip(history["components"], components):
        target.AddMultiDimComponent(source.vec)
    history["mesh_displacement"].AddMultiDimComponent(displacement.vec)
    history["times"].append(float(time))


def summarize_performance(result):
    """Extract compact nonlinear/Krylov totals from an ``FSI`` result.

    Timing values are populated when ``Profiling=True``.  Iteration counts
    are available for every run and make this helper useful in benchmark and
    refinement scripts without duplicating result-dictionary traversal.
    """
    krylov = result.get("krylov_timestep_timings", ())
    preconditioner = result.get("preconditioner_timestep_timings", ())
    memory_cleanup = result.get("memory_cleanup_timings", ())
    histories = result.get("newton_histories", ())
    iterations = sum((item.get("iterations", 0) for item in krylov))
    gmres_seconds = sum((item.get("seconds", 0.0) for item in krylov))
    summary = {
        "timesteps": len(histories),
        "newton_linearizations": sum(
            (item.get("gmres_iterations", 0) > 0 for history in histories for item in history)
        ),
        "gmres_iterations": int(iterations),
        "gmres_seconds": float(gmres_seconds),
        "gmres_ms_per_iteration": float(1000.0 * gmres_seconds / iterations if iterations else 0.0),
        "preconditioner_setup_seconds": float(sum((item.get("seconds", 0.0) for item in preconditioner))),
        "memory_cleanup_seconds": float(sum((item.get("seconds", 0.0) for item in memory_cleanup))),
        "fsi_seconds": float(result.get("profiling", {}).get("FSI total", 0.0)),
    }
    return summary


# Nondimensional finite-element equations, assembly and accepted Newton updates.
def _scaled_core_FSI(
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
    maxh=0.1,
    order=2,
    mesh_mu=1.0,
    mesh_lame_lambda=5.0,
    quadrature_order=None,
    preconditioner_options=None,
    gmres_tol=1e-08,
    gmres_maxsteps=150,
    newton_options=None,
    homogenise=False,
    print_iterations=True,
    Profiling=False,
    diagnostics=False,
    compile_tangent=False,
    exterior="periodic",
    geometry,
    initial_state=None,
    _outputs_only=False,
    _solver_session=None,
    save_history=False,
    loading_path=None,
    _time_scale=1.0,
    _time_offset=0.0,
):
    """Advance the monolithic six-field FSI/ALE state on the reference mesh.

    Fields are velocity, fluid pressure, solid pressure, two mean multipliers and
    relative ALE velocity. Exact component Jacobians retain periodic FreeDofs.
    Newton uses the original scaled residual, admissibility and line-search tests.
    Native SPD proxies on the reference configuration define a fixed residual metric.
    Frozen MUMPS correctors and compiled derivatives produce the endpoint tangent;
    previous history is held fixed. State/history arrays are copied on restart.

    Profiling reports timestep, Newton and homogenisation timings. Scientific
    diagnostics and microscopic histories are allocated only when requested.
    The common quadrature rule is max(2*order, 4*order-3, homogenisation_order)
    unless quadrature_order is supplied explicitly. Geometry must provide the
    periodic solid/fluid domains and interface labels. Live solve objects are
    released after use, except for the private workspace retained by FSISession.
    """
    dt = float(dt)
    t_final = float(t_final)
    maxh = float(maxh)
    E = float(E)
    nu = float(nu)
    mu_f = float(mu_f)
    eta_s = float(eta_s)
    zeta_s = float(zeta_s)
    mesh_mu = float(mesh_mu)
    mesh_lame_lambda = float(mesh_lame_lambda)
    profiler = Profiler(Profiling)
    # Records are per call; clearing here stops the class-level list growing across calls.
    _PETScBDDCCholesky.records.clear()
    fsi_started = perf_counter()

    # FSI checked the inputs (inputs.check_inputs); only state-dependent checks remain here.
    number_of_steps = int(round(t_final / dt))
    rate_controlled = F_rate is not None
    F_target_array = None if F_target is None else np.asarray(F_target, dtype=float)
    G_target_array = np.asarray(G_target, dtype=float)
    has_pressure_gradient_load = bool(
        (np.any(G_target_array != 0.0) if loading_path is None else np.any(loading_path[:, 10:] != 0.0))
        or homogenise
        or _solver_session is not None
    )
    P_target = float(P_target)
    identity_array = np.eye(3)
    state_signature = (E, nu, mu_f, eta_s, zeta_s, order, maxh, mesh_mu, mesh_lame_lambda)
    if initial_state is not None:
        if initial_state.get("exterior", "periodic") != exterior:
            raise ValueError(
                f"restart state is for exterior={initial_state.get('exterior', 'periodic')!r}, not {exterior!r}"
            )
        stored_signature = initial_state.get("signature")
        signatures_match = (
            isinstance(stored_signature, (tuple, list))
            and len(stored_signature) == len(state_signature)
            and (int(stored_signature[5]) == int(state_signature[5]))
            and np.allclose(
                np.asarray(stored_signature[:5] + stored_signature[6:], dtype=float),
                np.asarray(state_signature[:5] + state_signature[6:], dtype=float),
                rtol=1e-12,
                atol=1e-14,
            )
        )
        if not signatures_match:
            raise ValueError(
                f"restart requires the same material and mesh parameters; stored={stored_signature}, current={state_signature}"
            )
        F_initial = np.asarray(initial_state["F_macro"], dtype=float).copy()
        P_initial = float(initial_state["P_macro"])
        G_initial = np.asarray(initial_state["G_macro"], dtype=float).copy()
    else:
        F_initial, P_initial, G_initial = (identity_array, 0.0, np.zeros(3))
    if loading_path is not None:
        loading_path = np.array(loading_path, dtype=float, copy=True)
        prescribed_F_path = [row[:9].reshape(3, 3).copy() for row in loading_path]
        F_target_array = prescribed_F_path[-1].copy()
    elif rate_controlled:
        rate_array = None if callable(F_rate) else np.asarray(F_rate, dtype=float)
        prescribed_F_path = [F_initial.copy()]
        for step in range(number_of_steps):
            rate_value = np.asarray(F_rate(step * dt) if callable(F_rate) else rate_array, dtype=float)
            if rate_value.shape != (3, 3) or not np.all(np.isfinite(rate_value)):
                raise ValueError("F_rate callable must return finite (3, 3) arrays")
            prescribed_F_path.append(prescribed_F_path[-1] + dt * rate_value)
        F_target_array = prescribed_F_path[-1].copy()
    else:
        prescribed_F_path = [
            F_initial + step / number_of_steps * (F_target_array - F_initial)
            for step in range(number_of_steps + 1)
        ]
    prescribed_P_path = (
        np.linspace(P_initial, P_target, number_of_steps + 1)
        if loading_path is None
        else loading_path[:, 9].copy()
    )
    prescribed_G_path = (
        np.linspace(G_initial, G_target_array, number_of_steps + 1)
        if loading_path is None
        else loading_path[:, 10:].copy()
    )
    if any((np.linalg.det(value) <= 0.0 for value in prescribed_F_path)):
        raise ValueError("the path to F_target contains det(F) <= 0")
    if exterior == "fixed" and (
        any(not np.array_equal(value, identity_array) for value in prescribed_F_path)
        or np.any(np.asarray(prescribed_G_path) != 0.0)
    ):
        raise ValueError('exterior="fixed" supports pressure loading only: F = I and G = 0 throughout')
    options = {**newton_defaults(), **(newton_options or {})}
    options["maximum_iterations"] = int(options["maximum_iterations"])
    options["max_backtracks"] = int(options["max_backtracks"])
    options["jacobian_check_order"] = int(options["jacobian_check_order"])
    if options["linear_solver"] not in ("gmres", "mumps"):
        raise ValueError("newton_options['linear_solver'] must be 'gmres' or 'mumps'")
    pc_options = copy.deepcopy({**preconditioner_defaults(mu_f), **(preconditioner_options or {})})
    corrector_threads = pc_options["corrector_blas_threads"]
    pc_options["corrector_blas_threads"] = None if corrector_threads is None else int(corrector_threads)
    pc_options["homogenisation_order"] = int(pc_options["homogenisation_order"])
    # Use one quadrature rule consistently in the equations and their derivatives.
    if quadrature_order is None:
        quadrature_order = max(2 * order, 4 * order - 3, pc_options["homogenisation_order"])
    quadrature_order = int(quadrature_order)
    pc_options["homogenisation_order"] = quadrature_order

    def _freeze_solver_options(value):
        if isinstance(value, dict):
            return tuple(((key, _freeze_solver_options(item)) for key, item in sorted(value.items())))
        if isinstance(value, (list, tuple)):
            return tuple((_freeze_solver_options(item) for item in value))
        return value

    # The material values (and the pressure-mass scale) are NGSolve Parameters in the forms,
    # so a session takes new values in place (CellProblem.set_material): a dt change rescales
    # the viscosities, and a new history may change the material. Only whether the solid is
    # viscous changes the forms.
    session_key = (
        (order, maxh, mesh_mu, mesh_lame_lambda, bool(eta_s or zeta_s)),
        dt,
        quadrature_order,
        gmres_tol,
        gmres_maxsteps,
        _freeze_solver_options(options),
        _freeze_solver_options(
            {key: value for key, value in pc_options.items() if key != "pressure_mass_inverse_scale"}
        ),
        id(geometry),
        bool(print_iterations),
        bool(Profiling),
        diagnostics,
        compile_tangent,
        exterior,
    )
    if _solver_session is not None and _solver_session._cell is not None:
        if session_key != _solver_session._key:
            raise ValueError(
                "FSISession settings changed; close it before changing material, mesh or solver options"
            )
        if initial_state is not None and initial_state["mesh"] is not _solver_session._mesh:
            raise ValueError("FSISession reuse requires a restart on the same mesh")
        cell = _solver_session._cell
        cell.set_material(
            E, nu, mu_f=mu_f, eta_s=eta_s, zeta_s=zeta_s, pc_options=pc_options, state_signature=state_signature
        )
        return cell.advance(
            F_target_array,
            P_target,
            G_target_array,
            prescribed_F_path,
            prescribed_P_path,
            prescribed_G_path,
            initial_state,
            homogenise,
            _outputs_only,
            save_history,
            time_scale=_time_scale,
            time_offset=_time_offset,
        )
    cell = CellProblem(
        E=E,
        Profiling=Profiling,
        _solver_session=_solver_session,
        diagnostics=diagnostics,
        compile_tangent=compile_tangent,
        dt=dt,
        eta_s=eta_s,
        exterior=exterior,
        fsi_started=fsi_started,
        geometry=geometry,
        gmres_maxsteps=gmres_maxsteps,
        gmres_tol=gmres_tol,
        has_pressure_gradient_load=has_pressure_gradient_load,
        identity_array=identity_array,
        initial_state=initial_state,
        maxh=maxh,
        mesh_lame_lambda=mesh_lame_lambda,
        mesh_mu=mesh_mu,
        mu_f=mu_f,
        nu=nu,
        options=options,
        order=order,
        pc_options=pc_options,
        print_iterations=print_iterations,
        profiler=profiler,
        quadrature_order=quadrature_order,
        state_signature=state_signature,
        zeta_s=zeta_s,
    )

    if _solver_session is not None:
        _solver_session._cell = cell
        _solver_session._key = session_key
        _solver_session._mesh = cell.discretisation.mesh
    return cell.advance(
        F_target_array,
        P_target,
        G_target_array,
        prescribed_F_path,
        prescribed_P_path,
        prescribed_G_path,
        initial_state,
        homogenise,
        _outputs_only,
        save_history,
        time_scale=_time_scale,
        time_offset=_time_offset,
    )


class CellProblem:
    """Workspace for one periodic cell: mesh, spaces, forms, block system and preconditioners.

    Built by _scaled_core_FSI after its input checks; FSISession keeps the
    bound ``advance`` method and reuses the workspace for later increments.
    """

    def __init__(
        self,
        *,
        E,
        Profiling,
        _solver_session,
        diagnostics,
        compile_tangent,
        dt,
        eta_s,
        exterior,
        fsi_started,
        geometry,
        gmres_maxsteps,
        gmres_tol,
        has_pressure_gradient_load,
        identity_array,
        initial_state,
        maxh,
        mesh_lame_lambda,
        mesh_mu,
        mu_f,
        nu,
        options,
        order,
        pc_options,
        print_iterations,
        profiler,
        quadrature_order,
        state_signature,
        zeta_s,
    ):
        self.settings = SolverSettings(
            options=options,
            pc_options=pc_options,
            gmres_tol=gmres_tol,
            gmres_maxsteps=gmres_maxsteps,
            print_iterations=print_iterations,
            profiling=Profiling,
            profiler=profiler,
            diagnostics=diagnostics,
            compile_tangent=compile_tangent,
        )
        self._solver_session = _solver_session
        self.dt = dt
        self.fsi_started = fsi_started
        self.has_pressure_gradient_load = has_pressure_gradient_load
        self.identity_array = identity_array
        self.quadrature_order = quadrature_order
        self.state_signature = state_signature
        self.exterior = exterior
        stage_started = self.settings.profiler.start()
        self.geometry_label = getattr(geometry, "__name__", repr(geometry))
        if initial_state is None:
            rve = geometry()
            if rve is None:
                raise ValueError("geometry callable returned None")
        self.settings.profiler.finish("Geometry setup", stage_started)
        stage_started = self.settings.profiler.start()
        if initial_state is None:
            mesh = generate_cell_mesh(rve, maxh, order)
        else:
            mesh = initial_state["mesh"]
            mesh.UnsetDeformation()
        self.settings.profiler.finish("Meshing", stage_started)
        volume_integration_rules = {
            element_type: IntegrationRule(element_type, order=self.quadrature_order)
            for element_type in {element.type for element in mesh.Elements(VOL)}
        }
        self.material = Material.from_elastic(
            E, nu, eta_s=eta_s, zeta_s=zeta_s, mu_f=mu_f, mesh_mu=mesh_mu, mesh_lame_lambda=mesh_lame_lambda
        )
        self.material_parameters = MaterialParameters(self.material, dt=self.dt)

        stage_started = self.settings.profiler.start()
        if exterior == "fixed":
            missing = sorted(set(FIXED_EXTERIOR_FACES) - set(mesh.GetBoundaries()))
            if missing:
                raise ValueError(f'exterior="fixed" needs the geometry to name the faces {", ".join(missing)}')
            fixed = "|".join(FIXED_EXTERIOR_FACES)
            V = Periodic(VectorH1(mesh, order=order, dirichlet=fixed))
            W_delta = Periodic(VectorH1(mesh, order=order, dirichlet="interface|" + fixed, definedon="fluid"))
        else:
            V = Periodic(VectorH1(mesh, order=order))
            W_delta = Periodic(VectorH1(mesh, order=order, dirichlet="interface", definedon="fluid"))
        P_f = Periodic(H1(mesh, order=order - 1, definedon="fluid"))
        P_s = Periodic(H1(mesh, order=order - 1, definedon="solid"))
        R = NumberSpace(mesh)
        R_u = VectorValued(NumberSpace(mesh), dim=3)
        field_spaces = (V, P_f, P_s, R_u, R, W_delta)
        total_state_dofs = sum((space.ndof for space in field_spaces))
        free_dof_blocks = (*(space.FreeDofs() for space in field_spaces),)
        self.discretisation = Discretisation(
            mesh=mesh,
            order=order,
            V=V,
            P_f=P_f,
            P_s=P_s,
            W_delta=W_delta,
            field_spaces=field_spaces,
            volume_integration_rules=volume_integration_rules,
            block_layout=BlockLayout(free_dof_blocks),
            free_dof_blocks=free_dof_blocks,
            total_state_dofs=total_state_dofs,
            number_of_fields=len(field_spaces),
        )
        self.settings.profiler.finish("Space setup", stage_started)
        self.gfu = BlockSolution(FIELD_NAMES, self.discretisation.field_spaces, self.discretisation.total_state_dofs)
        self.mesh_displacement = GridFunction(VectorH1(self.discretisation.mesh, order=self.discretisation.order), name="total_mesh_displacement")
        self.mesh_velocity = GridFunction(VectorH1(self.discretisation.mesh, order=self.discretisation.order), name="mesh_fluctuation_velocity")
        self.solid_rate_of_deformation = GridFunction(
            MatrixValued(L2(self.discretisation.mesh, order=self.discretisation.order, definedon="solid"), 3,
                         symmetric=True), name="solid_rate_of_deformation"
        )
        # The full 3 x 3 space. MatrixValued(space, 3, 3) reads the second 3 as symmetric=True,
        # which kept only sym(F) of the solid's history: local rotations of the solid were lost
        # every step and rigid rotations gave spurious stress (found 25 Oct 2026 in Cell-Optimisation).
        F_space = MatrixValued(L2(self.discretisation.mesh, order=self.discretisation.order, definedon="solid"), 3)
        self.F_old = GridFunction(F_space, name="F_old")
        self.F_new = GridFunction(F_space, name="F_new")
        self.F_old.Set(Id(3))
        if initial_state is not None:

            def restore_vector(target, values):
                values = np.asarray(values, dtype=float)
                target_values = target.vec.FV().NumPy()
                if values.shape != target_values.shape or not np.isfinite(values).all():
                    raise ValueError("incompatible RVE restart field")
                target_values[:] = values

            restore_vector(self.F_old, initial_state["F_solid"])
            restore_vector(self.mesh_displacement, initial_state["mesh_displacement"])
            if len(initial_state["unknowns"]) != len(self.gfu.components):
                raise ValueError("incompatible RVE restart block count")
            for field, values in zip(self.gfu.components, initial_state["unknowns"]):
                restore_vector(field, values)
        if self.settings.profiling:
            self.settings.profiler.summary(
                f"mesh: vertices={int(self.discretisation.mesh.nv)}, elements={int(self.discretisation.mesh.ne)}, geometry={self.geometry_label}"
            )
            self.settings.profiler.summary("finite-element space DOFs:")
            profiling_spaces = (
                ("V (velocity)", self.discretisation.V),
                ("P_f (fluid pressure)", self.discretisation.P_f),
                ("P_s (solid pressure)", self.discretisation.P_s),
                ("R_u (lambda_u velocity multipliers)", R_u),
                ("R (lambda_pf pressure multiplier)", R),
                ("W_delta (ALE fluctuation)", self.discretisation.W_delta),
                ("F_space (solid F_old)", F_space),
            )
            for label, space in profiling_spaces:
                self.settings.profiler.summary(f"  {label}: {int(space.ndof)}")
            self.settings.profiler.summary(f"total block-system DOFs: {int(self.discretisation.total_state_dofs)}")
            self.settings.profiler.summary("state storage: six component GridFunctions + BlockVector (no compound FE space)")
        reference_volume = Integrate(1, mesh, order=4)
        jacobian_measure_tolerance = 1e-14 * max(1.0, reference_volume)
        self.admissibility = AdmissibilityCheck(mesh, self.settings.options["jacobian_check_order"], jacobian_measure_tolerance)
        pos = CF((x, y, z))
        # Cell centre, taken from the reference mesh's bounding box. For a cell that fills its box
        # this is the volume centroid, and it is exact in floating point.
        vertex_coordinates = np.array([vertex.point for vertex in mesh.vertices], dtype=float)
        cell_centre = 0.5 * (vertex_coordinates.min(axis=0) + vertex_coordinates.max(axis=0))
        xbar = CF(tuple(float(value) for value in cell_centre))
        solid_region = mesh.Materials("solid")
        fluid_region = mesh.Materials("fluid")
        self.geometry = CellGeometry(
            solid_region=solid_region,
            fluid_region=fluid_region,
            reference_volume=reference_volume,
            reference_solid_volume=max(float(Integrate(1, mesh, definedon=solid_region, order=4)), 1e-30),
            reference_fluid_volume=max(float(Integrate(1, mesh, definedon=fluid_region, order=4)), 1e-30),
            pos=pos,
            xbar=xbar,
        )

        self.discretisation.mesh.UnsetDeformation()
        self.F_n = GridFunction(F_space, name="F_n")
        self.F_n.Set(Id(3))
        self.displacement_n = GridFunction(self.mesh_displacement.space, name="mesh_displacement_n")
        self.load_parameters = LoadParameters()
        self.E_dot = self.load_parameters.strain_rate
        P0_parameter = self.load_parameters.pressure
        G_parameter = self.load_parameters.gradient
        displacement_cf = CF(tuple(self.displacement_n.components))
        self.U_macro = self.E_dot * (self.geometry.pos + displacement_cf - self.geometry.xbar)
        stage_started = self.settings.profiler.start()
        self.system, self.preconditioner = build_system(
            self.material,
            self.material_parameters,
            self.settings,
            self.discretisation,
            self.F_n,
            self.E_dot,
            P0_parameter,
            G_parameter,
            self.displacement_n,
            dt=self.dt,
            has_pressure_gradient_load=self.has_pressure_gradient_load,
            load_derivative_parameters=self.load_parameters.parameters,
        )
        self.settings.profiler.finish("Reusable block system construction", stage_started)
        self.newton_linear_cache = {"backend": None, "pc": None, "key": None}


    def set_material(self, E, nu, *, mu_f, eta_s, zeta_s, pc_options, state_signature):
        """Take the (scaled) material of the next FSISession call: a new dt rescales the
        viscosities and the pressure-mass scale; a new history may change E and nu.

        The forms read MaterialParameters, so they and any compiled code stay; the
        residual metric is refactorised and the constant solid-pressure tangent block
        reassembled when the values change.
        """
        material = Material.from_elastic(
            E, nu, eta_s=eta_s, zeta_s=zeta_s, mu_f=mu_f,
            mesh_mu=self.material.mesh_mu, mesh_lame_lambda=self.material.mesh_lame_lambda,
        )
        if material != self.material:
            if material.solid_pressure_compliance != self.material.solid_pressure_compliance:
                self.system.full_tangent.blocks.pop((PS_FIELD, PS_FIELD), None)
            self.material = material
            self.material_parameters.set(material)
            self.preconditioner.metric_stale = True
        self.settings = replace(self.settings, pc_options=pc_options)
        self.preconditioner.pressure_inverse_scale = float(pc_options["pressure_mass_inverse_scale"])
        self.state_signature = state_signature


    def set_mesh_state(self, displacement):
        self.discretisation.mesh.UnsetDeformation()
        self.discretisation.mesh.SetDeformation(displacement)


    def build_mesh_displacement_candidate(self, displacement_n, state, U_macro):
        velocity = state.components[U_FIELD]
        relative_mesh_velocity = state.components[ALE_FIELD]
        old_displacement = CF(tuple(displacement_n.components))
        candidate = GridFunction(self.mesh_displacement.space)
        candidate.Set(
            self.discretisation.mesh.MaterialCF(
                {
                    "solid": old_displacement + self.dt * (velocity + U_macro),
                    "fluid": old_displacement + self.dt * (velocity + relative_mesh_velocity + U_macro),
                }
            )
        )
        return candidate


    def advance(
        self,
        F_target_array,
        P_target,
        G_target_array,
        prescribed_F_path,
        prescribed_P_path,
        prescribed_G_path,
        initial_state,
        homogenise,
        _outputs_only,
        save_history,
        *,
        time_scale=1.0,
        time_offset=0.0,
    ):
        number_of_steps = len(prescribed_F_path) - 1
        if self._solver_session is not None and self._solver_session._calls:
            self.fsi_started = perf_counter()
            self.settings.profiler.reset()
            _PETScBDDCCholesky.records.clear()

            def restore_cached(field, values):
                values = np.asarray(values, dtype=float)
                target = field.vec.FV().NumPy()
                if target.shape != values.shape or not np.isfinite(values).all():
                    raise ValueError("incompatible RVE restart field")
                target[:] = values

            if initial_state is None:
                # A new history on the same cell: the reference state at rest, as for a new cell.
                self.F_old.Set(Id(3))
                self.mesh_displacement.vec[:] = 0.0
                for field in self.gfu.components:
                    field.vec[:] = 0.0
            else:
                restore_cached(self.F_old, initial_state["F_solid"])
                restore_cached(self.mesh_displacement, initial_state["mesh_displacement"])
                if len(initial_state["unknowns"]) != len(self.gfu.components):
                    raise ValueError("incompatible RVE restart block count")
                for field, values in zip(self.gfu.components, initial_state["unknowns"]):
                    restore_cached(field, values)
            self.discretisation.mesh.UnsetDeformation()

        homogenisation = Homogenisation(self)

        solution_history = None
        if save_history:
            solution_history = _new_solution_history(FIELD_NAMES, self.gfu.components, self.mesh_displacement)
            _append_solution_history(solution_history, 0.0, self.gfu.components, self.mesh_displacement)
        max_displacements = [0.0]
        prescribed_times = [0.0]
        newton_histories = []
        deformation_check_history = []
        preconditioner_timestep_timings = []
        krylov_timestep_timings = []
        memory_cleanup_timings = []
        F_macro_previous = prescribed_F_path[0]
        quality_sample_order = max(6, 2 * self.discretisation.order - 2)
        step_quality_summary = None
        for step_index in range(1, number_of_steps + 1):
            self.settings.profiler.timestep = int(step_index)
            self.discretisation.mesh.UnsetDeformation()
            time_value = step_index * self.dt
            if self.settings.print_iterations or self.settings.profiling:
                # Physical time for display: the loop runs in time-scaled units (time_scale = physical dt).
                print(f"timestep {step_index}/{number_of_steps}: t={time_offset + time_value * time_scale:g}")
            timestep_started = perf_counter()
            F_macro_next = prescribed_F_path[step_index]
            pressure_target = float(prescribed_P_path[step_index])
            stage_started = self.settings.profiler.start()
            self.F_n.vec.data = self.F_old.vec
            self.displacement_n.vec.data = self.mesh_displacement.vec
            base_preconditioner_time = self.preconditioner.begin_timestep()
            self.settings.profiler.finish("Timestep: state preparation", stage_started)
            F_macro_start = F_macro_previous.copy()
            pressure_start = float(prescribed_P_path[step_index - 1])
            G_start_array = prescribed_G_path[step_index - 1]

            def solve_load_level(level):
                self.preconditioner.reset_adaptive_level()
                F_level = F_macro_start + level * (F_macro_next - F_macro_start)
                F_macro_inc = F_level @ np.linalg.inv(F_macro_start)
                L_level = (F_macro_inc - self.identity_array) / self.dt
                pressure_level = pressure_start + level * (pressure_target - pressure_start)
                gradient_level = G_start_array + level * (prescribed_G_path[step_index] - G_start_array)
                self.load_parameters.set(L_level, pressure_level, gradient_level)
                nonlinear_started = self.settings.profiler.start()
                history_level = solve_newton(
                    self.gfu,
                    self.system,
                    self.preconditioner,
                    self.F_n,
                    self.E_dot,
                    self.displacement_n,
                    settings=self.settings,
                    discretisation=self.discretisation,
                    geometry=self.geometry,
                    admissibility=self.admissibility,
                    dt=self.dt,
                    linear_cache=self.newton_linear_cache,
                )
                self.settings.profiler.finish("Timestep: nonlinear solve", nonlinear_started)
                if not history_level or not history_level[-1].get(
                    "converged",
                    history_level[-1].get(
                        "residual_after_step", history_level[-1].get("residual_norm", np.inf)
                    )
                    <= self.settings.options["tolerance"],
                ):
                    raise NonlinearConvergenceFailure("continuation load level did not converge")
                return history_level

            def snapshot_state():
                snapshot = [component.vec.CreateVector() for component in self.gfu.components]
                for destination, component in zip(snapshot, self.gfu.components):
                    destination.data = component.vec
                return snapshot

            def restore_state(snapshot):
                for component, source in zip(self.gfu.components, snapshot):
                    component.vec.data = source

            state_before_timestep = snapshot_state()
            histories = []
            continuation_used = False
            try:
                history = solve_load_level(1.0)
                histories.append(history)
            except (NonlinearConvergenceFailure, NonlinearAdmissibilityFailure) as direct_failure:
                restore_state(state_before_timestep)
                continuation_used = True
                if self.settings.print_iterations or self.settings.profiling:
                    print("Newton target solve stalled; using nonlinear load continuation")
                levels = [0.25, 0.5, 0.75, 1.0]
                level_index = 0
                previous_level = 0.0
                while level_index < len(levels):
                    level = levels[level_index]
                    level_snapshot = snapshot_state()
                    try:
                        history = solve_load_level(level)
                    except (NonlinearConvergenceFailure, NonlinearAdmissibilityFailure):
                        restore_state(level_snapshot)
                        interval = level - previous_level
                        if interval <= 1.0 / 64.0:
                            raise direct_failure
                        levels.insert(level_index, previous_level + 0.5 * interval)
                        continue
                    histories.append(history)
                    previous_level = level
                    level_index += 1
            history = [item for stage_history in histories for item in stage_history]
            newton_histories.append(history)
            if continuation_used and self.settings.profiling:
                self.settings.profiler.summary(f"timestep {step_index}: continuation levels={len(histories)}")
            preconditioner_elapsed = base_preconditioner_time + sum(
                (item["timings"].get("preconditioner", 0.0) for item in history)
            )
            preconditioner_updates = sum(("preconditioner" in item["timings"] for item in history))
            preconditioner_timestep_timings.append(
                {
                    "timestep": int(step_index),
                    "time": float(time_value),
                    "seconds": float(preconditioner_elapsed),
                    "base_refresh_seconds": float(base_preconditioner_time),
                    "newton_updates": int(preconditioner_updates),
                }
            )
            krylov_elapsed = sum((item["timings"].get("gmres", 0.0) for item in history))
            krylov_iterations = sum((item.get("gmres_iterations", 0) for item in history))
            krylov_solves = sum((item.get("gmres_iterations", 0) > 0 for item in history))
            seconds_per_iteration = krylov_elapsed / krylov_iterations if krylov_iterations else 0.0
            krylov_timestep_timings.append(
                {
                    "timestep": int(step_index),
                    "time": float(time_value),
                    "seconds": float(krylov_elapsed),
                    "linear_solves": int(krylov_solves),
                    "iterations": int(krylov_iterations),
                    "seconds_per_iteration": float(seconds_per_iteration),
                }
            )
            self.settings.profiler.summary(
                f"timestep {step_index}/{number_of_steps}: preconditioner setup {preconditioner_elapsed:.3f} s ({preconditioner_updates} Newton update(s)); GMRES {krylov_iterations} iteration(s) in {krylov_elapsed:.3f} s ({1000.0 * seconds_per_iteration:.2f} ms/iteration)"
            )
            linear_details = [
                attempt.get("linear_backend", {})
                for row in history
                for attempt in row.get("linear_attempts", ())
            ]
            setup_time = sum((item.get("setup_seconds", 0.0) for item in linear_details))
            solve_time = sum((item.get("gmres_seconds", 0.0) for item in linear_details))
            check_time = sum((item.get("physical_residual_check_seconds", 0.0) for item in linear_details))
            self.settings.profiler.summary(
                f"Newton linear work (included above): PETSc setup {setup_time:.3f} s; KSP solve {solve_time:.3f} s; physical residual checks {check_time:.3f} s"
            )
            stage_started = self.settings.profiler.start()
            velocity = self.gfu.components[U_FIELD]
            relative_mesh_velocity = self.gfu.components[ALE_FIELD]
            F_A_n = Id(3) + Grad(self.displacement_n)
            F_A_s_new = F_A_n + self.dt * (Grad(velocity) + self.E_dot * F_A_n)
            self.F_new.Set(F_A_s_new * Inv(F_A_n) * self.F_n, definedon=self.geometry.solid_region)
            solid_admissible = not self.settings.options["check_solid_jacobian"] or self.admissibility.jacobian(
                self.F_new, self.settings.options["minimum_solid_jacobian"], definedon=self.geometry.solid_region
            )
            if not solid_admissible:
                raise RuntimeError("converged state has a non-admissible solid Jacobian")
            mesh_displacement_new = self.build_mesh_displacement_candidate(self.displacement_n, self.gfu, self.U_macro)
            mesh_admissible = not self.settings.options["check_mesh_jacobian"] or self.admissibility.total_mesh(
                mesh_displacement_new, self.settings.options["minimum_mesh_jacobian"]
            )
            if not mesh_admissible:
                raise RuntimeError("converged state has a non-admissible mesh Jacobian")
            if self.settings.options["check_mesh_jacobian"]:
                try:
                    step_quality_summary = _final_ale_mesh_quality(
                        self.discretisation.mesh, mesh_displacement_new, sample_order=quality_sample_order
                    )[2]
                finally:
                    # The sampling deforms the mesh; the rate fields below are set on the reference mesh.
                    self.discretisation.mesh.UnsetDeformation()
                _reject_invalid_sampled_geometry(
                    step_quality_summary, self.settings.options["minimum_mesh_jacobian"]
                )
            F_A_n_post = Id(3) + Grad(self.displacement_n)
            F_A_s_post = Id(3) + Grad(mesh_displacement_new)
            L_s_post = (Grad(velocity) + self.E_dot * F_A_n_post) * Inv(F_A_s_post)
            self.solid_rate_of_deformation.Set(Sym(L_s_post), definedon=self.geometry.solid_region)
            self.mesh_velocity.Set(
                self.discretisation.mesh.MaterialCF({"solid": velocity, "fluid": velocity + relative_mesh_velocity})
            )
            self.set_mesh_state(mesh_displacement_new)
            fluid_volume = Integrate(1, self.discretisation.mesh, definedon=self.geometry.fluid_region, order=self.settings.options["jacobian_check_order"])
            total_volume = Integrate(1, self.discretisation.mesh, order=self.settings.options["jacobian_check_order"])
            geometric_porosity = float(fluid_volume / total_volume)
            if self.settings.options["check_porosity"] and (
                not np.isfinite(geometric_porosity) or geometric_porosity <= self.settings.options["minimum_porosity"]
            ):
                raise RuntimeError("the updated fluid domain reached the configured minimum porosity")
            self.mesh_displacement.vec.data = mesh_displacement_new.vec
            self.set_mesh_state(self.mesh_displacement)
            self.F_old.vec.data = self.F_new.vec
            max_displacements.append(float(np.max(np.abs(self.mesh_displacement.vec.FV().NumPy()))))
            prescribed_times.append(float(time_value))
            deformation_check_history.append(
                {
                    "time": float(time_value),
                    "solid_jacobian_admissible": solid_admissible,
                    "mesh_jacobian_admissible": mesh_admissible,
                    "geometric_porosity": geometric_porosity,
                }
            )
            if solution_history is not None:
                _append_solution_history(solution_history, time_value, self.gfu.components, self.mesh_displacement)
            F_macro_previous = F_macro_next
            self.settings.profiler.finish("Timestep: state and mesh update", stage_started)
            cleanup_started = self.settings.profiler.start()
            del state_before_timestep
            del histories
            del mesh_displacement_new
            del solve_load_level
            del snapshot_state
            del restore_state
            collected_objects, heap_trimmed = _collect_solver_memory(True)
            cleanup_elapsed = self.settings.profiler.finish("Timestep: memory cleanup", cleanup_started)
            memory_cleanup_timings.append(
                {
                    "timestep": int(step_index),
                    "time": float(time_value),
                    "seconds": float(cleanup_elapsed),
                    "collected_objects": int(collected_objects),
                    "heap_trimmed": heap_trimmed,
                }
            )
            self.settings.profiler.finish("Timestep total", timestep_started)
        if self._solver_session is None and self.newton_linear_cache["pc"] is not None:
            self.newton_linear_cache["pc"].close()
            self.newton_linear_cache["pc"] = None
        newton_solver_metadata = {
            "fsi_bddc_variant": self.preconditioner.fsi_bddc_variant,
            "high_order_fsi": self.preconditioner.high_order_fsi,
            "adaptive_iteration_threshold": self.preconditioner.adaptive_iteration_threshold,
            "adaptive_refresh_count": self.preconditioner.adaptive_refresh_count,
        }
        homogenised_response = None
        if homogenise:
            homogenised_response = homogenisation.compute_homogenised_response(
                prescribed_F_path, P_target, G_target_array, _outputs_only
            )
        if self.settings.diagnostics:
            interface_correction_norm = float(
                np.sqrt(
                    max(
                        0.0,
                        Integrate(
                            InnerProduct(self.gfu.components[ALE_FIELD], self.gfu.components[ALE_FIELD]),
                            self.discretisation.mesh,
                            BND,
                            definedon=self.discretisation.mesh.Boundaries("interface"),
                            order=max(2, self.discretisation.order),
                        ),
                    )
                )
            )
        stage_started = self.settings.profiler.start()
        if step_quality_summary is not None and not self.settings.diagnostics:
            # The last step's accepted map is the final mesh; only the plotting fields need a new pass.
            mesh_quality = mesh_ale_jacobian_min = None
            mesh_quality_summary = step_quality_summary
        else:
            mesh_quality, mesh_ale_jacobian_min, mesh_quality_summary = _final_ale_mesh_quality(
                self.discretisation.mesh, self.mesh_displacement, sample_order=quality_sample_order, store_fields=self.settings.diagnostics
            )
        self.settings.profiler.finish("Final mesh quality", stage_started)
        self.settings.profiler.finish("FSI total", self.fsi_started)
        if self.settings.profiling:
            self.settings.profiler.summary(
                f"completed {len(max_displacements) - 1} physical timestep(s); total FSI time {self.settings.profiler.totals.get('FSI total', 0.0):.3f} s"
            )
            self.settings.profiler.summary(
                f"preconditioner setup total {sum((item['seconds'] for item in preconditioner_timestep_timings)):.3f} s"
            )
            self.settings.profiler.summary(
                f"memory cleanup total {sum((item['seconds'] for item in memory_cleanup_timings)):.3f} s"
            )
        self.settings.profiler.print_setup_breakdown()
        self.settings.profiler.print_summary_table()
        profiling_averages = {
            label: elapsed / self.settings.profiler.counts[label] for label, elapsed in self.settings.profiler.totals.items()
        }
        result = {
            "gfu": self.gfu,
            "mesh": self.discretisation.mesh,
            "mesh_displacement": self.mesh_displacement,
            **(
                {"mesh_quality": mesh_quality, "mesh_ale_jacobian_min": mesh_ale_jacobian_min}
                if self.settings.diagnostics
                else {}
            ),
            "mesh_quality_summary": mesh_quality_summary,
            "mesh_velocity": self.mesh_velocity,
            "solid_rate_of_deformation": self.solid_rate_of_deformation,
            "F_old": self.F_old,
            "linear_system": self.system,
            "preconditioner": self.preconditioner,
            "homogenisation_preconditioner": None,
            "max_displacements": max_displacements,
            "prescribed_times": np.asarray(prescribed_times),
            "prescribed_F_history": np.stack(prescribed_F_path),
            "prescribed_P_history": np.asarray(prescribed_P_path).copy(),
            "prescribed_G_history": np.asarray(prescribed_G_path).copy(),
            "prescribed_F_rate_history": np.stack(
                [(prescribed_F_path[i + 1] - prescribed_F_path[i]) / self.dt for i in range(number_of_steps)]
            ),
            "F_target": F_target_array.copy(),
            "P_target": P_target,
            "G_target": G_target_array.copy(),
            "geometry": self.geometry_label,
            "homogenised_response": homogenised_response,
            "newton_histories": newton_histories,
            "newton_solver_metadata": [
                history[-1].get("newton_backend", {}) for history in newton_histories if history
            ],
            "deformation_check_history": deformation_check_history,
            **({"interface_correction_norm": interface_correction_norm} if self.settings.diagnostics else {}),
            "total_dofs": int(self.discretisation.total_state_dofs),
            "solid_pressure_dofs": int(self.discretisation.P_s.ndof),
            "component_dofs": tuple((int(component.space.ndof) for component in self.gfu.components)),
            "field_order": tuple(self.system.field_order),
            "dof_ranges": dict(self.system.dof_ranges),
            "mesh_mu": self.material.mesh_mu,
            "mesh_lame_lambda": self.material.mesh_lame_lambda,
            "preconditioner_options": dict(self.settings.pc_options),
            "quadrature_order": self.quadrature_order,
            "fsi_preconditioner_variant": newton_solver_metadata["fsi_bddc_variant"],
            "fsi_preconditioner_order_aware": bool(newton_solver_metadata["high_order_fsi"]),
            "fsi_adaptive_refresh_iteration_threshold": int(
                newton_solver_metadata["adaptive_iteration_threshold"]
            ),
            "adaptive_preconditioner_refreshes": int(newton_solver_metadata["adaptive_refresh_count"]),
            "native_velocity_coarse_metadata": [
                dict(row) for row in _PETScBDDCCholesky.records
            ],
            "native_velocity_coarse_backend": "PETSc/MUMPS Cholesky experimental exact BDDC coarse inverse",
            "setup_breakdown": self.settings.profiler.setup_snapshot(),
            "setup_breakdown_accounting": {
                "enabled": bool(self.settings.profiling),
                "parent_child_rows_overlap": True,
                "assemble_includes_registered_native_setup": True,
                "internal_native_factorization_count_known": False,
                "legacy_preconditioner_setup": "begin_timestep elapsed plus history preconditioner update elapsed",
                "adaptive_refresh": "ordinary adaptive refresh is outside the legacy preconditioner setup total; failure refresh can be included in a preconditioner-update or GMRES timer",
                "metric": "residual-metric factorisation, once per cell and after a dt or material change; inside begin_timestep",
                "PETSc_setup": "separate from these native call timers; contained in Newton GMRES stage",
            },
            "profiling": dict(self.settings.profiler.totals),
            "profiling_counts": dict(self.settings.profiler.counts),
            "profiling_averages": profiling_averages,
            "preconditioner_timestep_timings": list(preconditioner_timestep_timings),
            "krylov_timestep_timings": list(krylov_timestep_timings),
            "memory_cleanup_timings": list(memory_cleanup_timings),
            "residual_metric": "fixed timestep-lagged positive block operators",
            "geometry_map_formulation": "reference pullback with fully coupled ALE map",
            "block_assembly": "component nonlinear forms assembled directly into BlockMatrix",
            "state_storage": "component GridFunctions with BlockVector",
            "assembled_block_keys": tuple(self.system.assembled_block_keys),
        }
        if solution_history is not None:
            solution_history["times"] = np.asarray(solution_history["times"], dtype=float)
            result["history"] = solution_history
        result["restart_state"] = {
            "signature": self.state_signature,
            "exterior": self.exterior,
            "mesh": self.discretisation.mesh,
            "F_macro": F_target_array.copy(),
            "P_macro": P_target,
            "G_macro": G_target_array.copy(),
            "F_solid": self.F_old.vec.FV().NumPy().copy(),
            "mesh_displacement": self.mesh_displacement.vec.FV().NumPy().copy(),
            "unknowns": tuple((field.vec.FV().NumPy().copy() for field in self.gfu.components)),
        }
        result["performance_summary"] = summarize_performance(result)
        if self._solver_session is None:
            _detach_finished_preconditioner(self.preconditioner)
        result["linear_system"] = None
        result["preconditioner"] = None
        result["homogenisation_preconditioner"] = None
        homogenisation.metric = None
        if self._solver_session is None:
            self.preconditioner = None
            self.system = None
        _collect_solver_memory(True)
        result["workspace_reused"] = bool(self._solver_session is not None and self._solver_session._calls > 0)
        if self._solver_session is not None:
            self._solver_session._calls += 1
        return result
