"""The cell problem's equations: constitutive laws, block forms and the six-field system (eqs. 32-37)."""

from __future__ import annotations
import numpy as np
from ngsolve import (
    BilinearForm,
    BlockMatrix,
    CoefficientFunction,
    Cof,
    Det,
    Grad,
    GridFunction,
    Id,
    InnerProduct,
    Inv,
    LinearForm,
    Parameter,
    Sym,
    Trace,
    dx,
    log,
)

from .layout import (
    ALE_FIELD,
    FIELD_NAMES,
    LAMBDA_PF_FIELD,
    LAMBDA_U_FIELD,
    PF_FIELD,
    PS_FIELD,
    U_FIELD,
)
from .residual_metric import BlockFSIALEPreconditioner, _make_fsi_bddc, make_pressure_preconditioner


def solid_cauchy_stress(F, pressure=None, *, mu, lam):
    J = Det(F)
    B = F * F.trans
    deviatoric_base = mu / J * (B - Id(3))
    if pressure is None:
        return deviatoric_base + lam * log(J) / J * Id(3)
    return deviatoric_base - pressure * Id(3)


def solid_viscous_stress(D, *, eta, zeta):
    """Kelvin--Voigt viscous Cauchy stress in the current configuration."""
    return 2.0 * eta * (D - Trace(D) / 3.0 * Id(3)) + zeta * Trace(D) * Id(3)


def mesh_first_piola(F, *, mu, lame_lambda):
    """Derivative of the reference compressible Neo-Hookean energy.

    psi(F) = mu/2 (tr(F.T F) - 3 - 2 log(J))
             + lame_lambda/2 log(J)^2.

    Both coefficients are constants here; there is no J-dependent mesh
    stiffening.
    """
    J = Det(F)
    FinvT = Inv(F).trans
    return mu * (F - FinvT) + lame_lambda * log(J) * FinvT



def _hm_solid_map(old_map, velocity_gradient, rate, step):
    """Shared direct residual map; no GridFunction.Set geometry transfer."""
    return old_map + step * (velocity_gradient + rate * old_map)


def _hm_fluid_map(old_map, velocity_gradient, relative_gradient, rate, step):
    return _hm_solid_map(old_map, velocity_gradient, rate, step) + step * relative_gradient


# The field equations, shared by the periodic cell (build_system) and the resolved column
# (resolved/). Each returns the integrand of one residual row for given field expressions
# (trial proxies or GridFunctions) and test functions; ``E_dot``, ``P0`` and ``G`` are the
# cell's macro loads (zero in the resolved column). F_A_n is the ALE map of the last step.


def momentum_terms(coefficients, material, *, velocity, fluid_pressure, solid_pressure, relative_mesh_velocity, test,
                   F_A_n, solid_history_jacobian, solid_history_metric, E_dot, P0, G, dt):
    """Momentum row: the solid first Piola stress and its map, and the fluid integrand.

    Returns (solid_piola, J_s, fluid_integrand); the solid integrand is
    InnerProduct(solid_piola, Grad(test)), to which the cell adds its velocity-mean multiplier.
    ``G`` None leaves out the pressure-gradient load.
    """
    F_A_s = _hm_solid_map(F_A_n, Grad(velocity), E_dot, dt)
    F_A_f = _hm_fluid_map(F_A_n, Grad(velocity), Grad(relative_mesh_velocity), E_dot, dt)
    A_s = Inv(F_A_s)
    A_f = Inv(F_A_f)
    J_s = Det(F_A_s)
    J_f = Det(F_A_f)
    C_s = Cof(F_A_s)
    C_f = Cof(F_A_f)
    velocity_gradient = Grad(velocity) + E_dot * F_A_n
    L_f = velocity_gradient * A_f
    solid_piola = (
        coefficients.mu_s / solid_history_jacobian * (F_A_s * solid_history_metric - A_s.trans) - solid_pressure * C_s
    )
    if material.eta_s or material.zeta_s:
        solid_piola += solid_viscous_stress(Sym(velocity_gradient * A_s), eta=coefficients.eta_s, zeta=coefficients.zeta_s) * C_s
    fluid_integrand = (
        J_f * 2.0 * coefficients.mu_f * InnerProduct(Sym(L_f), Sym(Grad(test) * A_f))
        - fluid_pressure * InnerProduct(C_f, Grad(test))
        - P0 * InnerProduct(C_f, Grad(test))
    )
    if G is not None:
        fluid_integrand += J_f * InnerProduct(G, test)
    return solid_piola, J_s, fluid_integrand


def continuity_terms(*, velocity, relative_mesh_velocity, test, F_A_n, E_dot, dt):
    """Fluid incompressibility row: (integrand, J_f); the cell adds its pressure-mean multiplier times J_f."""
    F_A_f = _hm_fluid_map(F_A_n, Grad(velocity), Grad(relative_mesh_velocity), E_dot, dt)
    J_f = Det(F_A_f)
    return -test * InnerProduct(Cof(F_A_f), Grad(velocity) + E_dot * F_A_n), J_f


def solid_pressure_integrand(coefficients, *, velocity, solid_pressure, test, F_A_n, solid_history_jacobian, E_dot, dt):
    """Solid pressure row: compressible Neo-Hookean volumetric law in mixed form."""
    F_A_s = _hm_solid_map(F_A_n, Grad(velocity), E_dot, dt)
    J_material = Det(F_A_s) * solid_history_jacobian
    return test * (
        coefficients.solid_pressure_compliance * solid_pressure
        + coefficients.solid_pressure_log_coefficient * log(J_material) / J_material
    )


def mesh_motion_integrand(coefficients, *, velocity, relative_mesh_velocity, test, F_A_n, A_n, J_n, E_dot, dt):
    """ALE mesh row (fluid): incremental Neo-Hookean mesh energy of this step's mesh map."""
    F_A_f = _hm_fluid_map(F_A_n, Grad(velocity), Grad(relative_mesh_velocity), E_dot, dt)
    F_mesh = F_A_f * A_n
    return dt * J_n * InnerProduct(
        mesh_first_piola(F_mesh, mu=coefficients.mesh_mu, lame_lambda=coefficients.mesh_lame_lambda) * A_n.trans,
        Grad(test),
    )


class BlockTangentBlocks:
    """Direct component-form tangent blocks.

    Nonlinear blocks are differentiated with respect to one active
    trial field; simple linear couplings are assembled analytically.
    The constant solid-pressure block and an exact transpose coupling
    share work and storage across tangent updates.  No compound form is
    assembled or split afterwards.
    """

    def __init__(self, forms, block_sizes, profiler):
        self.profiler = profiler
        self.forms = dict(forms)
        self.block_sizes = tuple((int(size) for size in block_sizes))
        self.blocks = {}
        self.linear_blocks = {
            (U_FIELD, PF_FIELD),
            (U_FIELD, PS_FIELD),
            (U_FIELD, LAMBDA_U_FIELD),
            (PF_FIELD, LAMBDA_PF_FIELD),
            (PS_FIELD, PS_FIELD),
        }
        self.constant_blocks = {(PS_FIELD, PS_FIELD)}
        self.transpose_sources = {(LAMBDA_PF_FIELD, PF_FIELD): (PF_FIELD, LAMBDA_PF_FIELD)}

    def assemble(self, state_components):
        for (row, column), form in self.forms.items():
            if (row, column) in self.constant_blocks and (row, column) in self.blocks:
                continue
            block_started = self.profiler.start()
            if (row, column) in self.linear_blocks:
                form.Assemble()
            else:
                form.AssembleLinearization(state_components[column].vec)
            self.profiler.finish(
                f"Newton: tangent block {FIELD_NAMES[row]}<-{FIELD_NAMES[column]}", block_started
            )
            self.blocks[row, column] = form.mat
        for key, source in self.transpose_sources.items():
            if key not in self.blocks:
                self.blocks[key] = self.blocks[source].T

    def get(self, row, column):
        return self.blocks.get((row, column))


class LoadParameters:
    """Mutable load coefficients shared by all reusable block forms."""

    def __init__(self):
        self._strain_rate = tuple((Parameter(0.0) for _ in range(9)))
        self._pressure = Parameter(0.0)
        self._gradient = tuple((Parameter(0.0) for _ in range(3)))
        self.parameters = (*self._strain_rate, self._pressure, *self._gradient)
        self.strain_rate = CoefficientFunction(self._strain_rate, dims=(3, 3))
        self.pressure = self._pressure
        self.gradient = CoefficientFunction(self._gradient)

    def set(self, strain_rate, pressure, gradient):
        strain_rate_values = np.asarray(strain_rate, dtype=float).reshape(9)
        gradient_values = np.asarray(gradient, dtype=float).reshape(3)
        for parameter, value in zip(self._strain_rate, strain_rate_values):
            parameter.Set(float(value))
        self._pressure.Set(float(pressure))
        for parameter, value in zip(self._gradient, gradient_values):
            parameter.Set(float(value))


class MaterialParameters:
    """Mutable material coefficients shared by all reusable block forms.

    The forms read these Parameters instead of literals, so the generated code
    does not depend on the material or on the time step (the scaled viscosities
    are value/(mu_s*dt)). An FSISession then keeps its forms and compiled code
    across a dt change, and every material reuses the same cached libraries.
    ``set`` also refreshes the proxy scales derived from the material.
    """

    names = (
        "mu_s",
        "mu_f",
        "eta_s",
        "zeta_s",
        "mesh_mu",
        "mesh_lame_lambda",
        "solid_pressure_compliance",
        "solid_pressure_log_coefficient",
        "solid_pressure_schur_scale",
        "solid_volumetric_proxy_scale",
        "solid_proxy_scale",
    )

    def __init__(self, material, *, dt):
        self.dt = dt
        for name in self.names:
            setattr(self, name, Parameter(0.0))
        self.set(material)

    def set(self, material):
        solid_pressure_schur_scale = material.solid_pressure_compliance + abs(
            material.solid_pressure_log_coefficient
        ) / (2.0 * material.mu_s)
        values = dict(
            mu_s=material.mu_s,
            mu_f=material.mu_f,
            eta_s=material.eta_s,
            zeta_s=material.zeta_s,
            mesh_mu=material.mesh_mu,
            mesh_lame_lambda=material.mesh_lame_lambda,
            solid_pressure_compliance=material.solid_pressure_compliance,
            solid_pressure_log_coefficient=material.solid_pressure_log_coefficient,
            solid_pressure_schur_scale=solid_pressure_schur_scale,
            solid_volumetric_proxy_scale=self.dt
            * abs(material.solid_pressure_log_coefficient)
            / solid_pressure_schur_scale,
            solid_proxy_scale=max(material.mu_s, material.mu_f, 1e-30),
        )
        for name, value in values.items():
            getattr(self, name).Set(float(value))


class BlockFSIALESystem:
    """Six-field residual/Jacobian assembled directly by component blocks."""

    def __init__(
        self,
        block_forms,
        residual_forms,
        pressure_mass,
        solid_pressure_mass,
        velocity_proxy,
        mesh_proxy,
        state_components,
        load_derivative_factory,
        homogenisation_velocity_proxy=None,
        homogenisation_pressure_mass=None,
        homogenisation_solid_pressure_mass=None,
        homogenisation_mesh_proxy=None,
        *,
        mesh,
        field_spaces,
        block_layout,
        profiler,
    ):
        self._mesh = mesh
        self._layout = block_layout
        self._profiler = profiler
        block_sizes = tuple((space.ndof for space in field_spaces))
        self.block_forms = dict(block_forms)
        self.residual_forms = tuple(residual_forms)
        self.load_derivative_factory = load_derivative_factory
        self.load_derivative_forms = None
        self.pressure_mass = pressure_mass
        self.solid_pressure_mass = solid_pressure_mass
        self.velocity_proxy = velocity_proxy
        self.mesh_proxy = mesh_proxy
        self.homogenisation_velocity_proxy = homogenisation_velocity_proxy
        self.homogenisation_pressure_mass = homogenisation_pressure_mass
        self.homogenisation_solid_pressure_mass = homogenisation_solid_pressure_mass
        self.homogenisation_mesh_proxy = homogenisation_mesh_proxy
        self.state_components = tuple(state_components)
        self.full_tangent = BlockTangentBlocks(self.block_forms, block_sizes, self._profiler)
        self.mesh_velocity_block = None
        self.mesh_matrix = None
        self.a_uu = self.block_forms[U_FIELD, U_FIELD]
        self.b = self.block_forms[PF_FIELD, U_FIELD]
        self.bt = self.block_forms[U_FIELD, PF_FIELD]
        self.c = self.block_forms[PF_FIELD, LAMBDA_PF_FIELD]
        self.b_s = self.block_forms[PS_FIELD, U_FIELD]
        self.bt_s = self.block_forms[U_FIELD, PS_FIELD]
        self.c_s = self.block_forms[PS_FIELD, PS_FIELD]
        self.d = self.block_forms[LAMBDA_U_FIELD, U_FIELD]
        self.e = self.block_forms[U_FIELD, LAMBDA_U_FIELD]
        self.field_order = FIELD_NAMES
        offsets = np.cumsum((0, *block_sizes))
        self.dof_ranges = {
            name: (int(offsets[index]), int(offsets[index + 1]))
            for index, name in enumerate(self.field_order)
        }
        self.block_matrix = None
        self.assembled_block_keys = (*self.block_forms, *self.full_tangent.transpose_sources)
        self.mat = None

    def AssembleLoadDerivatives(self):
        """Assemble exact residual derivatives with respect to loads.

        The tangent and all derivative forms use the same accepted state
        stored in ``state_components``.  Each form differentiates the
        symbolic residual with respect to one mutable load Parameter;
        no plus/minus residual states are evaluated.
        """
        self._mesh.UnsetDeformation()
        started = self._profiler.start()
        if self.load_derivative_forms is None:
            self.load_derivative_forms = self.load_derivative_factory()
            self.load_derivative_factory = None
        for derivative_rows in self.load_derivative_forms:
            for form in derivative_rows:
                if form is None:
                    continue
                form.Assemble()
        return self._profiler.finish("Homogenisation: exact residual derivative assembly", started)

    def _build_matrix(self):
        self.block_matrix = BlockMatrix(
            [
                [self.full_tangent.get(row, column) for column in range(self._layout.number_of_fields)]
                for row in range(self._layout.number_of_fields)
            ]
        )
        self.mat = self.block_matrix

    def _set_state(self, x):
        for component, value in zip(self.state_components, x):
            component.vec.data = value

    def Apply(self, x, y):
        self._mesh.UnsetDeformation()
        self._set_state(x)
        self._layout.zero(y)
        for row, form in enumerate(self.residual_forms):
            row_started = self._profiler.start()
            form.Assemble()
            self._profiler.finish(f"Newton: residual row {FIELD_NAMES[row]}", row_started)
            y[row].data = form.vec

    def ApplyResidual(self, x, y):
        self.Apply(x, y)

    def AssembleLinearization(self, x):
        self._mesh.UnsetDeformation()
        self._set_state(x)
        tangent_started = self._profiler.start()
        self.full_tangent.assemble(self.state_components)
        self._linearization_revision = getattr(self, "_linearization_revision", 0) + 1
        self._profiler.finish("Newton: component block tangent assembly", tangent_started)
        self.mesh_velocity_block = self.full_tangent.get(ALE_FIELD, U_FIELD)
        self.mesh_matrix = self.full_tangent.get(ALE_FIELD, ALE_FIELD)
        if self.block_matrix is None:
            self._build_matrix()


def build_system(
    material,
    coefficients,
    settings,
    discretisation,
    F_n,
    E_dot,
    P0,
    G,
    displacement_n,
    *,
    dt,
    has_pressure_gradient_load,
    load_derivative_parameters=None,
):
    """Assemble one monolithic residual from independent field blocks.

    Each nonzero block is a nonlinear component-space form.  Its selected
    trial proxy is the Newton variable and all other fields are ordinary
    coefficient GridFunctions.  Thus ``Apply`` evaluates a complete
    residual row and ``AssembleLinearization`` differentiates that row
    with respect to exactly one column; no nonlinear form on the compound
    product space is assembled.

    The forms read the material through ``coefficients`` (MaterialParameters);
    ``material`` only decides whether the viscous solid terms exist.
    """
    u_proxy, v_phys = discretisation.V.TnT()
    p_proxy, q_test = discretisation.P_f.TnT()
    state_components = tuple(
        (GridFunction(space, name=f"component_state_{index}") for index, space in enumerate(discretisation.field_spaces))
    )
    F_A_n = Id(3) + Grad(displacement_n)
    A_n = Inv(F_A_n)
    J_n = Det(F_A_n)
    solid_history_map = A_n * F_n
    solid_history_metric = solid_history_map * solid_history_map.trans
    solid_history_jacobian = Det(solid_history_map)
    trial_spaces = discretisation.field_spaces
    test_spaces = trial_spaces
    block_forms = {}
    block_proxies = {}
    load_derivative_parameters = tuple(load_derivative_parameters or ())

    strain_rate_parameters = load_derivative_parameters[:9]

    def add_integrand(form, integrand, region, derivative_parameter=None, native=False):
        if derivative_parameter is not None:
            integrand = integrand.Diff(derivative_parameter)
        if settings.compile_tangent and (native or isinstance(form, BilinearForm)):
            # Tangent blocks and the velocity-row strain-rate sensitivities (about
            # 85% of the sensitivity assembly) only: native code needs a C++
            # compiler and is rebuilt for every cell, so other forms stay interpreted.
            # wait=False compiles in the background; assembly stays interpreted until
            # each block's library is ready, so results vary at round-off with timing.
            integrand = integrand.Compile(realcompile=True, wait=False)
        else:
            integrand = integrand.Compile()
        form += integrand * dx(region, intrules=discretisation.volume_integration_rules)

    def make_block(row, column):
        trial_proxy = trial_spaces[column].TnT()[0]
        test_proxy = test_spaces[row].TnT()[1]
        form = BilinearForm(trialspace=trial_spaces[column], testspace=test_spaces[row])
        block_forms[row, column] = form
        block_proxies[row, column] = (trial_proxy, test_proxy)
        return form

    def expression_fields(column, active_proxy):
        fields = list(state_components)
        fields[column] = active_proxy
        return fields

    def solid_ale_map(velocity):
        return _hm_solid_map(F_A_n, Grad(velocity), E_dot, dt)

    def fluid_ale_map(velocity, relative_mesh_velocity):
        return _hm_fluid_map(F_A_n, Grad(velocity), Grad(relative_mesh_velocity), E_dot, dt)

    def add_physical_row(form, column, active_proxy, test_proxy, derivative_parameter=None):
        u_expr, p_f_expr, p_s_expr, rho_expr, lam_expr, delta_expr = expression_fields(
            column, active_proxy
        )
        native = any(derivative_parameter is parameter for parameter in strain_rate_parameters)
        solid_piola, J_s, fluid_integrand = momentum_terms(
            coefficients, material, velocity=u_expr, fluid_pressure=p_f_expr, solid_pressure=p_s_expr,
            relative_mesh_velocity=delta_expr, test=test_proxy, F_A_n=F_A_n,
            solid_history_jacobian=solid_history_jacobian, solid_history_metric=solid_history_metric,
            E_dot=E_dot, P0=P0, G=G if has_pressure_gradient_load else None, dt=dt,
        )
        add_integrand(
            form,
            InnerProduct(solid_piola, Grad(test_proxy)) + J_s * InnerProduct(rho_expr, test_proxy),
            "solid",
            derivative_parameter,
            native=native,
        )
        add_integrand(form, fluid_integrand, "fluid", derivative_parameter, native=native)

    def add_pressure_row(form, column, active_proxy, test_proxy, derivative_parameter=None):
        u_expr, _, _, _, lam_expr, delta_expr = expression_fields(column, active_proxy)
        continuity, J_f = continuity_terms(
            velocity=u_expr, relative_mesh_velocity=delta_expr, test=test_proxy, F_A_n=F_A_n, E_dot=E_dot, dt=dt
        )
        add_integrand(form, continuity + J_f * lam_expr * test_proxy, "fluid", derivative_parameter)

    def add_solid_pressure_row(form, column, active_proxy, test_proxy, derivative_parameter=None):
        u_expr, _, p_s_expr, _, _, _ = expression_fields(column, active_proxy)
        integrand = solid_pressure_integrand(
            coefficients, velocity=u_expr, solid_pressure=p_s_expr, test=test_proxy, F_A_n=F_A_n,
            solid_history_jacobian=solid_history_jacobian, E_dot=E_dot, dt=dt,
        )
        add_integrand(form, integrand, "solid", derivative_parameter)

    def add_pressure_constraint_row(form, column, active_proxy, test_proxy, derivative_parameter=None):
        u_expr, p_f_expr, _, _, _, delta_expr = expression_fields(column, active_proxy)
        F_A_f = fluid_ale_map(u_expr, delta_expr)
        J_f = Det(F_A_f)
        add_integrand(form, J_f * test_proxy * p_f_expr, "fluid", derivative_parameter)

    def add_velocity_constraint_row(form, column, active_proxy, test_proxy, derivative_parameter=None):
        u_expr, _, _, _, _, _ = expression_fields(column, active_proxy)
        F_A_s = solid_ale_map(u_expr)
        J_s = Det(F_A_s)
        add_integrand(form, J_s * InnerProduct(u_expr, test_proxy), "solid", derivative_parameter)

    def add_mesh_row(form, column, active_proxy, test_proxy, derivative_parameter=None):
        u_expr, _, _, _, _, delta_expr = expression_fields(column, active_proxy)
        integrand = mesh_motion_integrand(
            coefficients, velocity=u_expr, relative_mesh_velocity=delta_expr, test=test_proxy,
            F_A_n=F_A_n, A_n=A_n, J_n=J_n, E_dot=E_dot, dt=dt,
        )
        add_integrand(form, integrand, "fluid", derivative_parameter)

    for column in (U_FIELD, ALE_FIELD):
        form = make_block(U_FIELD, column)
        add_physical_row(form, column, *block_proxies[U_FIELD, column])
    state_solid_map = solid_ale_map(state_components[U_FIELD])
    state_fluid_map = fluid_ale_map(state_components[U_FIELD], state_components[ALE_FIELD])
    for column in (PF_FIELD, PS_FIELD, LAMBDA_U_FIELD):
        form = make_block(U_FIELD, column)
        active, test = block_proxies[U_FIELD, column]
        if column == PF_FIELD:
            integrand = -active * InnerProduct(Cof(state_fluid_map), Grad(test))
            region = "fluid"
        elif column == PS_FIELD:
            integrand = -active * InnerProduct(Cof(state_solid_map), Grad(test))
            region = "solid"
        else:
            integrand = Det(state_solid_map) * InnerProduct(active, test)
            region = "solid"
        add_integrand(form, integrand, region)
    for column in (U_FIELD, ALE_FIELD):
        form = make_block(PF_FIELD, column)
        add_pressure_row(form, column, *block_proxies[PF_FIELD, column])
    form = make_block(PF_FIELD, LAMBDA_PF_FIELD)
    active, test = block_proxies[PF_FIELD, LAMBDA_PF_FIELD]
    add_integrand(form, Det(state_fluid_map) * active * test, "fluid")
    form = make_block(LAMBDA_U_FIELD, U_FIELD)
    add_velocity_constraint_row(form, U_FIELD, *block_proxies[LAMBDA_U_FIELD, U_FIELD])
    for column in (U_FIELD, ALE_FIELD):
        form = make_block(LAMBDA_PF_FIELD, column)
        add_pressure_constraint_row(form, column, *block_proxies[LAMBDA_PF_FIELD, column])
    form = make_block(PS_FIELD, U_FIELD)
    add_solid_pressure_row(form, U_FIELD, *block_proxies[PS_FIELD, U_FIELD])
    form = make_block(PS_FIELD, PS_FIELD)
    active, test = block_proxies[PS_FIELD, PS_FIELD]
    add_integrand(form, coefficients.solid_pressure_compliance * active * test, "solid")
    for column in (U_FIELD, ALE_FIELD):
        form = make_block(ALE_FIELD, column)
        add_mesh_row(form, column, *block_proxies[ALE_FIELD, column])
    velocity_proxy = BilinearForm(discretisation.V, symmetric=True)
    velocity_proxy += (
        J_n * 2.0 * coefficients.mu_f * InnerProduct(Sym(Grad(u_proxy) * A_n), Sym(Grad(v_phys) * A_n))
    ).Compile() * dx("fluid")
    velocity_proxy += (
        dt
        * J_n
        * (
            2.0 * coefficients.mu_s * InnerProduct(Sym(Grad(u_proxy) * A_n), Sym(Grad(v_phys) * A_n))
            + settings.pc_options["velocity_mean_regularization"]
            * coefficients.solid_proxy_scale
            * InnerProduct(u_proxy, v_phys)
            + coefficients.solid_volumetric_proxy_scale * Trace(Grad(u_proxy) * A_n) * Trace(Grad(v_phys) * A_n)
        )
    ).Compile() * dx("solid")
    if material.eta_s or material.zeta_s:
        velocity_proxy += (
            J_n
            * (
                2.0 * coefficients.eta_s * InnerProduct(Sym(Grad(u_proxy) * A_n), Sym(Grad(v_phys) * A_n))
                + coefficients.zeta_s * Trace(Grad(u_proxy) * A_n) * Trace(Grad(v_phys) * A_n)
            )
        ).Compile() * dx("solid")
    pressure_mass = BilinearForm(discretisation.P_f)
    pressure_mass += (J_n * p_proxy * q_test).Compile() * dx("fluid")
    ps_proxy, qs_test = discretisation.P_s.TnT()
    solid_pressure_mass = BilinearForm(discretisation.P_s)
    solid_pressure_mass += (J_n * coefficients.solid_pressure_schur_scale * ps_proxy * qs_test).Compile() * dx("solid")
    mesh_pre_trial, mesh_pre_test = discretisation.W_delta.TnT()
    mesh_pre_grad_trial = Grad(mesh_pre_trial) * A_n
    mesh_pre_grad_test = Grad(mesh_pre_test) * A_n
    mesh_preconditioner_form = BilinearForm(discretisation.W_delta, symmetric=True)
    mesh_preconditioner_form += (
        dt
        * J_n
        * (
            2.0 * coefficients.mesh_mu * InnerProduct(Sym(mesh_pre_grad_trial), Sym(mesh_pre_grad_test))
            + coefficients.mesh_lame_lambda * Trace(mesh_pre_grad_trial) * Trace(mesh_pre_grad_test)
        )
    ).Compile() * dx("fluid")
    u_state = state_components[U_FIELD]
    delta_state = state_components[ALE_FIELD]
    F_A_s_state = solid_ale_map(u_state)
    F_A_f_state = fluid_ale_map(u_state, delta_state)
    A_s_state = Inv(F_A_s_state)
    A_f_state = Inv(F_A_f_state)
    J_s_state = Det(F_A_s_state)
    J_f_state = Det(F_A_f_state)
    hom_velocity_proxy = BilinearForm(discretisation.V, symmetric=True)
    hom_velocity_proxy += (
        J_f_state
        * 2.0
        * coefficients.mu_f
        * InnerProduct(Sym(Grad(u_proxy) * A_f_state), Sym(Grad(v_phys) * A_f_state))
    ).Compile() * dx("fluid")
    hom_velocity_proxy += (
        dt
        * J_s_state
        * (
            2.0 * coefficients.mu_s * InnerProduct(Sym(Grad(u_proxy) * A_s_state), Sym(Grad(v_phys) * A_s_state))
            + settings.pc_options["velocity_mean_regularization"]
            * coefficients.solid_proxy_scale
            * InnerProduct(u_proxy, v_phys)
            + coefficients.solid_volumetric_proxy_scale
            * Trace(Grad(u_proxy) * A_s_state)
            * Trace(Grad(v_phys) * A_s_state)
        )
    ).Compile() * dx("solid")
    if material.eta_s or material.zeta_s:
        hom_velocity_proxy += (
            J_s_state
            * (
                2.0 * coefficients.eta_s * InnerProduct(Sym(Grad(u_proxy) * A_s_state), Sym(Grad(v_phys) * A_s_state))
                + coefficients.zeta_s * Trace(Grad(u_proxy) * A_s_state) * Trace(Grad(v_phys) * A_s_state)
            )
        ).Compile() * dx("solid")
    hom_pressure_mass = BilinearForm(discretisation.P_f)
    hom_pressure_mass += (J_f_state * p_proxy * q_test).Compile() * dx("fluid")
    hom_solid_pressure_mass = BilinearForm(discretisation.P_s)
    hom_solid_pressure_mass += (
        J_s_state * coefficients.solid_pressure_schur_scale * ps_proxy * qs_test
    ).Compile() * dx("solid")
    mesh_state_jacobian = J_n * Det(F_A_f_state * A_n)
    hom_mesh_pre_trial, hom_mesh_pre_test = discretisation.W_delta.TnT()
    hom_mesh_metric_trial = Grad(hom_mesh_pre_trial) * A_n
    hom_mesh_metric_test = Grad(hom_mesh_pre_test) * A_n
    hom_mesh_proxy = BilinearForm(discretisation.W_delta, symmetric=True)
    hom_mesh_proxy += (
        dt
        * dt
        * mesh_state_jacobian
        * (
            2.0 * coefficients.mesh_mu * InnerProduct(Sym(hom_mesh_metric_trial), Sym(hom_mesh_metric_test))
            + coefficients.mesh_lame_lambda * Trace(hom_mesh_metric_trial) * Trace(hom_mesh_metric_test)
        )
    ).Compile() * dx("fluid")
    residual_row_specs = (
        (U_FIELD, U_FIELD, add_physical_row),
        (PF_FIELD, U_FIELD, add_pressure_row),
        (PS_FIELD, PS_FIELD, add_solid_pressure_row),
        (LAMBDA_U_FIELD, U_FIELD, add_velocity_constraint_row),
        (LAMBDA_PF_FIELD, U_FIELD, add_pressure_constraint_row),
        (ALE_FIELD, ALE_FIELD, add_mesh_row),
    )
    residual_forms = []
    for row, column, add_row in residual_row_specs:
        test_proxy = test_spaces[row].TnT()[1]
        form = LinearForm(test_spaces[row])
        add_row(form, column, state_components[column], test_proxy)
        residual_forms.append(form)

    def make_load_derivative_forms():
        """Build sensitivities only when a tangent is actually requested.

        Newton-only and finite-difference output trials never need these
        forms. A session builds them once, then reassembles on later states.
        """
        forms = []
        for parameter_index, parameter in enumerate(load_derivative_parameters):
            derivative_rows = []
            for row, column, add_row in residual_row_specs:
                form = None
                if parameter_index < 9 or row == U_FIELD:
                    test_proxy = test_spaces[row].TnT()[1]
                    form = LinearForm(test_spaces[row])
                    add_row(
                        form, column, state_components[column], test_proxy, derivative_parameter=parameter
                    )
                derivative_rows.append(form)
            forms.append(tuple(derivative_rows))
        return tuple(forms)

    system = BlockFSIALESystem(
        block_forms,
        residual_forms,
        pressure_mass,
        solid_pressure_mass,
        velocity_proxy,
        mesh_preconditioner_form,
        state_components,
        make_load_derivative_forms,
        hom_velocity_proxy,
        hom_pressure_mass,
        hom_solid_pressure_mass,
        hom_mesh_proxy,
        mesh=discretisation.mesh,
        field_spaces=discretisation.field_spaces,
        block_layout=discretisation.block_layout,
        profiler=settings.profiler,
    )
    # Residual metric: the dual norm of the same four proxies on the reference configuration
    # (J = 1, A = I). It does not change between timesteps, so its BDDC inverses are factorised
    # once per cell (and again after a dt or material change) instead of in every timestep.
    metric_velocity = BilinearForm(discretisation.V, symmetric=True)
    metric_velocity += (
        2.0 * coefficients.mu_f * InnerProduct(Sym(Grad(u_proxy)), Sym(Grad(v_phys)))
    ).Compile() * dx("fluid")
    metric_velocity += (
        dt
        * (
            2.0 * coefficients.mu_s * InnerProduct(Sym(Grad(u_proxy)), Sym(Grad(v_phys)))
            + settings.pc_options["velocity_mean_regularization"]
            * coefficients.solid_proxy_scale
            * InnerProduct(u_proxy, v_phys)
            + coefficients.solid_volumetric_proxy_scale * Trace(Grad(u_proxy)) * Trace(Grad(v_phys))
        )
    ).Compile() * dx("solid")
    if material.eta_s or material.zeta_s:
        metric_velocity += (
            2.0 * coefficients.eta_s * InnerProduct(Sym(Grad(u_proxy)), Sym(Grad(v_phys)))
            + coefficients.zeta_s * Trace(Grad(u_proxy)) * Trace(Grad(v_phys))
        ).Compile() * dx("solid")
    metric_pressure = BilinearForm(discretisation.P_f)
    metric_pressure += (p_proxy * q_test).Compile() * dx("fluid")
    metric_solid_pressure = BilinearForm(discretisation.P_s)
    metric_solid_pressure += (coefficients.solid_pressure_schur_scale * ps_proxy * qs_test).Compile() * dx("solid")
    metric_mesh = BilinearForm(discretisation.W_delta, symmetric=True)
    metric_mesh += (
        dt
        * (
            2.0 * coefficients.mesh_mu * InnerProduct(Sym(Grad(mesh_pre_trial)), Sym(Grad(mesh_pre_test)))
            + coefficients.mesh_lame_lambda * Trace(Grad(mesh_pre_trial)) * Trace(Grad(mesh_pre_test))
        )
    ).Compile() * dx("fluid")
    preconditioner = BlockFSIALEPreconditioner(
        system,
        pressure_inverse_scale=settings.pc_options["pressure_mass_inverse_scale"],
        velocity_preconditioner=_make_fsi_bddc(metric_velocity, mean_nullspace=True),
        pressure_preconditioner=make_pressure_preconditioner(metric_pressure),
        solid_pressure_preconditioner=make_pressure_preconditioner(metric_solid_pressure),
        mesh_preconditioner=_make_fsi_bddc(metric_mesh),
        metric_forms=(metric_velocity, metric_pressure, metric_solid_pressure, metric_mesh),
        mesh_inverse_scale=1.0 / max(abs(dt), 1e-30),
        order_aware=True,
        adaptive_iteration_threshold=(
            min(int(settings.gmres_maxsteps), max(10, int(0.25 * settings.gmres_maxsteps))) if discretisation.order >= 3 else 80
        ),
        order=discretisation.order,
        profiler=settings.profiler,
    )
    return (system, preconditioner)
