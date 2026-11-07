"""ResolvedColumn: the fully resolved microstructure of a confined consolidation column.

A stack of ``cells`` square-channel cells (geometry.square_column), periodic
in x and y, solved with the periodic cell solver's own field equations
(periodic_fsi.formulation: momentum_terms, continuity_terms,
solid_pressure_integrand, mesh_motion_integrand), fields, mesher, ALE
parameterisation and nondimensionalisation. The macro loads and the cell's
pressure-mean multiplier are absent; boundary conditions take their place:

- bottom (z = 0): a symmetry plane, impermeable: zero normal velocity of the
  solid, the fluid and the mesh, sideways motion free. z = 0 is a mirror
  plane of the square-channel microstructure, so the column is exactly half
  of a column of 2 x cells drained at both ends (Terzaghi's double-drainage
  equivalence), with no artificial boundary layer at the base. (A clamped,
  no-slip base leaves an almost sealed fluid element at each corner of the
  channel's bottom opening whose pressure cannot drain.)
- sideways: periodic. Base and platen allow sliding, so the rigid sideways
  translation is removed as in the periodic cell (its lambda_u): the mean
  sideways velocity of the solid is zero, by two Lagrange multipliers.
- top (z = cells): a rigid, porous, frictionless platen. The solid's top face
  moves with one platen velocity V_p (normal component only; it slides
  freely sideways) and carries the applied force. ``platen="bonded"`` also
  fixes the face sideways (no slip), as the earlier resolved bundle did. The fluid drains freely
  through it with zero traction, and the fluid mesh's top face follows the
  platen.

Fields per step, nondimensional as in the cell (lengths by the cell length,
stresses by the solid shear modulus, time by the step dt, so dt = 1 here):
fluctuation-free velocity u, fluid pressure p_f, solid pressure p_s, relative
mesh velocity delta in the fluid (mesh velocity u + delta; zero on the
solid--fluid interface).

The platen conditions are eliminated exactly by a trial map T and a test map
S on the assembled system (as in the earlier resolved bundle): the top
solid-face normal velocities are V_p times the interpolated constant; the
top fluid-face normal mesh velocities are V_p minus the fluid velocity there;
the platen's test is the virtual rigid motion of the solid top face, and
the eliminated mesh rows are dropped.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ngsolve import (
    NumberSpace,
    VectorValued,
    BND,
    CF,
    VOL,
    BilinearForm,
    FESpace,
    GridFunction,
    H1,
    Id,
    IfPos,
    Integrate,
    IntegrationRule,
    Inv,
    L2,
    MatrixValued,
    Periodic,
    TaskManager,
    VectorH1,
    Det,
    Grad,
    InnerProduct,
    dx,
    z,
)

from periodic_fsi.cell_data import Material
from periodic_fsi.core import generate_cell_mesh
from periodic_fsi.formulation import (
    MaterialParameters,
    continuity_terms,
    mesh_motion_integrand,
    momentum_terms,
    solid_pressure_integrand,
)
from periodic_fsi.newton import AdmissibilityCheck

BOTTOM = "zmin_solid|zmin_fluid"
U, PF, PS, DELTA, LAMBDA = range(5)


def _parallel(method):
    """Run an NGSolve-heavy method (assembly, Set, Integrate) under TaskManager, with the threads of
    SetNumThreads. Only these: the MUMPS solve between them must run without it, because idle
    TaskManager threads keep spinning and starve the factorisation's OpenMP threads (on Aire a step
    took 3-5 times longer with the whole time loop inside one TaskManager). Nesting is harmless."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with TaskManager():
            return method(*args, **kwargs)

    return wrapper


def _indices(mask):
    return np.flatnonzero(np.asarray(list(mask), dtype=bool))


class ResolvedColumn:
    """Spaces, forms, platen maps and state of one resolved column (no time loop)."""

    @_parallel
    def __init__(self, geometry, *, cells, E, nu, mu_f, eta_s=0.0, zeta_s=0.0, maxh, order=2,
                 quadrature_order=None, mesh_mu=0.01, mesh_lame_lambda=0.05, check_order=6, mesh=None,
                 platen="frictionless"):
        if platen not in ("frictionless", "bonded"):
            raise ValueError("platen must be 'frictionless' or 'bonded'")
        self.platen = platen
        self.cells = int(cells)
        self.order = int(order)
        self.E, self.nu = float(E), float(nu)
        self.mu_s = self.E / (2.0 * (1.0 + self.nu))
        self.physical = dict(mu_f=float(mu_f), eta_s=float(eta_s), zeta_s=float(zeta_s))
        # Mesh motion as the cell solver scales it (driver._dimensional_FSI): mu = 1, lambda/mu kept.
        self.mesh_law = dict(mesh_mu=1.0, mesh_lame_lambda=float(mesh_lame_lambda) / float(mesh_mu))
        if quadrature_order is None:
            quadrature_order = max(2 * order, 4 * order - 3)  # the cell solver's default
        if mesh is None:
            mesh = generate_cell_mesh(geometry, maxh, order)  # the cell solver's mesher
        self.mesh = mesh
        rules = {t: IntegrationRule(t, order=int(quadrature_order)) for t in {el.type for el in mesh.Elements(VOL)}}
        self.rules = rules

        # Spaces: the cell's, periodic in x and y only; the base is a symmetry plane (normal
        # components fixed); two multipliers fix the mean sideways solid velocity.
        self.V_base = VectorH1(mesh, order=order, dirichletz=BOTTOM)
        self.W_base = VectorH1(mesh, order=order, dirichlet="interface", dirichletz="zmin_fluid", definedon="fluid")
        V = Periodic(self.V_base)
        P_f = Periodic(H1(mesh, order=order - 1, definedon="fluid"))
        P_s = Periodic(H1(mesh, order=order - 1, definedon="solid"))
        W = Periodic(self.W_base)
        R = VectorValued(NumberSpace(mesh), dim=2)
        self.spaces = (V, P_f, P_s, W, R)
        self.X = X = FESpace([V, P_f, P_s, W, R])
        self.state = GridFunction(X, name="resolved_state")
        self.displacement = GridFunction(VectorH1(mesh, order=order), name="mesh_displacement")
        self.displacement_n = GridFunction(self.displacement.space)
        # The full 3 x 3 space (MatrixValued(space, 3, 3) would be the symmetric one, dropping the
        # solid's local rotations from its F history).
        F_space = MatrixValued(L2(mesh, order=order, definedon="solid"), 3)
        self.F_old = GridFunction(F_space)
        self.F_old.Set(Id(3))
        self.F_n = GridFunction(F_space)
        self.F_n.Set(Id(3))
        self.F_new = GridFunction(F_space)

        # Material: nondimensional like the cell; the viscosities carry 1/dt (set_step).
        self.material = Material.from_elastic(self.E / self.mu_s, self.nu, eta_s=0.0, zeta_s=0.0, mu_f=1.0,
                                              **self.mesh_law)
        self.viscous_solid = bool(eta_s or zeta_s)
        self.coefficients = MaterialParameters(self.material, dt=1.0)
        self.step_dt = None
        self._build_forms()
        self._build_maps()
        self.admissibility = AdmissibilityCheck(mesh, check_order, 1e-14 * self.cells)
        self.platen_displacement = 0.0  # nondimensional, cumulative
        self.reference_fluid_volume = float(Integrate(1, mesh, definedon=mesh.Materials("fluid"), order=4))

    # ----- forms ---------------------------------------------------------------------------
    def _build_forms(self):
        X, coefficients = self.X, self.coefficients
        u, pf, ps, delta, lam = X.TrialFunction()
        v, qf, qs, w, mu = X.TestFunction()
        F_A_n = Id(3) + Grad(self.displacement_n)
        A_n = Inv(F_A_n)
        J_n = Det(F_A_n)
        history = A_n * self.F_n
        zero_rate = CF((0,) * 9, dims=(3, 3))
        material = self.material if not self.viscous_solid else _Viscous()
        solid_piola, J_s, fluid_integrand = momentum_terms(
            coefficients, material, velocity=u, fluid_pressure=pf, solid_pressure=ps, relative_mesh_velocity=delta,
            test=v, F_A_n=F_A_n, solid_history_jacobian=Det(history), solid_history_metric=history * history.trans,
            E_dot=zero_rate, P0=0.0, G=None, dt=1.0,
        )
        continuity, _ = continuity_terms(velocity=u, relative_mesh_velocity=delta, test=qf, F_A_n=F_A_n,
                                         E_dot=zero_rate, dt=1.0)
        volumetric = solid_pressure_integrand(coefficients, velocity=u, solid_pressure=ps, test=qs, F_A_n=F_A_n,
                                              solid_history_jacobian=Det(history), E_dot=zero_rate, dt=1.0)
        mesh = mesh_motion_integrand(coefficients, velocity=u, relative_mesh_velocity=delta, test=w,
                                     F_A_n=F_A_n, A_n=A_n, J_n=J_n, E_dot=zero_rate, dt=1.0)
        a = BilinearForm(X, symmetric=False)
        # Mean sideways solid velocity zero (the cell's velocity_constraint_row, x and y only).
        sideways = J_s * (lam[0] * v[0] + lam[1] * v[1] + mu[0] * u[0] + mu[1] * u[1])
        a += (InnerProduct(solid_piola, Grad(v)) + sideways + volumetric).Compile() * dx("solid", intrules=self.rules)
        a += (fluid_integrand + continuity + mesh).Compile() * dx("fluid", intrules=self.rules)
        self.form = a
        self.residual_vector = self.state.vec.CreateVector()

    @_parallel
    def set_step(self, dt):
        """Nondimensionalise for a step of physical length dt (s): viscosities / (mu_s dt)."""
        self.step_dt = float(dt)
        scale = self.mu_s * self.step_dt
        self.material = Material.from_elastic(
            self.E / self.mu_s, self.nu, eta_s=self.physical["eta_s"] / scale,
            zeta_s=self.physical["zeta_s"] / scale, mu_f=self.physical["mu_f"] / scale, **self.mesh_law)
        self.coefficients.set(self.material)

    # ----- platen elimination ---------------------------------------------------------------
    def _build_maps(self):
        X, mesh = self.X, self.mesh
        V, P_f, P_s, W, _ = self.spaces
        offsets = [X.Range(i).start for i in range(5)]
        self.offsets = offsets
        z_u = set(range(self.V_base.Range(2).start, self.V_base.Range(2).stop))
        z_w = set(range(self.W_base.Range(2).start, self.W_base.Range(2).stop))
        # Periodic() keeps whole-vector Dirichlet flags but drops component ones (dirichletz), so a
        # DOF is free only if the periodic space and its base space both say so.
        free_u = set(_indices(V.FreeDofs())) & set(_indices(self.V_base.FreeDofs()))
        free_w = set(_indices(W.FreeDofs())) & set(_indices(self.W_base.FreeDofs()))
        top_face = set(_indices(V.GetDofs(mesh.Boundaries("zmax_solid"))))
        top_solid = (top_face & z_u) & free_u
        if self.platen == "bonded":  # no slip under the platen: the solid's top face moves vertically only
            free_u -= top_face - z_u
        # Fluid top face: pairs (W dof, V dof) of the same normal basis function.
        pairs = {}
        for el in mesh.Elements(BND):
            if el.mat != "zmax_fluid":
                continue
            for j, i in zip(W.GetDofNrs(el), V.GetDofNrs(el)):
                if j in z_w and j in free_w and i in z_u:
                    pairs[j] = i
        # The interpolated constant e_z: vertex coefficients one, higher-order ones zero.
        lift_u = GridFunction(V)
        lift_u.Set(CF((0, 0, 1)))
        lift_w = GridFunction(W)
        lift_w.Set(CF((0, 0, 1)))
        lift_u, lift_w = lift_u.vec.FV().NumPy().copy(), lift_w.vec.FV().NumPy().copy()

        independent_u = sorted(free_u - top_solid)
        independent_w = sorted(free_w - set(pairs))
        independent_pf = sorted(_indices(P_f.FreeDofs()))
        independent_ps = sorted(_indices(P_s.FreeDofs()))
        column_u = {dof: k for k, dof in enumerate(independent_u)}
        plate = len(independent_u)
        start_pf = plate + 1
        start_ps = start_pf + len(independent_pf)
        start_w = start_ps + len(independent_ps)
        start_lambda = start_w + len(independent_w)
        self.size = start_lambda + 2
        self.plate = plate
        self.blocks = dict(u=slice(0, plate), platen=slice(plate, plate + 1), p_f=slice(start_pf, start_ps),
                           p_s=slice(start_ps, start_w), delta=slice(start_w, start_lambda),
                           lambda_=slice(start_lambda, self.size))
        trial, test = [], []  # (full row, reduced column, value)
        for k, dof in enumerate(independent_u):
            trial.append((offsets[U] + dof, k, 1.0))
            test.append((offsets[U] + dof, k, 1.0))
        for dof in top_solid:
            trial.append((offsets[U] + dof, plate, lift_u[dof]))
            test.append((offsets[U] + dof, plate, lift_u[dof]))  # virtual rigid platen motion
        for k, dof in enumerate(independent_pf):
            trial.append((offsets[PF] + dof, start_pf + k, 1.0))
            test.append((offsets[PF] + dof, start_pf + k, 1.0))
        for k, dof in enumerate(independent_ps):
            trial.append((offsets[PS] + dof, start_ps + k, 1.0))
            test.append((offsets[PS] + dof, start_ps + k, 1.0))
        for k, dof in enumerate(independent_w):
            trial.append((offsets[DELTA] + dof, start_w + k, 1.0))
            test.append((offsets[DELTA] + dof, start_w + k, 1.0))
        for k in range(2):
            trial.append((offsets[LAMBDA] + k, start_lambda + k, 1.0))
            test.append((offsets[LAMBDA] + k, start_lambda + k, 1.0))
        # delta_z = V_p lift - u_z on the fluid top face (mesh follows the platen); no test rows.
        for j, i in pairs.items():
            trial.append((offsets[DELTA] + j, plate, lift_w[j]))
            if i in column_u:
                trial.append((offsets[DELTA] + j, column_u[i], -1.0))
            elif i in top_solid:
                trial.append((offsets[DELTA] + j, plate, -lift_u[i]))
        n = X.ndof
        make = lambda e: coo_matrix(([v for _, _, v in e], ([r for r, _, _ in e], [c for _, c, _ in e])),  # noqa: E731
                                    shape=(n, self.size)).tocsr()
        self.T, self.S = make(trial), make(test)
        self.top_fluid_pairs = len(pairs)

    # ----- residual, tangent, state ---------------------------------------------------------
    def set_reduced(self, y):
        self.state.vec.FV().NumPy()[:] = self.T @ y

    @_parallel
    def residual(self, y, load):
        """Reduced residual for the nondimensional platen force ``load`` (compressive positive)."""
        self.set_reduced(y)
        self.form.Apply(self.state.vec, self.residual_vector)
        r = np.asarray(self.S.T @ self.residual_vector.FV().NumPy()).ravel()
        r[self.plate] += load
        return r

    @_parallel
    def tangent(self, y):
        self.set_reduced(y)
        self.form.AssembleLinearization(self.state.vec)
        values, columns, rows = self.form.mat.CSR()
        n = self.X.ndof
        full = csr_matrix((np.asarray(values), np.asarray(columns), np.asarray(rows)), shape=(n, n))
        return (self.S.T @ full @ self.T).tocsr()

    def fields(self):
        return self.state.components

    def trial_maps(self):
        """The solid and fluid ALE maps of the current state (E_dot = 0, dt = 1)."""
        u, _, _, delta, _ = self.fields()
        F_A_n = Id(3) + Grad(self.displacement_n)
        return F_A_n + Grad(u), F_A_n + Grad(u) + Grad(delta)

    @_parallel
    def admissible(self, minimum=1e-8):
        """Positive solid and fluid ALE Jacobians on the reference mesh for the current state."""
        F_s, F_f = self.trial_maps()
        return (self.admissibility.jacobian(F_s, minimum, definedon=self.mesh.Materials("solid"))
                and self.admissibility.jacobian(F_f, minimum, definedon=self.mesh.Materials("fluid")))

    @_parallel
    def begin_step(self):
        self.displacement_n.vec.data = self.displacement.vec
        self.F_n.vec.data = self.F_old.vec

    @_parallel
    def accept(self, y, minimum=1e-8):
        """Update the mesh displacement and solid F after a converged step (as CellProblem.advance)."""
        self.set_reduced(y)
        u, _, _, delta, _ = self.fields()
        F_s, _ = self.trial_maps()
        F_A_n = Id(3) + Grad(self.displacement_n)
        self.F_new.Set(F_s * Inv(F_A_n) * self.F_n, definedon=self.mesh.Materials("solid"))
        if not self.admissibility.jacobian(self.F_new, minimum, definedon=self.mesh.Materials("solid")):
            raise RuntimeError("converged state has a non-admissible solid Jacobian")
        candidate = GridFunction(self.displacement.space)
        old = CF(tuple(self.displacement_n.components))
        candidate.Set(self.mesh.MaterialCF({"solid": old + u, "fluid": old + u + delta}))
        if not self.admissibility.total_mesh(candidate, minimum):
            raise RuntimeError("converged state has a non-admissible mesh Jacobian")
        self.displacement.vec.data = candidate.vec
        self.F_old.vec.data = self.F_new.vec
        self.platen_displacement += float(y[self.plate])

    # ----- recorded quantities (physical units need cell_length) ----------------------------
    @_parallel
    def summary(self, cell_length):
        """Settlement, fluid volume, nodal max pressure and per-cell averages, physical units."""
        mesh = self.mesh
        _, pf, _, _, _ = self.fields()
        J = Det(Id(3) + Grad(self.displacement))
        fluid = mesh.Materials("fluid")
        rule = dict(definedon=fluid, order=4)
        cell_pressure, cell_content, cell_displacement = [], [], []
        solid = mesh.Materials("solid")
        for k in range(self.cells):
            inside = IfPos(z - k, IfPos(k + 1 - z, 1.0, 0.0), 0.0)
            volume = float(Integrate(inside * J, mesh, **rule))
            cell_content.append(volume)  # fluid volume per reference cell volume (= 1)
            cell_pressure.append(self.mu_s * float(Integrate(inside * J * pf, mesh, **rule)) / volume)
            solid_volume = float(Integrate(inside, mesh, definedon=solid, order=4))
            cell_displacement.append(cell_length * float(Integrate(inside * self.displacement[2], mesh,
                                                                   definedon=solid, order=4)) / solid_volume)
        pressures = pf.vec.FV().NumPy()  # order-1 H1: vertex values
        # max_pressure is the largest cell-averaged pressure: the nodal maximum is spoiled by the
        # four corners of the channel's sealed bottom opening, where the fixed bottom meets the
        # solid wall and the corner pressure is almost unconstrained (it keeps an arbitrary value
        # after drainage; velocities and cell averages are unaffected). It is kept as a diagnostic.
        return dict(
            settlement=-cell_length * self.platen_displacement,
            fluid_volume=cell_length**3 * float(sum(cell_content)),
            max_pressure=float(np.max(np.abs(cell_pressure))),
            nodal_max_pressure=self.mu_s * float(np.max(np.abs(pressures))) if len(pressures) else 0.0,
            cell_pressure=np.array(cell_pressure),
            cell_content=np.array(cell_content),
            cell_displacement=np.array(cell_displacement),
        )


class _Viscous:
    """momentum_terms only reads eta_s/zeta_s from ``material`` to decide whether the viscous terms exist."""

    eta_s = zeta_s = 1.0
