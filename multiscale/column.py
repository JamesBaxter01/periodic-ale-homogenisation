"""Column1D: confined one-dimensional consolidation, the macro problem of the first FE² example.

A column of height H along Z (tensor direction 3) and cross-section area A,
laterally confined (F_11 = F_22 = 1, no shear), fixed and sealed at the
bottom, loaded by a compressive traction and drained (p = 0) at the top.
Displacement and pressure use Lagrange elements of any order on
Gauss–Lobatto nodes (default cubic / quadratic), integrated by
``gauss_points`` Gauss points per element: one material point each.
Elements are graded geometrically towards the drained top: ``grading`` is
the length of the bottom element over that of the top one (1 = uniform).

Unknowns y = [u (n_u), p (n_p)]. Residual, by row block:

    R_u = int B_u P_33 A dZ + traction(t) A e_top
    R_p = int [N_p (m - m_old) / dt - B_p Q_3] A dZ

The pressure row of the drained top node is not solved; minus its residual
is the outflow rate there (the conservative reaction).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre


def lobatto_nodes(order):
    """Gauss–Lobatto nodes on [-1, 1]: the ends and the roots of P_order'."""
    if order < 1:
        raise ValueError("element order must be at least 1")
    inner = legendre.Legendre.basis(order).deriv().roots() if order > 1 else []
    return np.r_[-1.0, np.sort(np.real(inner)), 1.0]


def lagrange(nodes, xi):
    """Values and xi-derivatives of the Lagrange basis on ``nodes`` at the points ``xi``."""
    coefficients = np.linalg.inv(np.vander(nodes, increasing=True))  # column i: basis function i
    powers = np.vander(np.atleast_1d(xi), len(nodes), increasing=True)
    d_powers = np.zeros_like(powers)
    d_powers[:, 1:] = powers[:, :-1] * np.arange(1, len(nodes))
    return powers @ coefficients, d_powers @ coefficients


class Column1D:
    def __init__(self, *, height, area, elements, traction, stress_scale, modulus, displacement_order=3,
                 pressure_order=2, gauss_points=4, grading=1.0):
        self.height = float(height)
        self.area = float(area)
        self.elements = int(elements)
        self.traction = traction  # callable t -> compressive traction (Pa)
        # Residual scales: the load (Pa) and the content change it causes (stress_scale / modulus).
        self.stress_scale = float(stress_scale)
        self.modulus = float(modulus)
        self.ku, self.kp = int(displacement_order), int(pressure_order)
        self.n_u = self.elements * self.ku + 1
        self.n_p = self.elements * self.kp + 1
        self.size = self.n_u + self.n_p
        if not grading >= 1.0:
            raise ValueError("grading (bottom over top element length) must be at least 1")
        self.grading = float(grading)
        # Element lengths shrink by a constant ratio from the bottom to the top.
        ratio = self.grading ** (-1.0 / max(self.elements - 1, 1))
        lengths = ratio ** np.arange(self.elements)
        self.edges = np.r_[0.0, np.cumsum(lengths)] * self.height / lengths.sum()
        self.edges[-1] = self.height
        self.lengths = np.diff(self.edges)
        xi, weights = legendre.leggauss(int(gauss_points))
        Nu, dNu = lagrange(lobatto_nodes(self.ku), xi)
        Np, dNp = lagrange(lobatto_nodes(self.kp), xi)
        # Per material point (element-major): its element's dofs, shape values and weight.
        self.u_dofs, self.p_dofs, self.Z, self.weights = [], [], [], []
        self.Nu, self.Bu, self.Np, self.Bp = [], [], [], []
        for e in range(self.elements):
            h = self.lengths[e]
            for q in range(len(xi)):
                self.u_dofs.append(e * self.ku + np.arange(self.ku + 1))
                self.p_dofs.append(self.n_u + e * self.kp + np.arange(self.kp + 1))
                self.Z.append(self.edges[e] + (xi[q] + 1.0) / 2.0 * h)
                self.weights.append(weights[q] * h / 2.0 * self.area)
                self.Nu.append(Nu[q])
                self.Bu.append(dNu[q] * 2.0 / h)
                self.Np.append(Np[q])
                self.Bp.append(dNp[q] * 2.0 / h)
        for name in ("u_dofs", "p_dofs", "Z", "weights", "Nu", "Bu", "Np", "Bp"):
            setattr(self, name, np.array(getattr(self, name)))
        self.n_points = len(self.weights)
        self.bottom_u = 0
        self.top_u = self.n_u - 1
        self.top_p = self.size - 1
        self.free = np.setdiff1d(np.arange(self.size), [self.bottom_u, self.top_p])
        self.node_Z_u = self._node_positions(self.ku)
        self.node_Z_p = self._node_positions(self.kp)

    def _node_positions(self, order):
        local = (lobatto_nodes(order) + 1.0) / 2.0
        return np.r_[np.concatenate([self.edges[e] + local[:-1] * self.lengths[e] for e in range(self.elements)]),
                     self.height]

    def row_scale(self, dt):
        """Residual row scales: momentum rows by the load on the area, mass rows by the volume
        change rate the load causes in one element of mean length."""
        h = self.height / self.elements
        return np.r_[np.full(self.n_u, self.stress_scale * self.area),
                     np.full(self.n_p, self.area * h * self.stress_scale / self.modulus / dt)]

    def error_scales(self, rest_tangent):
        """Sizes for the time-step error estimate: the load, the strain and content change it
        causes, the settlement, and the flux the load drives across one element of mean length
        with the permeability of the cells at rest (-dQ_3/dG_3)."""
        strain = self.stress_scale / self.modulus
        permeability = float(np.max(-rest_tangent[:, 12, 12]))
        h = self.height / self.elements
        return dict(stress=self.stress_scale, p=self.stress_scale, content=strain, u=strain * self.height,
                    flux=permeability * self.stress_scale / h)

    def summary(self, y, residual=None):
        p = self.pressure(y)
        return dict(settlement=-float(y[self.top_u]), max_pressure=float(np.max(np.abs(p))),
                    base_pressure=float(p[0]))

    def point_coordinates(self):
        return dict(Z_points=self.Z)

    def initial_state(self):
        return np.zeros(self.size)

    def apply_boundary_conditions(self, y, t=0.0):
        y = y.copy()
        y[self.bottom_u] = 0.0
        y[self.top_p] = 0.0
        return y

    def displacement(self, y):
        return y[: self.n_u]

    def pressure(self, y):
        return y[self.n_u:]

    def point_inputs(self, y):
        """The 13 cell inputs at every material point, and the axial stretch and pressure gradient."""
        u = np.einsum("qa,qa->q", self.Bu, y[self.u_dofs])
        p = np.einsum("qa,qa->q", self.Np, y[self.p_dofs])
        dp = np.einsum("qa,qa->q", self.Bp, y[self.p_dofs])
        stretch = 1.0 + u
        inputs = np.zeros((self.n_points, 13))
        inputs[:, 0] = inputs[:, 4] = 1.0
        inputs[:, 8] = stretch
        inputs[:, 9] = p
        inputs[:, 12] = dp / stretch  # spatial gradient G = F^-T grad_X p
        return inputs

    def residual(self, y, outputs, old_content, dt, t):
        R = np.zeros(self.size)
        momentum = self.weights[:, None] * self.Bu * outputs[:, 8, None]
        mass = self.weights[:, None] * (self.Np * ((outputs[:, 9] - old_content) / dt)[:, None]
                                        - self.Bp * outputs[:, 12, None])
        np.add.at(R, self.u_dofs, momentum)
        np.add.at(R, self.p_dofs, mass)
        R[self.top_u] += self.traction(t) * self.area
        return R

    def tangent(self, inputs, tangents, dt):
        """dR/dy (dense): test maps ^T C input maps, summed over the material points."""
        K = np.zeros((self.size, self.size))
        rows = (8, 9, 12)  # P_33, fluid content, flux_3
        for q in range(self.n_points):
            dofs = np.r_[self.u_dofs[q], self.p_dofs[q]]
            nu = len(self.u_dofs[q])
            stretch, G3 = inputs[q, 8], inputs[q, 12]
            d_inputs = np.zeros((3, len(dofs)))  # d(F_33, P0, G_3)/d(local dofs)
            d_inputs[0, :nu] = self.Bu[q]
            d_inputs[1, nu:] = self.Np[q]
            d_inputs[2, :nu] = -G3 / stretch * self.Bu[q]
            d_inputs[2, nu:] = self.Bp[q] / stretch
            test = np.zeros((3, len(dofs)))  # residual rows per unit output
            test[0, :nu] = self.Bu[q]
            test[1, nu:] = self.Np[q] / dt
            test[2, nu:] = -self.Bp[q]
            C = tangents[q][np.ix_(rows, rows)]
            K[np.ix_(dofs, dofs)] += self.weights[q] * test.T @ C @ d_inputs
        return K

    def outflow_rate(self, R):
        """Fluid volume leaving through the drained top per unit time (m^3/s)."""
        return -R[self.top_p]

    def fluid_volume(self, outputs):
        return float(self.weights @ outputs[:, 9])
