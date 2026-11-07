"""LinearPoroelastic: a closed-form Biot law in the cell's 13-input layout.

For testing the macro code without NGSolve. Small-strain Biot written in F:

    eps = sym(F - I)
    P   = lam tr(eps) I + 2 G eps - alpha P0 I
    m   = phi0 + alpha tr(eps) + P0 / M
    Q   = -kappa G_spatial            (kappa = permeability / fluid viscosity)

It has no history, so ``commit`` only counts steps. In 1-D confined
consolidation it gives Terzaghi's problem with
c_v = kappa / (1/M + alpha^2 / (lam + 2 G)).
"""

from __future__ import annotations

import numpy as np

from .interface import Response, check_inputs

_I = np.eye(3)


class LinearPoroelastic:
    def __init__(self, n_points, *, K, G, kappa, alpha=1.0, biot_modulus=np.inf, porosity=0.3):
        self.n_points = int(n_points)
        self.lam = K - 2.0 * G / 3.0
        self.G = float(G)
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.inverse_M = 0.0 if np.isinf(biot_modulus) else 1.0 / biot_modulus
        self.porosity = float(porosity)
        self.commits = 0
        self._tangent = self._constant_tangent()

    def consolidation_coefficient(self):
        return self.kappa / (self.inverse_M + self.alpha**2 / (self.lam + 2.0 * self.G))

    def _constant_tangent(self):
        C = np.zeros((13, 13))
        for a in range(9):
            i, j = divmod(a, 3)
            for b in range(9):
                k, l = divmod(b, 3)
                # d P_ij / d F_kl of lam tr(eps) I + 2G sym(F - I)
                C[a, b] = self.lam * _I[i, j] * _I[k, l] + self.G * (_I[i, k] * _I[j, l] + _I[i, l] * _I[j, k])
            C[a, 9] = -self.alpha * _I[i, j]
        C[9, :9] = self.alpha * _I.reshape(9)
        C[9, 9] = self.inverse_M
        C[10:, 10:] = -self.kappa * _I
        return C

    def evaluate(self, inputs, dt, *, tangent=True):
        x = check_inputs(inputs, self.n_points)
        eps = x[:, :9].reshape(-1, 3, 3) - _I
        eps = 0.5 * (eps + eps.transpose(0, 2, 1))
        tr = np.trace(eps, axis1=1, axis2=2)
        p = x[:, 9]
        P = self.lam * tr[:, None, None] * _I + 2.0 * self.G * eps - self.alpha * p[:, None, None] * _I
        y = np.empty((self.n_points, 13))
        y[:, :9] = P.reshape(-1, 9)
        y[:, 9] = self.porosity + self.alpha * tr + self.inverse_M * p
        y[:, 10:] = -self.kappa * x[:, 10:]
        C = np.broadcast_to(self._tangent, (self.n_points, 13, 13)).copy() if tangent else None
        return Response(y, C)

    def commit(self):
        self.commits += 1

    def save_checkpoint(self, folder):  # no history to keep
        pass

    def load_checkpoint(self, folder):
        pass

    def close(self):
        pass
