"""FiniteStrainPoroelastic: a closed-form finite-strain Biot law in the cell's 13-input layout.

A single-scale reference for the macro problems at finite strain (mesh
convergence, a continuum baseline), not a verification model:
LinearPoroelastic stays the one with closed-form solutions. Compressible
neo-Hookean skeleton, Biot coupling and Darcy flow pulled back to the
reference configuration:

    P = mu F + (lam ln J - mu) F^-T - alpha p J F^-T      (first Piola, total)
    m = phi0 + alpha (J - 1) + p / M                     (content per reference volume)
    Q = -kappa J F^-1 G                                  (nominal flux; G = F^-T grad_X p)

Frame-indifferent (a rigid rotation gives no stress), with no history; for
small strains it reduces to LinearPoroelastic with the same lam, mu (= G),
alpha, M and kappa. The tangent is exact.
"""

from __future__ import annotations

import numpy as np

from .interface import Response, check_inputs

_I = np.eye(3)


class FiniteStrainPoroelastic:
    def __init__(self, n_points, *, K, G, kappa, alpha=1.0, biot_modulus=np.inf, porosity=0.3):
        self.n_points = int(n_points)
        self.lam = K - 2.0 * G / 3.0
        self.mu = float(G)
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.inverse_M = 0.0 if np.isinf(biot_modulus) else 1.0 / biot_modulus
        self.porosity = float(porosity)
        self.commits = 0

    def evaluate(self, inputs, dt, *, tangent=True):
        x = check_inputs(inputs, self.n_points)
        F = x[:, :9].reshape(-1, 3, 3)
        p, G = x[:, 9], x[:, 10:]
        J = np.linalg.det(F)
        Finv = np.linalg.inv(F)
        FinvT = Finv.transpose(0, 2, 1)
        logJ = np.log(J)
        scale = self.lam * logJ - self.mu - self.alpha * p * J  # coefficient of F^-T in P
        P = self.mu * F + scale[:, None, None] * FinvT
        FinvG = np.einsum("qij,qj->qi", Finv, G)
        y = np.empty((self.n_points, 13))
        y[:, :9] = P.reshape(-1, 9)
        y[:, 9] = self.porosity + self.alpha * (J - 1.0) + self.inverse_M * p
        y[:, 10:] = -self.kappa * J[:, None] * FinvG
        if not tangent:
            return Response(y, None)
        C = np.zeros((self.n_points, 13, 13))
        # d(F^-T)_iJ / dF_kL = -Finv_Jk Finv_Li;  d(ln J) / dF_kL = Finv_Lk;  dJ / dF_kL = J Finv_Lk.
        dFinvT = -np.einsum("qJk,qLi->qiJkL", Finv, Finv)
        dscale = (self.lam - self.alpha * p * J)[:, None, None] * Finv.transpose(0, 2, 1)  # d scale / dF_kL
        dP = (self.mu * np.einsum("ik,JL->iJkL", _I, _I)[None]
              + np.einsum("qiJ,qkL->qiJkL", FinvT, dscale) + scale[:, None, None, None, None] * dFinvT)
        C[:, :9, :9] = dP.reshape(-1, 9, 9)
        C[:, :9, 9] = (-self.alpha * J[:, None, None] * FinvT).reshape(-1, 9)
        C[:, 9, :9] = (self.alpha * J[:, None, None] * FinvT).reshape(-1, 9)
        C[:, 9, 9] = self.inverse_M
        # Q_I = -kappa J Finv_Ij G_j: d/dF_kL = -kappa J (Finv_Lk (Finv G)_I - Finv_Ik (Finv G)_L).
        dQ = -self.kappa * J[:, None, None, None] * (np.einsum("qLk,qI->qIkL", Finv, FinvG)
                                                      - np.einsum("qIk,qL->qIkL", Finv, FinvG))
        C[:, 10:, :9] = dQ.reshape(-1, 3, 9)
        C[:, 10:, 10:] = -self.kappa * J[:, None, None] * Finv
        return Response(y, C)

    def commit(self):
        self.commits += 1

    def save_checkpoint(self, folder):  # no history to keep
        pass

    def load_checkpoint(self, folder):
        pass

    def close(self):
        pass
