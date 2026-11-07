"""Macro Newton solve of one time step, with Armijo backtracking.

Knows nothing of the material model beyond ``evaluate``: a
``MaterialPointFailure`` at a trial is treated like a residual increase
(halve the step).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from .interface import MaterialPointFailure


class MacroStepFailure(RuntimeError):
    """Newton did not converge for this dt; the driver retries with a smaller one."""


@dataclass
class NewtonOptions:
    tolerance: float = 1e-8  # on the scaled residual (inf-norm), see Problem row scales
    max_iterations: int = 15
    max_backtracks: int = 8
    backtrack_factor: float = 0.5
    armijo: float = 1e-4


@dataclass
class Step:
    y: np.ndarray
    inputs: np.ndarray
    outputs: np.ndarray
    tangents: np.ndarray
    residual: np.ndarray
    norms: list
    info: dict


def solve_step(problem, model, y_start, old_content, dt, t, row_scale, options=NewtonOptions(), log=None):
    """Return the converged Step at time t, or raise MacroStepFailure / MaterialPointFailure."""
    free = problem.free

    def evaluate(y):
        inputs = problem.point_inputs(y)
        response = model.evaluate(inputs, dt)
        R = problem.residual(y, response.outputs, old_content, dt, t)
        norm = float(np.max(np.abs(R[free] / row_scale[free])))
        return Step(y, inputs, response.outputs, response.tangent, R, [], dict(response.info)), norm

    y = problem.apply_boundary_conditions(y_start, t)
    current, norm = evaluate(y)
    norms = [norm]
    for iteration in range(options.max_iterations + 1):
        if log:
            log(f"    Newton {iteration}: scaled residual {norm:.3e}")
        if norm <= options.tolerance:
            current.norms = norms
            return current
        if iteration == options.max_iterations:
            break
        K = problem.tangent(current.inputs, current.tangents, dt)
        delta = np.zeros_like(y)
        try:
            if sparse.issparse(K):  # 3-D problems
                K = K.tocsr()[free][:, free].tocsc()
                with warnings.catch_warnings():
                    warnings.simplefilter("error", sparse_linalg.MatrixRankWarning)
                    delta[free] = sparse_linalg.spsolve(K, -current.residual[free])
                if not np.isfinite(delta).all():
                    raise np.linalg.LinAlgError("non-finite sparse solve")
            else:
                delta[free] = np.linalg.solve(K[np.ix_(free, free)], -current.residual[free])
        except (np.linalg.LinAlgError, sparse_linalg.MatrixRankWarning) as error:
            raise MacroStepFailure(f"singular macro tangent at t={t:g}") from error
        factor = 1.0
        for _ in range(options.max_backtracks + 1):
            try:
                trial, trial_norm = evaluate(current.y + factor * delta)
            except MaterialPointFailure as error:
                if log:
                    log(f"    step {factor:g} rejected by material points: {error}")
                factor *= options.backtrack_factor
                continue
            if trial_norm <= (1.0 - options.armijo * factor) * norm:
                current, norm = trial, trial_norm
                break
            factor *= options.backtrack_factor
        else:
            raise MacroStepFailure(f"line search failed at t={t:g} (scaled residual {norm:.3e})")
        norms.append(norm)
    raise MacroStepFailure(f"Newton did not converge at t={t:g} in {options.max_iterations} iterations "
                           f"(scaled residual {norm:.3e})")
