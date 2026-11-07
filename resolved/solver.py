"""Newton solve of one resolved step with a direct MUMPS factorisation (petsc4py)."""

from __future__ import annotations

import numpy as np


class StepFailure(RuntimeError):
    """Newton did not converge or every trial left the admissible set; retry with a smaller dt."""


def mumps_solve(matrix, rhs, threads=None):
    """Solve the reduced (nonsymmetric) system with MUMPS LU; check the residual.

    ``threads`` sets the BLAS/OpenMP threads of the factorisation (the job scripts pin both
    to one thread for everything else); None keeps the process settings.
    """
    from contextlib import nullcontext
    from petsc4py import PETSc
    from threadpoolctl import threadpool_limits

    matrix = matrix.tocsr()
    matrix.sort_indices()
    A = PETSc.Mat().createAIJ(size=matrix.shape, csr=(matrix.indptr.astype(PETSc.IntType),
                                                       matrix.indices.astype(PETSc.IntType), matrix.data))
    ksp = PETSc.KSP().create()
    limits = threadpool_limits(limits=int(threads)) if threads else nullcontext()
    try:
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        b, x = A.createVecLeft(), A.createVecRight()
        b.array[:] = rhs
        with limits:
            ksp.solve(b, x)  # factorisation and solve
        solution = x.array.copy()
    finally:
        ksp.destroy()
        A.destroy()
    error = np.linalg.norm(matrix @ solution - rhs)
    if not np.isfinite(solution).all() or error > 1e-8 * np.linalg.norm(rhs) + 1e-300:
        raise StepFailure(f"MUMPS solution failed its residual check ({error:.2e})")
    return solution


def solve_step(column, y, load, *, tolerance=1e-8, absolute=1e-14, max_iterations=20, max_backtracks=12,
               log=None):
    """Return (y, norms) at convergence: ||r|| <= max(absolute, tolerance * max(||r(0)||, |load|)).

    ||r(0)|| is the residual with every velocity and pressure zero (the step's
    unbalanced load); |load| keeps the target above round-off when a step is tiny.
    The guess ``y`` (e.g. the last step's velocities rescaled to this dt) is used
    only if its state is admissible and its residual is below that of the zero
    state, as the cell solver guards its restart guess.
    """
    zero = np.zeros_like(y)
    r_zero = column.residual(zero, load)
    reference = float(np.linalg.norm(r_zero))
    target = max(absolute, tolerance * max(reference, abs(load)))
    r = column.residual(y, load)
    norm = float(np.linalg.norm(r))
    if not (np.isfinite(norm) and norm < reference and column.admissible()):
        if log:
            log(f"    guess rejected (residual {norm:.3e} vs zero state {reference:.3e}); starting from rest")
        y, r, norm = zero, column.residual(zero, load), reference
    norms = [norm]
    for iteration in range(max_iterations + 1):
        if log:
            log(f"    Newton {iteration}: residual {norm:.3e} (target {target:.1e})")
        if norm <= target:
            column.set_reduced(y)
            return y, norms
        if iteration == max_iterations:
            break
        delta = mumps_solve(column.tangent(y), -r, threads=getattr(column, "solver_threads", None))
        factor = 1.0
        for _ in range(max_backtracks + 1):
            trial = y + factor * delta
            r_trial = column.residual(trial, load)
            n_trial = float(np.linalg.norm(r_trial))
            if np.isfinite(n_trial) and n_trial < (1 - 1e-4 * factor) * norm and column.admissible():
                y, r, norm = trial, r_trial, n_trial
                break
            factor *= 0.5
        else:
            raise StepFailure(f"line search failed (residual {norm:.3e})")
        norms.append(norm)
    raise StepFailure(f"Newton did not converge in {max_iterations} iterations (residual {norm:.3e})")
