"""PETSc linear solvers: GMRES with the block preconditioner, MUMPS correctors, BDDC coarse solve."""

from __future__ import annotations
from builtins import sum as _backend_sum
from contextlib import nullcontext
from time import perf_counter
import weakref
import numpy as np
import ngsolve as ngs
from ngsolve import CoefficientFunction, GridFunction
from petsc4py import PETSc
import scipy as sp
import scipy.sparse
from threadpoolctl import threadpool_limits

from .layout import NativeCSRMatrix, _validate_build

if not hasattr(ngs.la, "SparseFactorizationInterface"):
    raise RuntimeError("Experimental coarse solver requires NGSolve SparseFactorizationInterface")


# Exact MUMPS coarse solve for the retained native BDDC residual metrics.
class _PETScBDDCCholesky(ngs.la.SparseFactorizationInterface):
    records = []

    def __init__(self, matrix):
        super().__init__(matrix)
        self._petsc_resources = []
        self.record = {
            "coarse_full_size": int(matrix.height),
            "factor_calls": 0,
            "solve_calls": 0,
            "factor_seconds": 0.0,
            "conversion_seconds": 0.0,
            "factorization_seconds": 0.0,
            "solve_seconds": 0.0,
            "coarse_residual_diagnostic": "verified in independent probe; native solver acceptance unchanged",
            "closed": False,
        }
        type(self).records.append(self.record)

    def close(self):
        for resource in reversed(getattr(self, "_petsc_resources", [])):
            resource.destroy()
        self._petsc_resources = []
        if hasattr(self, "record"):
            self.record["closed"] = True

    def __del__(self):
        try:
            if not PETSc.Sys.isFinalized():
                self.close()
        except Exception:
            pass

    def Factor(self):
        self.close()
        self.record["closed"] = False
        started = perf_counter()
        inner = self.GetInnerMatrix()
        if PETSc.COMM_WORLD.getSize() != 1:
            raise RuntimeError("Prototype supports one MPI rank only")
        if inner.is_complex or PETSc.ScalarType != np.float64:
            raise RuntimeError("Prototype requires real double precision")
        if tuple(inner.entrysizes) != (1, 1):
            raise RuntimeError("Prototype requires scalar CSR entries")
        values, indices, indptr = inner.CSR()
        csr = sp.sparse.csr_matrix(
            (np.array(values), np.array(indices), np.array(indptr)), shape=(inner.height, inner.width)
        )
        if self.is_symmetric_storage:
            csr = (csr + csr.T - sp.sparse.diags(csr.diagonal())).tocsr()
        skew = (csr - csr.T).tocsr()
        relative_skew = float(
            np.max(np.abs(skew.data), initial=0.0) / max(np.max(np.abs(csr.data), initial=0.0), 1e-300)
        )
        if relative_skew > 1e-12:
            raise RuntimeError(f"Coarse matrix is not symmetric: {relative_skew}")
        self.record["relative_entrywise_skew"] = relative_skew
        csr.sort_indices()
        self.record.update(
            coarse_compact_size=int(inner.height),
            symmetric_storage=bool(self.is_symmetric_storage),
            compact_nnz=int(csr.nnz),
        )
        try:
            self._csr = csr
            self._matrix = PETSc.Mat().createAIJ(
                size=csr.shape,
                csr=(csr.indptr.astype(PETSc.IntType), csr.indices.astype(PETSc.IntType), csr.data),
                comm=PETSc.COMM_SELF,
            )
            self._petsc_resources.append(self._matrix)
            self._matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            self._matrix.setOption(PETSc.Mat.Option.SPD, True)
            self.record["conversion_seconds"] += perf_counter() - started
            factor_started = perf_counter()
            self._ksp = PETSc.KSP().create(PETSc.COMM_SELF)
            self._petsc_resources.append(self._ksp)
            self._ksp.setType("preonly")
            self._ksp.setOperators(self._matrix)
            pc = self._ksp.getPC()
            pc.setType("cholesky")
            pc.setFactorSolverType("mumps")
            self._ksp.setErrorIfNotConverged(True)
            self._ksp.setUp()
            self.record["factorization_seconds"] += perf_counter() - factor_started
            self._rhs, self._solution = self._matrix.createVecs()
            self._petsc_resources.extend((self._rhs, self._solution))
            self.record["factor_calls"] += 1
            self.record["factor_seconds"] += perf_counter() - started
        except Exception:
            self.close()
            raise

    def Solve(self, rhs, solution):
        started = perf_counter()
        self._rhs.array[:] = rhs.FV().NumPy()
        self._ksp.solve(self._rhs, self._solution)
        solution.FV().NumPy()[:] = self._solution.array_r
        if not np.isfinite(solution.FV().NumPy()).all():
            raise RuntimeError("MUMPS coarse solution is non-finite")
        self.record["solve_calls"] += 1
        self.record["solve_seconds"] += perf_counter() - started


_NATIVE_BDDC_MUMPS_INVERSE = "prototype_petsc_bddc_mumps"
ngs.la.RegisterInverseType(_NATIVE_BDDC_MUMPS_INVERSE, _PETScBDDCCholesky)


# One frozen coupled tangent and factorization serve all 13 correctors.
class PETScMUMPSSolver:
    """Direct frozen corrector solver with the existing ``solve`` contract.

    ``tol`` and ``atol`` measure the final original-native residual; they do
    not change equations or factorization.  ``force_gmres`` only identifies a
    caller-requested defect refinement, for compatibility with the original
    corrector API.  Every call applies the existing direct factors once.
    The caller's unchanged full scaled and per-field checks remain decisive.

    ``blas_threads`` above one limits BLAS and OpenMP during MUMPS setup and
    solves; one limits BLAS only during solves; None keeps process settings.
    """

    def __init__(self, *, system, free_dof_blocks, template, blas_threads=1):
        started = perf_counter()
        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise RuntimeError("MUMPS correctors require real-double PETSc")
        if PETSc.COMM_WORLD.getSize() != 1:
            raise RuntimeError("The compact six-field MUMPS bridge is sequential only")
        if not PETSc.Sys.hasExternalPackage("mumps"):
            raise RuntimeError("This PETSc build does not provide MUMPS")
        self._closed = False
        self._busy = False
        self.system = system
        self.native_matrix = system.mat
        self.matrix = self.ksp = self.rhs = self.solution = self.residual = None
        self.native_output = None
        self._converters = []
        self._temporary_matrices = []
        self._block_cache = {}
        self.blas_threads = None if blas_threads is None else max(1, int(blas_threads))
        self.masks = tuple((np.asarray(list(mask), dtype=bool) for mask in free_dof_blocks))
        if len(self.masks) != 6 or template.nblocks != 6:
            raise ValueError("Expected the six independent FSI/ALE field masks")
        for field, mask in enumerate(self.masks):
            array = template[field].FV().NumPy()
            if array.dtype != np.float64 or array.ndim != 1 or len(array) != len(mask):
                raise ValueError("Native fields must match their real-double FreeDofs masks")
        self.free = tuple((np.flatnonzero(mask) for mask in self.masks))
        self.sizes = tuple((len(indices) for indices in self.free))
        self.offsets = np.cumsum((0, *self.sizes))
        self.sections = tuple((slice(int(a), int(b)) for a, b in zip(self.offsets[:-1], self.offsets[1:])))
        self.ndof = int(self.offsets[-1])
        if self.ndof == 0:
            raise ValueError("The tangent has no independent free DOFs")
        full_ndof = _backend_sum((len(mask) for mask in self.masks))
        self.metadata = {
            "mode": "petsc_mumps_direct",
            "operator_backend": "exact_compact_aij",
            "preconditioner_backend": "mumps_lu_direct",
            "petsc_version": PETSc.Sys.getVersion(),
            "full_dofs": full_ndof,
            "active_dofs": self.ndof,
            "active_field_dofs": list(self.sizes),
            "compact_field_order": list(range(6)),
            "factor_solver": "mumps",
            "factor_reuse": True,
            "symbolic_factorizations": 0,
            "numeric_factorizations": 0,
            "full_tangent_copies": 1,
            "matrix_bytes": 0,
            "copied_sparse_bytes": 0,
            "matrix_nonzeros": 0,
            "native_scratch_bytes": full_ndof * 8,
            "compact_work_bytes": 3 * self.ndof * 8,
            "retained_corrector_solutions": 0,
            "recycle_vectors": 0,
            "diagonal_regularization": False,
            "mumps_modified_pivots": 0,
            "setup_blas_threads": (
                self.blas_threads
                if self.blas_threads is not None and self.blas_threads > 1
                else "process_settings"
            ),
            "blas_threads": self.blas_threads,
            "setup_seconds": 0.0,
            "conversion_seconds": 0.0,
            "factor_setup_seconds": 0.0,
            "factor_apply_seconds": 0.0,
            "native_residual_seconds": 0.0,
            "solve_seconds": 0.0,
            "blas_context_seconds": 0.0,
            "packing_seconds": 0.0,
            "solve_calls": 0,
            "direct_factor_applications": 0,
            "iterations": 0,
            "iteration_metric": "direct_factor_applications",
            "matvec_calls": 0,
            "matvec_seconds": 0.0,
            "preconditioner_calls": 0,
            "preconditioner_seconds": 0.0,
            "fallback_solve_calls": 0,
            "forced_gmres_calls": 0,
            "defect_refinement_calls": 0,
            "solves": [],
        }
        try:
            with self._thread_limits(setup=True):
                self._build()
            self.native_output = template.CreateVector()
        except BaseException:
            self.close()
            raise
        self.metadata["setup_seconds"] = perf_counter() - started

    def _thread_limits(self, *, setup=False):
        if self.blas_threads is None or (setup and self.blas_threads == 1):
            return nullcontext()
        if self.blas_threads == 1:
            return threadpool_limits(limits=1, user_api="blas")
        return threadpool_limits(limits=self.blas_threads)

    def _explain_thread_mismatch(self, error):
        factor = self.ksp.getPC().getFactorMatrix()
        if int(factor.getMumpsInfog(1)) == -58:
            expected = int(factor.getMumpsInfog(2))
            raise RuntimeError(
                f"MUMPS requires consistent OpenMP thread counts between analysis and solve (INFOG(1)=-58, analysis threads={expected}). Keep process OpenMP settings unchanged after factorization, or select an explicit corrector_blas_threads value above one to use the same scoped setting for MUMPS setup and solves. Check OMP_DYNAMIC and OMP_THREAD_LIMIT if the requested count cannot be created."
            ) from error

    def _block(self, row, col):
        key = (row, col)
        if key in self._block_cache:
            return self._block_cache[key]
        transpose_source = getattr(self.system.full_tangent, "transpose_sources", {}).get(key)
        if transpose_source is not None:
            source = self._block(*transpose_source)
            matrix = source.transpose(out=PETSc.Mat())
            self._temporary_matrices.append(matrix)
        else:
            native = self.system.full_tangent.get(row, col)
            if native is None:
                matrix = PETSc.Mat().createAIJ(
                    (self.sizes[row], self.sizes[col]), nnz=0, comm=PETSc.COMM_SELF
                )
                matrix.assemble()
                self._temporary_matrices.append(matrix)
            else:
                bridge = NativeCSRMatrix(native, self.masks[row], self.masks[col])
                self._converters.append(bridge)
                matrix = bridge.mat
        self._block_cache[key] = matrix
        return matrix

    def _release_block_storage(self):
        self._block_cache.clear()
        for matrix in reversed(self._temporary_matrices):
            matrix.destroy()
        self._temporary_matrices.clear()
        for bridge in self._converters:
            bridge.close()
        self._converters.clear()

    def _build(self):
        conversion_started = perf_counter()
        nest = None
        try:
            nest = PETSc.Mat().createNest(
                [[self._block(i, j) for j in range(6)] for i in range(6)], comm=PETSc.COMM_SELF
            )
            nest.assemble()
            self.matrix = nest.convert(PETSc.Mat.Type.AIJ, out=PETSc.Mat())
        finally:
            if nest is not None:
                nest.destroy()
            self._release_block_storage()
        self.metadata["conversion_seconds"] = perf_counter() - conversion_started
        nonzeros = int(self.matrix.getInfo()["nz_used"])
        index_bytes = np.dtype(PETSc.IntType).itemsize
        csr_bytes = nonzeros * (8 + index_bytes) + (self.ndof + 1) * index_bytes
        self.metadata.update(
            matrix_nonzeros=nonzeros, matrix_bytes=int(csr_bytes), copied_sparse_bytes=int(csr_bytes)
        )
        factor_started = perf_counter()
        self.ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp.setOptionsPrefix("rve_frozen_mumps_%x_" % id(self))
        self.ksp.setType(PETSc.KSP.Type.PREONLY)
        self.ksp.setOperators(self.matrix)
        self.ksp.setInitialGuessNonzero(False)
        self.ksp.setErrorIfNotConverged(True)
        pc = self.ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        pc.setFactorShift(PETSc.Mat.FactorShiftType.NONE, 0.0)
        pc.setFactorSetUpSolverType()
        factor = pc.getFactorMatrix()

        # Ordering controls.
        factor.setMumpsIcntl(28, 1)
        factor.setMumpsIcntl(7, 5)

        # Existing pivot controls.
        factor.setMumpsCntl(4, -1.0)
        factor.setMumpsIcntl(24, 0)

        self.ksp.setUp()
        modified_pivots = int(factor.getMumpsInfog(25))
        if modified_pivots:
            raise RuntimeError("MUMPS modified pivots despite disabled regularization")
        factor_status = int(factor.getMumpsInfog(1))
        if factor_status < 0:
            raise RuntimeError(f"MUMPS factorization failed with INFOG(1)={factor_status}")
        self.metadata.update(
            {
                "symbolic_factorizations": 1,
                "numeric_factorizations": 1,
                "mumps_modified_pivots": modified_pivots,
                "mumps_factor_status": factor_status,
                "mumps_factor_nonzeros": int(factor.getMumpsInfog(29)),
                "mumps_factor_memory_mb": int(factor.getMumpsInfog(22)),
                "mumps_factor_memory_note": "MUMPS INFOG(22), integer MB; excludes other process allocations",
            }
        )
        pc.setReusePreconditioner(True)
        self.metadata["factor_setup_seconds"] = perf_counter() - factor_started
        self.solution, self.rhs = self.matrix.createVecs()
        self.residual = self.rhs.duplicate()

    def pack(self, native, destination=None):
        if destination is None:
            destination = np.empty(self.ndof, dtype=np.float64)
        for field, (indices, section) in enumerate(zip(self.free, self.sections)):
            np.take(native[field].FV().NumPy(), indices, out=destination[section], mode="clip")
        return destination

    def unpack(self, compact, native):
        for field, (indices, section) in enumerate(zip(self.free, self.sections)):
            array = native[field].FV().NumPy()
            array.fill(0.0)
            array[indices] = compact[section]

    def solve(self, rhs, destination, *, tol=1e-10, atol=0.0, force_gmres=False):
        if self._closed:
            raise RuntimeError("The frozen MUMPS solver is closed")
        if self._busy:
            raise RuntimeError("The frozen MUMPS solver is not reentrant")
        tol, atol = (float(tol), float(atol))
        if not np.isfinite(tol) or not 0 <= tol < 1 or (not np.isfinite(atol)) or (atol < 0):
            raise ValueError("Residual tolerances must be finite, with 0 <= tol < 1 and atol >= 0")
        started = perf_counter()
        self._busy = True
        try:
            with self._thread_limits():
                stats = self._solve(rhs, destination, tol=tol, atol=atol, defect_retry=bool(force_gmres))
        except PETSc.Error as error:
            self._explain_thread_mismatch(error)
            raise
        finally:
            self._busy = False
        elapsed = perf_counter() - started
        stats["blas_context_seconds"] = max(0.0, elapsed - stats["solve_seconds"])
        stats["solve_seconds"] = elapsed
        self.metadata["solve_calls"] += 1
        self.metadata["forced_gmres_calls"] += int(force_gmres)
        self.metadata["defect_refinement_calls"] += int(force_gmres)
        for key in (
            "direct_factor_applications",
            "iterations",
            "factor_apply_seconds",
            "native_residual_seconds",
            "solve_seconds",
            "blas_context_seconds",
            "packing_seconds",
            "matvec_calls",
            "matvec_seconds",
            "preconditioner_calls",
            "preconditioner_seconds",
        ):
            self.metadata[key] += stats[key]
        self.metadata["solves"].append(stats.copy())
        del self.metadata["solves"][:-64]
        return stats

    def _solve(self, rhs, destination, *, tol, atol, defect_retry):
        started = perf_counter()
        packing_started = perf_counter()
        self.pack(rhs, self.rhs.getArray())
        self.solution.set(0.0)
        packing_seconds = perf_counter() - packing_started
        rhs_norm = float(self.rhs.norm())
        target = max(atol, tol * rhs_norm)
        factor_started = perf_counter()
        self.ksp.solve(self.rhs, self.solution)
        factor_seconds = perf_counter() - factor_started
        reason = int(self.ksp.getConvergedReason())
        compact_solution = self.solution.getArray(readonly=True)
        if not np.isfinite(compact_solution).all():
            raise RuntimeError("MUMPS returned a non-finite correction")
        packing_started = perf_counter()
        self.unpack(compact_solution, destination)
        packing_seconds += perf_counter() - packing_started
        residual_started = perf_counter()
        self.native_matrix.Mult(destination, self.native_output)
        self.pack(self.native_output, self.residual.getArray())
        self.residual.aypx(-1.0, self.rhs)
        native_residual = float(self.residual.norm())
        residual_seconds = perf_counter() - residual_started
        info = (
            0
            if reason > 0 and np.isfinite(native_residual) and (native_residual <= 2.0 * target)
            else max(1, abs(reason))
        )
        return {
            "info": info,
            "iterations": 1,
            "iteration_metric": "direct_factor_applications",
            "direct_factor_applications": 1,
            "factor_apply_seconds": factor_seconds,
            "petsc_converged_reason": reason,
            "native_residual": native_residual,
            "true_residual_norm": native_residual,
            "true_residual_target": target,
            "native_residual_seconds": residual_seconds,
            "matvec_calls": 1,
            "matvec_seconds": residual_seconds,
            "preconditioner_calls": 1,
            "preconditioner_seconds": factor_seconds,
            "packing_seconds": packing_seconds,
            "solve_seconds": perf_counter() - started,
            "defect_retry": defect_retry,
            "forced_gmres": defect_retry,
            "gmres_iterations": 0,
            "outer_iterations": 0,
            "fallback_used": False,
            "fallback_reason": None,
        }

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for name in ("ksp", "solution", "rhs", "residual", "matrix"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)
        self._release_block_storage()
        self.native_output = self.native_matrix = self.system = None

    def __enter__(self):
        if self._closed:
            raise RuntimeError("The frozen MUMPS solver is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            if not PETSc.Sys.isFinalized():
                self.close()
        except Exception:
            pass


class _NativeMatrixContext:

    def __init__(self, solver):
        self._solver = weakref.ref(solver)

    def mult(self, mat, x, y):
        self._solver()._apply_matrix(x, y)


# Reusable Newton GMRES and the fixed-work, field-based GAMG block action.
class ReusableNewtonKSP:
    """One GMRES KSP with the direct PETSc GAMG block action per Newton solve.

    Independent DOFs are restricted/prolonged through the unchanged native
    operator. Left preconditioning, modified Gram--Schmidt, zero initial
    guesses and the original iteration budget are retained. The caller's
    native scaled residual check is authoritative; PETSc's recursive norm
    remains diagnostic. Newton inherits the process's existing thread limits.
    """

    def __init__(self, *, native_matrix, free_dof_blocks, template, direct_pc, maxiter=500, restart=50):
        started = perf_counter()
        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise RuntimeError("This solver requires a real-double PETSc build")
        if PETSc.COMM_WORLD.getSize() != 1:
            raise RuntimeError("The compact six-field PETSc bridge is sequential only")
        self._closed = False
        self._active_stats = None
        self.ksp = self.operator = self.rhs = self.solution = self.residual = None
        self.native_matrix = native_matrix
        if direct_pc is None:
            raise ValueError("PETSc Newton requires its direct GAMG block preconditioner")
        self.direct_pc = direct_pc
        self.maxiter = max(1, int(maxiter))
        self.restart = min(self.maxiter, max(1, int(restart)))
        masks = tuple((np.asarray(list(mask), dtype=bool) for mask in free_dof_blocks))
        if len(masks) != template.nblocks:
            raise ValueError("A FreeDofs mask is required for each native field")
        for index, mask in enumerate(masks):
            array = template[index].FV().NumPy()
            if array.dtype != np.float64 or array.ndim != 1 or len(array) != len(mask):
                raise ValueError("Native fields must match real-double FreeDofs masks")
        self.free = tuple((np.flatnonzero(mask) for mask in masks))
        self.offsets = np.cumsum((0, *(len(indices) for indices in self.free)))
        self.ndof = int(self.offsets[-1])
        if self.ndof == 0:
            raise ValueError("The compact system has no independent free DOFs")
        self._slices = tuple(
            (slice(int(begin), int(end)) for begin, end in zip(self.offsets[:-1], self.offsets[1:]))
        )
        full_ndof = _backend_sum((len(mask) for mask in masks))
        self.native_input = template.CreateVector()
        self.native_output = template.CreateVector()
        self._input_arrays = tuple((self.native_input[i].FV().NumPy() for i in range(len(masks))))
        self._output_arrays = tuple((self.native_output[i].FV().NumPy() for i in range(len(masks))))
        self.metadata = {
            "mode": "petsc_gmres",
            "operator_backend": "native_restricted_python_mat",
            "preconditioner_backend": "direct_petsc_block_pc",
            "petsc_version": PETSc.Sys.getVersion(),
            "full_dofs": full_ndof,
            "active_dofs": self.ndof,
            "restart": self.restart,
            "recycle_vectors": 0,
            "matrix_bytes": 0,
            "native_scratch_bytes": 2 * full_ndof * 8,
            "compact_work_bytes": 3 * self.ndof * 8,
            "pc_side": "left",
            "norm_type": "preconditioned",
            "orthogonalization": "modified_gram_schmidt",
            "divergence_growth_limit": None,
            "blas_threads": None,
            "setup_seconds": 0.0,
            "blas_context_seconds": 0.0,
            "solve_seconds": 0.0,
            "solve_calls": 0,
            "fallback_solve_calls": 0,
            "forced_gmres_calls": 0,
            "iterations": 0,
            "matvec_calls": 0,
            "matvec_seconds": 0.0,
            "preconditioner_calls": 0,
            "preconditioner_seconds": 0.0,
            "packing_seconds": 0.0,
            "solves": [],
        }
        try:
            self.operator = PETSc.Mat().createPython(
                (self.ndof, self.ndof), context=_NativeMatrixContext(self), comm=PETSc.COMM_SELF
            )
            self.operator.setUp()
            self.ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            self.ksp.setType("gmres")
            prefix = "native_compact_%x_" % id(self)
            self.ksp.setOptionsPrefix(prefix)
            options = PETSc.Options(prefix)
            options["ksp_gmres_modifiedgramschmidt"] = True
            try:
                self.ksp.setFromOptions()
            finally:
                del options["ksp_gmres_modifiedgramschmidt"]
            self.ksp.setOperators(self.operator)
            self.ksp.setGMRESRestart(self.restart)
            self.ksp.setPCSide(PETSc.PC.Side.LEFT)
            self.ksp.setNormType(PETSc.KSP.NormType.PRECONDITIONED)
            self.ksp.setInitialGuessNonzero(False)
            self.ksp.setErrorIfNotConverged(False)
            pc = self.ksp.getPC()
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setReusePreconditioner(True)
            direct_pc.attach(self.ksp)
            self.solution, self.rhs = self.operator.createVecs()
            self.residual = self.rhs.duplicate()
            self.ksp.setUp()
        except Exception:
            self.close()
            raise
        self.metadata["setup_seconds"] = perf_counter() - started
        self.metadata.update(
            ksp_creations=1,
            duplicate_residual_products=0,
            residual_acceptance="original_native_scaled_check_in_Newton",
        )

    def pack(self, native_vector, destination=None):
        """Restrict a block vector; an existing compact destination avoids copies."""
        if destination is None:
            destination = np.empty(self.ndof, dtype=np.float64)
        for block, (indices, compact_slice) in enumerate(zip(self.free, self._slices)):
            np.take(native_vector[block].FV().NumPy(), indices, out=destination[compact_slice])
        return destination

    def unpack(self, compact_vector, native_vector):
        """Prolong independent entries and zero all constrained/slave entries."""
        for block, (indices, compact_slice) in enumerate(zip(self.free, self._slices)):
            array = native_vector[block].FV().NumPy()
            array.fill(0.0)
            array[indices] = compact_vector[compact_slice]

    def _apply_matrix(self, x, y):
        started = perf_counter()
        compact_input = x.getArray(readonly=True)
        compact_output = y.getArray()
        for array, indices, compact_slice in zip(self._input_arrays, self.free, self._slices):
            array.fill(0.0)
            array[indices] = compact_input[compact_slice]
        action = self.native_matrix
        action.Mult(self.native_input, self.native_output)
        for array, indices, compact_slice in zip(self._output_arrays, self.free, self._slices):
            np.take(array, indices, out=compact_output[compact_slice])
        if self._active_stats is not None:
            prefix = "matvec"
            self._active_stats[prefix + "_calls"] += 1
            self._active_stats[prefix + "_seconds"] += perf_counter() - started

    def solve(self, rhs, destination, *, tol=1e-10, atol=0.0, force_gmres=False):
        """Write a correction and return iteration, status, timing and residual data.

        PETSc's reason is reported independently of the exact restricted residual.
        The caller's full native residual acceptance remains authoritative
        and may request another defect solve.
        """
        if self._closed:
            raise RuntimeError("The PETSc compact solver is closed")
        if self._active_stats is not None:
            raise RuntimeError("Shared native scratch makes this solver non-reentrant")
        tol, atol = (float(tol), float(atol))
        if not np.isfinite(tol) or not 0.0 <= tol < 1.0 or (not np.isfinite(atol)) or (atol < 0.0):
            raise ValueError("Krylov tolerances must be finite with 0 <= rtol < 1 and atol >= 0")
        started = perf_counter()
        stats = self._solve(rhs, destination, tol=tol, atol=atol, force_gmres=force_gmres)
        elapsed = perf_counter() - started
        stats["blas_context_seconds"] = max(0.0, elapsed - stats["solve_seconds"])
        stats["solve_seconds"] = elapsed
        self.metadata["solve_calls"] += 1
        self.metadata["forced_gmres_calls"] += int(force_gmres)
        for key in (
            "solve_seconds",
            "blas_context_seconds",
            "iterations",
            "matvec_calls",
            "matvec_seconds",
            "preconditioner_calls",
            "preconditioner_seconds",
            "packing_seconds",
        ):
            self.metadata[key] += stats[key]
        self.metadata["solves"].append(stats.copy())
        del self.metadata["solves"][:-64]
        return stats

    def _solve(self, rhs, destination, *, tol, atol, force_gmres):
        started = perf_counter()
        stats = {
            "matvec_calls": 0,
            "matvec_seconds": 0.0,
            "preconditioner_calls": 0,
            "preconditioner_seconds": 0.0,
            "packing_seconds": 0.0,
            "iterations": 0,
            "forced_gmres": bool(force_gmres),
            "fallback_used": False,
            "gmres_restart": self.restart,
            "gmres_iteration_budget": self.maxiter,
        }
        pc_before = dict(self.direct_pc.metadata) if self.direct_pc is not None else None
        self._active_stats = stats
        try:
            packing_started = perf_counter()
            self.pack(rhs, self.rhs.getArray())
            self.solution.set(0.0)
            stats["packing_seconds"] += perf_counter() - packing_started
            self.ksp.setTolerances(
                rtol=float(tol), atol=float(atol), divtol=float("inf"), max_it=self.maxiter
            )
            krylov_started = perf_counter()
            self.ksp.solve(self.rhs, self.solution)
            stats["gmres_seconds"] = perf_counter() - krylov_started
            stats["iterations"] = int(self.ksp.getIterationNumber())
            stats["gmres_iterations"] = stats["iterations"]
            stats["petsc_converged_reason"] = int(self.ksp.getConvergedReason())
            stats["petsc_residual_estimate"] = float(self.ksp.getResidualNorm())
            stats["iteration_metric"] = "petsc_gmres_inner_steps"
            values = self.solution.getArray(readonly=True)
            if not np.isfinite(values).all():
                raise RuntimeError("PETSc returned a non-finite Newton direction")
            packing_started = perf_counter()
            self.unpack(values, destination)
            stats["packing_seconds"] += perf_counter() - packing_started
        finally:
            self._active_stats = None
        if self.direct_pc is not None:
            stats["preconditioner_calls"] = int(
                self.direct_pc.metadata.get("apply_calls", 0) - pc_before.get("apply_calls", 0)
            )
            stats["preconditioner_seconds"] = float(
                self.direct_pc.metadata.get("apply_seconds", 0.0) - pc_before.get("apply_seconds", 0.0)
            )
        stats["solve_seconds"] = perf_counter() - started
        return stats

    def close(self):
        """Destroy PETSc objects promptly; safe to call repeatedly."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for name in ("ksp", "solution", "rhs", "residual", "operator"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)
        self._input_arrays = self._output_arrays = ()
        self.native_input = self.native_output = None
        self.native_matrix = None
        self.direct_pc = None

    def __enter__(self):
        if self._closed:
            raise RuntimeError("The PETSc compact solver is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            if not PETSc.Sys.isFinalized():
                self.close()
        except Exception:
            pass




class _ReusablePETScPCContext:

    def __init__(self, owner):
        self.owner = weakref.ref(owner)

    def apply(self, pc, source, destination):
        owner = self.owner()
        if owner is None:
            raise RuntimeError("Reusable PETSc block preconditioner was released")
        owner.apply(source, destination)


class ReusablePETScBlockPC:
    """Fixed PETSc block sweep on the six-field FSI/ALE tangent.

    Velocity/ALE use GAMG, both pressure masses use Jacobi, and exact mean
    constraints use small Schur solves. Solid-pressure feedback is retained.
    Numerical proxy revisions reuse sparse graphs and unchanged PCs. Native
    actions are used only by the independent physical acceptance checks.

    Compatible vector proxies use actual node-major ordering and projected
    FE translations; mismatched component masks retain component ordering.
    A block without admissible translations uses fixed ILU(0). Every PC
    application does fixed work, as required by ordinary GMRES.
    """

    _couplings = {
        "B": (1, 0),
        "Bt": (0, 1),
        "Bs": (2, 0),
        "Bst": (0, 2),
        "D": (3, 0),
        "E": (0, 3),
        "C": (1, 4),
        "Wu": (5, 0),
    }
    _proxy_fields = {"velocity": 0, "pressure": 1, "solid_pressure": 2, "mesh": 5}

    def __init__(self, system, native_pc, free_dof_blocks, template):
        _validate_build()
        self.masks = tuple((np.asarray(list(mask), dtype=bool) for mask in free_dof_blocks))
        self.free = tuple((np.flatnonzero(mask) for mask in self.masks))
        if len(self.free) != 6 or template.nblocks != 6:
            raise ValueError("Reusable PETSc block PC requires the six-field FSI layout")
        self.sizes = tuple((int(len(indices)) for indices in self.free))
        if self.sizes[3:5] != (3, 1) or min(self.sizes[:5]) < 1:
            raise ValueError(
                "Expected nonempty velocity/pressure fields, three velocity and one pressure mean multiplier"
            )
        self.offsets = np.cumsum((0, *self.sizes))
        self.slices = tuple((slice(int(a), int(b)) for a, b in zip(self.offsets[:-1], self.offsets[1:])))
        for field, mask in enumerate(self.masks):
            if len(mask) != len(template[field].FV().NumPy()):
                raise ValueError("Free-DOF mask and native field dimensions differ")
        self._closed = False
        self._bridges = {}
        self._proxies = {}
        self._pcs = {}
        self._ordering = {}
        self._nullspaces = {}
        self._nullvectors = {}
        self._revisions = {}
        self._vectors = []
        self._attached_pcs = []
        self._context = _ReusablePETScPCContext(self)
        self._source = []
        self._destination = []
        self._mean_basis = []
        self._mean_inverse = None
        self.pressure_inverse_scale = self.mesh_inverse_scale = 1.0
        self.metadata = {
            "mode": "petsc_reusable_gamg",
            "preconditioner_backend": "direct_petsc_block_pc",
            "active_dofs": int(self.offsets[-1]),
            "field_dofs": list(self.sizes),
            "block_size": 1,
            "requested_vector_ordering": "nodal",
            "proxy_vector_ordering": {},
            "proxy_block_sizes": {},
            "vector_ordering_fallbacks": {},
            "proxy_scatter_seconds": 0.0,
            "field_order": ["u", "p_f", "p_s", "lambda_u", "lambda_pf", "w"],
            "pressure_pc": "jacobi",
            "solid_pressure_feedback": True,
            "near_nullspace": "projected_FE_translations",
            "near_nullspace_modes": {},
            "near_nullspace_skipped_modes": {},
            "proxy_fallbacks": {},
            "proxy_methods": {},
            "hypre_near_nullspace_used": None,
            "fixed_work_per_application": True,
            "native_actions_per_application": 0,
            "full_native_vector_copies_per_application": 0,
            "updates": 0,
            "apply_calls": 0,
            "apply_seconds": 0.0,
            "update_seconds": 0.0,
            "constraint_setup_seconds": 0.0,
            "matrix_conversion_seconds": 0.0,
            "proxy_setup_seconds": 0.0,
            "proxy_updates": {name: 0 for name in self._proxy_fields},
            "proxy_reuses": {name: 0 for name in self._proxy_fields},
            "proxy_apply_calls": {name: 0 for name in self._proxy_fields},
            "proxy_apply_seconds": {name: 0.0 for name in self._proxy_fields},
            "attachments": 0,
            "couplings": {},
            "proxies": {},
        }
        try:
            for size in self.sizes:
                self._source.append(self._new_vec(size))
                self._destination.append(self._new_vec(size))
            self._u_rhs = self._new_vec(self.sizes[0])
            self._u_correction = self._new_vec(self.sizes[0])
            self._rho_correction = self._new_vec(3)
            self._rho_tmp = self._new_vec(3)
            self._rho_zero = self._new_vec(3)
            self._rho_zero.set(0.0)
            self._rho_unit = self._new_vec(3)
            self._p_rhs = self._new_vec(self.sizes[1])
            self._p_tmp = self._new_vec(self.sizes[1])
            self._c_unit = self._new_vec(self.sizes[1])
            self._pressure_one = self._new_vec(self.sizes[1])
            self._ps_rhs = self._new_vec(self.sizes[2])
            self._w_rhs = self._new_vec(self.sizes[5])
            self._lambda_unit = self._new_vec(1)
            self._lambda_unit.set(1.0)
            self._mean_basis = [self._new_vec(self.sizes[0]) for unused in range(3)]
            space = system.state_components[1].space
            pressure_constant = GridFunction(space)
            pressure_constant.Set(1.0)
            self._pressure_one.array[:] = pressure_constant.vec.FV().NumPy()[self.free[1]]
            if not np.isfinite(self._pressure_one.array_r).all():
                raise ValueError("Projected pressure constant is non-finite")
            self.update(system, native_pc, proxy_revision=getattr(native_pc, "proxy_revisions", 0))
        except Exception:
            self.close()
            raise

    def _new_vec(self, size):
        vec = PETSc.Vec().createSeq(int(size), comm=PETSc.COMM_SELF)
        self._vectors.append(vec)
        return vec

    @staticmethod
    def _proxy_forms(system, native_pc):
        mesh_form = getattr(native_pc, "mesh_proxy", None)
        if mesh_form is None:
            mesh_form = (
                getattr(native_pc, "_current_mesh_proxy", None)
                if getattr(native_pc, "_using_current_state_proxies", False)
                else getattr(native_pc, "_lagged_mesh_proxy", None)
            )
        if mesh_form is None:
            mesh_form = system.mesh_proxy
        return {
            "velocity": getattr(native_pc, "velocity_preconditioner_form", system.velocity_proxy),
            "pressure": getattr(native_pc, "pressure_mass", system.pressure_mass),
            "solid_pressure": getattr(native_pc, "solid_pressure_mass", system.solid_pressure_mass),
            "mesh": mesh_form,
        }

    def _configure_vector_ordering(self, name, space, bridge):
        """Permute only compatible FE vector proxies; retain one compact CSR.

        NativeCSRMatrix already stores a reusable native-value gather. We
        permute that gather and its graph once, rather than retaining both
        component and nodal PETSc matrices on every numerical refresh.
        """
        self.metadata["proxy_vector_ordering"][name] = "component"
        self.metadata["proxy_block_sizes"][name] = 1
        field = self._proxy_fields[name]
        base = space
        while getattr(base, "base_space", None) is not None:
            base = base.base_space
        try:
            components = tuple(base.components)
            counts = tuple((int(component.ndof) for component in components))
            ranges = tuple(
                ((int(base.Range(i).start), int(base.Range(i).stop)) for i in range(len(components)))
            )
        except Exception:
            self.metadata["vector_ordering_fallbacks"][name] = "component_layout_is_not_exposed_by_FE_space"
            return
        full_size = len(self.masks[field])
        if len(counts) != 3 or len(set(counts)) != 1 or sum(counts) != full_size:
            self.metadata["vector_ordering_fallbacks"][
                name
            ] = "vector_components_do_not_have_equal_scalar_layouts"
            return
        scalar_size = counts[0]
        if ranges != tuple(((i * scalar_size, (i + 1) * scalar_size) for i in range(3))):
            self.metadata["vector_ordering_fallbacks"][
                name
            ] = "FE_components_are_not_stored_as_contiguous_scalar_blocks"
            return
        masks = self.masks[field].reshape(3, scalar_size)
        if not all((np.array_equal(masks[0], masks[i]) for i in (1, 2))):
            self.metadata["vector_ordering_fallbacks"][name] = "component_free_DOF_masks_differ"
            return
        size = self.sizes[field]
        permutation = np.arange(size, dtype=PETSc.IntType).reshape(3, -1).T.reshape(-1)
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(size, dtype=PETSc.IntType)
        row_counts = np.diff(bridge._pointers)[permutation].astype(np.int64)
        starts = bridge._pointers[permutation].astype(np.int64)
        selected = np.repeat(starts - np.r_[0, np.cumsum(row_counts[:-1])], row_counts)
        selected += np.arange(len(bridge._positions))
        columns = inverse[bridge._columns[selected]]
        rows = np.repeat(np.arange(size), row_counts)
        ordering = np.lexsort((columns, rows))
        positions = bridge._positions[selected[ordering]]
        new_columns = columns[ordering].astype(PETSc.IntType, copy=False)
        new_pointers = np.r_[0, np.cumsum(row_counts)].astype(PETSc.IntType)
        new_values = bridge._values[selected[ordering]]
        nodal_mat = PETSc.Mat().createAIJ(
            size=(size, size), csr=(new_pointers, new_columns, new_values), comm=PETSc.COMM_SELF
        )
        try:
            nodal_mat.setBlockSize(3)
            nodal_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
            nodal_mat.assemble()
        except Exception:
            nodal_mat.destroy()
            raise
        bridge.mat.destroy()
        bridge.mat = nodal_mat
        bridge._positions = positions
        bridge._columns = new_columns
        bridge._pointers = new_pointers
        bridge._values = new_values
        bridge.metadata["storage_order"] = "node_major"
        bridge.metadata["block_size"] = 3
        data = {
            "permutation": permutation,
            "is": None,
            "scatter": None,
            "rhs": self._new_vec(size),
            "solution": self._new_vec(size),
        }
        self._ordering[name] = data
        data["rhs"].setBlockSize(3)
        data["solution"].setBlockSize(3)
        data["is"] = PETSc.IS().createGeneral(permutation, comm=PETSc.COMM_SELF)
        data["scatter"] = PETSc.Scatter().create(self._source[field], data["is"], data["rhs"], None)
        self.metadata["proxy_vector_ordering"][name] = "nodal"
        self.metadata["proxy_block_sizes"][name] = 3

    def _attach_near_modes(self, name, space, bridge):
        field = self._proxy_fields[name]
        constant = GridFunction(space)
        vectors = []
        skipped = 0
        pending = None
        try:
            for component in range(3):
                constant.Set(
                    CoefficientFunction(tuple((1.0 if index == component else 0.0 for index in range(3))))
                )
                coefficients = constant.vec.FV().NumPy()
                zero_threshold = 1e-12 * max(1.0, float(np.linalg.norm(coefficients)))
                pending = bridge.mat.createVecRight()
                compact_mode = coefficients[self.free[field]]
                if name in self._ordering:
                    compact_mode = compact_mode[self._ordering[name]["permutation"]]
                pending.array[:] = compact_mode
                for previous in vectors:
                    pending.axpy(-previous.dot(pending), previous)
                norm = float(pending.norm())
                if not np.isfinite(norm):
                    raise ValueError("Projected vector FE near-nullspace mode is non-finite")
                if norm <= zero_threshold:
                    pending.destroy()
                    pending = None
                    skipped += 1
                    continue
                pending.scale(1.0 / norm)
                vectors.append(pending)
                pending = None
            self.metadata["near_nullspace_modes"][name] = len(vectors)
            self.metadata["near_nullspace_skipped_modes"][name] = skipped
            if vectors:
                nullspace = PETSc.NullSpace().create(vectors=vectors, constant=False, comm=PETSc.COMM_SELF)
                bridge.mat.setNearNullSpace(nullspace)
                self._nullspaces[name] = nullspace
            else:
                self.metadata["proxy_fallbacks"][name] = {
                    "requested": "gamg",
                    "used": "ilu",
                    "reason": "no_nonzero_FE_translations_after_constraint_restriction",
                }
            self._nullvectors[name] = vectors
        except Exception:
            if pending is not None:
                pending.destroy()
            for mode in vectors:
                mode.destroy()
            raise

    def _configure_pc(self, name, bridge):
        method = "gamg" if name in ("velocity", "mesh") else "jacobi"
        fallback = self.metadata["proxy_fallbacks"].get(name)
        if fallback is not None:
            method = fallback["used"]
            fallback["ignored_amg_options"] = []
        self.metadata["proxy_methods"][name] = method
        pc = PETSc.PC().create(comm=PETSc.COMM_SELF)
        self._pcs[name] = pc
        pc.setOperators(bridge.mat)
        pc.setType(method)
        options = {}
        if method == "gamg":
            pc.setGAMGType("agg")
            options = {
                "pc_gamg_reuse_interpolation": True,
                "pc_gamg_threshold": 0.02,
                "pc_mg_cycle_type": "v",
                "mg_levels_ksp_type": "chebyshev",
                "mg_levels_ksp_max_it": 2,
                "mg_levels_ksp_norm_type": "none",
                "mg_levels_pc_type": "jacobi",
                "mg_coarse_ksp_type": "preonly",
                "mg_coarse_pc_type": "lu",
            }
        elif method == "ilu":
            pc.setFactorLevels(0)
        prefix = "fsi_reusable_%x_%s_" % (id(self), name)
        pc.setOptionsPrefix(prefix)
        database = PETSc.Options(prefix)
        try:
            for key, value in options.items():
                database[key] = value
            pc.setFromOptions()
            pc.setUp()
        finally:
            for key in options:
                del database[key]
        return pc

    def update(self, system, native_pc, proxy_revision=0):
        if self._closed:
            raise RuntimeError("Reusable PETSc preconditioner is closed")
        started = perf_counter()
        forms = self._proxy_forms(system, native_pc)
        self.pressure_inverse_scale = float(native_pc.pressure_inverse_scale)
        self.mesh_inverse_scale = float(native_pc.mesh_inverse_scale)
        if not np.isfinite([self.pressure_inverse_scale, self.mesh_inverse_scale]).all():
            raise ValueError("Proxy inverse scales must be finite")
        matrix_started = perf_counter()
        for name, (row, column) in self._couplings.items():
            matrix = system.full_tangent.get(row, column)
            if matrix is None:
                raise ValueError("Missing exact Newton coupling: " + name)
            if name not in self._bridges:
                self._bridges[name] = NativeCSRMatrix(matrix, self.masks[row], self.masks[column])
            else:
                self._bridges[name].update(matrix)
            self.metadata["couplings"][name] = self._bridges[name].metadata
        self.metadata["matrix_conversion_seconds"] += perf_counter() - matrix_started
        for name, field in self._proxy_fields.items():
            if self.sizes[field] == 0:
                if name in self._revisions:
                    self.metadata["proxy_reuses"][name] += 1
                self._revisions[name] = "empty"
                self.metadata["proxy_methods"][name] = "empty"
                self.metadata["proxy_vector_ordering"][name] = "empty"
                self.metadata["proxy_block_sizes"][name] = 1
                self.metadata["near_nullspace_modes"][name] = 0
                self.metadata["near_nullspace_skipped_modes"][name] = 3
                self.metadata["proxy_fallbacks"][name] = {
                    "requested": "gamg",
                    "used": "empty",
                    "reason": "no_free_DOFs_in_constrained_ALE_block",
                }
                self.metadata["proxies"][name] = {
                    "compact_shape": [0, 0],
                    "nonzeros": 0,
                    "updates": 0,
                    "compact_csr_bytes": 0,
                    "gather_bytes": 0,
                }
                continue
            revision = proxy_revision.get(name, 0) if isinstance(proxy_revision, dict) else proxy_revision
            signature = (id(forms[name]), revision)
            if name in self._proxies and self._revisions.get(name) == signature:
                self.metadata["proxy_reuses"][name] += 1
                continue
            setup_started = perf_counter()
            if name not in self._proxies:
                bridge = NativeCSRMatrix(forms[name].mat, self.masks[field])
                self._proxies[name] = bridge
                if name in ("velocity", "mesh"):
                    self._configure_vector_ordering(name, system.state_components[field].space, bridge)
                    self._attach_near_modes(name, system.state_components[field].space, bridge)
                bridge.mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
                self._configure_pc(name, bridge)
            else:
                bridge = self._proxies[name]
                bridge.update(forms[name].mat)
                if name not in self._pcs:
                    self._configure_pc(name, bridge)
                else:
                    pc = self._pcs[name]
                    pc.setReusePreconditioner(False)
                    pc.setOperators(bridge.mat)
                    pc.setUp()
            self._pcs[name].setReusePreconditioner(True)
            self._revisions[name] = signature
            self.metadata["proxy_setup_seconds"] += perf_counter() - setup_started
            self.metadata["proxy_updates"][name] += 1
            self.metadata["proxies"][name] = bridge.metadata
        constraint_started = perf_counter()
        schur = np.empty((3, 3))
        for index, basis in enumerate(self._mean_basis):
            self._rho_unit.set(0.0)
            self._rho_unit.array[index] = 1.0
            self._bridges["E"].mat.mult(self._rho_unit, self._u_rhs)
            self._apply_proxy("velocity", self._u_rhs, basis)
            self._bridges["D"].mat.mult(basis, self._rho_tmp)
            schur[:, index] = self._rho_tmp.array_r
        if not np.isfinite(schur).all() or np.linalg.cond(schur) > 1000000000000.0:
            raise RuntimeError("PETSc velocity-mean Schur matrix is singular or ill-conditioned")
        self._mean_inverse = np.linalg.inv(schur)
        self._bridges["C"].mat.mult(self._lambda_unit, self._c_unit)
        self._pressure_normalization = float(self._pressure_one.dot(self._c_unit))
        if not np.isfinite(self._pressure_normalization) or abs(self._pressure_normalization) < 1e-14:
            raise RuntimeError("PETSc pressure-mean constraint is singular")
        self.metadata["constraint_setup_seconds"] += perf_counter() - constraint_started
        self.metadata["updates"] += 1
        self.metadata["update_seconds"] += perf_counter() - started

    def attach(self, ksp):
        if self._closed:
            raise RuntimeError("Cannot attach a closed block preconditioner")
        pc = ksp.getPC()
        pc.setType("python")
        pc.setPythonContext(self._context)
        self.metadata["attachments"] += 1

    def _apply_proxy(self, name, source, destination):
        started = perf_counter()
        destination.set(0.0)
        ordering = self._ordering.get(name)
        if self.sizes[self._proxy_fields[name]] == 0:
            pass
        elif ordering is None:
            self._pcs[name].apply(source, destination)
        else:
            scatter_started = perf_counter()
            ordering["scatter"].scatter(
                source, ordering["rhs"], addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )
            self.metadata["proxy_scatter_seconds"] += perf_counter() - scatter_started
            ordering["solution"].set(0.0)
            self._pcs[name].apply(ordering["rhs"], ordering["solution"])
            scatter_started = perf_counter()
            ordering["scatter"].scatter(
                ordering["solution"],
                destination,
                addv=PETSc.InsertMode.INSERT_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )
            self.metadata["proxy_scatter_seconds"] += perf_counter() - scatter_started
        if name == "pressure":
            destination.scale(self.pressure_inverse_scale)
        elif name == "mesh":
            destination.scale(self.mesh_inverse_scale)
        self.metadata["proxy_apply_calls"][name] += 1
        self.metadata["proxy_apply_seconds"][name] += perf_counter() - started

    def _velocity_saddle(self, rhs, mean_rhs, velocity, multiplier):
        self._apply_proxy("velocity", rhs, velocity)
        self._bridges["D"].mat.mult(velocity, self._rho_tmp)
        self._rho_tmp.axpy(-1.0, mean_rhs)
        multiplier.array[:] = self._mean_inverse @ self._rho_tmp.array_r
        for coefficient, basis in zip(multiplier.array_r, self._mean_basis):
            velocity.axpy(-float(coefficient), basis)

    def apply(self, source, destination):
        if self._closed:
            raise RuntimeError("Reusable PETSc preconditioner is closed")
        started = perf_counter()
        source_array = source.array_r
        destination_array = destination.array
        if len(source_array) != self.offsets[-1] or len(destination_array) != self.offsets[-1]:
            raise ValueError("Compact PC vector dimensions differ from the six-field layout")
        placed = []
        try:
            for field, region in enumerate(self.slices):
                self._source[field].placeArray(source_array[region])
                placed.append(self._source[field])
                self._destination[field].placeArray(destination_array[region])
                placed.append(self._destination[field])
            ru, rp, rs, rr, rl, rw = self._source
            u, p, ps, rho, lam, w = self._destination
            self._velocity_saddle(ru, rr, u, rho)
            self._bridges["B"].mat.mult(u, self._p_rhs)
            self._p_rhs.aypx(-1.0, rp)
            pressure_multiplier = float(self._pressure_one.dot(self._p_rhs)) / self._pressure_normalization
            lam.array[0] = pressure_multiplier
            mean_pressure = float(rl.array_r[0]) / self._pressure_normalization
            self._c_unit.copy(self._p_tmp)
            self._p_tmp.scale(pressure_multiplier)
            self._p_tmp.axpy(-1.0, self._p_rhs)
            self._apply_proxy("pressure", self._p_tmp, p)
            p.axpy(mean_pressure, self._pressure_one)
            self._bridges["Bt"].mat.mult(p, self._u_rhs)
            self._velocity_saddle(self._u_rhs, self._rho_zero, self._u_correction, self._rho_correction)
            u.axpy(-1.0, self._u_correction)
            rho.axpy(-1.0, self._rho_correction)
            self._bridges["Bs"].mat.mult(u, self._ps_rhs)
            self._ps_rhs.aypx(-1.0, rs)
            self._apply_proxy("solid_pressure", self._ps_rhs, ps)
            self._bridges["Bst"].mat.mult(ps, self._u_rhs)
            self._velocity_saddle(self._u_rhs, self._rho_zero, self._u_correction, self._rho_correction)
            u.axpy(-1.0, self._u_correction)
            rho.axpy(-1.0, self._rho_correction)
            self._bridges["Wu"].mat.mult(u, self._w_rhs)
            self._w_rhs.aypx(-1.0, rw)
            self._apply_proxy("mesh", self._w_rhs, w)
            if not np.isfinite(destination_array).all():
                raise RuntimeError("PETSc AMG block action produced non-finite values")
            self.metadata["apply_calls"] += 1
        finally:
            for vector in reversed(placed):
                vector.resetArray()
            self.metadata["apply_seconds"] += perf_counter() - started

    def release_proxy_pcs(self):
        """Destroy the proxy PCs; the next update sets them up afresh.

        The CSR bridges, nodal orderings and near-nullspaces are kept, so a
        later Newton solve skips those, while GAMG builds a new hierarchy
        instead of reusing an interpolation from an earlier state.
        """
        for pc in self._pcs.values():
            pc.destroy()
        self._pcs.clear()
        self._revisions.clear()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.release_proxy_pcs()
        for nullspace in self._nullspaces.values():
            nullspace.destroy()
        self._nullspaces.clear()
        for vectors in self._nullvectors.values():
            for vector in vectors:
                vector.destroy()
        self._nullvectors.clear()
        for bridge in (*self._bridges.values(), *self._proxies.values()):
            bridge.close()
        self._bridges.clear()
        self._proxies.clear()
        for ordering in self._ordering.values():
            for name in ("scatter", "is"):
                if ordering[name] is not None:
                    ordering[name].destroy()
        self._ordering.clear()
        for vector in self._vectors:
            vector.destroy()
        self._vectors.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class PETScNewtonSolver:
    """Reusable PETSc GMRES with the production GAMG block preconditioner."""

    def __init__(
        self, *, mat, atol, tol, maxiter, system, preconditioner, cache, diagnostics, free_dof_blocks, template
    ):
        self.system = system
        self.preconditioner = preconditioner
        self.cache = cache
        self.diagnostics = diagnostics
        self.free_dof_blocks = free_dof_blocks
        self.template = template
        self.mat = mat
        self.tol, self.atol, self.maxiter = (tol, atol, maxiter)
        self.linear_metadata = {}

    def Solve(self, rhs, destination):
        setup_started = perf_counter()
        key = (self.system._linearization_revision, tuple(self.preconditioner.proxy_revisions.values()))
        if self.cache["key"] != key:
            if self.cache["pc"] is None:
                self.cache["pc"] = ReusablePETScBlockPC(
                    system=self.system,
                    native_pc=self.preconditioner,
                    free_dof_blocks=self.free_dof_blocks,
                    template=self.template,
                )
            else:
                self.cache["pc"].update(
                    self.system, self.preconditioner, proxy_revision=self.preconditioner.proxy_revisions
                )
            self.cache["key"] = key
        pc = self.cache["pc"]
        if self.cache["backend"] is None:
            self.cache["backend"] = ReusableNewtonKSP(
                native_matrix=self.mat,
                free_dof_blocks=self.free_dof_blocks,
                template=self.template,
                maxiter=self.maxiter,
                direct_pc=pc,
            )
            self.diagnostics["ksp_creations"] += 1
        backend = self.cache["backend"]
        backend.native_matrix = self.mat
        setup_seconds = perf_counter() - setup_started
        self.diagnostics["setup_seconds"] += setup_seconds
        info = backend.solve(rhs, destination, tol=self.tol, atol=self.atol)
        self.iterations = int(info["iterations"]) + 1
        self.residuals = [float(info["petsc_residual_estimate"])]
        self.linear_metadata = dict(info, setup_seconds=setup_seconds)


class DirectNewtonSolver:
    """Newton directions by a fresh MUMPS LU factorisation of the current Jacobian (newton_options
    linear_solver="mumps"), with the PETScNewtonSolver contract. For steps far longer than the cell's
    internal times, where the block preconditioner of the GMRES solver loses effect."""

    def __init__(
        self, *, mat, atol, tol, maxiter, system, preconditioner, cache, diagnostics, free_dof_blocks, template
    ):
        self.system = system
        self.free_dof_blocks = free_dof_blocks
        self.template = template
        self.tol, self.atol = (tol, atol)
        self.linear_metadata = {}

    def Solve(self, rhs, destination):
        started = perf_counter()
        try:
            with PETScMUMPSSolver(
                system=self.system, free_dof_blocks=self.free_dof_blocks, template=self.template, blas_threads=None
            ) as solver:
                info = solver.solve(rhs, destination, tol=min(self.tol, 1e-12), atol=self.atol)
        except (PETSc.Error, RuntimeError) as error:
            from .newton import NonlinearConvergenceFailure

            raise NonlinearConvergenceFailure(f"MUMPS Newton direction failed: {error}") from error
        self.iterations = 2  # one factor application (callers count iterations - 1)
        self.residuals = [float(info.get("native_residual", 0.0))]
        self.linear_metadata = dict(info, setup_seconds=perf_counter() - started - info.get("solve_seconds", 0.0))
