"""Compact free-DOF layout and the NGSolve-to-PETSc CSR bridge."""

from __future__ import annotations
from time import perf_counter
import numpy as np
from ngsolve import BlockVector, GridFunction, Norm
from petsc4py import PETSc

# The six state fields, in block order.
U_FIELD, PF_FIELD, PS_FIELD, LAMBDA_U_FIELD, LAMBDA_PF_FIELD, ALE_FIELD = range(6)
FIELD_NAMES = ("u", "p_f", "p_s", "lambda_u", "lambda_pf", "delta_w")


def _validate_build():
    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise RuntimeError("PETSc block inverses require a real-double PETSc build")
    if PETSc.COMM_WORLD.getSize() != 1:
        raise RuntimeError("This FreeDofs/CSR bridge supports one MPI rank")


def _mask_indices(mask, size):
    values = np.asarray(list(mask), dtype=bool)
    if values.shape != (size,):
        raise ValueError("FreeDofs mask does not match the native matrix dimension")
    return np.flatnonzero(values).astype(PETSc.IntType)


# Compact PETSc bridges contain independent periodic DOFs only.
class NativeCSRMatrix:
    """Compact scalar-entry AIJ block with a reusable native-value gather.

    Rectangular blocks are supported.  ``row_free`` and ``col_free`` are
    boolean masks, rather than lists of indices.  The finite-element graph
    must remain unchanged on update; changing it requires a new instance.
    NGSolve VectorH1 matrices already have scalar entries in component-major
    order, so no artificial PETSc block size is imposed.
    """

    def __init__(self, native_matrix, row_free, col_free=None):
        _validate_build()
        self.mat = None
        self._closed = False
        self.metadata = {"updates": 0, "conversion_seconds": 0.0}
        started = perf_counter()
        local = getattr(native_matrix, "local_mat", native_matrix)
        if tuple(local.entrysizes) != (1, 1):
            raise ValueError("CSR bridge expects scalar entries (including VectorH1)")
        self.shape = (local.height, local.width)
        self.row_free = _mask_indices(row_free, self.shape[0])
        self.col_free = _mask_indices(row_free if col_free is None else col_free, self.shape[1])
        values, columns, pointers = local.CSR()
        self._graph_owner = local
        self._source_columns = np.asarray(columns)
        self._source_pointers = np.asarray(pointers)
        mapping = np.full(self.shape[1], -1, dtype=PETSc.IntType)
        mapping[self.col_free] = np.arange(len(self.col_free), dtype=PETSc.IntType)
        self._symmetric_storage = "SparseMatrixSymmetric" in type(local).__name__
        if self._symmetric_storage:
            row_map = np.full(self.shape[0], -1, dtype=PETSc.IntType)
            row_map[self.row_free] = np.arange(len(self.row_free), dtype=PETSc.IntType)
            native_rows = np.repeat(np.arange(self.shape[0]), np.diff(self._source_pointers).astype(np.int64))
            native_cols = self._source_columns
            forward = (row_map[native_rows] >= 0) & (mapping[native_cols] >= 0)
            backward = (
                (native_rows != native_cols) & (row_map[native_cols] >= 0) & (mapping[native_rows] >= 0)
            )
            rows = np.r_[row_map[native_rows[forward]], row_map[native_cols[backward]]]
            cols = np.r_[mapping[native_cols[forward]], mapping[native_rows[backward]]]
            positions = np.r_[np.flatnonzero(forward), np.flatnonzero(backward)]
            ordering = np.lexsort((cols, rows))
            self._positions = positions[ordering]
            self._columns = cols[ordering].astype(PETSc.IntType, copy=False)
            counts = np.bincount(rows, minlength=len(self.row_free))
        else:
            row_counts = np.diff(self._source_pointers)[self.row_free].astype(np.int64)
            starts = self._source_pointers[self.row_free].astype(np.int64)
            offsets = np.repeat(starts - np.r_[0, np.cumsum(row_counts[:-1])], row_counts)
            positions = offsets + np.arange(int(np.sum(row_counts)))
            compact_columns = mapping[self._source_columns[positions]]
            keep = compact_columns >= 0
            rows = np.repeat(np.arange(len(self.row_free)), row_counts)
            counts = np.bincount(rows[keep], minlength=len(self.row_free))
            self._positions = positions[keep]
            self._columns = compact_columns[keep].astype(PETSc.IntType, copy=False)
        self._pointers = np.r_[0, np.cumsum(counts)].astype(PETSc.IntType)
        self._values = np.empty(len(self._positions), dtype=PETSc.ScalarType)
        np.take(np.asarray(values), self._positions, out=self._values)
        try:
            self.mat = PETSc.Mat().createAIJ(
                size=(len(self.row_free), len(self.col_free)),
                csr=(self._pointers, self._columns, self._values),
                comm=PETSc.COMM_SELF,
            )
            self.mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
            self.mat.assemble()
        except Exception:
            self.close()
            raise
        self.metadata.update(
            {
                "updates": 1,
                "native_shape": self.shape,
                "expanded_symmetric_storage": self._symmetric_storage,
                "compact_shape": (len(self.row_free), len(self.col_free)),
                "nonzeros": len(self._positions),
                "compact_csr_bytes": self._pointers.nbytes + self._columns.nbytes + self._values.nbytes,
                "gather_bytes": self._positions.nbytes,
                "conversion_seconds": perf_counter() - started,
            }
        )

    def update(self, native_matrix):
        if self._closed:
            raise RuntimeError("PETSc matrix is closed")
        started = perf_counter()
        local = getattr(native_matrix, "local_mat", native_matrix)
        values, columns, pointers = local.CSR()
        if (
            (local.height, local.width) != self.shape
            or tuple(local.entrysizes) != (1, 1)
            or ("SparseMatrixSymmetric" in type(local).__name__) != self._symmetric_storage
            or (not np.array_equal(np.asarray(pointers), self._source_pointers))
            or (not np.array_equal(np.asarray(columns), self._source_columns))
        ):
            raise ValueError("Finite-element graph changed; create a new CSR bridge")
        np.take(np.asarray(values), self._positions, out=self._values)
        self.mat.setValuesCSR(
            self._pointers, self._columns, self._values, addv=PETSc.InsertMode.INSERT_VALUES
        )
        self.mat.assemble()
        self.metadata["updates"] += 1
        self.metadata["conversion_seconds"] += perf_counter() - started

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.mat is not None:
            self.mat.destroy()
            self.mat = None
        for name in (
            "_graph_owner",
            "_source_columns",
            "_source_pointers",
            "_positions",
            "_columns",
            "_pointers",
            "_values",
        ):
            setattr(self, name, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class BlockLayout:
    """Per-field FreeDofs masks and in-place operations on block state vectors."""

    def __init__(self, free_dof_blocks):
        self.free_dof_blocks = tuple(free_dof_blocks)
        self.number_of_fields = len(self.free_dof_blocks)

    def zero(self, vector):
        for block in range(self.number_of_fields):
            vector[block][:] = 0.0

    def copy(self, destination, source):
        for block in range(self.number_of_fields):
            destination[block].data = source[block]

    def axpy(self, destination, alpha, source):
        for block in range(self.number_of_fields):
            destination[block].data += alpha * source[block]

    def apply(self, matrix, source, destination):
        """Materialise a BlockMatrix product into an indexable BlockVector.

        NGSolve's ``matrix * block_vector`` overload returns a lazy
        ``DynamicVectorExpression``.  The explicit residual checks need the
        individual field blocks, so use the concrete ``Mult`` interface.
        """
        matrix.Mult(source, destination)
        return destination

    def norm(self, vector):
        return float(np.sqrt(sum((Norm(vector[block]) ** 2 for block in range(self.number_of_fields)))))

    def mask(self, vector):
        for block, block_freedofs in enumerate(self.free_dof_blocks):
            vector[block].data[~block_freedofs] = 0.0


class BlockSolution:
    """Independent GridFunctions with a reconstructed BlockVector on load."""

    def __init__(self, field_names, field_spaces, total_state_dofs, name="monolithic_fsi_ale_state"):
        self.spaces = tuple(field_spaces)
        self.ndof = total_state_dofs
        self.components = tuple(
            (
                GridFunction(space, name=f"{name}_{field_name}")
                for field_name, space in zip(field_names, self.spaces)
            )
        )
        self._rebuild_vec()

    def _rebuild_vec(self):
        self.vec = BlockVector([component.vec for component in self.components])

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("vec", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._rebuild_vec()
