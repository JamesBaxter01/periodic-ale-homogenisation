"""Self-contained NGSolve snapshots. Load only trusted pickle files."""
from pathlib import Path
import hashlib
import os
import pickle
import platform
import sys

SNAPSHOT_SCHEMA = 1


def _restore_matrix_space(base_space):
    # Preserve the solver's exact constructor (including its symmetry flag).
    # Native MatrixValued unpickling aborts in NGSolve 6.2.2606 on macOS;
    # constructing the same native space avoids that pybind11 failure.
    # The symmetric space: the rate of deformation, and the F history of snapshots
    # saved before the solid's F history became the full space.
    from ngsolve import MatrixValued
    return MatrixValued(base_space, 3, 3)


def _restore_full_matrix_space(base_space):
    # The full 3 x 3 space: the solid's F history (since 26 Oct 2026).
    from ngsolve import MatrixValued
    return MatrixValued(base_space, 3)


class SnapshotPickler(pickle.Pickler):
    def reducer_override(self, obj):
        from ngsolve import MatrixValued
        if isinstance(obj, MatrixValued):
            if len(obj.components) == 6 and "symmetric" in str(obj.flags):
                return _restore_matrix_space, (obj.components[0],)
            if len(obj.components) == 9:
                return _restore_full_matrix_space, (obj.components[0],)
            raise ValueError("Unexpected MatrixValued space; update the archive adapter")
        return NotImplemented


def solver_sha256(entry):
    """Digest of a solver: its entry file plus every module of the periodic_fsi package beside it.

    Each file adds its path relative to the entry's folder and its bytes, in sorted
    order, so editing, adding or removing a package module changes the digest.
    periodic_fsi/sequence.py computes the same digest for HDF5 history metadata.
    """
    entry = Path(entry)
    digest = hashlib.sha256()
    for path in [entry, *sorted((entry.parent / "periodic_fsi").glob("*.py"))]:
        digest.update(path.relative_to(entry.parent).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_snapshot(path, result, *, case, settings, increment, time,
                  compression, solver_sha256, previous_restart_state):
    """One pickle stream preserves shared mesh/space references.

    Never serialize solver matrices, preconditioners, or the custom BlockSolution.
    Persistent solution and history GridFunctions are retained. Corrector field
    snapshots are not exported by the authoritative publication solver. Temporary Krylov/proxy/workspace fields are not physical state.
    """
    import ngsolve
    import numpy as np
    import netgen
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result_copy = {k: v for k, v in result.items() if k not in {
        "gfu", "linear_system", "preconditioner", "homogenisation_preconditioner"}}
    fields = dict(zip(result["field_order"], result["gfu"].components))
    result_copy["solution_fields"] = fields
    result_copy["solution_spaces"] = dict(zip(fields, result["gfu"].spaces))
    snapshot = {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "case": dict(case), "settings": dict(settings),
        "increment": increment, "time": time, "compression": compression,
        "is_unloaded_reference": increment == 0,
        "reference_for": [family for family, key in settings.get("reference_cases", {}).items() if key == case["id"]],
        "solver_sha256": solver_sha256,
        "versions": {"python": sys.version, "ngsolve": ngsolve.__version__,
                     "netgen": getattr(netgen, "__version__", "unknown"),
                     "numpy": np.__version__, "platform": platform.platform()},
        "mesh_storage": "reference coordinates; call SetDeformation(mesh_displacement) for current geometry",
        "field_units": "internal nondimensional; see result.nondimensional_scales",
        "response_units": "physical; *_hat arrays are dimensionless",
        "previous_restart_state": previous_restart_state,
        "result": result_copy,
    }
    mesh = result["mesh"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    mesh.UnsetDeformation()
    try:
        with temporary.open("wb") as stream:
            SnapshotPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(snapshot)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        mesh.SetDeformation(result["mesh_displacement"])
        if temporary.exists():
            temporary.unlink()
    return {"path": str(path.name), "bytes": path.stat().st_size,
            "sha256": file_sha256(path), "increment": increment,
            "time": time, "compression": compression}


def load_snapshot(path, *, deformed=False, expected_sha256=None):
    """Load a trusted snapshot in a compatible NGSolve environment."""
    import ngsolve  # Registers native pickle types.
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"Snapshot checksum mismatch: {path}")
    with Path(path).open("rb") as stream:
        snapshot = pickle.load(stream)
    if snapshot.get("snapshot_schema") != SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported snapshot schema")
    result = snapshot["result"]
    result["mesh"].UnsetDeformation()
    if deformed:
        result["mesh"].SetDeformation(result["mesh_displacement"])
    return snapshot


def mark_retained_snapshots(status, keep=1):
    """Plan retention before committing status; preserve every numeric step index."""
    if keep < 1:
        raise ValueError("At least one restart checkpoint must be retained")
    items = status["snapshots"]
    for index, item in enumerate(items):
        item["retained"] = index >= len(items) - keep
    status["snapshot_bytes"] = sum(item["bytes"] for item in items if item["retained"])
    status["snapshot_retention"] = keep


def remove_retired_snapshots(folder, status):
    """Call only AFTER status is committed, so interruption preserves restart.

    Delete only explicitly retired files from the committed index. Small numeric
    files and the index itself remain available for analysis and resume.
    """
    for item in status["snapshots"]:
        if item.get("retained", True):
            continue
        name = item["path"]
        if Path(name).name != name or not name.endswith(".pkl"):
            raise ValueError("Invalid checkpoint filename")
        (Path(folder) / "states" / name).unlink(missing_ok=True)
