"""Solve periodic FSI/ALE cells and compute their homogenised response and tangent.

The entry point is ``FSI(...)``; ``FSISession`` reuses one cell's workspace
across sequential calls. Parameters, returned entries, exceptions and examples
are documented in the "Solver interface" section of README.md. The solver
itself is the ``periodic_fsi`` package next to this file; this module only
makes that package importable when the file is loaded by path, and re-exports
its public names.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path

# Consumers load this file by path, so put its directory on sys.path to find
# the sibling package, and refuse a package already imported from elsewhere.
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import periodic_fsi as _periodic_fsi

if _Path(_periodic_fsi.__file__).resolve().parent != _HERE / "periodic_fsi":
    raise ImportError(
        f"periodic_fsi already imported from {_periodic_fsi.__file__}, not {_HERE}; "
        "load solvers from different checkouts in separate processes"
    )
from periodic_fsi.core import generate_cell_mesh, summarize_performance  # noqa: E402,F401
from periodic_fsi.layout import BlockSolution  # noqa: E402,F401
from periodic_fsi.api import FSI, FSISession  # noqa: E402,F401
from periodic_fsi.result import FSIResult  # noqa: E402,F401
from periodic_fsi.driver import restore_accepted_mesh  # noqa: E402,F401
from periodic_fsi.profiling import format_profile_table  # noqa: E402,F401

__all__ = [
    "BlockSolution",
    "FSI",
    "FSIResult",
    "FSISession",
    "format_profile_table",
    "generate_cell_mesh",
    "restore_accepted_mesh",
    "summarize_performance",
]
