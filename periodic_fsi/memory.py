"""Releasing native solver memory between solves."""

from __future__ import annotations
import gc


def _collect_solver_memory(trim_native_heap=True):
    """Collect unreachable wrappers and optionally return free glibc pages."""
    collected = gc.collect()
    trimmed = False
    if trim_native_heap:
        try:
            import ctypes

            libc = ctypes.CDLL(None)
            trim = getattr(libc, "malloc_trim", None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trimmed = bool(trim(0))
        except (AttributeError, OSError):
            pass
    return (collected, trimmed)


def _detach_finished_preconditioner(preconditioner):
    """Drop Python-owned native resources only after the last application.

    This handles the lagged/current aliases and scratch vectors together.
    It does not mutate the shared system/forms or the physical state.
    The detached object must never be applied again.
    """
    if preconditioner is not None:
        for name in tuple(vars(preconditioner)):
            setattr(preconditioner, name, None)
