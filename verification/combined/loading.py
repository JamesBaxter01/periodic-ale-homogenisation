"""The verification loading (paper Section 4; hpc/README.md), as a loading path for FSI.

With s(t) = (1 - cos(pi t / T)) / 2 on 0 <= t <= T:

    F33 = 1 - compression * s(t)          (20 % compression)
    F12 = shear * s(t)                    (breaks the cell's symmetry)
    p   = pressure * sin(pi t / T)        (rises and falls: storage power changes sign)
    grad p = gradient * s(t) * d / |d|    (all three flux components nonzero)

The 13 inputs are F row-major (9), p, and the spatial pressure gradient (3), the
column order of FSI's loading_path. The settings come from configs/verification.json.
"""

import numpy as np

INPUT_NAMES = [f"F{i}{j}" for i in range(1, 4) for j in range(1, 4)] + ["p", "g1", "g2", "g3"]
OUTPUT_NAMES = [f"P{i}{j}" for i in range(1, 4) for j in range(1, 4)] + ["m", "Q1", "Q2", "Q3"]


def ramp(t, T):
    """s(t) = (1 - cos(pi t / T)) / 2: 0 at t = 0, 1 at t = T, zero rate at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.asarray(t, dtype=float) / T))


def inputs(t, T, loading):
    """The 13 inputs at time(s) t for duration T; shape (13,) or (len(t), 13)."""
    t = np.asarray(t, dtype=float)
    s = ramp(t, T)
    direction = np.asarray(loading["gradient_direction"], dtype=float)
    direction = direction / np.linalg.norm(direction)
    z = np.zeros(t.shape + (13,))
    z[..., 0] = z[..., 4] = 1.0
    z[..., 8] = 1.0 - loading["compression"] * s
    z[..., 1] = loading["shear"] * s
    z[..., 9] = loading["pressure"] * np.sin(np.pi * t / T)
    z[..., 10:] = loading["gradient"] * s[..., None] * direction
    return z


def times(T, steps):
    """The N + 1 time points t_k = k T / N."""
    return T * np.arange(steps + 1) / steps


def loading_path(T, steps, loading):
    """(N + 1, 13) loading_path for FSI: row k applies at t_k = k T / N."""
    return inputs(times(T, steps), T, loading)

