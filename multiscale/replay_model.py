"""ReplayModel: a saved FE² history played back as a material model, with no cell solves.

The history of a finished run (driver.run with save_tangents) holds, for every accepted step n and
every point, the converged inputs x_n, outputs y_n and one-step tangent C_n. Step n is answered by the
first-order expansion about the stored point,

    y(x) = y_n + C_n (x - x_n),    tangent C_n,

which is exact at x_n. The macro residual vanishes there, so the macro Newton solve of step n converges
back to the stored solution (to its tolerance) in an iteration or two. Replaying the run on the same
macro problem and time grid therefore reproduces it in seconds: a check that the saved data is
complete and consistent, and the best any surrogate trained on it could do on this run.

The step is the number of commits so far (the rest evaluation is row 0). The history has data only at
the stored dt of each step: any other dt raises MaterialPointFailure, which makes the driver cut the
step exactly as it did when the original run failed a step, so the replay follows the stored grid.
It cannot go past the last stored step, and away from the stored inputs it is only a linearisation.
"""

from __future__ import annotations

import numpy as np

from .interface import MaterialPointFailure, Response, check_inputs


class ReplayModel:
    def __init__(self, n_points, *, history, dt_tolerance=1e-9):
        data = np.load(history)
        if "tangents" not in data.files:
            raise ValueError(f"{history} has no tangents (run with save_tangents=True)")
        self.inputs, self.outputs, self.tangents = data["inputs"], data["outputs"], data["tangents"]
        self.dt = data["dt"]
        self.n_points = int(n_points)
        if self.inputs.shape[1] != self.n_points:
            raise ValueError(f"{history} has {self.inputs.shape[1]} points, not {self.n_points}")
        self.dt_tolerance = float(dt_tolerance)
        self.step = 0  # the stored row the next evaluation answers
        self.calls = 0
        self.offset = 0.0  # max |x - x_n| of the latest evaluation
        self.offsets = []  # ... of the committed evaluation of every step: ~0 when the replay lands on the stored points

    @property
    def steps(self):
        return len(self.dt)

    def evaluate(self, inputs, dt, *, tangent=True):
        x = check_inputs(inputs, self.n_points)
        n = self.step
        if n >= self.steps:
            raise RuntimeError(f"the history ends after step {self.steps - 1}")
        if n > 0 and abs(dt - self.dt[n]) > self.dt_tolerance * self.dt[n]:
            raise MaterialPointFailure(f"no stored data at dt = {dt:.10g} s for step {n} (stored {self.dt[n]:.10g} s)",
                                       np.arange(self.n_points))
        self.calls += 1
        dx = x - self.inputs[n]
        self.offset = float(np.max(np.abs(dx)))
        C = self.tangents[n]
        y = self.outputs[n] + np.einsum("nij,nj->ni", C, dx)
        return Response(y, C.copy() if tangent else None)

    def commit(self):
        self.offsets.append(self.offset)
        self.step += 1

    def close(self):
        pass
