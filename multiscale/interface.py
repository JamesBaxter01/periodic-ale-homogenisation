"""The seam between the macroscale and any material-point model.

The macro side sees only ``MaterialModel``: a batched ``evaluate`` and a
``commit``. A finite element cell, a surrogate and an analytic law all
implement it, so they are interchangeable.

Contract:

- ``evaluate(inputs, dt)`` depends only on the committed history, the inputs
  and dt. Trial evaluations never change the committed history, so line
  searches, backtracking and dt cuts are safe.
- ``commit()`` makes the most recent evaluation of every point its new
  history. Only the driver calls it, once per accepted macro step.
- ``tangent`` is d outputs(t_n) / d inputs(t_n) with the history committed
  at t_(n-1) held fixed: the consistent tangent of one macro step.
- An inadmissible trial raises ``MaterialPointFailure``.
- Construction: ``Model(n_points, **settings)``, so ``build_model`` makes any
  model from a name and its settings, and an example swaps models by
  changing one setting (see ``build_model``).
- Optional: ``save_checkpoint(folder)`` / ``load_checkpoint(folder)`` write
  and restore the committed history, for runs long enough to need resuming
  (the FE cell). A model without them is simply not checkpointed.
"""

from __future__ import annotations

import importlib

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

# The cell solver's layout (periodic_fsi.homogenisation.INPUT_NAMES / OUTPUT_NAMES).
# Inputs: F row-major, macro pressure P0, SPATIAL pressure gradient G = F^-T grad_X p.
INPUT_NAMES = tuple(f"F_{i + 1}{j + 1}" for i in range(3) for j in range(3)) + ("P0", "G_1", "G_2", "G_3")
# Outputs: total nominal (first Piola) stress row-major, fluid content per
# reference volume, nominal fluid flux.
OUTPUT_NAMES = tuple(f"P_{i + 1}{j + 1}" for i in range(3) for j in range(3)) + (
    "fluid_content",
    "flux_1",
    "flux_2",
    "flux_3",
)
N_INPUTS = N_OUTPUTS = 13
STRESS = slice(0, 9)
CONTENT = 9
FLUX = slice(10, 13)


class MaterialPointFailure(RuntimeError):
    """A trial a material point cannot accept (no convergence, inverted cell, ...).

    The macro solver treats it as "this trial step is too large", not as a bug.
    """

    def __init__(self, message, points=()):
        super().__init__(message)
        self.points = np.asarray(points, dtype=int)


@dataclass(frozen=True)
class Response:
    """Outputs of one batched evaluation, in physical units."""

    outputs: np.ndarray  # (n, 13)
    tangent: np.ndarray | None  # (n, 13, 13) d outputs / d inputs, or None
    info: Mapping[str, np.ndarray] = field(default_factory=dict)  # per-point diagnostics


class MaterialModel(Protocol):
    n_points: int

    def __init__(self, n_points: int, **settings) -> None: ...

    def evaluate(self, inputs: np.ndarray, dt: float, *, tangent: bool = True) -> Response: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


# Built-in models by name; build_model also takes "package.module:Class" for any other (a surrogate).
MODELS = {
    "fe": "multiscale.fe_cell:FECellModel",
    "stationary_fe": "multiscale.stationary_cell:StationaryCellModel",
    "linear": "multiscale.linear_model:LinearPoroelastic",
    "finite_strain": "multiscale.hyperelastic_model:FiniteStrainPoroelastic",
    "replay": "multiscale.replay_model:ReplayModel",
}


def model_class(kind):
    """The class named by ``kind``: a key of MODELS or an import path "package.module:Class"."""
    path = MODELS.get(kind, kind)
    if ":" not in path:
        raise ValueError(f"unknown material model {kind!r}: use one of {sorted(MODELS)} or 'package.module:Class'")
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def build_model(kind, n_points, **settings):
    """``Checked(Model(n_points, **settings))`` for the model named by ``kind`` (see model_class)."""
    return Checked(model_class(kind)(n_points, **settings))


def check_inputs(inputs, n_points):
    """Return inputs as a float (n_points, 13) array, or raise ValueError."""
    inputs = np.asarray(inputs, dtype=float)
    if inputs.shape != (n_points, N_INPUTS):
        raise ValueError(f"inputs must have shape ({n_points}, {N_INPUTS}), not {inputs.shape}")
    if not np.isfinite(inputs).all():
        raise ValueError("inputs must be finite")
    return inputs


class Checked:
    """Wrap a model: reject non-finite outputs or tangents and non-positive fluid content.

    Violations become ``MaterialPointFailure`` so the macro step backtracks.
    """

    def __init__(self, model):
        self.model = model
        self.n_points = model.n_points

    def evaluate(self, inputs, dt, *, tangent=True):
        inputs = check_inputs(inputs, self.n_points)
        F = inputs[:, :9].reshape(-1, 3, 3)
        bad = np.flatnonzero(np.linalg.det(F) <= 0)
        if len(bad):
            raise MaterialPointFailure(f"non-positive det F at points {bad.tolist()}", bad)
        response = self.model.evaluate(inputs, dt, tangent=tangent)
        y = np.asarray(response.outputs)
        if y.shape != (self.n_points, N_OUTPUTS):
            raise RuntimeError(f"model returned outputs of shape {y.shape}")
        bad = np.flatnonzero(~np.isfinite(y).all(axis=1) | (y[:, CONTENT] <= 0))
        if tangent:
            C = response.tangent
            if C is None or np.shape(C) != (self.n_points, N_OUTPUTS, N_INPUTS):
                raise RuntimeError("model returned no tangent or one of the wrong shape")
            bad = np.union1d(bad, np.flatnonzero(~np.isfinite(C).all(axis=(1, 2))))
        if len(bad):
            raise MaterialPointFailure(f"non-finite response or non-positive content at points {bad.tolist()}", bad)
        return response

    def commit(self):
        self.model.commit()

    def close(self):
        self.model.close()

    def __getattr__(self, name):
        # Counters and settings of the wrapped model (calls, cell_seconds, ...).
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
