"""StationaryCellModel: the periodic FSI cell without history (a stationary, memoryless subscale).

Every evaluation solves a fresh cell, with no state carried between evaluations: from rest, one
backward-Euler step of STEPS[0] to the current inputs, then steps held at them. Each step is far longer
than the cell's internal times (the slowest, the cell-scale drainage of its fluid content, is about 7 s),
so each removes all but about tau/dt ~ 1e-2 of the remaining departure from equilibrium, and three
leave about 1e-7: the response and tangent are those of the steady cell problem (the pore fluid in
steady flow through the current geometry, the skeleton in equilibrium with no viscous stress), the
large-step limit of the same discrete equations as the cell with history. They depend on the current
inputs only, as the equilibrated parameters of the paper's Table 5 do.

Newton directions use a direct MUMPS factorisation (newton_options linear_solver="mumps"): the GMRES
block preconditioner, built for transient steps, fails at such steps under large pressure gradients
(the top of the cyclic column). Steps much longer still (1e4 s) make the system singular.

Each evaluation records ``equilibrium``, the change of its outputs over the last step (stress and fluid
content relative, flux in m/s); the driver saves the largest over the cells at every step, so a run
shows its own convergence. Same cell, settings and parallel workers as FECellModel; compare with
FECellModel, which carries the transient microscopic state between steps.
"""

from __future__ import annotations

from .fe_cell import FECellModel, FreshEquilibrium

STEPS = (1000.0, 1000.0, 1000.0)  # (s): one to the inputs, then two held there


class StationaryCellModel(FECellModel):
    def __init__(self, n_points, *, steps=STEPS, **settings):
        newton = dict(settings.pop("newton_options", None) or {}, linear_solver="mumps")
        super().__init__(n_points, newton_options=newton, **settings)
        self.relaxation = FreshEquilibrium(tuple(map(float, steps)))

    def evaluate(self, inputs, dt, *, tangent=True):
        return super().evaluate(inputs, self.relaxation, tangent=tangent)
