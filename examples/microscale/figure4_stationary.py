"""Stationary cell response for Figure 4: the cell without history.

Press Run (or python run.py figure4-stationary [--smoke]). The stationary
response depends only on the current inputs: here the compression c (F = diag(1, 1, 1 - c)) under the
background pressure gradient of Figure 4, at p = 0. At each of POINTS compressions from 0 to COMPRESSION, a
fresh cell (multiscale/stationary_cell.py, the stationary cell of the cyclic consolidation study) is taken
from rest to the inputs and held there by long backward-Euler steps with direct (MUMPS) Newton solves, to
the steady cell problem: steady flow through the current pore geometry, no skeleton viscous stress. No
state passes between compressions, so the response is the same for the elastic and the viscoelastic solid
and for every pulse duration, and plot_paper_figures.py reads it at c(t). Each point records its
equilibrium check, the change of its outputs over the last step (stress and fluid content relative, flux
in m/s). On Calder: examples/microscale/hpc/submit_figure4_stationary.sh.

Saves results/paper/figure4_stationary.npz (compression, flux, fluid content, nominal stress, equilibrium).
"""

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from paper_common import command_line, save_results
from geometry import square_geo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from multiscale.stationary_cell import STEPS, StationaryCellModel  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: edit here and press Run.
# ---------------------------------------------------------------------------
K, G = 50.0, 50.0                 # as figure4_cycles.py
MU_F = 1.0
CHANNEL_WIDTH = 0.33
RVE_LENGTH = 1.0
MAXH, ORDER, QUADRATURE_ORDER = 0.09, 2, None
MESH_MU, MESH_LAME_LAMBDA = 0.01, 0.05
COMPRESSION = 0.2                 # peak compression of the Figure 4 pulse
PRESSURE_GRADIENT = [-1.0, 0.0, 0.0]
POINTS = 41                       # compressions from 0 to COMPRESSION
THREADS = 4                       # cells solved in parallel, one thread each (--threads)
# ---------------------------------------------------------------------------
SMOKE = dict(POINTS=3, COMPRESSION=0.001, MAXH=0.7)


def main():
    compressions = np.linspace(0.0, COMPRESSION, POINTS)
    inputs = np.array([np.r_[np.diag([1.0, 1.0, 1.0 - c]).ravel(), 0.0, PRESSURE_GRADIENT] for c in compressions])
    started = time.perf_counter()
    cell = dict(K=K, G=G, mu_f=MU_F, rve_length=RVE_LENGTH, maxh=MAXH, order=ORDER, quadrature_order=QUADRATURE_ORDER,
                mesh_mu=MESH_MU, mesh_lame_lambda=MESH_LAME_LAMBDA, geometry=partial(square_geo, CHANNEL_WIDTH),
                print_iterations=False, compile_tangent=False)
    with StationaryCellModel(POINTS, workers=min(THREADS, POINTS), threads_per_worker=1, **cell) as model:
        response = model.evaluate(inputs, 0.0, tangent=False)
    outputs, equilibrium = response.outputs, response.info["equilibrium"]
    for c, row, change in zip(compressions, outputs, equilibrium):
        print(f"  c = {c:.4f}: Q1 = {row[10]:.6g}, m_f = {row[9]:.6f}, last-step change (stress, content, flux) "
              f"{change[0]:.1e}, {change[1]:.1e}, {change[2]:.1e}")
    settings = dict(K=K, G=G, mu_f=MU_F, channel_width=CHANNEL_WIDTH, rve_length=RVE_LENGTH, maxh=MAXH, order=ORDER,
                    compression=COMPRESSION, pressure_gradient=PRESSURE_GRADIENT, points=POINTS, steps=list(STEPS))
    save_results("figure4_stationary", settings, time.perf_counter() - started, compression=compressions,
                 flux=outputs[:, 10:], fluid_content=outputs[:, 9], stress=outputs[:, :9].reshape(-1, 3, 3),
                 equilibrium=equilibrium)


if __name__ == "__main__":
    command_line(globals(), SMOKE)
    main()
