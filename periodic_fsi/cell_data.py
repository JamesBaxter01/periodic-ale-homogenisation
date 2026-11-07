"""Read-only groups of CellProblem data, built once per cell.

The Newton solve, the system assembly and the homogenisation take these
groups instead of the whole cell, so each one shows what it depends on.
An FSISession time-step change replaces ``Material`` and ``SolverSettings``
with copies holding the rescaled viscosities and pressure-mass scale.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Nondimensional material and mesh-motion parameters."""

    mu_s: float
    lam_s: float
    eta_s: float
    zeta_s: float
    mu_f: float
    mesh_mu: float
    mesh_lame_lambda: float
    solid_pressure_compliance: float
    solid_pressure_log_coefficient: float

    @classmethod
    def from_elastic(cls, E, nu, *, eta_s, zeta_s, mu_f, mesh_mu, mesh_lame_lambda):
        """Lamé parameters from (E, nu); the solid-pressure row is scaled by max(|lam_s|, mu_s)."""
        mu_s = E / (2.0 * (1.0 + nu))
        lam_s = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        solid_pressure_row_scale = max(abs(lam_s), mu_s)
        return cls(
            mu_s=mu_s,
            lam_s=lam_s,
            eta_s=eta_s,
            zeta_s=zeta_s,
            mu_f=mu_f,
            mesh_mu=mesh_mu,
            mesh_lame_lambda=mesh_lame_lambda,
            solid_pressure_compliance=1.0 / solid_pressure_row_scale,
            solid_pressure_log_coefficient=lam_s / solid_pressure_row_scale,
        )


@dataclass(frozen=True)
class SolverSettings:
    """Validated Newton and linear-solver options, output switches and the run's profiler."""

    options: dict
    pc_options: dict
    gmres_tol: float
    gmres_maxsteps: int
    print_iterations: bool
    profiling: bool
    profiler: object
    diagnostics: bool
    compile_tangent: bool


@dataclass(frozen=True)
class Discretisation:
    """Reference mesh, finite-element spaces, quadrature and the block layout of the six fields."""

    mesh: object
    order: int
    V: object
    P_f: object
    P_s: object
    W_delta: object
    field_spaces: tuple
    volume_integration_rules: dict
    block_layout: object
    free_dof_blocks: tuple
    total_state_dofs: int
    number_of_fields: int


@dataclass(frozen=True)
class CellGeometry:
    """Solid and fluid regions, reference volumes, the position field and the cell centre."""

    solid_region: object
    fluid_region: object
    reference_volume: float
    reference_solid_volume: float
    reference_fluid_volume: float
    pos: object
    xbar: object
