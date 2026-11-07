"""FSIResult: the value FSI returns, a frozen dataclass that reads like the old result dict."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True, eq=False)
class FSIResult(Mapping):
    """Result of one FSI call; see "Solver interface" in README.md for each entry.

    Entries are attributes (``result.restart_state``) and also read-only
    mapping items (``result["restart_state"]``, ``get``, ``keys``, ``items``,
    ``in``), in the order the solver produced them. An entry the call did not
    produce (for example ``history`` without ``save_history=True``) is absent
    from the mapping, as it was from the dict; its attribute is None.
    ``restart_state``, ``homogenised_response`` and the other nested values
    stay plain dicts.
    """

    # Final state.
    gfu: Any = None
    mesh: Any = None
    mesh_displacement: Any = None
    mesh_quality: Any = None
    mesh_ale_jacobian_min: Any = None
    mesh_quality_summary: Any = None
    mesh_velocity: Any = None
    solid_rate_of_deformation: Any = None
    F_old: Any = None
    # Live solver objects; always None on return.
    linear_system: Any = None
    preconditioner: Any = None
    homogenisation_preconditioner: Any = None
    # Loading record.
    max_displacements: Any = None
    prescribed_times: Any = None
    prescribed_F_history: Any = None
    prescribed_P_history: Any = None
    prescribed_G_history: Any = None
    prescribed_F_rate_history: Any = None
    F_target: Any = None
    P_target: Any = None
    G_target: Any = None
    geometry: Any = None
    homogenised_response: Any = None
    # Newton and acceptance diagnostics.
    newton_histories: Any = None
    newton_solver_metadata: Any = None
    deformation_check_history: Any = None
    interface_correction_norm: Any = None
    # Discretisation and solver configuration.
    total_dofs: Any = None
    solid_pressure_dofs: Any = None
    component_dofs: Any = None
    field_order: Any = None
    dof_ranges: Any = None
    mesh_mu: Any = None
    mesh_lame_lambda: Any = None
    preconditioner_options: Any = None
    quadrature_order: Any = None
    fsi_preconditioner_variant: Any = None
    fsi_preconditioner_order_aware: Any = None
    fsi_adaptive_refresh_iteration_threshold: Any = None
    adaptive_preconditioner_refreshes: Any = None
    native_velocity_coarse_metadata: Any = None
    native_velocity_coarse_backend: Any = None
    # Timings and profiling.
    setup_breakdown: Any = None
    setup_breakdown_accounting: Any = None
    profiling: Any = None
    profiling_counts: Any = None
    profiling_averages: Any = None
    preconditioner_timestep_timings: Any = None
    krylov_timestep_timings: Any = None
    memory_cleanup_timings: Any = None
    # Formulation labels.
    residual_metric: Any = None
    geometry_map_formulation: Any = None
    block_assembly: Any = None
    state_storage: Any = None
    assembled_block_keys: Any = None
    # Restart, field history and summaries.
    history: Any = None
    restart_state: Any = None
    performance_summary: Any = None
    workspace_reused: Any = None
    # Physical units (added by the dimensional wrapper).
    E: Any = None
    mu_s: Any = None
    mu_f: Any = None
    eta_s: Any = None
    zeta_s: Any = None
    dt: Any = None
    t_final: Any = None
    rve_length: Any = None
    loading_path: Any = None
    nondimensional_scales: Any = None
    # Sequence calls (variable dt or compact response history).
    requested_times: Any = None
    requested_loading_path: Any = None
    time_steps: Any = None
    rejected_steps: Any = None
    response_history: Any = None
    history_file: Any = None
    # Entries present in this result, in the solver's order.
    _keys: tuple = field(default=(), repr=False)

    @classmethod
    def from_dict(cls, values):
        unknown = [key for key in values if key not in _ENTRY_NAMES]
        if unknown:
            raise KeyError(f"FSIResult has no field for result entries {unknown}")
        return cls(**values, _keys=tuple(values))

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys


_ENTRY_NAMES = frozenset(entry.name for entry in fields(FSIResult)) - {"_keys"}
