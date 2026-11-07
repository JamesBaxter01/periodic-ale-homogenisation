"""Native NGSolve BDDC proxies for the six-field tangent.

The Newton residual metric is the dual norm of the proxies on the reference
configuration (BDDC inverses factorised once per cell); ReusablePETScBlockPC
reads the timestep-lagged proxy forms. The correctors are accepted in the same
metric (ResidualMetric).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ngsolve import Preconditioner

from .linear import _NATIVE_BDDC_MUMPS_INVERSE
from .profiling import profiled_setup


def _make_fsi_bddc(form, mean_nullspace=False):
    if mean_nullspace:
        return Preconditioner(form, "bddc", inverse=_NATIVE_BDDC_MUMPS_INVERSE)
    return Preconditioner(form, "bddc")

def make_pressure_preconditioner(form):
    return Preconditioner(form, "bddc")

class BlockFSIALEPreconditioner:
    """Block proxy forms and BDDC inverses for the six-field tangent.

    The BDDC inverses define the Newton residual norm; with ``metric_forms``
    they belong to separate reference-configuration forms, refactorised only
    while ``metric_stale``. ``ReusablePETScBlockPC`` reads the lagged proxy
    forms, scales and revisions, including the current-state proxies swapped
    in by the adaptive refresh.
    """

    def __init__(
        self,
        system,
        pressure_inverse_scale=1.0,
        pressure_preconditioner=None,
        mesh_preconditioner=None,
        mesh_inverse_scale=1.0,
        order_aware=False,
        adaptive_iteration_threshold=80,
        velocity_preconditioner=None,
        solid_pressure_preconditioner=None,
        metric_forms=None,
        *,
        order,
        profiler,
    ):
        self.profiler = profiler
        # Forms of the residual metric when it is separate from the lagged proxies; they are
        # assembled (factorising their registered BDDC inverses) only while metric_stale.
        self.metric_forms = metric_forms
        self.metric_stale = metric_forms is not None
        self.proxy_revisions = dict(velocity=0, pressure=0, solid_pressure=0, mesh=0)
        self.system = system
        self.order_aware = bool(order_aware)
        self.polynomial_order = int(order)
        self.high_order_fsi = bool(self.order_aware and self.polynomial_order >= 3)
        self.adaptive_iteration_threshold = max(1, int(adaptive_iteration_threshold))
        self.fsi_bddc_variant = "vector_bddc"
        self.velocity_preconditioner_form = system.velocity_proxy
        self.velocity_preconditioner = velocity_preconditioner
        if self.velocity_preconditioner is None:
            self.velocity_preconditioner = _make_fsi_bddc(
                self.velocity_preconditioner_form, mean_nullspace=True
            )
        self.pressure_mass = system.pressure_mass
        self.solid_pressure_mass = system.solid_pressure_mass
        if pressure_preconditioner is None:
            raise RuntimeError("pressure preconditioner was not built")
        if mesh_preconditioner is None:
            raise RuntimeError("mesh preconditioner was not built")
        self.pressure_preconditioner = pressure_preconditioner
        self.solid_pressure_preconditioner = solid_pressure_preconditioner
        if self.solid_pressure_preconditioner is None:
            self.solid_pressure_preconditioner = make_pressure_preconditioner(self.solid_pressure_mass)
        self.mesh_preconditioner = mesh_preconditioner
        self.pressure_inverse_scale = float(pressure_inverse_scale)
        self.mesh_inverse_scale = float(mesh_inverse_scale)
        self._lagged_mesh_inverse_scale = float(mesh_inverse_scale)
        self._lagged_velocity_proxy = system.velocity_proxy
        self._lagged_pressure_mass = system.pressure_mass
        self._lagged_solid_pressure_mass = system.solid_pressure_mass
        self._lagged_mesh_proxy = system.mesh_proxy
        self._current_velocity_proxy = getattr(system, "homogenisation_velocity_proxy", None)
        self._current_pressure_mass = getattr(system, "homogenisation_pressure_mass", None)
        self._current_solid_pressure_mass = getattr(system, "homogenisation_solid_pressure_mass", None)
        self._current_mesh_proxy = getattr(system, "homogenisation_mesh_proxy", None)
        self._using_current_state_proxies = False
        self._velocity_is_current = False
        self.adaptive_refresh_count = 0
        self.adaptive_last_refresh_iteration = -1
        self.adaptive_max_refreshes = 2

    def _restore_lagged_proxies(self):
        """Restore the timestep-lagged proxy objects after a refresh."""
        if not self._using_current_state_proxies:
            return
        self.velocity_preconditioner_form = self._lagged_velocity_proxy
        self.pressure_mass = self._lagged_pressure_mass
        self.solid_pressure_mass = self._lagged_solid_pressure_mass
        self.mesh_inverse_scale = self._lagged_mesh_inverse_scale
        self._using_current_state_proxies = False
        self._velocity_is_current = False
        for name in self.proxy_revisions:
            self.proxy_revisions[name] += 1

    @profiled_setup("adaptive", "current-state refresh total")
    def refresh_current_state_proxies(self, force_velocity=False):
        """Assemble current-state proxy forms for the adaptive FSI mode.

        The exact tangent remains assembled from the accepted Newton
        state. Pressure and ALE proxy forms are refreshed here; the PETSc
        block PC rebuilds from them. Velocity keeps the timestep-lagged
        proxy unless a linear-solve retry requests force_velocity=True.
        The BDDC inverses that define the residual norm are not changed.
        """
        current_forms = (
            self._current_velocity_proxy,
            self._current_pressure_mass,
            self._current_solid_pressure_mass,
            self._current_mesh_proxy,
        )
        if any((form is None for form in current_forms)):
            return 0.0
        started = self.profiler.start()
        refresh_velocity = force_velocity
        refresh_start = 0 if refresh_velocity else 1
        for field_name, form in zip(
            ("velocity", "fluid pressure", "solid pressure", "ALE")[refresh_start:],
            current_forms[refresh_start:],
        ):
            self.profiler.setup_call(
                "adaptive", f"{field_name} proxy Assemble (includes registered setup)", form.Assemble
            )
        for name in ("velocity", "pressure", "solid_pressure", "mesh")[refresh_start:]:
            self.proxy_revisions[name] += 1
        if refresh_velocity:
            self.velocity_preconditioner_form = self._current_velocity_proxy
        self.pressure_mass = self._current_pressure_mass
        self.solid_pressure_mass = self._current_solid_pressure_mass
        self.mesh_inverse_scale = 1.0
        self._using_current_state_proxies = True
        self._velocity_is_current = refresh_velocity or self._velocity_is_current
        return self.profiler.finish("Preconditioner: adaptive current-state refresh", started)

    def maybe_refresh_for_newton(self, current_residual_norm, iteration, history):
        """Refresh adaptive proxies only when their lag becomes costly.

        For the order-two production path, the first refresh remains
        aligned with the accurate Newton end game (the same residual
        threshold at which the linear tolerance is tightened).  In the
        high-order FSI path, a long preceding Krylov solve is also a
        sufficient signal: high-order interior modes can make the
        timestep-lagged proxy ineffective well before the residual falls
        below one.  A subsequent refresh is permitted after separated
        Newton updates if another solve is unusually long.
        """
        if self.adaptive_refresh_count >= self.adaptive_max_refreshes:
            return 0.0
        if self._current_velocity_proxy is None:
            return 0.0
        accurate_end_game = (
            np.isfinite(float(current_residual_norm)) and float(current_residual_norm) <= 1.0
        )
        previous_linear_solve_was_long = (
            bool(history)
            and int(history[-1].get("gmres_iterations", 0)) >= self.adaptive_iteration_threshold
        )
        sufficiently_separated = iteration - self.adaptive_last_refresh_iteration >= 2
        should_refresh = (
            not self._using_current_state_proxies
            and (accurate_end_game or (self.high_order_fsi and previous_linear_solve_was_long))
            or (
                self._using_current_state_proxies
                and sufficiently_separated
                and previous_linear_solve_was_long
            )
        )
        if not should_refresh:
            return 0.0
        elapsed = self.refresh_current_state_proxies()
        if self._using_current_state_proxies:
            self.adaptive_refresh_count += 1
            self.adaptive_last_refresh_iteration = int(iteration)
        return elapsed

    def refresh_for_linear_failure(self, iteration, gmres_iterations):
        """Refresh the FSI proxy before retrying a failed Krylov solve.

        A high-order solve can exhaust its GMRES budget while the current
        state is still outside the usual end-game residual range.  When
        this happens, activate the current-state velocity/ALE proxies
        before the next retry.  If those proxies are already active, repeating the same
        refresh at an unchanged Newton state cannot improve the action
        and is therefore skipped.  The homogenisation preconditioner
        never calls this method.
        """
        if self.adaptive_refresh_count >= self.adaptive_max_refreshes:
            return 0.0
        if self._current_velocity_proxy is None:
            return 0.0
        if self._using_current_state_proxies and self._velocity_is_current:
            return 0.0
        elapsed = self.refresh_current_state_proxies(force_velocity=True)
        if not self._using_current_state_proxies:
            return 0.0
        self.adaptive_refresh_count += 1
        self.adaptive_last_refresh_iteration = int(iteration)
        return elapsed

    def reset_adaptive_level(self):
        """Return the adaptive mode to its lagged start-of-level state.

        A nonlinear continuation solve can retry the same physical
        timestep from a restored state.  A proxy refreshed for the
        failed load level must not be carried into that retry, because
        its coefficients correspond to the discarded state.
        """
        self._restore_lagged_proxies()
        self.adaptive_refresh_count = 0
        self.adaptive_last_refresh_iteration = -1

    @profiled_setup("lagged", "begin_timestep total")
    def begin_timestep(self):
        """Refresh timestep-dependent SPD proxies and reusable state.

        The nonlinear component forms and their sparse patterns persist
        across timesteps.  Only coefficient-dependent proxy values and
        their registered preconditioners need refreshing after ``F_n``
        and ``d_n`` change.
        """
        self._restore_lagged_proxies()
        self.adaptive_refresh_count = 0
        self.adaptive_last_refresh_iteration = -1
        started = self.profiler.start()
        for field_name, form in (
            ("velocity", self.system.velocity_proxy),
            ("fluid pressure", self.system.pressure_mass),
            ("solid pressure", self.system.solid_pressure_mass),
            ("ALE", self.system.mesh_proxy),
        ):
            self.profiler.setup_call(
                "lagged", f"{field_name} proxy Assemble (includes registered setup)", form.Assemble
            )
        for name in self.proxy_revisions:
            self.proxy_revisions[name] += 1
        if self.metric_forms is not None:
            if self.metric_stale:
                for field_name, form in zip(("velocity", "fluid pressure", "solid pressure", "ALE"), self.metric_forms):
                    self.profiler.setup_call(
                        "metric", f"{field_name} metric Assemble (includes registered setup)", form.Assemble
                    )
                self.metric_stale = False
            return self.profiler.finish("Preconditioner: timestep proxy refresh", started)
        for field_name, preconditioner in (
            ("velocity", self.velocity_preconditioner),
            ("fluid pressure", self.pressure_preconditioner),
            ("solid pressure", self.solid_pressure_preconditioner),
            ("ALE", self.mesh_preconditioner),
        ):
            update = getattr(preconditioner, "Update", None)
            if update is not None:
                self.profiler.setup_call("lagged", f"{field_name} inverse explicit Update", update)
        return self.profiler.finish("Preconditioner: timestep proxy refresh", started)


@dataclass(frozen=True)
class ResidualMetric:
    """The four BDDC inverses of the residual metric; the correctors are accepted in it.

    Taken from the cell's Newton preconditioner before that is released, so the
    inverses stay alive for the homogenisation.
    """

    velocity: object
    pressure: object
    solid_pressure: object
    mesh: object

    @classmethod
    def of(cls, preconditioner):
        return cls(
            preconditioner.velocity_preconditioner.mat,
            preconditioner.pressure_preconditioner.mat,
            preconditioner.solid_pressure_preconditioner.mat,
            preconditioner.mesh_preconditioner.mat,
        )
