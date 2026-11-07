"""Public entry points: FSI and FSISession."""

from __future__ import annotations
import gc
import numpy as np

from .driver import _FSI_fixed
from .inputs import check_inputs
from .result import FSIResult
from .sequence import _fsi_sequence, _fsi_sequence_path, _fsi_time_grid


class FSISession:
    """Reuse FE spaces, forms and registered preconditioners across increments.

    Bind geometry, mesh and solver settings at construction.
    dt may change between calls: restart guesses are rescaled and the workspace,
    including its forms and any compiled code, is kept. A call without
    initial_state starts a new history at rest on the same cell; it may also
    change the material (elastic constants, viscosities), except whether the
    solid is viscous. The cell length and other settings remain fixed.
    Call with F_target/P_target/G_target, t_final, homogenise and initial_state.
    Calls are sequential. Returned FE fields are live views reused by the next
    call; restart arrays and homogenised output arrays are independent snapshots.
    Use close() or a context manager to release the retained workspace.
    """

    def __init__(self, **settings):
        self.settings = dict(settings)
        self._cell = None
        self._key = None
        self._mesh = None
        self._busy = False
        self._calls = 0
        self._scales = None

    def __call__(self, **step):
        if self._busy:
            raise RuntimeError("FSISession does not allow concurrent calls")
        kwargs = dict(self.settings)
        kwargs.update(step)
        if kwargs.get("tangent_method", "linear correctors") not in ("linear correctors", None):
            raise ValueError("FSISession supports linear correctors or tangent_method=None only")
        self._busy = True
        try:
            return FSI(**kwargs, _solver_session=self)
        except Exception:
            self.close()
            raise
        finally:
            self._busy = False

    def close(self):
        self._cell = None
        self._key = None
        self._mesh = None
        self._calls = 0
        self._scales = None
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def FSI(
    E=None,
    nu=None,
    dt=None,
    t_final=None,
    mu_f=None,
    F_target=None,
    P_target=None,
    G_target=None,
    *,
    K=None,
    G=None,
    mu=None,
    lam=None,
    eta_s=0.0,
    zeta_s=0.0,
    F_rate=None,
    rve_length=1.0,
    tangent_method="linear correctors",
    central_difference_step=0.0001,
    save_history=False,
    loading_path=None,
    loading_times=None,
    save_response_history=None,
    history_file=None,
    history_metadata=None,
    time_step_options=None,
    **kwargs,
):
    """Solve target loads or a sampled loading path, with fixed or variable steps.

    Existing calls are unchanged: scalar dt and t_final with F_target/P_target/
    G_target (or F_rate), or loading_path=(N+1,13) with scalar dt. A fixed-step
    path can omit t_final; its duration is inferred. The last three input columns
    are the PHYSICAL SPATIAL pressure gradient. F and Piola stress are row-major.

    For variable steps use dt=[dt_0,...,dt_(N-1)] with N+1 loading_path rows.
    Path row i applies at sum(dt[:i]); repeated rows are holds. t_final can be
    omitted or must equal sum(dt). Alternatively give loading_times starting at
    zero; omit dt, or provide dt consistent with those intervals. Variable dt
    also works with target loads (linear ramp in physical time) and F_rate.

    With homogenise=True, variable-grid calls return response_history containing
    time, dt, inputs (13), outputs (13), tangent (13x13), validity masks and scalar
    diagnostics at EVERY accepted increment. To request this also for existing
    fixed-step calls, set save_response_history=True. Its reference tangent is
    invalid/NaN. Old restart states without stored homogenised outputs have an
    invalid/NaN initial response rather than an invented relaxation step.

    history_file='run.h5' optionally streams the same compact data directly from
    this solver (h5py required only then; existing files are never overwritten).
    Set save_response_history=False to stream without retaining compact response
    arrays in RAM. File readers must use committed_rows. This is not a microscopic
    checkpoint: process restart requires replay. history_metadata accepts your
    JSON-like material/geometry/campaign descriptions in addition to recorded
    settings, source hash and the exact input grid. No external runner is needed.

    save_history retains its ORIGINAL meaning: optional microscopic field frames,
    not compact response data. It remains False by default. Variable-grid field
    histories include scales_per_frame; velocity coefficients must be interpreted
    using the corresponding frame's scale.

    Sequence calls land on every supplied time point. On numerical failure they
    can subdivide an interval, interpolating its loading endpoints. Configure
    time_step_options with max_retries (8), minimum_dt (min requested dt/1024),
    retry_factor (0.5), growth_factor (1.5). This is convergence recovery, NOT an
    accuracy estimator. Original fixed-step calls without new options follow the
    original implementation. The final homogenised_response tangent always holds
    the previous accepted state and the FINAL accepted dt fixed, not the whole
    history or a requested interval that was subdivided. Actual accepted grids
    are returned as prescribed_times/time_steps/loading_path; requested grids are
    also retained. Reference and candidate states remain internal to this file.

    ``diagnostics=True`` adds Hill--Mandel audits, an interface residual and
    mesh-quality plotting fields. The sampled geometry validity summary and
    numerical acceptance checks are always evaluated. Von Mises fields are
    neither calculated nor stored. Newton directions use PETSc GMRES with GAMG
    block preconditioning. Frozen correctors reuse one full MUMPS LU factor.
    Both retain the original explicit residual acceptance. Their serial
    independent-DOF adapters are included directly in this standalone module.
    """
    variable = loading_times is not None or (dt is not None and np.ndim(dt) != 0)
    sequence = (
        variable or bool(save_response_history) or history_file is not None or (time_step_options is not None)
    )
    checked = check_inputs(
        sequence=sequence,
        E=E,
        nu=nu,
        K=K,
        G=G,
        mu=mu,
        lam=lam,
        dt=dt,
        t_final=t_final,
        mu_f=mu_f,
        F_target=F_target,
        P_target=P_target,
        G_target=G_target,
        F_rate=F_rate,
        eta_s=eta_s,
        zeta_s=zeta_s,
        rve_length=rve_length,
        tangent_method=tangent_method,
        central_difference_step=central_difference_step,
        save_history=save_history,
        save_response_history=save_response_history,
        loading_path=loading_path,
        kwargs=kwargs,
    )
    common = dict(
        E=E,
        nu=nu,
        mu_f=mu_f,
        K=K,
        G=G,
        mu=mu,
        lam=lam,
        eta_s=eta_s,
        zeta_s=zeta_s,
        rve_length=rve_length,
        tangent_method=tangent_method,
        central_difference_step=central_difference_step,
        **kwargs,
    )
    if not sequence:
        t_final, F_target, P_target, G_target, loading_path = checked
        result = _FSI_fixed(
            **common,
            dt=dt,
            t_final=t_final,
            F_target=F_target,
            P_target=P_target,
            G_target=G_target,
            F_rate=F_rate,
            save_history=save_history,
            loading_path=loading_path,
        )
        return FSIResult.from_dict(result)
    n_intervals = None
    if loading_path is not None:
        supplied = np.asarray(loading_path)
        if supplied.ndim != 2 or supplied.shape[1] != 13 or supplied.shape[0] < 2:
            raise ValueError("loading_path must have shape (N+1,13)")
        n_intervals = len(supplied) - 1
    times = _fsi_time_grid(dt, t_final, loading_times, n_intervals)
    path = _fsi_sequence_path(
        times, loading_path, F_target, P_target, G_target, F_rate, kwargs.get("initial_state")
    )
    if save_response_history is None:
        save_response_history = bool(kwargs.get("homogenise", False))
    session = common.pop("_solver_session", None)
    owned = session is None and tangent_method in ("linear correctors", None)
    if owned:
        session = FSISession()
    try:
        result = _fsi_sequence(
            times=times,
            path=path,
            fixed=common,
            save_history=bool(save_history),
            save_response_history=bool(save_response_history),
            history_file=history_file,
            history_metadata=history_metadata,
            time_step_options=time_step_options,
            session=session,
        )
    finally:
        if owned:
            session.close()
    return FSIResult.from_dict(result)
