"""Loading paths, time grids and response-history output (HDF5)."""

from __future__ import annotations
from time import perf_counter
import numpy as np

from .driver import _FSI_fixed, restore_accepted_mesh
from .inputs import _validate_loading_path


# Variable-step loading, response histories and optional HDF5 streaming.
def _fsi_time_grid(dt, t_final, loading_times, n_intervals=None):
    """Validate a call-relative grid; dt may be scalar or one value per interval."""
    if loading_times is not None:
        if np.iscomplexobj(loading_times):
            raise ValueError("loading_times must be real")
        times = np.array(loading_times, dtype=float, copy=True)
        if (
            times.ndim != 1
            or len(times) < 2
            or (not np.isfinite(times).all())
            or (times[0] != 0)
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("loading_times must start at zero and strictly increase")
        steps = np.diff(times)
        if dt is not None:
            supplied = np.asarray(dt)
            if np.iscomplexobj(supplied) or supplied.dtype.kind == "b" or supplied.ndim > 1:
                raise ValueError("dt must be a positive scalar or interval vector")
            if supplied.ndim == 1 and supplied.shape != steps.shape:
                raise ValueError("dt must contain one value per loading_times interval")
            if not np.isfinite(supplied).all() or not np.allclose(supplied, steps, rtol=1e-12, atol=0):
                raise ValueError("dt and loading_times describe different time grids")
    else:
        if dt is None or np.iscomplexobj(dt):
            raise ValueError("Supply dt or loading_times")
        supplied = np.asarray(dt)
        if (
            supplied.dtype.kind == "b"
            or supplied.ndim > 1
            or (not np.isfinite(supplied).all())
            or np.any(supplied <= 0)
        ):
            raise ValueError("dt must be a finite positive scalar or interval vector")
        if supplied.ndim == 0:
            if n_intervals is None:
                if (
                    t_final is None
                    or not np.isscalar(t_final)
                    or (not np.isfinite(t_final))
                    or (t_final <= 0)
                ):
                    raise ValueError("Target loading with scalar dt requires positive t_final")
                n_intervals = int(round(t_final / float(supplied)))
                if n_intervals < 1 or not np.isclose(
                    n_intervals * float(supplied), t_final, rtol=1e-10, atol=0
                ):
                    raise ValueError("t_final must be an integer multiple of scalar dt")
            steps = np.full(n_intervals, float(supplied))
        else:
            steps = np.array(supplied, dtype=float, copy=True)
            if len(steps) == 0:
                raise ValueError("dt interval vector cannot be empty")
        times = np.r_[0.0, np.cumsum(steps)]
    if n_intervals is not None and len(steps) != n_intervals:
        raise ValueError("dt must have len(loading_path)-1 intervals")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Time grid overflow or intervals below floating-point resolution")
    if t_final is not None:
        if (
            not np.isscalar(t_final)
            or not np.isfinite(t_final)
            or t_final <= 0
            or (not np.isclose(times[-1], t_final, rtol=1e-10, atol=0))
        ):
            raise ValueError("t_final must equal the sum of the time intervals")
    return times


def _fsi_sequence_path(times, loading_path, F_target, P_target, G_target, F_rate, initial_state):
    if initial_state is None:
        initial = np.r_[np.eye(3).ravel(), 0.0, np.zeros(3)]
    else:
        initial = np.r_[
            np.asarray(initial_state["F_macro"]).ravel(),
            initial_state["P_macro"] * initial_state["stress_scale"],
            np.asarray(initial_state["G_macro"])
            * initial_state["stress_scale"]
            / initial_state["rve_length"],
        ]
    if loading_path is not None:
        if F_rate is not None or any((v is not None for v in (F_target, P_target, G_target))):
            raise ValueError("loading_path replaces targets and F_rate")
        path, _ = _validate_loading_path(loading_path, 1.0, len(times) - 1, initial)
    else:
        if P_target is None or G_target is None:
            raise TypeError("P_target and G_target are required without loading_path")
        if (F_target is None) == (F_rate is None):
            raise ValueError("Supply exactly one of F_target and F_rate")
        if (
            np.iscomplexobj(P_target)
            or np.iscomplexobj(G_target)
            or (F_target is not None and np.iscomplexobj(F_target))
        ):
            raise ValueError("Loading must be real")
        gradient = np.asarray(G_target, dtype=float)
        if gradient.shape != (3,) or not np.isscalar(P_target):
            raise ValueError("P_target must be scalar and G_target must have shape (3,)")
        path = np.tile(initial, (len(times), 1))
        fraction = times / times[-1]
        path[:, 9] += fraction * (float(P_target) - initial[9])
        path[:, 10:] += fraction[:, None] * (gradient - initial[10:])
        if F_target is not None:
            F = np.asarray(F_target, dtype=float)
            if F.shape != (3, 3):
                raise ValueError("F_target must have shape (3,3)")
            path[:, :9] += fraction[:, None] * (F.ravel() - initial[:9])
        else:
            for i, step in enumerate(np.diff(times)):
                rate = F_rate(times[i]) if callable(F_rate) else F_rate
                if np.iscomplexobj(rate):
                    raise ValueError("F_rate must be real")
                rate = np.asarray(rate, dtype=float)
                if rate.shape != (3, 3) or not np.isfinite(rate).all():
                    raise ValueError("F_rate must provide finite (3,3) tensors")
                path[i + 1, :9] = path[i, :9] + step * rate.ravel()
        path, _ = _validate_loading_path(path, 1.0, len(times) - 1, initial)
    for left, right in zip(path[:-1, :9], path[1:, :9]):
        A, B = (left.reshape(3, 3), (right - left).reshape(3, 3))
        ss = np.linspace(0, 1, 4)
        polynomial = np.polynomial.Polynomial.fit(ss, [np.linalg.det(A + s * B) for s in ss], 3).convert()
        points = [0.0, 1.0] + [
            float(x.real) for x in polynomial.deriv().roots() if abs(x.imag) < 1e-10 and 0 < x.real < 1
        ]
        if any((np.linalg.det(A + s * B) <= 0 for s in points)):
            raise ValueError("Interpolated loading crosses det(F)<=0")
    return path


def _fsi_json(value):
    """Small metadata only; deliberately never serialize FE objects."""
    if isinstance(value, np.ndarray):
        return _fsi_json(value.tolist())
    if isinstance(value, np.generic):
        return _fsi_json(value.item())
    if isinstance(value, dict):
        return {str(k): _fsi_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_fsi_json(v) for v in value]
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class _FSIHistoryFile:
    """Optional append-only compact HDF5 output, entirely inside this module."""

    def __init__(self, filename, metadata):
        import json
        import h5py
        from pathlib import Path

        self.json = json
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "x")
        f = self.file
        try:
            request = f.create_group("requested")
            request.create_dataset("time", data=metadata["requested_times"], compression="gzip")
            request.create_dataset("inputs", data=metadata["requested_loading_path"], compression="gzip")
            header = {
                k: v for k, v in metadata.items() if k not in ("requested_times", "requested_loading_path")
            }
            f.attrs.update(
                schema_version=1,
                committed_rows=0,
                status="running",
                metadata=json.dumps(_fsi_json(header), allow_nan=False),
            )
            for name, shape, dtype in [
                ("time", (), "f8"),
                ("dt", (), "f8"),
                ("inputs", (13,), "f8"),
                ("outputs", (13,), "f8"),
                ("tangent", (13, 13), "f8"),
                ("output_valid", (), "?"),
                ("tangent_valid", (), "?"),
                ("wall_seconds", (), "f8"),
            ]:
                f.create_dataset(
                    name,
                    shape=(0,) + shape,
                    maxshape=(None,) + shape,
                    dtype=dtype,
                    chunks=(64,) + shape,
                    compression="gzip",
                    compression_opts=1,
                )
            for name in ("diagnostics", "rejected_attempts"):
                f.create_dataset(
                    name, shape=(0,), maxshape=(None,), chunks=(64,), dtype=h5py.string_dtype("utf-8")
                )
            f.flush()
        except BaseException:
            f.close()
            raise

    def append(self, row):
        f = self.file
        n = int(f.attrs["committed_rows"])
        for name, value in row.items():
            d = f[name]
            d.resize(n + 1, axis=0)
            d[n] = self.json.dumps(_fsi_json(value), allow_nan=False) if name == "diagnostics" else value
        f.flush()
        f.attrs.modify("committed_rows", n + 1)
        f.flush()

    def rejected(self, row):
        d = self.file["rejected_attempts"]
        n = len(d)
        d.resize(n + 1, axis=0)
        d[n] = self.json.dumps(_fsi_json(row), allow_nan=False)
        self.file.flush()

    def finish(self, status, message=""):
        try:
            self.file.attrs["status"] = status
            self.file.attrs["message"] = message
            self.file.flush()
        finally:
            self.file.close()


def _fsi_compact_diagnostics(result):
    response = result.get("homogenised_response") or {}
    return _fsi_json(
        dict(
            mesh_quality=result.get("mesh_quality_summary"),
            deformation_checks=result.get("deformation_check_history"),
            newton_iterations=sum((len(h) for h in result.get("newton_histories", []))),
            balances={
                k: v
                for k, v in response.get("hill_mandel_diagnostics", {}).items()
                if isinstance(v, (float, int, bool, np.generic))
            },
            corrector_iterations=response.get("corrector_gmres_iterations"),
        )
    )


def _fsi_checked_response(result, target, homogenise, *, tangent=True):
    histories = result.get("newton_histories", [])
    if not histories or any((not h or not h[-1].get("converged", False) for h in histories)):
        raise RuntimeError("FSI increment did not converge")
    quality = result.get("mesh_quality_summary", {})
    if any(
        (
            quality.get(k, 0)
            for k in ("inverted_element_count", "collapsed_element_count", "nonfinite_element_count")
        )
    ):
        raise RuntimeError("FSI increment has invalid sampled mesh geometry")
    if not homogenise:
        return (None, None)
    response = result["homogenised_response"]
    y = np.asarray(response["outputs"])
    C = np.asarray(response["jacobian"]) if tangent else np.full((13, 13), np.nan)
    if y.shape != (13,) or C.shape != (13, 13) or (not np.isfinite(y).all()) or (tangent and not np.isfinite(C).all()):
        raise RuntimeError("FSI increment has missing or nonfinite outputs/tangents")
    if not np.allclose(response["inputs"], target, rtol=1e-10, atol=1e-12):
        raise RuntimeError("FSI increment returned inconsistent loading")
    if y[9] <= 0:
        raise RuntimeError("FSI increment has nonpositive fluid content")
    return (y.copy(), C.copy())


def _fsi_sequence(
    *,
    times,
    path,
    fixed,
    save_history,
    save_response_history,
    history_file,
    history_metadata,
    time_step_options,
    session,
):
    """Advance through a requested time grid, subdividing intervals on failure.

    ``session`` is the FSISession reused between increments (None for central
    differences); the caller owns it and closes it.
    """
    import hashlib
    import inspect
    from pathlib import Path

    options = {
        "max_retries": 8,
        "minimum_dt": float(np.min(np.diff(times))) / 1024.0,
        "retry_factor": 0.5,
        "growth_factor": 1.5,
    }
    if time_step_options:
        unknown = set(time_step_options) - set(options)
        if unknown:
            raise ValueError(f"Unknown time_step_options: {sorted(unknown)}")
        options.update(time_step_options)
    if (
        isinstance(options["max_retries"], bool)
        or not isinstance(options["max_retries"], (int, np.integer))
        or options["max_retries"] < 0
    ):
        raise ValueError("max_retries must be a nonnegative integer")
    if not np.isfinite(options["minimum_dt"]) or options["minimum_dt"] <= 0:
        raise ValueError("minimum_dt must be finite and positive")
    if not 0 < options["retry_factor"] < 1 or not 1 <= options["growth_factor"] <= 2:
        raise ValueError("Invalid retry/growth factors")
    homogenise = bool(fixed.get("homogenise", False))
    tangents = fixed.get("tangent_method", "linear correctors") is not None
    collect = bool(save_response_history or history_file is not None)
    if collect and (not homogenise):
        raise ValueError("Compact response history requires homogenise=True")
    fixed = dict(fixed)
    state = fixed.pop("initial_state", None)
    metadata = dict(
        input_names=[f"F{i}{j}" for i in range(1, 4) for j in range(1, 4)] + ["p", "gp1", "gp2", "gp3"],
        output_names=[f"P{i}{j}" for i in range(1, 4) for j in range(1, 4)] + ["mf", "Q01", "Q02", "Q03"],
        units="physical; consistent with supplied moduli, cell length, and times",
        input_unit_dimensions=["1"] * 9 + ["stress"] + ["stress/length"] * 3,
        output_unit_dimensions=["stress"] * 9 + ["1"] + ["length/time"] * 3,
        tensor_order="row-major",
        gradient_convention="spatial",
        time_origin="start of this FSI call",
        tangent_definition="dy[n+1]/dz[n+1] at fixed accepted state and dt",
        settings={k: v for k, v in fixed.items() if k != "geometry"},
        geometry=getattr(fixed.get("geometry"), "__qualname__", repr(fixed.get("geometry"))),
        requested_times=times,
        requested_loading_path=path,
        time_step_options=options,
        user_metadata=history_metadata or {},
    )
    try:
        # The same digest as verification/state_io.py's solver_sha256: entry file plus package.
        root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for source in [root / "homogenisation_microscale.py", *sorted(Path(__file__).resolve().parent.glob("*.py"))]:
            digest.update(source.relative_to(root).as_posix().encode() + b"\0")
            digest.update(source.read_bytes())
        metadata["solver_sha256"] = digest.hexdigest()
        metadata["geometry_source_sha256"] = hashlib.sha256(
            Path(inspect.getfile(fixed["geometry"])).read_bytes()
        ).hexdigest()
    except (OSError, TypeError, KeyError):
        pass
    writer = None
    rows = [] if save_response_history else None
    actual_times = [0.0]
    actual_loads = [path[0].copy()]
    actual_displacements = [0.0]
    rejections = []
    field_history = None
    t = 0.0
    result = None

    def solve(previous, target, step, fields=False, outputs_only=False, time_offset=None):
        kwargs = dict(
            fixed,
            initial_state=previous,
            dt=step,
            t_final=step,
            F_target=target[:9].reshape(3, 3),
            P_target=float(target[9]),
            G_target=target[10:],
            save_history=fields,
        )
        if session is not None:
            kwargs["_solver_session"] = session
        if outputs_only:
            kwargs["_outputs_only"] = True
        # Printed times count from the start of the sequence.
        kwargs["_time_offset"] = t if time_offset is None else time_offset
        return _FSI_fixed(**kwargs)

    def append(row):
        if writer is not None:
            writer.append(row)
        if rows is not None:
            rows.append(row)

    try:
        if history_file is not None:
            writer = _FSIHistoryFile(history_file, metadata)
        if state is None:
            # The initial row records outputs at rest but no tangent, so this solve skips the
            # correctors (central differences keep their own path).
            outputs_only = homogenise and fixed.get("tangent_method", "linear correctors") in ("linear correctors", None)
            # Prints as t=0: this solve creates the state at rest at the start of the sequence.
            result = solve(None, path[0], float(times[1]), outputs_only=outputs_only, time_offset=-float(times[1]))
            _fsi_checked_response(result, path[0], homogenise, tangent=not outputs_only)
            state = result["restart_state"]
        if collect:
            initial_output = state.get("homogenised_outputs")
            initial_valid = (
                initial_output is not None
                and np.asarray(initial_output).shape == (13,)
                and np.isfinite(initial_output).all()
            )
            append(
                dict(
                    time=0.0,
                    dt=0.0,
                    inputs=path[0].copy(),
                    outputs=np.array(initial_output, copy=True) if initial_valid else np.full(13, np.nan),
                    tangent=np.full((13, 13), np.nan),
                    output_valid=bool(initial_valid),
                    tangent_valid=False,
                    wall_seconds=0.0,
                    diagnostics={"initial_reference": True, "output_available": bool(initial_valid)},
                )
            )
        for interval, end in enumerate(times[1:]):
            start = float(times[interval])
            end = float(end)
            cap = end - start
            while t < end:
                step = min(cap, end - t)
                tries = 0
                started = perf_counter()
                while True:
                    endpoint = min(t + step, end)
                    if end - endpoint <= 8 * np.finfo(float).eps * max(1.0, abs(end)):
                        endpoint = end
                    step = endpoint - t
                    if step <= 0:
                        raise RuntimeError("Time step is below floating-point resolution")
                    fraction = (endpoint - start) / (end - start)
                    target = path[interval] + fraction * (path[interval + 1] - path[interval])
                    tries += 1
                    result = None
                    try:
                        result = solve(state, target, step, fields=save_history)
                        y, C = _fsi_checked_response(result, target, homogenise, tangent=tangents)
                        break
                    except (RuntimeError, FloatingPointError) as error:
                        result = None
                        if session is not None:
                            session.close()
                        restore_accepted_mesh(state)
                        rejected = dict(time=t, trial_time=endpoint, dt=step, error=str(error))
                        rejections.append(rejected)
                        if writer is not None:
                            writer.rejected(rejected)
                        if (
                            tries > options["max_retries"]
                            or step * options["retry_factor"] < options["minimum_dt"]
                        ):
                            raise RuntimeError(f"Cannot advance FSI from t={t:g}: {error}") from error
                        step *= options["retry_factor"]
                if collect:
                    append(
                        dict(
                            time=endpoint,
                            dt=step,
                            inputs=target.copy(),
                            outputs=y,
                            tangent=C,
                            output_valid=True,
                            tangent_valid=tangents,
                            wall_seconds=perf_counter() - started,
                            diagnostics=_fsi_compact_diagnostics(result),
                        )
                    )
                if save_history:
                    segment = result["history"]
                    if field_history is None:
                        field_history = segment
                        field_history["times"] = list(np.asarray(segment["times"]) + t)
                        field_history["scales_per_frame"] = [dict(segment["scales"]), dict(segment["scales"])]
                    else:
                        for destination, source in zip(field_history["components"], segment["components"]):
                            destination.AddMultiDimComponent(source.vecs[-1])
                        for name in ("mesh_displacement",):
                            field_history[name].AddMultiDimComponent(segment[name].vecs[-1])
                        field_history["times"].append(endpoint)
                        field_history["scales_per_frame"].append(dict(segment["scales"]))
                state = result["restart_state"]
                t = endpoint
                actual_times.append(t)
                actual_loads.append(target.copy())
                actual_displacements.append(float(result["max_displacements"][-1]))
                cap = step * options["growth_factor"]
        actual_times = np.asarray(actual_times)
        actual_loads = np.asarray(actual_loads)
        result["requested_times"] = times.copy()
        result["requested_loading_path"] = path.copy()
        result["prescribed_times"] = actual_times
        result["time_steps"] = np.diff(actual_times)
        result["max_displacements"] = actual_displacements
        result["loading_path"] = actual_loads
        result["prescribed_F_history"] = actual_loads[:, :9].reshape(-1, 3, 3)
        result["prescribed_P_history"] = actual_loads[:, 9]
        result["prescribed_G_history"] = actual_loads[:, 10:]
        result["prescribed_F_rate_history"] = (
            np.diff(result["prescribed_F_history"], axis=0) / np.diff(actual_times)[:, None, None]
        )
        result["t_final"] = float(times[-1])
        result["rejected_steps"] = rejections
        if field_history is not None:
            field_history["times"] = actual_times.copy()
            field_history["scales_per_frame"] = [dict(x) for x in field_history["scales_per_frame"]]
            if any(
                (
                    x["time"] != field_history["scales_per_frame"][0]["time"]
                    for x in field_history["scales_per_frame"]
                )
            ):
                field_history.pop("scales", None)
                field_history["field_units"] = "internal dimensionless coefficients; see scales_per_frame"
            result["history"] = field_history
        if rows is not None:
            result["response_history"] = {
                key: [r[key] for r in rows] if key == "diagnostics" else np.asarray([r[key] for r in rows])
                for key in rows[0]
            }
            result["response_history"]["metadata"] = _fsi_json(metadata)
        if writer is not None:
            completed_writer = writer
            writer = None
            completed_writer.finish("complete")
            result["history_file"] = str(Path(history_file).resolve())
        return result
    except BaseException as error:
        try:
            restore_accepted_mesh(state)
        finally:
            if writer is not None:
                writer.finish(
                    "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                    str(error),
                )
                writer = None
        raise
