"""FECellModel: one periodic FSI cell per material point, through the public FSISession.

Each point's committed ``restart_state`` lives in the process that solves it
(NGSolve objects cannot cross processes). One ``FSISession`` per process
serves all of its points: every call restores the point's own restart state,
so a point's result does not depend on which points were solved before it
(checked to round-off on 5 Oct 2026).

``workers=0`` solves every point in this process. ``workers=N`` starts N
persistent processes, each owning the points i with i % N == lane.
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import pickle
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .interface import MaterialPointFailure, Response, check_inputs

ROOT = Path(__file__).resolve().parents[1]

# The coupling owns the loading, the history and the time step of every cell.
OWNED = {"F_target", "P_target", "G_target", "F_rate", "dt", "t_final", "initial_state", "homogenise",
         "loading_path", "loading_times", "tangent_method", "save_history", "save_response_history",
         "history_file", "time_step_options"}


@dataclass(frozen=True)
class FreshEquilibrium:
    """Passed as dt: evaluate a fresh cell, memoryless by construction. It starts at rest, takes one
    backward-Euler step of STEPS[0] to the inputs and holds them for the remaining steps; the response is
    that of the last step, and ``equilibrium`` its change over that step."""

    steps: tuple = (1000.0, 1000.0, 1000.0)


REST = np.r_[np.eye(3).ravel(), 0.0, np.zeros(3)]  # the inputs of a cell at rest


def _change(outputs):
    """Change of the outputs between the last two rows: (largest stress change relative to the largest stress
    component, fluid content change relative to the content, flux change in m/s)."""
    new, old = outputs[-1], outputs[-2]
    return (float(np.max(np.abs(new[:9] - old[:9])) / max(np.max(np.abs(new[:9])), 1e-300)),
            float(abs(new[9] - old[9]) / max(abs(new[9]), 1e-300)), float(np.linalg.norm(new[10:] - old[10:])))


def _load_solver():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import homogenisation_microscale

    return homogenisation_microscale


def _accepted(result):
    """Raise MaterialPointFailure unless every Newton solve converged on admissible geometry."""
    histories = result.get("newton_histories") or []
    if not histories or any(not h or not h[-1].get("converged", False) for h in histories):
        raise MaterialPointFailure("cell Newton solve did not converge")
    quality = result.get("mesh_quality_summary") or {}
    if any(quality.get(k, 0) for k in ("inverted_element_count", "collapsed_element_count",
                                       "nonfinite_element_count")):
        raise MaterialPointFailure("cell has invalid sampled mesh geometry")


class _Points:
    """The points one process owns: committed and trial restart states, one FSISession."""

    def __init__(self, settings, threads):
        from ngsolve import SetNumThreads

        SetNumThreads(threads)
        # The cell's MUMPS corrector solves raise BLAS/OpenMP to corrector_blas_threads (13 by
        # default), overriding OMP_NUM_THREADS: several workers on one node then each spin that
        # many threads (8x slower on Aire with 8 workers). Instead the cell keeps the process
        # settings (None) and evaluate() limits BLAS and OpenMP together to this worker's budget
        # for the whole call, so MUMPS analysis and solves see the same count (INFOG(1)=-58
        # otherwise, where OpenBLAS is built with OpenMP).
        self.threads = int(threads)
        settings = dict(settings)
        options = dict(settings.get("preconditioner_options") or {})
        options.setdefault("corrector_blas_threads", None)
        settings["preconditioner_options"] = options
        self.session = _load_solver().FSISession(**settings)
        self.committed = {}
        self.trial = {}
        self.cache = {}  # point -> (key, outputs, tangent, state, info): a revisited trial is free
        self.verbose = bool(settings.get("print_iterations") or settings.get("Profiling"))

    def evaluate(self, payload):
        from ngsolve import TaskManager

        points, rows, dt = payload
        outputs, tangents, newton, seconds, equilibrium = [], [], [], [], []
        for point, row in zip(points, rows):
            key = (row.tobytes(), dt)
            cached = self.cache.get(point)
            fresh = cached is None or cached[0] != key
            if fresh:
                started = perf_counter()
                if isinstance(dt, FreshEquilibrium):
                    call = dict(dt=list(dt.steps), loading_path=np.vstack([REST, np.tile(row, (len(dt.steps), 1))]),
                                initial_state=None, homogenise=True, save_response_history=True)
                else:
                    call = dict(F_target=row[:9].reshape(3, 3), P_target=float(row[9]), G_target=row[10:].copy(),
                                dt=float(dt), t_final=float(dt), initial_state=self.committed.get(point),
                                homogenise=True)
                try:
                    with TaskManager(), threadpool_limits(limits=self.threads), contextlib.ExitStack() as stack:
                        if not self.verbose:
                            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        result = self.session(**call)
                    _accepted(result)
                except MaterialPointFailure as error:
                    raise MaterialPointFailure(f"point {point}: {error}", [point]) from error
                except RuntimeError as error:
                    # The session closes itself on failure; the next call rebuilds its workspace.
                    raise MaterialPointFailure(f"point {point}: {error}", [point]) from error
                response = result["homogenised_response"]
                change = (_change(np.asarray(result["response_history"]["outputs"], dtype=float))
                          if isinstance(dt, FreshEquilibrium) else (np.nan,) * 3)
                cached = (key, np.array(response["outputs"], dtype=float),
                          np.array(response["jacobian"], dtype=float), result["restart_state"],
                          (sum(len(h) for h in result["newton_histories"]), perf_counter() - started), change)
                self.cache[point] = cached
            self.trial[point] = cached[3]
            outputs.append(cached[1])
            tangents.append(cached[2])
            newton.append(cached[4][0] if fresh else 0)  # a cache hit costs no solve
            seconds.append(cached[4][1] if fresh else 0.0)
            equilibrium.append(cached[5])
        return dict(outputs=np.array(outputs), tangents=np.array(tangents), newton=np.array(newton),
                    seconds=np.array(seconds), equilibrium=np.array(equilibrium))

    def commit(self, points):
        missing = set(points) - set(self.trial)
        if missing:
            raise RuntimeError(f"commit before every point was evaluated: {sorted(missing)}")
        self.committed.update(self.trial)
        self.trial.clear()
        self.cache.clear()

    def save(self, folder):
        """Pickle every committed point to folder/point_<i>.pkl (the restart state, mesh included)."""
        folder = Path(folder)
        for point, state in self.committed.items():
            with (folder / f"point_{point:05d}.pkl").open("wb") as stream:
                pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, payload):
        """Make the pickled states of these points the committed history, on one shared mesh.

        FSISession requires every restart state it is given to sit on its own mesh object, so the
        points' separately pickled (identical) meshes are replaced by the first one, after checking
        that they are the same mesh.
        """
        folder, points = payload
        if self.committed or self.trial:
            raise RuntimeError("load a checkpoint into fresh cells only")
        mesh = vertices = None
        for point in points:
            with (Path(folder) / f"point_{point:05d}.pkl").open("rb") as stream:
                state = pickle.load(stream)
            coordinates = np.array([v.point for v in state["mesh"].vertices])
            if mesh is None:
                mesh, vertices = state["mesh"], coordinates
            elif coordinates.shape != vertices.shape or np.max(np.abs(coordinates - vertices)) > 1e-12:
                raise RuntimeError(f"point {point} was saved on a different cell mesh")
            state["mesh"] = mesh
            self.committed[point] = state

    def close(self):
        self.session.close()


def _serve(connection, settings, threads):
    """Worker process loop: answer (command, payload) messages until 'close'."""
    points = None
    try:
        points = _Points(settings, threads)
        connection.send(("ok", None))
    except Exception:
        connection.send(("error", traceback.format_exc()))
        return
    while True:
        try:
            command, payload = connection.recv()
        except (EOFError, OSError):  # the parent process has gone: stop instead of waiting forever
            points.close()
            return
        if command == "close":
            points.close()
            connection.send(("ok", None))
            return
        try:
            connection.send(("ok", getattr(points, command)(payload)))
        except MaterialPointFailure as error:
            connection.send(("failure", (str(error), error.points.tolist())))
        except Exception:
            connection.send(("error", traceback.format_exc()))


class FECellModel:
    """The periodic FSI cell at every material point.

    ``settings`` are FSISession keywords except those the coupling owns
    (loading, dt, history, tangent): material (K/G or E/nu, mu_f, eta_s,
    zeta_s), ``geometry``, ``rve_length``, ``maxh``, ``order``, solver options.
    ``workers`` cell processes (0 = this one), each with ``threads_per_worker``
    NGSolve threads. Tangents always come from the linear correctors.
    """

    def __init__(self, n_points, *, workers=0, threads_per_worker=1, **settings):
        clash = OWNED & set(settings)
        if clash:
            raise ValueError(f"the coupling sets these cell inputs itself: {sorted(clash)}")
        self.n_points = int(n_points)
        self.settings = dict(settings)
        self.workers = min(int(workers), self.n_points)
        self.calls = 0
        self.newton_iterations = 0
        self.cell_seconds = 0.0
        self._lanes = []
        self._local = None
        if self.workers == 0:
            self._local = _Points(self.settings, threads_per_worker)
            return
        context = multiprocessing.get_context("spawn")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(name, "1")
        try:
            for _ in range(self.workers):
                parent, child = context.Pipe()
                process = context.Process(target=_serve, args=(child, self.settings, threads_per_worker), daemon=True)
                process.start()
                self._lanes.append((process, parent))
            for _, connection in self._lanes:
                self._reply(connection.recv())
        except BaseException:
            self.close()
            raise

    def _groups(self):
        lanes = max(self.workers, 1)
        return [np.arange(lane, self.n_points, lanes) for lane in range(lanes)]

    @staticmethod
    def _reply(message):
        status, value = message
        if status == "failure":
            raise MaterialPointFailure(value[0], value[1])
        if status == "error":
            raise RuntimeError("cell worker failed:\n" + value)
        return value

    def _dispatch(self, command, payloads):
        if self._local is not None:
            return [getattr(self._local, command)(payloads[0])]
        for (_, connection), payload in zip(self._lanes, payloads):
            connection.send((command, payload))
        replies = [connection.recv() for _, connection in self._lanes]  # drain every lane, even on failure
        failures = [value for status, value in replies if status == "failure"]
        errors = [value for status, value in replies if status == "error"]
        if errors:
            raise RuntimeError("cell worker failed:\n" + "\n".join(errors))
        if failures:
            raise MaterialPointFailure("; ".join(f[0] for f in failures), sum((f[1] for f in failures), []))
        return [value for _, value in replies]

    def evaluate(self, inputs, dt, *, tangent=True):
        inputs = check_inputs(inputs, self.n_points)
        groups = self._groups()
        dt = dt if isinstance(dt, FreshEquilibrium) else float(dt)
        replies = self._dispatch("evaluate", [(g.tolist(), inputs[g], dt) for g in groups])
        outputs = np.empty((self.n_points, 13))
        tangents = np.empty((self.n_points, 13, 13))
        newton = np.empty(self.n_points, dtype=int)
        seconds = np.empty(self.n_points)
        equilibrium = np.empty((self.n_points, 3))
        for g, reply in zip(groups, replies):
            outputs[g], tangents[g] = reply["outputs"], reply["tangents"]
            newton[g], seconds[g], equilibrium[g] = reply["newton"], reply["seconds"], reply["equilibrium"]
        fresh = seconds > 0
        self.calls += int(fresh.sum())
        self.newton_iterations += int(newton[fresh].sum())
        self.cell_seconds += float(seconds[fresh].sum())
        info = dict(newton=newton, seconds=seconds)
        if isinstance(dt, FreshEquilibrium):
            info["equilibrium"] = equilibrium
        return Response(outputs, tangents if tangent else None, info)

    def commit(self):
        self._dispatch("commit", [g.tolist() for g in self._groups()])

    def save_checkpoint(self, folder):
        """Write every point's committed history to folder (one file per point)."""
        Path(folder).mkdir(parents=True, exist_ok=True)
        self._dispatch("save", [str(folder)] * len(self._groups()))

    def load_checkpoint(self, folder):
        """Restore the committed histories written by save_checkpoint, before any evaluation.
        Any number of workers can load what any number saved."""
        missing = [i for i in range(self.n_points) if not (Path(folder) / f"point_{i:05d}.pkl").exists()]
        if missing:
            raise FileNotFoundError(f"checkpoint {folder} has no state for points {missing[:10]}")
        self._dispatch("load", [(str(folder), g.tolist()) for g in self._groups()])

    def close(self):
        if self._local is not None:
            self._local.close()
            self._local = None
        for process, connection in self._lanes:
            try:
                connection.send(("close", None))
                connection.recv()
            except (OSError, EOFError):
                pass
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
        self._lanes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
