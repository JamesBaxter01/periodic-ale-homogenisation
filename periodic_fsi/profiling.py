"""Phase timings, the nested native-setup call breakdown and the stage summary table for ``Profiling=True``."""

from __future__ import annotations
from time import perf_counter

_OUTPUT_KEYS = (
    "Homogenisation: calculating stress",
    "Homogenisation: calculating m.f.",
    "Homogenisation: calculating Q",
)
# (label, profiling keys, indent). Indented rows are parts of the row above them.
_SUMMARY_STAGES = (
    ("Setup: geometry, mesh, spaces, forms",
     ("Geometry setup", "Meshing", "Space setup", "Reusable block system construction"), 0),
    ("Timesteps", ("Timestep total",), 0),
    ("Newton tangent assembly", ("Newton: component block tangent assembly",), 1),
    ("Newton GMRES", ("Newton: GMRES solve",), 1),
    ("Residual evaluation and line search", ("Newton: residual evaluation", "Newton: line search"), 1),
    ("Proxy refresh (residual metric, PC proxies)", ("Preconditioner: timestep proxy refresh",), 1),
    ("State and mesh update", ("Timestep: state and mesh update",), 1),
    ("Homogenisation", ("Homogenisation: 13 linear correctors",), 0),
    ("Corrector MUMPS factorisation", ("Homogenisation: compact linear solver setup",), 1),
    ("Corrector solves", ("Homogenisation: solving corrector",), 1),
    ("Monolithic linearisation", ("Homogenisation: linearising monolithic tangent",), 1),
    ("Residual derivatives", ("Homogenisation: exact residual derivative assembly",), 1),
    ("Output derivatives (Jacobian columns)", ("Homogenisation: assembling exact Jacobian column",), 1),
    ("Outputs (stress, fluid content, flux)", _OUTPUT_KEYS, 1),
    ("Final mesh quality", ("Final mesh quality",), 0),
)


def format_profile_table(totals, title="Profiling summary"):
    """Return a table of the main stages of ``totals`` (``result["profiling"]``, or a sum over calls).

    Shares are of "FSI total". Indented rows are parts of the row above them;
    rows that did not run (or round to zero) are left out.
    """
    total = float(totals.get("FSI total", 0.0))
    if not total:
        return f"{title}: no profiling data (run with Profiling=True)"
    rule = "-" * 66
    lines = [title, f"{'stage':<48}{'seconds':>10}{'share':>8}", rule]
    for label, keys, indent in _SUMMARY_STAGES:
        if label == "Homogenisation" and keys[0] not in totals:
            keys = _OUTPUT_KEYS  # tangent_method=None: outputs only, no corrector timer
        if not any(key in totals for key in keys):
            continue
        seconds = sum(float(totals[key]) for key in keys if key in totals)
        if indent and seconds < 0.005:
            continue
        lines.append(f"{('  ' * indent + label)[:47]:<48}{seconds:>10.2f}{100 * seconds / total:>7.1f}%")
    lines += [rule, f"{'FSI total':<48}{total:>10.2f}{100.0:>7.1f}%"]
    return "\n".join(lines)


class Profiler:
    """Accumulate phase totals and inclusive setup-call timings.

    When disabled, nothing is timed or printed and setup calls run directly.
    ``timestep`` labels setup records; the caller sets it at each step.
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.totals = {}
        self.counts = {}
        self.setup_records = {}
        self.setup_stack = []
        self.timestep = 0

    def reset(self):
        self.totals.clear()
        self.counts.clear()
        self.setup_records.clear()
        self.setup_stack.clear()

    def setup_call(self, phase, name, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        parent = tuple((item[1] for item in self.setup_stack))
        key = (int(self.timestep), phase, parent, name)
        self.setup_stack.append((phase, name))
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = perf_counter() - started
            self.setup_stack.pop()
            entry = self.setup_records.setdefault(
                key,
                {
                    "timestep": key[0],
                    "phase": phase,
                    "name": name,
                    "parent": " / ".join(parent),
                    "depth": len(parent),
                    "seconds": 0.0,
                    "count": 0,
                    "timing_kind": "inclusive call boundary",
                },
            )
            entry["seconds"] += elapsed
            entry["count"] += 1

    def setup_component(self, name, function):
        phase = self.setup_stack[-1][0] if self.setup_stack else "unscoped"
        return self.setup_call(phase, name, function)

    def setup_snapshot(self):
        return [dict(record) for record in self.setup_records.values()]

    def print_setup_breakdown(self):
        if not self.enabled:
            return
        self.summary(
            "Native setup call breakdown: inclusive parent/child timings; do not sum nested rows"
        )
        self.summary(
            "  Assemble includes any registered native setup callbacks; explicit Update calls are counted separately"
        )
        records = sorted(
            self.setup_snapshot(),
            key=lambda row: (
                row["timestep"],
                row["phase"],
                tuple(row["parent"].split(" / ")) if row["parent"] else (),
                row["depth"],
                row["name"],
            ),
        )
        groups = {}
        for row in records:
            groups.setdefault((row["timestep"], row["phase"]), []).append(row)
        for (timestep, phase), rows in groups.items():
            self.summary(f"  timestep {timestep}, {phase}:")
            rows.sort(
                key=lambda row: tuple(
                    ([*row["parent"].split(" / ")] if row["parent"] else []) + [row["name"]]
                )
            )
            for row in rows:
                indent = "    " + "  " * row["depth"]
                self.summary(f"{indent}{row['name']}: {row['seconds']:.3f} s ({row['count']} calls)")

    def summary(self, message):
        if self.enabled:
            print(f"[profiling] {message}")

    def print_summary_table(self):
        if self.enabled:
            print("\n" + format_profile_table(self.totals), flush=True)

    def record(self, key, elapsed):
        if not self.enabled:
            return
        self.totals[key] = self.totals.get(key, 0.0) + elapsed
        self.counts[key] = self.counts.get(key, 0) + 1

    def start(self):
        return perf_counter() if self.enabled else None

    def finish(self, key, started):
        if started is None:
            return 0.0
        elapsed = perf_counter() - started
        self.record(key, elapsed)
        return elapsed


def profiled_setup(phase, name):
    """Time a method as a setup call of the instance's ``profiler``."""

    def decorate(function):

        def measured(self, *args, **kwargs):
            return self.profiler.setup_call(phase, name, function, self, *args, **kwargs)

        return measured

    return decorate
