"""Shared helpers for the paper-figure scripts (figure3_biot.py, figure4_cycles.py, figure4_stationary.py)."""

from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "verification"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from state_io import solver_sha256  # noqa: E402  (entry file plus periodic_fsi/)

DATA = ROOT / "results" / "paper"  # where the figure scripts save, and plot_paper_figures.py reads


def command_line(settings, smoke, modes=None):
    """Optional command-line options, used by run.py figure3/figure4; pressing Run needs none.

    --smoke applies the script's SMOKE settings (coarse mesh, tiny loads) and saves under
    results/paper_smoke/ unless --output is given; --output DIR sets the folder for the .npz;
    --threads N overrides THREADS. ``modes`` ({name: (help, settings)}) adds an option --name for
    each, applying its settings before --smoke. ``settings`` is the script's globals().
    """
    import argparse

    global DATA
    p = argparse.ArgumentParser(description=settings.get("__doc__"),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="coarse, small-load functionality check")
    p.add_argument("--output", type=Path, help=f"folder for the .npz (default {DATA})")
    p.add_argument("--threads", type=int, help="NGSolve threads (default THREADS)")
    for name, (text, _) in (modes or {}).items():
        p.add_argument(f"--{name}", action="store_true", help=text)
    a = p.parse_args()
    for name, (_, values) in (modes or {}).items():
        if getattr(a, name):
            settings.update(values)
    if a.smoke:
        settings.update(smoke)
        DATA = ROOT / "results" / "paper_smoke"
    if a.output:
        DATA = a.output
    if a.threads:
        settings["THREADS"] = a.threads


def compiler_environment():
    """For compile_tangent=True: ngscxx needs this environment's python3 first on PATH, and on this
    Mac the SDK's libc++ headers must win over a stale CommandLineTools copy. Fills in what is missing."""
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1")
    if sdk.is_dir() and "CPLUS_INCLUDE_PATH" not in os.environ:
        os.environ["CPLUS_INCLUDE_PATH"] = str(sdk)


def save_results(name, settings, seconds, **arrays):
    """Save DATA/<name>.npz (arrays plus a JSON 'metadata' string). An existing file of that name
    is kept as <name>_<date-time>.npz, so no earlier run is overwritten."""
    import numpy as np

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.npz"
    if path.exists():
        kept = path.with_name(f"{name}_{time.strftime('%Y%m%d-%H%M%S', time.localtime(path.stat().st_mtime))}.npz")
        path.rename(kept)
        print(f"Kept the previous {path.name} as {kept.name}")
    metadata = dict(settings=settings, seconds=seconds, solver_sha256=solver_sha256(ROOT / "homogenisation_microscale.py"),
                    saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata, default=float)), **arrays)
    print(f"Saved {path}  ({seconds / 60:.1f} min)")
    return path


def accumulate(totals, profiling):
    """Sum one call's result["profiling"] into totals, for format_profile_table."""
    for key, value in profiling.items():
        totals[key] = totals.get(key, 0.0) + float(value)
