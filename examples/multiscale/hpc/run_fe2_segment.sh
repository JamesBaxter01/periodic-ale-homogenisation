#!/bin/bash
#SBATCH --job-name=fe2-segment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# One segment of an FE² consolidation run, submitted by submit_consolidation.sh and submit_cyclic.sh:
#   sbatch run_fe2_segment.sh REPO OUTPUT WALLTIME_SECONDS MODE [consolidation_multiscale.py options]
# Starts the run, resumes it from OUTPUT's checkpoint (<output>_checkpoint/), or exits at once if it has
# ended, so the submit script can queue several segments in a row. MODE is smoke or full. A segment that
# stops at the wall limit with a checkpoint exits with code 76: the next segment (afterany) resumes it, and
# jobs that need the finished run (afterok on the last segment) are not started on a partial one.
set -euo pipefail
REPO="${1:?repository path}"
OUTPUT="${2:?history file}"
WALLTIME="${3:?walltime seconds}"
MODE="${4:?smoke or full}"
shift 4

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1
# Calder: the conda module is miniforge3 under calder/cpu, which a batch job starts without (Lmod refuses
# miniforge3 alone); it has no `source activate`, and conda's own
# activation scripts read unset variables, so -u is off around it.
module load calder/cpu miniforge3
set +u
eval "$(conda shell.bash hook)"
conda activate ngsolve-petsc
set -u

cd "$REPO"
echo "Host $(hostname), job ${SLURM_JOB_ID:-}, ${SLURM_CPUS_PER_TASK:-?} CPUs, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

status() {  # the run's status in OUTPUT: new, wall limit, t_max, stopped, failed ...
    python - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
path = Path(sys.argv[1])
if not path.exists():
    print("new")
else:
    with np.load(path, allow_pickle=False) as data:
        print(json.loads(str(data["metadata"])).get("status", "unknown"))
PY
}
STATUS="$(status)"
echo "Run $OUTPUT: status $STATUS"
case "$STATUS" in
    t_max|stopped) echo "Already ended; nothing to do."; exit 0 ;;
    failed*) echo "The run failed; not resuming (see the previous segment's log)." >&2; exit 1 ;;
esac
python -u examples/multiscale/consolidation_multiscale.py "--$MODE" --workers "${SLURM_CPUS_PER_TASK:-40}" \
    --threads 1 --output "$OUTPUT" --walltime "$WALLTIME" --resume "$@"
STATUS="$(status)"
echo "Run $OUTPUT: status $STATUS"
case "$STATUS" in
    t_max|stopped) exit 0 ;;
    "wall limit") echo "Stopped at the wall limit with a checkpoint; the next segment resumes it."; exit 76 ;;
    *) echo "Unexpected status $STATUS" >&2; exit 1 ;;
esac
