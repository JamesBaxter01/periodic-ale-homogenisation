#!/bin/bash
#SBATCH --job-name=resolved-refine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# One segment of a resolved run, submitted by submit_resolved_refinement.sh, submit_consolidation.sh and
# submit_cyclic.sh:
#   sbatch run_resolved_refinement.sh REPO RUN_FOLDER WALLTIME_SECONDS [consolidation_resolved.py options]
# Starts the run, resumes it from its checkpoint, or exits at once if it is already complete, so the
# submit script can queue several segments in a row; a segment that stops at its wall time with a
# checkpoint exits with code 76 (the next segment resumes it). analyse runs resolved_refinement.py instead:
#   sbatch run_resolved_refinement.sh REPO STUDY_FOLDER analyse BASE_FOLDER
set -euo pipefail
REPO="${1:?repository path}"
FOLDER="${2:?run folder}"
WALLTIME="${3:?walltime seconds, or analyse}"
shift 3

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

if [[ "$WALLTIME" == analyse ]]; then
    exec python -u examples/multiscale/resolved_refinement.py "$FOLDER" --base "${1:?base folder}"
fi

OUTPUT="$FOLDER/resolved.npz"
status() {  # the run's status in OUTPUT: new, walltime, complete, failed ...
    python - "$OUTPUT" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
path = Path(sys.argv[1])
if not path.exists():
    print("new")
else:
    with np.load(path, allow_pickle=False) as data:
        print(json.loads(str(data["metadata"])).get("status", "unknown"))
EOF
}
STATUS="$(status)"
echo "Run $FOLDER: status $STATUS"
case "$STATUS" in
    complete) echo "Already complete; nothing to do."; exit 0 ;;
    failed*) echo "The run failed; not resuming (see the previous segment's log)." >&2; exit 1 ;;
esac
RESUME=()
if [[ "$STATUS" != new && -f "$FOLDER/resolved_checkpoint.npz" ]]; then RESUME=(--resume); fi
python -u examples/multiscale/consolidation_resolved.py --full --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --output "$OUTPUT" --walltime "$WALLTIME" ${RESUME[@]+"${RESUME[@]}"} "$@"
STATUS="$(status)"
echo "Run $FOLDER: status $STATUS"
case "$STATUS" in
    complete) exit 0 ;;
    walltime) echo "Stopped at the wall time with a checkpoint; the next segment resumes it."; exit 76 ;;
    *) echo "Unexpected status $STATUS" >&2; exit 1 ;;
esac
