#!/bin/bash
#SBATCH --job-name=verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# One array task = one verification job (one history, or one FD tangent check).
# Submitted by submit.sh, which sets the array range, memory, time and dependencies:
#   sbatch --array=0-13 run_study.sh REPO CAMPAIGN STUDY [run.py options]
# STUDY is A, B, C, D or F, or "collect" to run analyse.py on the campaign.
# The #SBATCH values above are defaults; submit.sh overrides them per study.
set -euo pipefail
REPO="${1:?repository path}"
CAMPAIGN="${2:?campaign folder}"
STUDY="${3:?study (A, B, C, D, F or collect)}"
shift 3

# NGSolve's TaskManager uses the allocated CPUs (run.py: SetNumThreads); every
# other threaded library is pinned to one thread so they do not oversubscribe.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
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
echo "Host $(hostname), job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-}, ${SLURM_CPUS_PER_TASK:-?} CPUs, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
python -c 'import ngsolve, numpy; print("NGSolve", ngsolve.__version__, "NumPy", numpy.__version__, flush=True)'

if [[ "$STUDY" == "collect" ]]; then
    exec python -u verification/combined/analyse.py "$CAMPAIGN"
fi
exec python -u verification/combined/run.py --study "$STUDY" --task "${SLURM_ARRAY_TASK_ID:?run as an array job}" \
    --output "$CAMPAIGN" --threads "${SLURM_CPUS_PER_TASK:-32}" "$@"
