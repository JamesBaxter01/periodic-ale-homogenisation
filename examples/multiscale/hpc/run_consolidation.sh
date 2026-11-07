#!/bin/bash
#SBATCH --job-name=consolidation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# One job of the consolidation comparison, submitted by submit_consolidation.sh:
#   sbatch run_consolidation.sh REPO RUN_FOLDER fe2|resolved|compare MODE [extra options]
# MODE is smoke or full. The #SBATCH values are defaults; submit_consolidation.sh overrides them.
set -euo pipefail
REPO="${1:?repository path}"
FOLDER="${2:?run folder}"
PART="${3:?fe2, resolved or compare}"
MODE="${4:?smoke or full}"
shift 4

# NGSolve's TaskManager and the FE² worker processes use the allocated CPUs; every other
# threaded library is pinned to one thread so they do not oversubscribe.
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
echo "Host $(hostname), job ${SLURM_JOB_ID:-}, ${SLURM_CPUS_PER_TASK:-?} CPUs, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), $PART ($MODE)"
CPUS="${SLURM_CPUS_PER_TASK:-32}"
# The mode is passed explicitly, so SMOKE_TEST in the example files does not matter here.
SMOKE=(--full)
if [[ "$MODE" == smoke ]]; then SMOKE=(--smoke); fi

case "$PART" in
    fe2)
        # One cell process per CPU, one NGSolve thread each (40 cells: CPUS=40 solves each batch at once).
        exec python -u examples/multiscale/consolidation_multiscale.py "${SMOKE[@]}" \
            --workers "$CPUS" --threads 1 --output "$FOLDER/fe2.npz" "$@" ;;
    resolved)
        exec python -u examples/multiscale/consolidation_resolved.py "${SMOKE[@]}" \
            --threads "$CPUS" --output "$FOLDER/resolved.npz" "$@" ;;
    compare)
        exec python -u examples/multiscale/compare_consolidation.py --fe2 "$FOLDER/fe2.npz" \
            --resolved "$FOLDER/resolved.npz" --output "$FOLDER/comparison" ;;
    *) echo "unknown part $PART" >&2; exit 2 ;;
esac
