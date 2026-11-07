#!/bin/bash
#SBATCH --job-name=python-job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# One Python script as a Slurm job (used by submit_cyclic.sh and submit_figure4_stationary.sh):
#   sbatch run_python_job.sh REPO SCRIPT [script options]
# SCRIPT is a path relative to the repository, run with the job's CPUs as NGSolve threads.
set -euo pipefail
REPO="${1:?repository path}"
SCRIPT="${2:?script}"
shift 2

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
echo "Host $(hostname), job ${SLURM_JOB_ID:-}, ${SLURM_CPUS_PER_TASK:-?} CPUs, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown): $SCRIPT $*"
exec python -u "$SCRIPT" "$@"
