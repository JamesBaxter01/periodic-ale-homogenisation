#!/usr/bin/env bash
# The stationary cell of Figure 4 on Calder (figure4_stationary.py): one fresh, equilibrated cell per
# compression, all solved at once, one per CPU. Run with bash:
#   bash examples/microscale/hpc/submit_figure4_stationary.sh            submit
#   bash examples/microscale/hpc/submit_figure4_stationary.sh --check    show the sbatch command; submit nothing
#   bash examples/microscale/hpc/submit_figure4_stationary.sh --smoke    coarse mesh, three points (or SMOKE=1)
# Writes results/paper/figure4_stationary.npz (an earlier file is kept with its date). Afterwards, on the
# Mac: rsync that file into results/paper/ and run examples/microscale/plot_paper_figures.py.
set -euo pipefail

# --------------------------- EDIT THESE SETTINGS ---------------------------

CPUS=41                        # one cell per compression (POINTS in figure4_stationary.py)
MEM="96G"
TIME="04:00:00"                # about 3 minutes per cell at one thread; the cells run together
PARTITION=""; ACCOUNT=""
# Nodes to keep off (sbatch --exclude), e.g. "calder-cpu012" (Aire's node003 ran every job 8-9x slower on 26 Oct 2026).
EXCLUDE_NODES="${EXCLUDE_NODES-}"

# ------------------------- END OF EDITABLE SETTINGS ------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
CHECK=false; EXTRA=()
[[ "${SMOKE:-0}" == 1 ]] && set -- --smoke "$@"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK=true; shift ;;
        --smoke) EXTRA=(--smoke); CPUS=3; MEM="8G"; TIME="00:30:00"; shift ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done
LOGS="$REPO/results/paper/slurm"
COMMAND=(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
         --job-name=figure4-stationary --output="$LOGS/figure4-stationary-%j.out")
[[ -n "$PARTITION" ]] && COMMAND+=("--partition=$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMAND+=("--account=$ACCOUNT")
[[ -n "$EXCLUDE_NODES" ]] && COMMAND+=("--exclude=$EXCLUDE_NODES")
COMMAND+=("$REPO/examples/multiscale/hpc/run_python_job.sh" "$REPO" examples/microscale/figure4_stationary.py
          --threads "$CPUS" ${EXTRA[@]+"${EXTRA[@]}"})

echo "Repository: $REPO (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
if [[ "$CHECK" == true ]]; then echo "  ${COMMAND[*]}"; echo "Check only: nothing submitted."; exit 0; fi
mkdir -p "$LOGS"
JOB="$("${COMMAND[@]}")"
echo "figure4-stationary: job ${JOB%%;*}"
echo "Follow with: squeue -u \$USER;  tail -f $LOGS/figure4-stationary-${JOB%%;*}.out"
