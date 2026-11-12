#!/usr/bin/env bash
# The Section 5.2 mechanism check on Calder (figure4_pore_fields.py): the Figure 4 pulse at T = 0.2 and 2 s,
# every step, with the stress and dissipation split by mechanism and VTK snapshots of the pore fields.
# One job per solid. Run with bash:
#   bash examples/microscale/hpc/submit_figure4_pore_fields.sh            submit the two jobs
#   bash examples/microscale/hpc/submit_figure4_pore_fields.sh --check    show the sbatch commands; submit nothing
#   bash examples/microscale/hpc/submit_figure4_pore_fields.sh --smoke    coarse mesh, one short pulse (or SMOKE=1)
# Writes results/paper/pore_fields_<stamp>/figure4_pore_fields_<solid>.npz and vtk/.
# Afterwards, on the Mac: rsync that folder into results/paper/.
set -euo pipefail

# --------------------------- EDIT THESE SETTINGS ---------------------------

CPUS=4                         # NGSolve threads per job, as figure4_cycles.py
MEM="16G"
TIME="04:00:00"                # each job: settling (~6 min) + 2 pulses of 150 steps (~4 s per step), under 1 h
PARTITION=""; ACCOUNT=""
EXCLUDE_NODES="${EXCLUDE_NODES-}"

# ------------------------- END OF EDITABLE SETTINGS ------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
CHECK=false; EXTRA=(); PREFIX=""
[[ "${SMOKE:-0}" == 1 ]] && set -- --smoke "$@"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK=true; shift ;;
        --smoke) EXTRA=(--smoke); MEM="8G"; TIME="00:30:00"; PREFIX="smoke_"; shift ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done
OUT="$REPO/results/paper/pore_fields_${PREFIX}$(date +%Y%m%d_%H%M%S)"
LOGS="$OUT/slurm"

echo "Repository: $REPO (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "Output: $OUT"
[[ "$CHECK" == true ]] || mkdir -p "$LOGS"
for solid in elastic viscoelastic; do
    COMMAND=(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
             --job-name="pore-fields-$solid" --output="$LOGS/pore-fields-$solid-%j.out")
    [[ -n "$PARTITION" ]] && COMMAND+=("--partition=$PARTITION")
    [[ -n "$ACCOUNT" ]] && COMMAND+=("--account=$ACCOUNT")
    [[ -n "$EXCLUDE_NODES" ]] && COMMAND+=("--exclude=$EXCLUDE_NODES")
    COMMAND+=("$REPO/examples/multiscale/hpc/run_python_job.sh" "$REPO" examples/microscale/figure4_pore_fields.py
              --threads "$CPUS" --output "$OUT" --solids "$solid" ${EXTRA[@]+"${EXTRA[@]}"})
    if [[ "$CHECK" == true ]]; then echo "  ${COMMAND[*]}"; continue; fi
    JOB="$("${COMMAND[@]}")"
    echo "pore-fields-$solid: job ${JOB%%;*}"
done
if [[ "$CHECK" == true ]]; then echo "Check only: nothing submitted."; exit 0; fi
echo "Follow with: squeue -u \$USER;  tail -f $LOGS/pore-fields-*.out"
