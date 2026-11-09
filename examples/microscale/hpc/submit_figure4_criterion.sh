#!/usr/bin/env bash
# The history criterion on Calder (figure4_cycles.py --criterion): the compression pulse of Figure 4 at
# thirteen durations from 0.2 to 200 s, one job per solid, plus the T = 2 and 20 s pulses at 40 steps
# per pulse (--criterion-check, both solids). Run with bash:
#   bash examples/microscale/hpc/submit_figure4_criterion.sh            submit the three jobs
#   bash examples/microscale/hpc/submit_figure4_criterion.sh --check    show the sbatch commands; submit nothing
#   bash examples/microscale/hpc/submit_figure4_criterion.sh --smoke    coarse mesh, one short pulse (or SMOKE=1)
# Writes results/paper/criterion_<stamp>/figure4_criterion_elastic.npz, figure4_criterion_viscoelastic.npz
# and figure4_criterion_steps40.npz. Afterwards, on the Mac: rsync that folder into results/paper/.
set -euo pipefail

# --------------------------- EDIT THESE SETTINGS ---------------------------

CPUS=4                         # NGSolve threads per job, as the Section 5.2 rate sweep
MEM="16G"
TIME="24:00:00"                # each sweep job: about 3 h (13 pulses of 150 steps, about 5 s per step)
CHECK_TIME="06:00:00"          # the 40-step check: under 1 h
PARTITION=""; ACCOUNT=""
# Nodes to keep off (sbatch --exclude), e.g. "calder-cpu012" (Aire's node003 ran every job 8-9x slower on 26 Oct 2026).
EXCLUDE_NODES="${EXCLUDE_NODES-}"

# ------------------------- END OF EDITABLE SETTINGS ------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
CHECK=false; EXTRA=(); PREFIX=""
[[ "${SMOKE:-0}" == 1 ]] && set -- --smoke "$@"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK=true; shift ;;
        --smoke) EXTRA=(--smoke); MEM="8G"; TIME="00:30:00"; CHECK_TIME="00:30:00"; PREFIX="smoke_"; shift ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done
OUT="$REPO/results/paper/criterion_${PREFIX}$(date +%Y%m%d_%H%M%S)"
LOGS="$OUT/slurm"

command_for() {  # job name, time limit, then the script's options
    local name="$1" time="$2"; shift 2
    COMMAND=(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" --time="$time"
             --job-name="$name" --output="$LOGS/$name-%j.out")
    [[ -n "$PARTITION" ]] && COMMAND+=("--partition=$PARTITION")
    [[ -n "$ACCOUNT" ]] && COMMAND+=("--account=$ACCOUNT")
    [[ -n "$EXCLUDE_NODES" ]] && COMMAND+=("--exclude=$EXCLUDE_NODES")
    COMMAND+=("$REPO/examples/multiscale/hpc/run_python_job.sh" "$REPO" examples/microscale/figure4_cycles.py
              --threads "$CPUS" --output "$OUT" "$@" ${EXTRA[@]+"${EXTRA[@]}"})
}

echo "Repository: $REPO (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "Output: $OUT"
[[ "$CHECK" == true ]] || mkdir -p "$LOGS"
for job in "criterion-elastic $TIME --criterion --solids elastic" \
           "criterion-viscoelastic $TIME --criterion --solids viscoelastic" \
           "criterion-steps40 $CHECK_TIME --criterion-check"; do
    read -r -a ARGS <<< "$job"
    command_for "${ARGS[@]}"
    if [[ "$CHECK" == true ]]; then echo "  ${COMMAND[*]}"; continue; fi
    JOB="$("${COMMAND[@]}")"
    echo "${ARGS[0]}: job ${JOB%%;*}"
done
if [[ "$CHECK" == true ]]; then echo "Check only: nothing submitted."; exit 0; fi
echo "Follow with: squeue -u \$USER;  tail -f $LOGS/criterion-*.out"
