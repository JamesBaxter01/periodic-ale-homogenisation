#!/usr/bin/env bash
# Refinement check of the resolved consolidation reference (elastic step load). Run with bash, not sbatch:
#   bash examples/multiscale/hpc/submit_resolved_refinement.sh            submit everything
#   bash examples/multiscale/hpc/submit_resolved_refinement.sh --check    show the sbatch commands; submit nothing
#   bash examples/multiscale/hpc/submit_resolved_refinement.sh --smoke    coarse mesh, first SMOKE_STEPS steps, 30 min
#                                     (SMOKE=1 in the environment is the same)
#   --base DIR     the production run (default BASE below); --folder DIR   the study folder
#   --after JOBID  start once that job (the base run's FE²) has completed: submit before fe2.npz exists
#   --analyse-after JOBID   the analysis also waits for that job (the base run's resolved run)
# Two resolved runs of the production column, each against the production run BASE:
#   mesh: MESH_MAXH instead of the production 0.09, on the FE² run's time grid;
#   time: the production mesh, with every interval of the FE² time grid split into TIME_REFINE steps.
# Each run is a chain of SEGMENTS jobs: a segment stops at a checkpoint before its wall time and the next
# resumes it (or exits at once once the run is complete). An analysis job runs when both chains have ended
# and writes <folder>/refinement_summary.md; examples/multiscale/resolved_refinement.py can also be run
# at any time while the jobs are going.
set -euo pipefail

# --------------------------- EDIT THESE SETTINGS ---------------------------

BASE="results/consolidation/full_elastic_step_20261026_201037"   # production run (fe2.npz, resolved.npz)
FOLDER=""                      # Blank: results/consolidation/refinement_<date>_<time>

MESH_MAXH=0.075                # production 0.09: 2.4x the elements (h_eff / 1.34)
TIME_REFINE=2                  # production 1: every FE² interval split in two (832 steps instead of 416)

# Resolved: one process with CPUS NGSolve and MUMPS threads (8 was fastest per step on Aire).
CPUS=8
MESH_MEM="64G"; MESH_SEGMENTS=4      # about 2-4 days in all
TIME_MEM="32G"; TIME_SEGMENTS=2      # about 1.5 days in all
SEGMENT_TIME="48:00:00"        # Calder's limit
TIME_MARGIN=40                 # minutes before the limit at which a segment stops (a fine-mesh step can take ~15 min)
PARTITION=""                   # Blank: cluster default.
ACCOUNT=""                     # Blank: cluster default.
# Nodes to keep off (sbatch --exclude), e.g. "calder-cpu012" (Aire's node003 ran every job 8-9x slower on 26 Oct 2026).
EXCLUDE_NODES="${EXCLUDE_NODES-}"

SMOKE_STEPS=120                # --smoke: first steps of the FE² grid only (to t ~ 0.02 s), on a coarse mesh

# ------------------------- END OF EDITABLE SETTINGS ------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
JOB="$HERE/run_resolved_refinement.sh"
CHECK=false
AFTER=""; ANALYSE_AFTER=""
[[ "${SMOKE:-0}" == 1 ]] && set -- --smoke "$@"
SMOKE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK=true; shift ;;
        --smoke) SMOKE=true; shift ;;
        --base) BASE="${2:?--base needs the production run folder}"; shift 2 ;;
        --folder) FOLDER="${2:?--folder needs a directory}"; shift 2 ;;
        --after) AFTER="${2:?--after needs a job id}"; shift 2 ;;
        --analyse-after) ANALYSE_AFTER="${2:?--analyse-after needs a job id}"; shift 2 ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run 'bash submit_resolved_refinement.sh' from a login shell; it submits its own Slurm jobs." >&2
    exit 2
fi
[[ "$BASE" == /* ]] || BASE="$REPO/$BASE"
if [[ -z "$AFTER" && ! -f "$BASE/fe2.npz" ]]; then
    echo "BASE $BASE needs fe2.npz (its FE² run finished; the analysis also reads its resolved.npz)" >&2
    exit 2
fi
if [[ -z "$FOLDER" ]]; then
    FOLDER="$REPO/results/consolidation/refinement_$( [[ "$SMOKE" == true ]] && echo smoke_ )$(date +%Y%m%d_%H%M%S)"
fi
[[ "$FOLDER" == /* ]] || FOLDER="$PWD/$FOLDER"

GRID="$BASE/fe2.npz"
MESH_OPTIONS=(--maxh "$MESH_MAXH")
TIME_OPTIONS=(--time-refine "$TIME_REFINE")
if [[ "$SMOKE" == true ]]; then
    SEGMENT_TIME="00:30:00"; TIME_MARGIN=5; MESH_SEGMENTS=1; TIME_SEGMENTS=1; MESH_MEM="16G"; TIME_MEM="16G"
    MESH_OPTIONS=(--maxh 0.3 --grid-steps "$SMOKE_STEPS")
    TIME_OPTIONS=(--maxh 0.4 --time-refine "$TIME_REFINE" --grid-steps "$SMOKE_STEPS")
fi

seconds() {  # Slurm time [D-]HH:MM:SS -> seconds
    local t="$1" days=0
    if [[ "$t" == *-* ]]; then days="${t%%-*}"; t="${t#*-}"; fi
    IFS=: read -r h m s <<< "$t"
    echo $(( days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#${s:-0} ))
}
WALLTIME=$(( $(seconds "$SEGMENT_TIME") - 60 * TIME_MARGIN ))

COMMON=(--parsable --nodes=1 --ntasks=1)
if [[ -n "$PARTITION" ]]; then COMMON+=("--partition=$PARTITION"); fi
if [[ -n "$ACCOUNT" ]]; then COMMON+=("--account=$ACCOUNT"); fi
if [[ -n "$EXCLUDE_NODES" ]]; then COMMON+=("--exclude=$EXCLUDE_NODES"); fi

echo "Repository: $REPO (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "WARNING: tracked files are modified; the results record a dirty commit." >&2
fi
echo "Production run: $BASE"
echo "Study folder:   $FOLDER$( [[ "$SMOKE" == true ]] && echo ' (smoke)')"

submit() {  # submit NAME DEPENDENCY sbatch-options...; sets LAST
    local name="$1" dependency="$2" id
    shift 2
    local command=(sbatch "${COMMON[@]}")
    # With --after, every job also needs that job to have completed; if it failed, all are cancelled.
    if [[ -n "$AFTER" ]]; then dependency="${dependency:+$dependency,}afterok:$AFTER"; fi
    if [[ -n "$dependency" ]]; then command+=("--dependency=$dependency" --kill-on-invalid-dep=yes); fi
    command+=("$@")
    if [[ "$CHECK" == true ]]; then
        echo "  ${command[*]}"
        id="<$name>"
    else
        id="$("${command[@]}")"
        id="${id%%;*}"
        echo "  $name: job $id"
    fi
    LAST="$id"
}

chain() {  # chain NAME SEGMENTS MEM options...: SEGMENTS jobs, each after the previous one ends
    local name="$1" segments="$2" mem="$3" dependency="" k
    shift 3
    if [[ "$CHECK" == false ]]; then mkdir -p "$FOLDER/$name/slurm"; fi
    for (( k = 1; k <= segments; k++ )); do
        submit "$name-$k" "$dependency" --job-name="refine-$name" --cpus-per-task="$CPUS" --mem="$mem" \
            --time="$SEGMENT_TIME" --output="$FOLDER/$name/slurm/resolved-%j.out" \
            "$JOB" "$REPO" "$FOLDER/$name" "$WALLTIME" --skeleton elastic --step --time-grid "$GRID" "$@"
        dependency="afterany:$LAST"
    done
}

echo "Submitting:"
chain mesh "$MESH_SEGMENTS" "$MESH_MEM" "${MESH_OPTIONS[@]}"
MESH_LAST="$LAST"
chain time "$TIME_SEGMENTS" "$TIME_MEM" "${TIME_OPTIONS[@]}"
TIME_LAST="$LAST"
submit analyse "afterany:$MESH_LAST:$TIME_LAST${ANALYSE_AFTER:+:$ANALYSE_AFTER}" --job-name="refine-analyse" --cpus-per-task=1 --mem=8G \
    --time=01:00:00 --output="$FOLDER/analyse-%j.out" "$JOB" "$REPO" "$FOLDER" analyse "$BASE"

if [[ "$CHECK" == true ]]; then
    echo "Check only: nothing submitted."
else
    echo "Follow with: squeue -u \$USER;  tail -f $FOLDER/*/slurm/*.out"
    echo "Summary (when the analysis job has run): $FOLDER/refinement_summary.md"
fi
