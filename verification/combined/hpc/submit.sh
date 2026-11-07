#!/usr/bin/env bash
# Submit the verification campaign to Slurm. Run with bash, not sbatch:
#   bash verification/combined/hpc/submit.sh            submit studies A-D, F and the collect job
#   bash verification/combined/hpc/submit.sh --check    show what would be submitted; submit nothing
#   bash verification/combined/hpc/submit.sh --resume results/verification/combined_YYYYmmdd_HHMMSS
#        resubmit into an existing campaign: complete jobs are skipped, the others resume
#        from their last checkpoint.
#   --studies "F"   submit only these studies (overrides STUDIES below), e.g. to add
#        study F to a campaign started with A-D:
#        bash verification/combined/hpc/submit.sh --studies F --resume <campaign>
#   --campaign DIR  a new campaign in DIR instead of results/verification/combined_<date>_<time>
#   --smoke (or SMOKE=1 in the environment)   one job, every study on a coarse mesh with a few steps
#        (run.py --all --smoke), into results/verification/combined_smoke: a check, not verification data
# Each study is one array job (one task per history or FD check). A job stops at a
# checkpoint shortly before its wall-time limit (exit code 76); the study's follow-on
# segments (SEGMENTS_*: the same array again, each after the previous one) resume
# such jobs and skip complete ones in seconds. D waits for C's last segment, and the
# collect job (analyse.py) runs when every study's last segment has finished.
set -euo pipefail

# --------------------------- EDIT THESE SETTINGS ---------------------------

STUDIES="A B C D F"            # Studies to submit (D needs C's histories).
CAMPAIGN=""                    # Blank: results/verification/combined_<date>_<time> in the repository.

# One node, one task, CPUS threads per job. Memory, wall time per segment and number
# of segments per study (a history longer than one segment continues in the next);
# the finest mesh (A, maxh 0.03) and the longest history (B, N = 800) dominate.
CPUS=32                        # CPUs (NGSolve threads) per job; several jobs can share a 168-core node.
MAX_WALLTIME="48:00:00"        # The cluster's limit per job; every TIME_* must be within it.
MEM_A="256G";   TIME_A="48:00:00";   SEGMENTS_A=3
MEM_B="96G";    TIME_B="48:00:00";   SEGMENTS_B=2
MEM_C="96G";    TIME_C="24:00:00";   SEGMENTS_C=1
MEM_D="96G";    TIME_D="24:00:00";   SEGMENTS_D=1
MEM_F="96G";    TIME_F="24:00:00";   SEGMENTS_F=1
MEM_COLLECT="8G"; TIME_COLLECT="01:00:00"
TIME_MARGIN=20                 # Minutes before the limit by which a job stops at a checkpoint.
MAX_PARALLEL=""                # Blank: no limit; e.g. 4 runs at most 4 tasks of an array at once.
PARTITION=""                   # Blank: cluster default.
ACCOUNT=""                     # Blank: cluster default.
# Nodes to keep off (sbatch --exclude), e.g. "calder-cpu012" (Aire's node003 ran every job 8-9x slower on 26 Oct 2026).
EXCLUDE_NODES="${EXCLUDE_NODES-}"

# Restart the solver process every RESTART_EVERY steps (from its checkpoint) for
# meshes with maxh <= RESTART_MAXH: bounds any memory leak. Blank RESTART_EVERY: never.
RESTART_EVERY="25"
RESTART_MAXH="0.045"

# Login-node Python: standard library only (reads the job list; no NGSolve).
PREP_PYTHON="${PREP_PYTHON:-python3}"

# ------------------------- END OF EDITABLE SETTINGS ------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RUN="$REPO/verification/combined/run.py"
CHECK=false
SMOKE_RUN=false
[[ "${SMOKE:-0}" == 1 ]] && set -- --smoke "$@"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK=true; shift ;;
        --smoke) SMOKE_RUN=true; shift ;;
        --resume|--campaign) CAMPAIGN="${2:?$1 needs the campaign folder}"; shift 2 ;;
        --studies) STUDIES="${2:?--studies needs a list, e.g. \"F\" or \"A B\"}"; shift 2 ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run 'bash submit.sh' from a login shell; it submits its own Slurm jobs." >&2
    exit 2
fi
if [[ "$SMOKE_RUN" == true ]]; then
    LOGS="$REPO/results/verification/combined_smoke/slurm"
    COMMAND=(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00
             --job-name=verify-smoke --output="$LOGS/smoke-%j.out")
    [[ -n "$PARTITION" ]] && COMMAND+=("--partition=$PARTITION")
    [[ -n "$ACCOUNT" ]] && COMMAND+=("--account=$ACCOUNT")
    [[ -n "$EXCLUDE_NODES" ]] && COMMAND+=("--exclude=$EXCLUDE_NODES")
    COMMAND+=("$REPO/examples/multiscale/hpc/run_python_job.sh" "$REPO" verification/combined/run.py --all --smoke
              --threads 4)
    echo "Repository: $REPO (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
    if [[ "$CHECK" == true ]]; then echo "  ${COMMAND[*]}"; echo "Check only: nothing submitted."; exit 0; fi
    mkdir -p "$LOGS"
    JOB="$("${COMMAND[@]}")"
    echo "  verify-smoke: job ${JOB%%;*} (summary: results/verification/combined_smoke/analysis/summary.md)"
    exit 0
fi
if [[ -z "$CAMPAIGN" ]]; then
    CAMPAIGN="$REPO/results/verification/combined_$(date +%Y%m%d_%H%M%S)"
fi
CAMPAIGN="$("$PREP_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$CAMPAIGN")"

seconds() {  # Slurm time [D-]HH:MM:SS (or HH:MM, MM) -> seconds
    "$PREP_PYTHON" -c 'import sys; sys.path.insert(0, sys.argv[1]); from run import parse_time_limit; print(parse_time_limit(sys.argv[2]))' \
        "$REPO/verification/combined" "$1"
}
for STUDY in $STUDIES; do
    TIME_VAR="TIME_$STUDY"
    if (( $(seconds "${!TIME_VAR}") > $(seconds "$MAX_WALLTIME") )); then
        echo "$TIME_VAR=${!TIME_VAR} exceeds MAX_WALLTIME=$MAX_WALLTIME" >&2
        exit 2
    fi
done

COMMON=(--parsable --nodes=1 --ntasks=1)
if [[ -n "$PARTITION" ]]; then COMMON+=("--partition=$PARTITION"); fi
if [[ -n "$ACCOUNT" ]]; then COMMON+=("--account=$ACCOUNT"); fi
if [[ -n "$EXCLUDE_NODES" ]]; then COMMON+=("--exclude=$EXCLUDE_NODES"); fi
EXTRA=(--time-margin "$TIME_MARGIN")
if [[ -n "$RESTART_EVERY" ]]; then EXTRA+=(--restart-every "$RESTART_EVERY" --restart-maxh "$RESTART_MAXH"); fi

echo "Repository: $REPO (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "WARNING: tracked files are modified; the results will record a dirty commit." >&2
fi
echo "Campaign: $CAMPAIGN"
echo "Jobs (study, tasks, levels per duration):"
"$PREP_PYTHON" "$RUN" --list | grep -v '^#' | sed 's/^/  /'

if [[ "$CHECK" == false ]]; then
    "$PREP_PYTHON" "$RUN" --prepare --output "$CAMPAIGN" > /dev/null
    mkdir -p "$CAMPAIGN/slurm"
fi

SUBMITTED=""
submit() {  # submit NAME DEPENDENCY sbatch-options...; sets JOB_<NAME> to the job id
    local name="$1" dependency="$2" id
    shift 2
    local command=(sbatch "${COMMON[@]}")
    if [[ -n "$dependency" ]]; then command+=("--dependency=$dependency" --kill-on-invalid-dep=yes); fi
    command+=("$@")
    if [[ "$CHECK" == true ]]; then
        echo "  ${command[*]}"
        id="<$name-job$(( $(wc -w <<< "$SUBMITTED") + 1 ))>"
    else
        id="$("${command[@]}")"
        id="${id%%;*}"
        echo "  $name: job $id"
    fi
    printf -v "JOB_$name" '%s' "$id"
    SUBMITTED="$SUBMITTED $name=$id"
}

echo "Submitting:"
ALL=""
# Dependencies are afterany, not afterok: a job stopped at its time limit (exit 76)
# or a failure elsewhere must not cancel the follow-on segments, D or the collect job;
# D itself checks that its source history in C is complete.
for STUDY in $STUDIES; do
    TASKS="$("$PREP_PYTHON" "$RUN" --list | awk -v s="$STUDY" '$1 == s {print $2}')"
    MEM_VAR="MEM_$STUDY"; TIME_VAR="TIME_$STUDY"; SEGMENTS_VAR="SEGMENTS_$STUDY"
    ARRAY="0-$((TASKS - 1))${MAX_PARALLEL:+%$MAX_PARALLEL}"
    DEPENDENCY=""
    if [[ "$STUDY" == "D" && -n "${JOB_C:-}" ]]; then DEPENDENCY="afterany:$JOB_C"; fi  # C's last segment
    for ((SEGMENT = 1; SEGMENT <= ${!SEGMENTS_VAR}; SEGMENT++)); do
        submit "$STUDY" "$DEPENDENCY" --cpus-per-task="$CPUS" --array="$ARRAY" --job-name="verify-$STUDY$SEGMENT" \
            --mem="${!MEM_VAR}" --time="${!TIME_VAR}" --output="$CAMPAIGN/slurm/$STUDY$SEGMENT-%A_%a.out" \
            "$HERE/run_study.sh" "$REPO" "$CAMPAIGN" "$STUDY" --time-limit "${!TIME_VAR}" ${EXTRA[@]+"${EXTRA[@]}"}
        ID_VAR="JOB_$STUDY"
        DEPENDENCY="afterany:${!ID_VAR}"
    done
    ALL="${ALL:+$ALL:}${!ID_VAR}"
done
submit collect "afterany:$ALL" --job-name="verify-collect" --cpus-per-task=1 --mem="$MEM_COLLECT" \
    --time="$TIME_COLLECT" --output="$CAMPAIGN/slurm/collect-%j.out" \
    "$HERE/run_study.sh" "$REPO" "$CAMPAIGN" collect

if [[ "$CHECK" == true ]]; then
    echo "Check only: nothing submitted."
else
    echo "$(date): commit $(git -C "$REPO" rev-parse HEAD 2>/dev/null) jobs$SUBMITTED" >> "$CAMPAIGN/slurm/submitted.txt"
    echo "Follow with: squeue -u \$USER;  results and analysis in $CAMPAIGN (analysis/summary.md)."
    echo "If jobs are still unfinished after the last segment, run: bash $0 --resume $CAMPAIGN"
fi
