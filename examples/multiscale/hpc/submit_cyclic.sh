#!/usr/bin/env bash
# Cyclic consolidation: FE² with history, FE² with stationary cells, and the resolved column. Run with bash:
#   bash examples/multiscale/hpc/submit_cyclic.sh            submit everything
#   bash examples/multiscale/hpc/submit_cyclic.sh --check    show the sbatch commands; submit nothing
#   bash examples/multiscale/hpc/submit_cyclic.sh --smoke    coarse cells, one cycle, a few minutes per job
#                                     (SMOKE=1 in the environment is the same)
#   bash examples/multiscale/hpc/submit_cyclic.sh --folder DIR   the study folder (default below)
#   bash examples/multiscale/hpc/submit_cyclic.sh --periods "20 200" --skeletons elastic
#                                     other periods or skeletons than PERIODS and SKELETONS below (the history
#                                     criterion runs: long periods, compared with the cell sweep of Figure 4)
#   bash examples/multiscale/hpc/submit_cyclic.sh --stationary-only FOLDER
#                                     (re)run only the stationary-cell FE² runs in an existing study FOLDER,
#                                     then its comparisons and summary
# The consolidation column of the paper (same macro mesh, cells and resolved mesh) under the raised-cosine
# traction TRACTION (1 - cos(2 pi t / P)) / 2 for CYCLES cycles, for every period in PERIODS. Per period:
#   stationary_P<P>/fe2_stationary.npz       FE² with stationary cells (no history; the same for both skeletons)
#   <skeleton>_P<P>/fe2.npz                   FE² with history, for each skeleton in SKELETONS
#   <skeleton>_P<P>/resolved.npz              the resolved column on that FE² run's time grid
#   <skeleton>_P<P>/comparison/cyclic.*       compare_cyclic.py, when the three have ended
# and, at the end, results/cyclic/<date>/cyclic.md joining every comparison.
set -euo pipefail

# --------------------------- EDIT THESE SETTINGS ---------------------------

PERIODS="0.2 2"                # load periods (s), as the pulse durations of Figure 4
CYCLES=5
SKELETONS="elastic viscoelastic"
FOLDER=""                      # Blank: results/cyclic/<date>_<time>
CPUS_FE2=40; MEM_FE2="128G"; TIME_FE2="48:00:00"     # one cell process per CPU (40 cells)
FE2_SEGMENTS=2                 # 48 h jobs in a row for each FE² run with history (each resumes the last checkpoint)
STATIONARY_SEGMENTS=3          # 48 h jobs in a row for each stationary run (each cell call is three solves)
# Macro Newton tolerance of the stationary runs, per period ("period:tolerance"; others the default 1e-8). At
# T = 0.2 s the cells' solver tolerance over dt = 5 ms sets a residual floor near 1e-7.
STATIONARY_NEWTON="0.2:1e-6"
CPUS_RESOLVED=8; MEM_RESOLVED="32G"; TIME_RESOLVED="48:00:00"; TIME_MARGIN=40
RESOLVED_SEGMENTS=2            # the same for each resolved run
PARTITION=""; ACCOUNT=""
# Nodes to keep off (sbatch --exclude), e.g. "calder-cpu012" (Aire's node003 ran every job 8-9x slower on 26 Oct 2026).
EXCLUDE_NODES="${EXCLUDE_NODES-}"

# ------------------------- END OF EDITABLE SETTINGS ------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
CHECK=false; MODE=--full; STEPS=""; ONLY_STATIONARY=false
[[ "${SMOKE:-0}" == 1 ]] && set -- --smoke "$@"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK=true; shift ;;
        --folder) FOLDER="${2:?--folder needs a directory}"; shift 2 ;;
        --stationary-only) ONLY_STATIONARY=true; FOLDER="$(cd "${2:?study folder}" && pwd)"; shift 2 ;;
        --periods) PERIODS="${2:?--periods needs a list, e.g. \"20 200\"}"; shift 2 ;;
        --skeletons) SKELETONS="${2:?--skeletons needs a list, e.g. elastic}"; shift 2 ;;
        --smoke) MODE=--smoke; CYCLES=1; STEPS="--steps-per-cycle 8"; TIME_FE2="02:00:00"; TIME_RESOLVED="02:00:00"; TIME_MARGIN=10; CPUS_FE2=8; MEM_FE2="32G"; shift ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done
[[ -n "$FOLDER" ]] || FOLDER="$REPO/results/cyclic/$( [[ "$MODE" == --smoke ]] && echo smoke_ )$(date +%Y%m%d_%H%M%S)"
seconds() { local h m s; IFS=: read -r h m s <<< "$1"; echo $(( 10#$h * 3600 + 10#$m * 60 + 10#${s:-0} )); }
WALLTIME=$(( $(seconds "$TIME_RESOLVED") - 60 * TIME_MARGIN ))
WALLTIME_FE2=$(( $(seconds "$TIME_FE2") - 60 * TIME_MARGIN ))
COMMON=(--parsable --nodes=1 --ntasks=1)
[[ -n "$PARTITION" ]] && COMMON+=("--partition=$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMON+=("--account=$ACCOUNT")
[[ -n "$EXCLUDE_NODES" ]] && COMMON+=("--exclude=$EXCLUDE_NODES")

echo "Repository: $REPO (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "WARNING: tracked files are modified; the results record a dirty commit." >&2
fi
echo "Folder: $FOLDER"

submit() {  # submit NAME DEPENDENCY sbatch-arguments...; sets LAST
    local name="$1" dependency="$2"; shift 2
    local command=(sbatch "${COMMON[@]}")
    [[ -n "$dependency" ]] && command+=("--dependency=$dependency" --kill-on-invalid-dep=yes)
    command+=("$@")
    if [[ "$CHECK" == true ]]; then echo "  ${command[*]}"; LAST="<$name>"
    else LAST="$("${command[@]}")"; LAST="${LAST%%;*}"; echo "  $name: job $LAST"; fi
}

ALL=()
FOLDERS=()
for P in $PERIODS; do
    STATIONARY="$FOLDER/stationary_P$P"
    [[ "$CHECK" == false ]] && mkdir -p "$STATIONARY/slurm"
    if [[ "$ONLY_STATIONARY" == true && "$CHECK" == false && -e "$STATIONARY/fe2_stationary.npz" ]]; then
        # a rerun: set the previous run aside, or the segments would find it ended and exit at once
        OLD="$STATIONARY/superseded_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$OLD"
        mv "$STATIONARY"/fe2_stationary* "$OLD"/
        echo "  previous stationary run moved to $OLD"
    fi
    STATIONARY_JOB=""
    NEWTON=()
    for PAIR in $STATIONARY_NEWTON; do [[ "${PAIR%%:*}" == "$P" ]] && NEWTON=(--newton-tolerance "${PAIR#*:}"); done
    for (( k = 1; k <= STATIONARY_SEGMENTS; k++ )); do
        submit "stationary-P$P-$k" "${STATIONARY_JOB:+afterany:$STATIONARY_JOB}" --job-name="cyclic-stationary" \
            --cpus-per-task="$CPUS_FE2" --mem="$MEM_FE2" --time="$TIME_FE2" --output="$STATIONARY/slurm/fe2-%j.out" \
            "$HERE/run_fe2_segment.sh" "$REPO" "$STATIONARY/fe2_stationary.npz" "$WALLTIME_FE2" "${MODE#--}" \
            --no-compile-tangent --model stationary_fe --cyclic "$P" --cycles "$CYCLES" $STEPS ${NEWTON[@]+"${NEWTON[@]}"}
        STATIONARY_JOB="$LAST"; ALL+=("$LAST")
    done
    for S in $SKELETONS; do
        RUN="$FOLDER/${S}_P$P"
        FOLDERS+=("$RUN")
        if [[ "$CHECK" == false ]]; then mkdir -p "$RUN/slurm"; echo "$STATIONARY/fe2_stationary.npz" > "$RUN/stationary"; fi
        if [[ "$ONLY_STATIONARY" == true ]]; then  # the history and resolved runs are already queued or done
            submit "compare-$S-P$P" "afterany:$STATIONARY_JOB" --job-name="cyclic-compare" --cpus-per-task=1 \
                --mem=8G --time=01:00:00 --output="$RUN/slurm/compare-%j.out" "$HERE/run_python_job.sh" "$REPO" \
                examples/multiscale/compare_cyclic.py "$RUN"
            ALL+=("$LAST")
            continue
        fi
        # FE² and resolved: chains of segments, each resuming the previous one's checkpoint; the resolved run
        # needs the FE² run completed (its last segment exits 76 while unfinished).
        FE2_JOB=""
        for (( k = 1; k <= FE2_SEGMENTS; k++ )); do
            submit "fe2-$S-P$P-$k" "${FE2_JOB:+afterany:$FE2_JOB}" --job-name="cyclic-fe2" --cpus-per-task="$CPUS_FE2" \
                --mem="$MEM_FE2" --time="$TIME_FE2" --output="$RUN/slurm/fe2-%j.out" "$HERE/run_fe2_segment.sh" "$REPO" \
                "$RUN/fe2.npz" "$WALLTIME_FE2" "${MODE#--}" --no-compile-tangent --skeleton "$S" --cyclic "$P" \
                --cycles "$CYCLES" $STEPS
            FE2_JOB="$LAST"; ALL+=("$LAST")
        done
        RESOLVED_EXTRA=()
        [[ "$MODE" == --smoke ]] && RESOLVED_EXTRA=(--maxh 0.4)
        RESOLVED_JOB=""
        for (( k = 1; k <= RESOLVED_SEGMENTS; k++ )); do
            submit "resolved-$S-P$P-$k" "${RESOLVED_JOB:+afterany:$RESOLVED_JOB,}afterok:$FE2_JOB" \
                --job-name="cyclic-resolved" --cpus-per-task="$CPUS_RESOLVED" --mem="$MEM_RESOLVED" \
                --time="$TIME_RESOLVED" --output="$RUN/slurm/resolved-%j.out" "$HERE/run_resolved_refinement.sh" "$REPO" \
                "$RUN" "$WALLTIME" --skeleton "$S" --cyclic "$P" --time-grid "$RUN/fe2.npz" \
                ${RESOLVED_EXTRA[@]+"${RESOLVED_EXTRA[@]}"}
            RESOLVED_JOB="$LAST"; ALL+=("$LAST")
        done
        submit "compare-$S-P$P" "afterany:$RESOLVED_JOB:$STATIONARY_JOB" --job-name="cyclic-compare" --cpus-per-task=1 \
            --mem=8G --time=01:00:00 --output="$RUN/slurm/compare-%j.out" "$HERE/run_python_job.sh" "$REPO" \
            examples/multiscale/compare_cyclic.py "$RUN"
        ALL+=("$LAST")
    done
done
submit summary "afterany:$(IFS=:; echo "${ALL[*]}")" --job-name="cyclic-summary" --cpus-per-task=1 --mem=8G --time=01:00:00 \
    --output="$FOLDER/summary-%j.out" "$HERE/run_python_job.sh" "$REPO" examples/multiscale/compare_cyclic.py \
    "${FOLDERS[@]}" --output "$FOLDER"
if [[ "$CHECK" == true ]]; then echo "Check only: nothing submitted."
else echo "Follow with: squeue -u \$USER;  tail -f $FOLDER/*/slurm/*.out"; echo "Summary: $FOLDER/cyclic.md"; fi
