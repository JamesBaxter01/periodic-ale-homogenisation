#!/usr/bin/env bash
# Submit the FE² / resolved consolidation comparison to Slurm. Run with bash, not sbatch:
#   bash examples/multiscale/hpc/submit_consolidation.sh            submit (MODE and SKELETON below)
#   bash examples/multiscale/hpc/submit_consolidation.sh --full --skeleton viscoelastic
#        the same, choosing the mode (--smoke / --full) and skeleton without editing this file
#   bash examples/multiscale/hpc/submit_consolidation.sh --full --skeleton viscoelastic --step
#        step traction instead of the ramp, on a log-spaced time grid from 1e-6 s (LOAD below)
#   bash examples/multiscale/hpc/submit_consolidation.sh --check    show the sbatch commands; submit nothing
#   --folder DIR   the run folder (default below); SMOKE=1 in the environment is the same as --smoke
#   bash examples/multiscale/hpc/submit_consolidation.sh --resume results/consolidation/<folder>
#        resubmit the resolved run of an existing folder from its checkpoint (and the comparison).
# Three runs: the FE² run; the resolved run, which waits for FE² and solves exactly its accepted
# times (RESOLVED_ON_FE2_GRID); the comparison when both have ended. FE² and resolved are each a chain of
# segments (FE2_SEGMENTS, RESOLVED_SEGMENTS): a segment checkpoints before its wall time and the next one
# resumes it, so a slow node delays a run instead of ending it. Physics and numerics are the
# settings blocks of examples/multiscale/consolidation_multiscale.py and consolidation_resolved.py; MODE only
# chooses their full or smoke settings.
set -euo pipefail

# --------------------------- EDIT THESE SETTINGS ---------------------------

MODE="smoke"                   # smoke (coarse, minutes) or full (production meshes, hours); or --smoke / --full
SKELETON="elastic"             # elastic or viscoelastic (the examples' SKELETONS); or --skeleton NAME
LOAD="ramp"                    # ramp (the examples' RAMP_TIME) or step (their STEP settings); or --step
FOLDER=""                      # Blank: results/consolidation/<mode>_<skeleton>[_step]_<date>_<time> in the repository.
RESOLVED_ON_FE2_GRID=true      # true: resolved waits for FE² and solves its times; false: both run at once.

# FE²: one cell process per CPU (40 cells, so 40 CPUs solve each batch of cells at once).
CPUS_FE2=40;      MEM_FE2="128G";      TIME_FE2="48:00:00";      FE2_SEGMENTS=2
# Compiled cell tangents (faster per cell) make every worker run the C++ compiler at once, a few GB
# each: 8 workers exceeded 32 GB on Aire. false assembles the tangent interpreted (same results to
# round-off, slower per call). true needs NGSolve's ngscxx on the nodes and about 5 GB per worker.
COMPILE_FE2=false
# Resolved: one process with CPUS_RESOLVED NGSolve and MUMPS threads (about 5 GB at maxh 0.09). More threads
# are slower here: 70-80 s per step on 8 CPUs against 320 s on 32 (Aire, 236k unknowns, shared node).
CPUS_RESOLVED=8;  MEM_RESOLVED="32G";  TIME_RESOLVED="48:00:00"; RESOLVED_SEGMENTS=2
MEM_COMPARE="8G"; TIME_COMPARE="01:00:00"
SMOKE_TIME="02:00:00"          # Wall time of every job in smoke mode.
SMOKE_MAXH=0.4                 # Resolved mesh in smoke mode (SMOKE in consolidation_resolved.py).
TIME_MARGIN=40                 # Minutes before its limit at which a segment stops at a checkpoint (a step on a
                               # busy node can take many minutes).
# Nodes to keep off (sbatch --exclude), e.g. "calder-cpu012" (Aire's node003 ran every job 8-9x slower on 26 Oct 2026).
EXCLUDE_NODES="${EXCLUDE_NODES-}"
PARTITION=""                   # Blank: cluster default.
ACCOUNT=""                     # Blank: cluster default.

# ------------------------- END OF EDITABLE SETTINGS ------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
JOB="$HERE/run_consolidation.sh"
CHECK=false
RESUME=""
[[ "${SMOKE:-0}" == 1 ]] && set -- --smoke "$@"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK=true; shift ;;
        --folder) FOLDER="${2:?--folder needs a directory}"; shift 2 ;;
        --smoke|--full) MODE="${1#--}"; shift ;;
        --skeleton) SKELETON="${2:?--skeleton needs elastic or viscoelastic}"; shift 2 ;;
        --step) LOAD="step"; shift ;;
        --resume) RESUME="${2:?--resume needs the run folder}"; shift 2 ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run 'bash submit_consolidation.sh' from a login shell; it submits its own Slurm jobs." >&2
    exit 2
fi
if [[ "$MODE" != smoke && "$MODE" != full ]]; then echo "MODE must be smoke or full" >&2; exit 2; fi
if [[ "$SKELETON" != elastic && "$SKELETON" != viscoelastic ]]; then echo "SKELETON must be elastic or viscoelastic" >&2; exit 2; fi
if [[ "$LOAD" != ramp && "$LOAD" != step ]]; then echo "LOAD must be ramp or step" >&2; exit 2; fi
if [[ -n "$RESUME" ]]; then
    FOLDER="$RESUME"
    MODE="$(cat "$FOLDER/mode")"
    SKELETON="$(cat "$FOLDER/skeleton" 2>/dev/null || echo elastic)"
    LOAD="$(cat "$FOLDER/load" 2>/dev/null || echo ramp)"
elif [[ -z "$FOLDER" ]]; then
    SUFFIX=""
    if [[ "$LOAD" == step ]]; then SUFFIX="_step"; fi
    FOLDER="$REPO/results/consolidation/${MODE}_${SKELETON}${SUFFIX}_$(date +%Y%m%d_%H%M%S)"
fi
LOAD_OPTION=()
if [[ "$LOAD" == step ]]; then LOAD_OPTION=(--step); fi
[[ "$FOLDER" == /* ]] || FOLDER="$PWD/$FOLDER"
# After --resume has read the folder's mode: a resumed full run keeps the full resources and wall time.
if [[ "$MODE" == smoke ]]; then TIME_FE2="$SMOKE_TIME"; TIME_RESOLVED="$SMOKE_TIME"; MEM_FE2="32G"; MEM_RESOLVED="16G"; CPUS_FE2=8; CPUS_RESOLVED=8; fi

seconds() {  # Slurm time [D-]HH:MM:SS -> seconds
    local t="$1" days=0
    if [[ "$t" == *-* ]]; then days="${t%%-*}"; t="${t#*-}"; fi
    IFS=: read -r h m s <<< "$t"
    echo $(( days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#${s:-0} ))
}
WALLTIME=$(( $(seconds "$TIME_RESOLVED") - 60 * TIME_MARGIN ))
WALLTIME_FE2=$(( $(seconds "$TIME_FE2") - 60 * TIME_MARGIN ))

COMMON=(--parsable --nodes=1 --ntasks=1)
if [[ -n "$EXCLUDE_NODES" ]]; then COMMON+=("--exclude=$EXCLUDE_NODES"); fi
if [[ -n "$PARTITION" ]]; then COMMON+=("--partition=$PARTITION"); fi
if [[ -n "$ACCOUNT" ]]; then COMMON+=("--account=$ACCOUNT"); fi

echo "Repository: $REPO (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "WARNING: tracked files are modified; the results record a dirty commit." >&2
fi
echo "Run folder: $FOLDER ($MODE, $SKELETON, $LOAD load)"
if [[ "$CHECK" == false ]]; then
    mkdir -p "$FOLDER/slurm"
    echo "$MODE" > "$FOLDER/mode"
    echo "$SKELETON" > "$FOLDER/skeleton"
    echo "$LOAD" > "$FOLDER/load"
fi

submit() {  # submit NAME DEPENDENCY sbatch-options...; sets JOB_<NAME> and LAST
    local name="$1" dependency="$2" id
    shift 2
    local command=(sbatch "${COMMON[@]}")
    # Cancel rather than leave pending (DependencyNeverSatisfied) when an upstream job fails.
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
    printf -v "JOB_$name" '%s' "$id"
    LAST="$id"
}

echo "Submitting:"
if [[ -n "$RESUME" ]]; then
    RESOLVED_OPTIONS=(--walltime "$WALLTIME" --skeleton "$SKELETON" ${LOAD_OPTION[@]+"${LOAD_OPTION[@]}"} --resume)
    if [[ -f "$FOLDER/fe2.npz" && "$RESOLVED_ON_FE2_GRID" == true ]]; then RESOLVED_OPTIONS+=(--time-grid "$FOLDER/fe2.npz"); fi
    submit resolved "" --job-name="resolved-$MODE" --cpus-per-task="$CPUS_RESOLVED" \
        --mem="$MEM_RESOLVED" --time="$TIME_RESOLVED" --output="$FOLDER/slurm/resolved-%j.out" \
        "$JOB" "$REPO" "$FOLDER" resolved "$MODE" "${RESOLVED_OPTIONS[@]}"
    AFTER="afterany:$JOB_resolved"
else
    COMPILE_OPTION=--no-compile-tangent
    if [[ "$COMPILE_FE2" == true ]]; then COMPILE_OPTION=--compile-tangent; fi
    # FE²: FE2_SEGMENTS jobs in a row (afterany), each starting, resuming or (once the run has ended) exiting.
    FE2_LAST=""
    for (( k = 1; k <= FE2_SEGMENTS; k++ )); do
        submit "fe2_$k" "${FE2_LAST:+afterany:$FE2_LAST}" --job-name="fe2-$MODE" --cpus-per-task="$CPUS_FE2" \
            --mem="$MEM_FE2" --time="$TIME_FE2" --output="$FOLDER/slurm/fe2-%j.out" "$HERE/run_fe2_segment.sh" "$REPO" \
            "$FOLDER/fe2.npz" "$WALLTIME_FE2" "$MODE" "$COMPILE_OPTION" --skeleton "$SKELETON" \
            ${LOAD_OPTION[@]+"${LOAD_OPTION[@]}"}
        FE2_LAST="$LAST"
    done
    # Resolved: RESOLVED_SEGMENTS jobs in a row; on the FE² grid each also needs the FE² run to have completed
    # (the last FE² segment exits 76 if the run is unfinished), and all are cancelled if it did not.
    GRID=(); FIRST=""; ALSO=""
    if [[ "$RESOLVED_ON_FE2_GRID" == true ]]; then
        GRID=(--time-grid "$FOLDER/fe2.npz"); FIRST="afterok:$FE2_LAST"; ALSO=",afterok:$FE2_LAST"
    fi
    COARSE=()
    if [[ "$MODE" == smoke ]]; then COARSE=(--maxh "$SMOKE_MAXH"); fi
    RESOLVED_LAST=""
    for (( k = 1; k <= RESOLVED_SEGMENTS; k++ )); do
        DEPENDENCY="$FIRST"
        if [[ -n "$RESOLVED_LAST" ]]; then DEPENDENCY="afterany:$RESOLVED_LAST$ALSO"; fi
        submit "resolved_$k" "$DEPENDENCY" --job-name="resolved-$MODE" --cpus-per-task="$CPUS_RESOLVED" \
            --mem="$MEM_RESOLVED" --time="$TIME_RESOLVED" --output="$FOLDER/slurm/resolved-%j.out" \
            "$HERE/run_resolved_refinement.sh" "$REPO" "$FOLDER" "$WALLTIME" --skeleton "$SKELETON" \
            ${LOAD_OPTION[@]+"${LOAD_OPTION[@]}"} ${GRID[@]+"${GRID[@]}"} ${COARSE[@]+"${COARSE[@]}"}
        RESOLVED_LAST="$LAST"
    done
    AFTER="afterany:$RESOLVED_LAST"
    if [[ "$RESOLVED_ON_FE2_GRID" != true ]]; then AFTER="$AFTER:$FE2_LAST"; fi
fi
submit compare "$AFTER" --job-name="compare-$MODE" --cpus-per-task=1 --mem="$MEM_COMPARE" --time="$TIME_COMPARE" \
    --output="$FOLDER/slurm/compare-%j.out" "$JOB" "$REPO" "$FOLDER" compare "$MODE"
if [[ -z "$RESUME" && "$CHECK" == false ]]; then  # the jobs other runs wait for (read by submit_paper_runs.sh)
    echo "  fe2: job $FE2_LAST (last of $FE2_SEGMENTS segments)"
    echo "  resolved: job $RESOLVED_LAST (last of $RESOLVED_SEGMENTS segments)"
fi

if [[ "$CHECK" == true ]]; then
    echo "Check only: nothing submitted."
else
    echo "Follow with: squeue -u \$USER;  tail -f $FOLDER/slurm/*.out"
    echo "If the resolved run stopped at its time limit (status 'walltime'), run: bash $0 --resume $FOLDER"
fi
