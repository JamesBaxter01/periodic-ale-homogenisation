#!/usr/bin/env bash
# Every solver run behind the paper's figures and tables, from scratch, submitted at once. Run with bash on the
# cluster's login node (it submits its own Slurm jobs):
#   bash examples/hpc/submit_paper_runs.sh                   everything; the runs that read a step-load FE² run
#        wait for it in the queue (Slurm dependency afterok). The FE² and resolved columns run as chains of 48 h
#        segments that resume from a checkpoint, so a slow node delays a run instead of ending it; every job
#        keeps off EXCLUDE_NODES (settings below).
#   SMOKE=1 bash examples/hpc/submit_paper_runs.sh [...]     the same in smoke mode: coarse cells, short
#        histories, small jobs; a check of every job, not paper data (folders named smoke_...)
#   --check            show the sbatch commands; submit nothing
#   --only "GROUPS"    only these of: cells verification consolidation cyclic
#   --allow-dirty      submit although tracked code files are modified (refused otherwise: the paper's data
#                      must come from a commit)
#   --wave 2 RERUN     only the runs that read the step-load FE² runs, now, for the FE² runs of an earlier
#        submission (RERUN: the folder it printed, results/paper/rerun_<stamp>); for instance after an FE² run
#        had to be resumed, which cancels the jobs waiting for it
#
# Folders (all carry one date stamp; RERUN/folders.txt lists them, for --wave 2 and for rsync):
#   cells          RERUN/cells/: figure3_biot.npz (Figure 3), figure4_cycles.npz (Figure 4),
#                  figure4_rates.npz (nine-duration sweep), figure4_stationary.npz (Figure 4, Table 3);
#                  the Table 3 sweep itself: examples/microscale/hpc/submit_figure4_criterion.sh
#   verification   results/verification/combined_<stamp> (Tables 1-2, Section 4): verification/combined/hpc/submit.sh
#   consolidation  results/consolidation/full_{elastic,viscoelastic}_step_<stamp> (Figure 6, Table 4) and
#                  full_elastic_<stamp> (10 s ramp, Section 5.3.1): submit_consolidation.sh, FE² then resolved;
#                  after the step-load FE² runs:
#                    Table 5        <viscoelastic step>/table_5/table_5_consolidation_parameters.csv
#                    Figure 7       <viscoelastic step>/cell_replay_bottom.npz
#                    storage moment <elastic step>/cell_replay_top.npz, |K_rate|/|Q0| in its log (Section 5.3.1)
#                    refinement     results/consolidation/refinement_<stamp>: submit_resolved_refinement.sh
#                                   against <elastic step>
#   cyclic         results/cyclic/<stamp> (Figure 8, Table 6): submit_cyclic.sh
set -euo pipefail

# --------------------------- EDIT THESE SETTINGS ---------------------------

CELL_CPUS=4;    CELL_MEM="16G";   CELL_TIME="24:00:00"     # figure3_biot.py, figure4_cycles.py (4 threads, ~3-4 h each on a laptop)
STATIONARY_CPUS=41; STATIONARY_MEM="96G"; STATIONARY_TIME="04:00:00"  # figure4_stationary.py: one cell per compression
TABLE_CPUS=4;   TABLE_MEM="32G";  TABLE_TIME="04:00:00"    # consolidation_parameter_table.py: one cell process per state
REPLAY_CPUS=4;  REPLAY_MEM="16G"; REPLAY_TIME="12:00:00"   # consolidation_cell_replay.py: about an hour per point on a laptop
SMOKE_TIME="01:00:00"          # wall time of the cell, table and replay jobs in smoke mode
PARTITION=""; ACCOUNT=""       # blank: cluster default (the scripts called below have their own settings blocks)
# Nodes every job keeps off (sbatch --exclude; passed on to the scripts called below), e.g. a Calder
# calder-cpuNNN node running jobs far slower than the rest (Aire's node003 ran 8-9x slower on 26 Oct 2026).
EXCLUDE_NODES="${EXCLUDE_NODES-}"

# ------------------------- END OF EDITABLE SETTINGS ------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
MULTI="$REPO/examples/multiscale/hpc"
JOB="$MULTI/run_python_job.sh"
WAVE=all; RERUN=""; CHECK=false; DIRTY=false; ONLY="cells verification consolidation cyclic"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wave) [[ "${2:-}" == 2 ]] || { echo "--wave takes 2 and the RERUN folder" >&2; exit 2; }
                WAVE=2; RERUN="${3:?--wave 2 needs the RERUN folder of the earlier submission}"; shift 3 ;;
        --check) CHECK=true; shift ;;
        --only) ONLY="${2:?--only needs a list of groups}"; shift 2 ;;
        --allow-dirty) DIRTY=true; shift ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done
if [[ -n "${SLURM_JOB_ID:-}" ]]; then echo "Run from a login shell; this script submits its own jobs." >&2; exit 2; fi
SMOKE="${SMOKE:-0}"
MODE=full; PREFIX=""
if [[ "$SMOKE" == 1 ]]; then
    MODE=smoke; PREFIX="smoke_"; CELL_TIME="$SMOKE_TIME"; TABLE_TIME="$SMOKE_TIME"; REPLAY_TIME="$SMOKE_TIME"
    CELL_MEM="8G"; STATIONARY_CPUS=3; STATIONARY_MEM="8G"; STATIONARY_TIME="$SMOKE_TIME"
fi
export SMOKE EXCLUDE_NODES   # the scripts called below read SMOKE=1 as --smoke, and EXCLUDE_NODES

COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Repository: $REPO (commit $COMMIT), $MODE$( [[ "$WAVE" == 2 ]] && echo ', runs after the step-load FE² only')"
[[ -n "$EXCLUDE_NODES" ]] && echo "Keeping off nodes: $EXCLUDE_NODES"
MODIFIED="$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null | awk '{print $2}' | grep -v '^results/' || true)"
if [[ -n "$MODIFIED" ]]; then
    echo "Tracked files modified outside results/:" >&2; echo "$MODIFIED" | sed 's/^/  /' >&2
    if [[ "$DIRTY" != true && "$CHECK" != true ]]; then
        echo "Refused: the runs would not come from commit $COMMIT. Commit or restore them, or pass --allow-dirty." >&2
        exit 2
    fi
fi
CHECK_OPTION=()
[[ "$CHECK" == true ]] && CHECK_OPTION=(--check)

submit() {  # submit NAME DEPENDENCY CPUS MEM TIME LOG SCRIPT [options]: one Python script as a Slurm job
    local name="$1" dependency="$2" cpus="$3" mem="$4" time="$5" log="$6"
    shift 6
    local command=(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem" --time="$time"
                   --job-name="$name" --output="$log")
    [[ -n "$dependency" ]] && command+=("--dependency=afterok:$dependency" --kill-on-invalid-dep=yes)
    [[ -n "$PARTITION" ]] && command+=("--partition=$PARTITION")
    [[ -n "$ACCOUNT" ]] && command+=("--account=$ACCOUNT")
    [[ -n "$EXCLUDE_NODES" ]] && command+=("--exclude=$EXCLUDE_NODES")
    command+=("$JOB" "$REPO" "$@")
    if [[ "$CHECK" == true ]]; then echo "  ${command[*]}"; return; fi
    mkdir -p "$(dirname "$log")"
    local id; id="$("${command[@]}")"
    echo "  $name: job ${id%%;*}"
}

consolidation() {  # consolidation [submit_consolidation.sh options]: prints its output; sets FE2_JOB and RESOLVED_JOB
    local out
    out="$(bash "$MULTI/submit_consolidation.sh" "--$MODE" "$@" ${CHECK_OPTION[@]+"${CHECK_OPTION[@]}"})"
    echo "$out"
    FE2_JOB="$(sed -n 's/^  fe2: job \([0-9]*\).*/\1/p' <<< "$out")"
    RESOLVED_JOB="$(sed -n 's/^  resolved: job \([0-9]*\).*/\1/p' <<< "$out")"
}

after_step_fe2() {  # after_step_fe2 EL_FE2_JOB VE_FE2_JOB EL_RESOLVED_JOB: the runs that read the step-load FE² runs
    local el_fe2="$1" ve_fe2="$2" el_resolved="$3"
    echo; echo "=== after the step-load FE² runs: Table 5, Figure 7, storage moment, refined resolved reference"
    submit paper-table5 "$ve_fe2" "$TABLE_CPUS" "$TABLE_MEM" "$TABLE_TIME" "$VE/slurm/table5-%j.out" \
        examples/multiscale/consolidation_parameter_table.py "$VE/fe2.npz" --output "$VE/table_5"
    submit paper-figure7 "$ve_fe2" "$REPLAY_CPUS" "$REPLAY_MEM" "$REPLAY_TIME" "$VE/slurm/replay-bottom-%j.out" \
        examples/multiscale/consolidation_cell_replay.py bottom --run "$VE" --threads "$REPLAY_CPUS"
    submit paper-storage "$el_fe2" "$REPLAY_CPUS" "$REPLAY_MEM" "$REPLAY_TIME" "$EL/slurm/replay-top-%j.out" \
        examples/multiscale/consolidation_cell_replay.py top --storage --run "$EL" --threads "$REPLAY_CPUS"
    local options=(--base "$EL" --folder "$REFINEMENT")
    [[ -n "$el_fe2" ]] && options+=(--after "$el_fe2")
    [[ -n "$el_resolved" ]] && options+=(--analyse-after "$el_resolved")
    bash "$MULTI/submit_resolved_refinement.sh" "${options[@]}" ${CHECK_OPTION[@]+"${CHECK_OPTION[@]}"}
}

if [[ "$WAVE" == 2 ]]; then
    [[ "$RERUN" == /* ]] || RERUN="$PWD/$RERUN"
    [[ -f "$RERUN/folders.txt" ]] || { echo "$RERUN/folders.txt not found (the folder the submission printed)" >&2; exit 2; }
    folder() { awk -F= -v k="$1" '$1 == k {print $2}' "$RERUN/folders.txt"; }
    EL="$(folder elastic_step)"; VE="$(folder viscoelastic_step)"
    if [[ "$(folder mode)" != "$MODE" ]]; then echo "$RERUN is a $(folder mode) run; set SMOKE to match" >&2; exit 2; fi
    STATUS="$(cd "$REPO"; { module load calder/cpu miniforge3 && eval "$(conda shell.bash hook)" && conda activate ngsolve-petsc; } >/dev/null 2>&1 || true
              python - "$EL/fe2.npz" "$VE/fe2.npz" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
for path in map(Path, sys.argv[1:]):
    status = "missing"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            status = json.loads(str(data["metadata"])).get("status", "unknown")
    print(f"{path.parent.name}:{status}")
PY
)"
    echo "Step-load FE² runs: $(echo $STATUS)"
    for ENTRY in $STATUS; do
        case "${ENTRY#*:}" in
            stopped|t_max) ;;
            *) [[ "$CHECK" == true ]] || { echo "Wait until both FE² runs have ended (status stopped)." >&2; exit 2; } ;;
        esac
    done
    REFINEMENT="$REPO/results/consolidation/refinement_${PREFIX}$(date +%Y%m%d_%H%M%S)"
    after_step_fe2 "" "" ""
    [[ "$CHECK" == false ]] && echo "refinement=$REFINEMENT" >> "$RERUN/folders.txt"
    exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RERUN="$REPO/results/paper/rerun_${PREFIX}$STAMP"
EL="$REPO/results/consolidation/${MODE}_elastic_step_$STAMP"
VE="$REPO/results/consolidation/${MODE}_viscoelastic_step_$STAMP"
RAMP="$REPO/results/consolidation/${MODE}_elastic_$STAMP"
REFINEMENT="$REPO/results/consolidation/refinement_${PREFIX}$STAMP"
CYCLIC="$REPO/results/cyclic/${PREFIX}$STAMP"
CAMPAIGN="$REPO/results/verification/combined_$STAMP"
[[ "$SMOKE" == 1 ]] && CAMPAIGN="$REPO/results/verification/combined_smoke"
if [[ "$CHECK" == false ]]; then
    mkdir -p "$RERUN"
    { echo "commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"; echo "mode=$MODE"
      for GROUP in $ONLY; do  # only the folders of the groups submitted
          case "$GROUP" in
              cells) echo "cells=$RERUN/cells" ;;
              verification) echo "verification=$CAMPAIGN" ;;
              consolidation) echo "elastic_step=$EL"; echo "viscoelastic_step=$VE"; echo "elastic_ramp=$RAMP"
                             echo "refinement=$REFINEMENT" ;;
              cyclic) echo "cyclic=$CYCLIC" ;;
          esac
      done; } > "$RERUN/folders.txt"
fi
echo "Folder list: $RERUN/folders.txt"
for GROUP in $ONLY; do
    echo; echo "=== $GROUP"
    case "$GROUP" in
        cells)
            CELLS="$RERUN/cells"; EXTRA=(); [[ "$SMOKE" == 1 ]] && EXTRA=(--smoke)
            submit paper-figure3 "" "$CELL_CPUS" "$CELL_MEM" "$CELL_TIME" "$CELLS/slurm/figure3-%j.out" \
                examples/microscale/figure3_biot.py --threads "$CELL_CPUS" --output "$CELLS" ${EXTRA[@]+"${EXTRA[@]}"}
            submit paper-figure4 "" "$CELL_CPUS" "$CELL_MEM" "$CELL_TIME" "$CELLS/slurm/figure4-%j.out" \
                examples/microscale/figure4_cycles.py --threads "$CELL_CPUS" --output "$CELLS" ${EXTRA[@]+"${EXTRA[@]}"}
            submit paper-rates "" "$CELL_CPUS" "$CELL_MEM" "$CELL_TIME" "$CELLS/slurm/figure4-rates-%j.out" \
                examples/microscale/figure4_cycles.py --rates --threads "$CELL_CPUS" --output "$CELLS" \
                ${EXTRA[@]+"${EXTRA[@]}"}
            submit paper-stationary "" "$STATIONARY_CPUS" "$STATIONARY_MEM" "$STATIONARY_TIME" \
                "$CELLS/slurm/figure4-stationary-%j.out" examples/microscale/figure4_stationary.py \
                --threads "$STATIONARY_CPUS" --output "$CELLS" ${EXTRA[@]+"${EXTRA[@]}"} ;;
        verification)
            bash "$REPO/verification/combined/hpc/submit.sh" --campaign "$CAMPAIGN" ${CHECK_OPTION[@]+"${CHECK_OPTION[@]}"} ;;
        consolidation)
            consolidation --skeleton elastic --step --folder "$EL"
            EL_FE2="$FE2_JOB"; EL_RESOLVED="$RESOLVED_JOB"
            consolidation --skeleton viscoelastic --step --folder "$VE"
            VE_FE2="$FE2_JOB"
            consolidation --skeleton elastic --folder "$RAMP"
            if [[ "$CHECK" == true ]]; then
                after_step_fe2 "<fe2-elastic-step>" "<fe2-viscoelastic-step>" "<resolved-elastic-step>"
            elif [[ -n "$EL_FE2" && -n "$VE_FE2" && -n "$EL_RESOLVED" ]]; then
                after_step_fe2 "$EL_FE2" "$VE_FE2" "$EL_RESOLVED"
            else
                echo "WARNING: could not read the step-load FE² job ids; once both FE² runs have ended, submit the" \
                     "runs that read them with: bash examples/hpc/submit_paper_runs.sh --wave 2 ${RERUN#$REPO/}" >&2
            fi ;;
        cyclic)
            bash "$MULTI/submit_cyclic.sh" --folder "$CYCLIC" ${CHECK_OPTION[@]+"${CHECK_OPTION[@]}"} ;;
        *) echo "Unknown group $GROUP (cells, verification, consolidation, cyclic)" >&2; exit 2 ;;
    esac
done
echo
if [[ "$CHECK" == true ]]; then echo "Check only: nothing submitted."
else echo "Everything is queued (squeue -u \$USER). Folders: $RERUN/folders.txt"; fi
