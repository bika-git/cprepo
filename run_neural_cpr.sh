#!/usr/bin/env bash
# run_neural_cpr.sh - apply trained Neural CPR models to any Flow deck.
#
# Usage:
#   ./run_neural_cpr.sh --amg-model cpr_amg_policy.model --deck CASE.DATA
#   ./run_neural_cpr.sh --amg-model amg.model --abort-model abort.model --deck CASE.DATA \
#       --np 32 --threads 2 --mpi-args "--bind-to none"
#   ./run_neural_cpr.sh cpr_policy.model CASE.DATA --np 8        # 24-variant config policy
#
# Policies (Flow from opm-simulators branch neural-cpr-control):
#   --model        CPR variant policy:   OPM_NEURAL_CPR_WEIGHTS=model
#   --amg-model    neural AMG policy:    OPM_NEURAL_CPR_WEIGHTS=model OPM_NEURAL_CPR_ACTIONS=amg
#                                        OPM_NEURAL_CPR_MIN_GAIN=0.05
#   --abort-model  neural abort policy:  OPM_CPR_ABORT_MODEL=model OPM_CPR_ABORT_THRESHOLD=0.95
# --model and --amg-model both use OPM_NEURAL_CPR_WEIGHTS, so a run takes one
# of them; the abort policy combines with either (or with plain Flow).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FLOW=${FLOW:-$SCRIPT_DIR/build/opm-simulators/bin/flow}
MODEL=${MODEL:-}
AMG_MODEL=${AMG_MODEL:-}
ABORT_MODEL=${ABORT_MODEL:-}
DECK=${DECK:-}
OUTDIR=${OUTDIR:-}
NPROCS=${NPROCS:-1}
CONFIDENCE=${OPM_NEURAL_CPR_CONFIDENCE:-0.6}
FORBID_LABELS=${OPM_NEURAL_CPR_FORBID_LABELS:-}
MIN_GAIN=${OPM_NEURAL_CPR_MIN_GAIN:-}
ABORT_THRESHOLD=${OPM_CPR_ABORT_THRESHOLD:-0.95}
REUSE=0
REUSE_SETUP=${CPR_REUSE_SETUP:-4}
REUSE_INTERVAL=${CPR_REUSE_INTERVAL:-30}
MPI_ARGS=${MPI_ARGS:-}
THREADS=${THREADS:-}
FLOW_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  run_neural_cpr.sh [--model M | --amg-model M] [--abort-model M] --deck CASE.DATA [options] [-- FLOW_OPTIONS...]
  run_neural_cpr.sh MODEL.model CASE.DATA [options] [-- FLOW_OPTIONS...]

Models (at least one):
  --model PATH          CPR variant policy (24 configs; classifier or cost model).
  --amg-model PATH      Neural AMG policy (pressure-AMG prolongation damping and
                        coarsening target). Not together with --model.
  --abort-model PATH    Neural abort policy (gives up on doomed timestep attempts).

Options:
  --deck PATH           Flow deck to simulate.
  --flow PATH           Flow binary. Default: ./build/opm-simulators/bin/flow.
  --out DIR             Output directory. Default: ./tmp/neural_cpr_runs/<deck>_<timestamp>.
  --np N                MPI ranks. Default: 1.
                        On a 64-core node: --np 32 --threads 2 --mpi-args "--bind-to none".
  --threads N           Threads per rank (--threads-per-process=N, OMP_NUM_THREADS=N).
                        Default: Flow's.
  --min-gain G          Anchored decisions: leave the default only for a predicted
                        log-cost gain above G. Default: 0.05 with --amg-model; unset
                        (classifier confidence gate) with --model.
  --abort-threshold P   Abort above this probability of a chop. Default: 0.95.
  --confidence X        Confidence gate for classifier models. Default: 0.6.
  --forbid-labels LIST  Comma-separated CPR variant labels to forbid.
  --spe11-safe          Shortcut for --forbid-labels 16,...,23 (trueimpesanalytic).
                        Flow from neural-cpr-control applies this automatically on
                        decks without oil; kept for older builds.
  --reuse               Add explicit CPR reuse flags (--reuse-setup/--reuse-interval).
                        Default: none, i.e. Flow's defaults (setup 4, interval 30).
  --no-reuse            Do not add CPR reuse flags (default).
  --reuse-setup N       Default: 4.
  --reuse-interval N    Default: 30.
  --mpi-args "ARGS"     Extra arguments placed after mpirun, before -np.
  -h, --help            Show this help.

Environment overrides:
  FLOW, MODEL, AMG_MODEL, ABORT_MODEL, DECK, OUTDIR, NPROCS,
  OPM_NEURAL_CPR_CONFIDENCE, OPM_NEURAL_CPR_FORBID_LABELS, OPM_NEURAL_CPR_MIN_GAIN,
  OPM_CPR_ABORT_THRESHOLD, CPR_REUSE_SETUP, CPR_REUSE_INTERVAL, MPI_ARGS, THREADS.

Any arguments after "--" are passed directly to flow.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || die "--model requires a path"
            MODEL=$2
            shift 2
            ;;
        --amg-model)
            [[ $# -ge 2 ]] || die "--amg-model requires a path"
            AMG_MODEL=$2
            shift 2
            ;;
        --abort-model)
            [[ $# -ge 2 ]] || die "--abort-model requires a path"
            ABORT_MODEL=$2
            shift 2
            ;;
        --deck)
            [[ $# -ge 2 ]] || die "--deck requires a path"
            DECK=$2
            shift 2
            ;;
        --flow)
            [[ $# -ge 2 ]] || die "--flow requires a path"
            FLOW=$2
            shift 2
            ;;
        --out)
            [[ $# -ge 2 ]] || die "--out requires a directory"
            OUTDIR=$2
            shift 2
            ;;
        --np)
            [[ $# -ge 2 ]] || die "--np requires an integer"
            NPROCS=$2
            shift 2
            ;;
        --threads)
            [[ $# -ge 2 ]] || die "--threads requires an integer"
            THREADS=$2
            shift 2
            ;;
        --min-gain)
            [[ $# -ge 2 ]] || die "--min-gain requires a number"
            MIN_GAIN=$2
            shift 2
            ;;
        --abort-threshold)
            [[ $# -ge 2 ]] || die "--abort-threshold requires a number"
            ABORT_THRESHOLD=$2
            shift 2
            ;;
        --confidence)
            [[ $# -ge 2 ]] || die "--confidence requires a number"
            CONFIDENCE=$2
            shift 2
            ;;
        --forbid-labels)
            [[ $# -ge 2 ]] || die "--forbid-labels requires a comma-separated list"
            FORBID_LABELS=$2
            shift 2
            ;;
        --spe11-safe)
            FORBID_LABELS=16,17,18,19,20,21,22,23
            shift
            ;;
        --reuse)
            REUSE=1
            shift
            ;;
        --no-reuse)
            REUSE=0
            shift
            ;;
        --reuse-setup)
            [[ $# -ge 2 ]] || die "--reuse-setup requires an integer"
            REUSE_SETUP=$2
            REUSE=1
            shift 2
            ;;
        --reuse-interval)
            [[ $# -ge 2 ]] || die "--reuse-interval requires an integer"
            REUSE_INTERVAL=$2
            REUSE=1
            shift 2
            ;;
        --mpi-args)
            [[ $# -ge 2 ]] || die "--mpi-args requires a quoted argument string"
            MPI_ARGS=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            FLOW_ARGS+=("$@")
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            if [[ -z "$MODEL" && -z "$AMG_MODEL" && -z "$ABORT_MODEL" ]]; then
                MODEL=$1
            elif [[ -z "$DECK" ]]; then
                DECK=$1
            else
                FLOW_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

[[ -n "$MODEL$AMG_MODEL$ABORT_MODEL" ]] \
    || die "missing model; pass --model, --amg-model and/or --abort-model"
[[ -z "$MODEL" || -z "$AMG_MODEL" ]] \
    || die "--model and --amg-model both use OPM_NEURAL_CPR_WEIGHTS; pass only one"
for m in "$MODEL" "$AMG_MODEL" "$ABORT_MODEL"; do
    [[ -z "$m" || -f "$m" ]] || die "model not found: $m"
done
[[ -n "$DECK" ]] || die "missing deck path; pass --deck CASE.DATA"
[[ -f "$DECK" ]] || die "deck not found: $DECK"
[[ -x "$FLOW" ]] || die "flow binary not executable: $FLOW"
[[ "$NPROCS" =~ ^[0-9]+$ ]] || die "--np must be an integer"
(( NPROCS >= 1 )) || die "--np must be >= 1"
[[ -z "$THREADS" || "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer"

abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
[[ -n "$MODEL" ]] && MODEL=$(abspath "$MODEL")
[[ -n "$AMG_MODEL" ]] && AMG_MODEL=$(abspath "$AMG_MODEL")
[[ -n "$ABORT_MODEL" ]] && ABORT_MODEL=$(abspath "$ABORT_MODEL")

if [[ -z "$OUTDIR" ]]; then
    deck_base=$(basename "$DECK")
    deck_base=${deck_base%.*}
    stamp=$(date +%Y%m%d_%H%M%S)
    OUTDIR=$SCRIPT_DIR/tmp/neural_cpr_runs/${deck_base}_${stamp}
fi
mkdir -p "$OUTDIR"

RUN_CMD=()
if (( NPROCS > 1 )); then
    RUN_CMD=(mpirun)
    if [[ -n "$MPI_ARGS" ]]; then
        # Intentional shell splitting for user-provided MPI flags.
        # shellcheck disable=SC2206
        extra_mpi_args=($MPI_ARGS)
        RUN_CMD+=("${extra_mpi_args[@]}")
    fi
    RUN_CMD+=(-np "$NPROCS" "$FLOW")
else
    RUN_CMD=("$FLOW")
fi

if (( REUSE == 1 )); then
    FLOW_ARGS+=(--cpr-reuse-setup="$REUSE_SETUP")
    FLOW_ARGS+=(--cpr-reuse-interval="$REUSE_INTERVAL")
fi
if [[ -n "$THREADS" ]]; then
    FLOW_ARGS=(--threads-per-process="$THREADS" "${FLOW_ARGS[@]}")
    export OMP_NUM_THREADS="$THREADS"
fi

# Start from a clean policy environment, then set what was asked for.
unset OPM_NEURAL_CPR_WEIGHTS OPM_NEURAL_CPR_ACTIONS OPM_NEURAL_CPR_MIN_GAIN \
      OPM_NEURAL_CPR_CONFIDENCE OPM_NEURAL_CPR_FORBID_LABELS OPM_NEURAL_CPR_DISABLE \
      OPM_CPR_ABORT_MODEL OPM_CPR_ABORT_THRESHOLD OPM_CPR_SWEEP_LOG
if [[ -n "$AMG_MODEL" ]]; then
    export OPM_NEURAL_CPR_WEIGHTS="$AMG_MODEL"
    export OPM_NEURAL_CPR_ACTIONS=amg
    export OPM_NEURAL_CPR_MIN_GAIN="${MIN_GAIN:-0.05}"
elif [[ -n "$MODEL" ]]; then
    export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
    export OPM_NEURAL_CPR_CONFIDENCE="$CONFIDENCE"
    [[ -n "$MIN_GAIN" ]] && export OPM_NEURAL_CPR_MIN_GAIN="$MIN_GAIN"
else
    export OPM_NEURAL_CPR_DISABLE=1      # abort policy only, on Flow's default solver
fi
[[ -n "$FORBID_LABELS" ]] && export OPM_NEURAL_CPR_FORBID_LABELS="$FORBID_LABELS"
if [[ -n "$ABORT_MODEL" ]]; then
    export OPM_CPR_ABORT_MODEL="$ABORT_MODEL"
    export OPM_CPR_ABORT_THRESHOLD="$ABORT_THRESHOLD"
fi

echo "Neural CPR run"
echo "  flow        : $FLOW"
[[ -n "$MODEL" ]]       && echo "  model       : $MODEL (CPR variant policy)"
[[ -n "$AMG_MODEL" ]]   && echo "  amg model   : $AMG_MODEL (min gain ${OPM_NEURAL_CPR_MIN_GAIN})"
[[ -n "$ABORT_MODEL" ]] && echo "  abort model : $ABORT_MODEL (threshold $ABORT_THRESHOLD)"
echo "  deck        : $DECK"
echo "  out         : $OUTDIR"
echo "  np          : $NPROCS${THREADS:+ x $THREADS threads}"
[[ -n "$MODEL" && -z "$MIN_GAIN" ]] && echo "  confidence  : $CONFIDENCE"
[[ -n "$FORBID_LABELS" ]] && echo "  forbid      : $FORBID_LABELS"
if (( REUSE == 1 )); then
    echo "  reuse       : setup=$REUSE_SETUP interval=$REUSE_INTERVAL"
else
    echo "  reuse       : Flow defaults"
fi

set +e
"${RUN_CMD[@]}" --output-dir="$OUTDIR" "${FLOW_ARGS[@]}" "$DECK" >"$OUTDIR/flow.log" 2>&1
status=$?
set -e
echo "$status" >"$OUTDIR/exit_code"

if [[ "$status" -eq 0 ]]; then
    echo "Done. Log: $OUTDIR/flow.log"
else
    echo "Flow failed with exit code $status. Log: $OUTDIR/flow.log" >&2
fi
exit "$status"
