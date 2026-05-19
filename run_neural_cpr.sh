#!/usr/bin/env bash
# run_neural_cpr.sh - apply a trained Neural CPR .model to any Flow deck.
#
# Usage:
#   ./run_neural_cpr.sh --model cpr_policy.model --deck CASE.DATA
#   ./run_neural_cpr.sh cpr_policy.model CASE.DATA --np 8 --reuse
#
# The only required runtime contract is:
#   OPM_NEURAL_CPR_WEIGHTS=/path/to/model.model flow CASE.DATA

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FLOW=${FLOW:-$SCRIPT_DIR/build/opm-simulators/bin/flow}
MODEL=${MODEL:-}
DECK=${DECK:-}
OUTDIR=${OUTDIR:-}
NPROCS=${NPROCS:-1}
CONFIDENCE=${OPM_NEURAL_CPR_CONFIDENCE:-0.6}
FORBID_LABELS=${OPM_NEURAL_CPR_FORBID_LABELS:-}
REUSE=1
REUSE_SETUP=${CPR_REUSE_SETUP:-4}
REUSE_INTERVAL=${CPR_REUSE_INTERVAL:-10}
MPI_ARGS=${MPI_ARGS:-}
FLOW_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  run_neural_cpr.sh --model MODEL.model --deck CASE.DATA [options] [-- FLOW_OPTIONS...]
  run_neural_cpr.sh MODEL.model CASE.DATA [options] [-- FLOW_OPTIONS...]

Options:
  --model PATH          Trained Neural CPR .model file.
  --deck PATH           Flow deck to simulate.
  --flow PATH           Flow binary. Default: ./build/opm-simulators/bin/flow.
  --out DIR             Output directory. Default: ./tmp/neural_cpr_runs/<deck>_<timestamp>.
  --np N                MPI ranks. Default: 1.
  --confidence X        Neural confidence gate. Default: 0.6.
  --forbid-labels LIST  Comma-separated labels to forbid, e.g. 8,16,17,18,19,20,21,22,23.
  --spe11-safe          Shortcut for --forbid-labels 8,16,17,18,19,20,21,22,23.
  --reuse               Enable CPR reuse flags. This is the default.
  --no-reuse            Do not add cpr-reuse flags.
  --reuse-setup N       Default: 4.
  --reuse-interval N    Default: 10.
  --mpi-args "ARGS"     Extra arguments placed after mpirun, before -np.
  -h, --help            Show this help.

Environment overrides:
  FLOW, MODEL, DECK, OUTDIR, NPROCS, OPM_NEURAL_CPR_CONFIDENCE,
  OPM_NEURAL_CPR_FORBID_LABELS, CPR_REUSE_SETUP, CPR_REUSE_INTERVAL, MPI_ARGS.

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
            FORBID_LABELS=8,16,17,18,19,20,21,22,23
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
            shift 2
            ;;
        --reuse-interval)
            [[ $# -ge 2 ]] || die "--reuse-interval requires an integer"
            REUSE_INTERVAL=$2
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
            if [[ -z "$MODEL" ]]; then
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

[[ -n "$MODEL" ]] || die "missing model path; pass --model MODEL.model"
[[ -n "$DECK" ]] || die "missing deck path; pass --deck CASE.DATA"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"
[[ -f "$DECK" ]] || die "deck not found: $DECK"
[[ -x "$FLOW" ]] || die "flow binary not executable: $FLOW"
[[ "$NPROCS" =~ ^[0-9]+$ ]] || die "--np must be an integer"
(( NPROCS >= 1 )) || die "--np must be >= 1"

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

export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
export OPM_NEURAL_CPR_CONFIDENCE="$CONFIDENCE"
if [[ -n "$FORBID_LABELS" ]]; then
    export OPM_NEURAL_CPR_FORBID_LABELS="$FORBID_LABELS"
else
    unset OPM_NEURAL_CPR_FORBID_LABELS
fi

echo "Neural CPR run"
echo "  flow       : $FLOW"
echo "  model      : $MODEL"
echo "  deck       : $DECK"
echo "  out        : $OUTDIR"
echo "  np         : $NPROCS"
echo "  confidence : $CONFIDENCE"
if [[ -n "$FORBID_LABELS" ]]; then
    echo "  forbid     : $FORBID_LABELS"
fi
if (( REUSE == 1 )); then
    echo "  reuse      : setup=$REUSE_SETUP interval=$REUSE_INTERVAL"
else
    echo "  reuse      : disabled"
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
