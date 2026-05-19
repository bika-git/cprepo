#!/usr/bin/env bash
# compare_neural_cpr.sh - compare Flow default vs Neural CPR vs Neural CPR reuse.
#
# This is intended for apply-only testing on another machine:
#   ./compare_neural_cpr.sh --flow /path/to/flow --model model.model --deck SPE11B.DATA --np 1 --spe11-safe

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FLOW=${FLOW:-$SCRIPT_DIR/build/opm-simulators/bin/flow}
MODEL=${MODEL:-}
DECK=${DECK:-}
OUTDIR=${OUTDIR:-}
NPROCS=${NPROCS:-1}
CONFIDENCE=${OPM_NEURAL_CPR_CONFIDENCE:-0.6}
FORBID_LABELS=${OPM_NEURAL_CPR_FORBID_LABELS:-}
REUSE_SETUP=${CPR_REUSE_SETUP:-4}
REUSE_INTERVAL=${CPR_REUSE_INTERVAL:-10}
MPI_ARGS=${MPI_ARGS:-}
BACKGROUND=0
FLOW_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  compare_neural_cpr.sh --model MODEL.model --deck CASE.DATA [options] [-- FLOW_OPTIONS...]

Runs:
  1. default       : no OPM_NEURAL_CPR_WEIGHTS
  2. neural        : OPM_NEURAL_CPR_WEIGHTS set, no cpr-reuse flags
  3. neural-reuse  : OPM_NEURAL_CPR_WEIGHTS set, cpr-reuse flags enabled

Options:
  --model PATH          Trained Neural CPR .model file.
  --deck PATH           Flow deck to simulate.
  --flow PATH           Flow binary. Default: ./build/opm-simulators/bin/flow.
  --out DIR             Output root. Default: ./tmp/compare_neural_cpr/<deck>_<timestamp>.
  --np N                MPI ranks. Default: 1.
  --confidence X        Neural confidence gate. Default: 0.6.
  --forbid-labels LIST  Comma-separated labels to forbid.
  --spe11-safe          Shortcut for --forbid-labels 8,16,17,18,19,20,21,22,23.
  --reuse-setup N       Neural-reuse setup. Default: 4.
  --reuse-interval N    Neural-reuse interval. Default: 10.
  --mpi-args "ARGS"     Extra arguments placed after mpirun, before -np.
  --background          Launch all three runs and return immediately.
  -h, --help            Show this help.

Environment overrides:
  FLOW, MODEL, DECK, OUTDIR, NPROCS, OPM_NEURAL_CPR_CONFIDENCE,
  OPM_NEURAL_CPR_FORBID_LABELS, CPR_REUSE_SETUP, CPR_REUSE_INTERVAL, MPI_ARGS.

Any arguments after "--" are passed directly to flow for all three runs.
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
        --background)
            BACKGROUND=1
            shift
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

FLOW=$(cd "$(dirname "$FLOW")" && pwd)/$(basename "$FLOW")
MODEL=$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")
DECK=$(cd "$(dirname "$DECK")" && pwd)/$(basename "$DECK")

if [[ -z "$OUTDIR" ]]; then
    deck_base=$(basename "$DECK")
    deck_base=${deck_base%.*}
    stamp=$(date +%Y%m%d_%H%M%S)
    OUTDIR=$SCRIPT_DIR/tmp/compare_neural_cpr/${deck_base}_${stamp}
else
    mkdir -p "$OUTDIR"
    OUTDIR=$(cd "$OUTDIR" && pwd)
fi
mkdir -p "$OUTDIR"

make_run_cmd() {
    local -n _cmd=$1
    _cmd=()
    if (( NPROCS > 1 )); then
        _cmd=(mpirun)
        if [[ -n "$MPI_ARGS" ]]; then
            # Intentional shell splitting for user-provided MPI flags.
            # shellcheck disable=SC2206
            local extra_mpi_args=($MPI_ARGS)
            _cmd+=("${extra_mpi_args[@]}")
        fi
        _cmd+=(-np "$NPROCS" "$FLOW")
    else
        _cmd=("$FLOW")
    fi
}

summarize_one() {
    local name=$1 dir=$2 log=$2/flow.log
    local exit_code="?"
    [[ -f "$dir/exit_code" ]] && exit_code=$(cat "$dir/exit_code")
    local sim newton linear errors
    sim=$(grep "Simulation time:" "$log" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    newton=$(grep "Overall Newton Iterations" "$log" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)
    linear=$(grep "Overall Linear Iterations" "$log" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)
    errors=$(grep -E '^Errors[[:space:]]+[0-9]+' "$log" 2>/dev/null | awk '{print $2}' | tail -1 || true)
    printf "%-13s exit=%-3s wall=%-10s newton=%-8s linear=%-8s errors=%s\n" \
        "$name" "$exit_code" "${sim:--}" "${newton:--}" "${linear:--}" "${errors:--}"
}

summarize_all() {
    echo ""
    echo "Summary:"
    summarize_one default "$OUTDIR/default"
    summarize_one neural "$OUTDIR/neural"
    summarize_one neural-reuse "$OUTDIR/neural-reuse"
}

run_case() {
    local name=$1 mode=$2
    local case_dir=$OUTDIR/$name
    mkdir -p "$case_dir"

    local cmd=()
    make_run_cmd cmd

    local args=(--output-dir="$case_dir")
    if [[ "$mode" == "neural-reuse" ]]; then
        args+=(--cpr-reuse-setup="$REUSE_SETUP")
        args+=(--cpr-reuse-interval="$REUSE_INTERVAL")
    fi
    args+=("${FLOW_ARGS[@]}")
    args+=("$DECK")

    (
        cd "$SCRIPT_DIR"
        if [[ "$mode" == "default" ]]; then
            unset OPM_NEURAL_CPR_WEIGHTS
        else
            export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
            export OPM_NEURAL_CPR_CONFIDENCE="$CONFIDENCE"
        fi
        if [[ -n "$FORBID_LABELS" ]]; then
            export OPM_NEURAL_CPR_FORBID_LABELS="$FORBID_LABELS"
        else
            unset OPM_NEURAL_CPR_FORBID_LABELS
        fi

        echo "[$name] started: $(date -Is)"
        echo "[$name] out: $case_dir"
        set +e
        "${cmd[@]}" "${args[@]}" >"$case_dir/flow.log" 2>&1
        status=$?
        set -e
        echo "$status" >"$case_dir/exit_code"
        echo "[$name] finished: $(date -Is), exit=$status"
        exit "$status"
    ) >"$case_dir/launcher.log" 2>&1
}

launch_case_bg() {
    local name=$1 mode=$2
    local case_dir=$OUTDIR/$name
    mkdir -p "$case_dir"
    ( run_case "$name" "$mode" ) &
    echo "$!" >"$case_dir/launcher.pid"
    echo "  $name: $case_dir (pid $(cat "$case_dir/launcher.pid"))"
}

echo "Neural CPR comparison"
echo "  flow       : $FLOW"
echo "  model      : $MODEL"
echo "  deck       : $DECK"
echo "  out        : $OUTDIR"
echo "  np         : $NPROCS"
echo "  confidence : $CONFIDENCE"
if [[ -n "$FORBID_LABELS" ]]; then
    echo "  forbid     : $FORBID_LABELS"
fi
echo "  reuse      : setup=$REUSE_SETUP interval=$REUSE_INTERVAL"

if (( BACKGROUND == 1 )); then
    echo ""
    echo "Launching in background:"
    launch_case_bg default default
    launch_case_bg neural neural
    launch_case_bg neural-reuse neural-reuse
    echo ""
    echo "Watch logs with:"
    echo "  tail -f $OUTDIR/default/flow.log"
    echo "  tail -f $OUTDIR/neural/flow.log"
    echo "  tail -f $OUTDIR/neural-reuse/flow.log"
    exit 0
fi

run_case default default || true
run_case neural neural || true
run_case neural-reuse neural-reuse || true
summarize_all
