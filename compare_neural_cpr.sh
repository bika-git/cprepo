#!/usr/bin/env bash
# compare_neural_cpr.sh - compare plain Flow with the Neural CPR policies.
#
# Runs plain Flow plus one arm per policy given (see --help).  Intended for
# apply-only testing, also on another machine:
#   ./compare_neural_cpr.sh --flow /path/to/flow --amg-model amg.model \
#       --abort-model abort.model --deck SPE11A.DATA --np 32 --threads 2 \
#       --mpi-args "--bind-to none"

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
DEFAULT_REUSE_SETUP=${DEFAULT_CPR_REUSE_SETUP:-4}
DEFAULT_REUSE_INTERVAL=${DEFAULT_CPR_REUSE_INTERVAL:-30}
REUSE_SETUP=${CPR_REUSE_SETUP:-4}
REUSE_INTERVAL=${CPR_REUSE_INTERVAL:-10}
REUSE_ARM=1
MPI_ARGS=${MPI_ARGS:-}
THREADS=${THREADS:-}
BACKGROUND=0
FLOW_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  compare_neural_cpr.sh [--model M] [--amg-model M] [--abort-model M] --deck CASE.DATA [options] [-- FLOW_OPTIONS...]
  compare_neural_cpr.sh MODEL.model CASE.DATA [options] [-- FLOW_OPTIONS...]

Runs (plain Flow, plus one arm per policy given):
  default           policy disabled, upstream solver and CPR reuse defaults
  neural            --model: CPR variant policy, upstream CPR reuse defaults
  neural-reuse      --model: CPR variant policy, explicit CPR reuse flags
  neural-amg        --amg-model: neural AMG policy (prolongation damping and
                    coarsening target of the pressure AMG)
  neural-abort      --abort-model: neural abort policy on the default solver
  neural-amg+abort  --amg-model and --abort-model together

Options:
  --model PATH          CPR variant policy model (24 configs).
  --amg-model PATH      Neural AMG policy model.
  --abort-model PATH    Neural abort policy model.
  --deck PATH           Flow deck to simulate.
  --flow PATH           Flow binary. Default: ./build/opm-simulators/bin/flow.
  --out DIR             Output root. Default: ./tmp/compare_neural_cpr/<deck>_<timestamp>.
  --np N                MPI ranks. Default: 1.
                        On a 64-core node: --np 32 --threads 2 --mpi-args "--bind-to none".
  --threads N           Threads per rank (--threads-per-process=N, OMP_NUM_THREADS=N).
                        Default: Flow's.
  --min-gain G          Anchored decisions (cost and AMG models): leave the default
                        only for a predicted log-cost gain above G. Default for
                        neural-amg: 0.05.
  --abort-threshold P   Abort above this probability of a chop. Default: 0.95.
  --confidence X        Confidence gate for classifier models. Default: 0.6.
  --forbid-labels LIST  Comma-separated CPR variant labels to forbid.
  --spe11-safe          Shortcut for --forbid-labels 16,...,23 (trueimpesanalytic).
                        Flow from neural-cpr-control applies this automatically on
                        decks without oil.  Do not forbid label 8 there: on decks
                        without wells the guards already block the cprw labels, so
                        label 8 is Flow's default solver.
  --reuse-setup N       Neural-reuse setup. Default: 4.
  --reuse-interval N    Neural-reuse interval. Default: 10.
  --no-reuse-arm        Skip the neural-reuse arm.
  --default-reuse-setup N     Setup for the other arms. Default: 4 (Flow's).
  --default-reuse-interval N  Interval for the other arms. Default: 30 (Flow's).
  --mpi-args "ARGS"     Extra arguments placed after mpirun, before -np.
  --background          Launch all runs at once and return immediately (runs
                        then compete for cores, so timings are not comparable).
  -h, --help            Show this help.

Environment overrides:
  FLOW, MODEL, AMG_MODEL, ABORT_MODEL, DECK, OUTDIR, NPROCS,
  OPM_NEURAL_CPR_CONFIDENCE, OPM_NEURAL_CPR_FORBID_LABELS, OPM_NEURAL_CPR_MIN_GAIN,
  OPM_CPR_ABORT_THRESHOLD, DEFAULT_CPR_REUSE_SETUP, DEFAULT_CPR_REUSE_INTERVAL,
  CPR_REUSE_SETUP, CPR_REUSE_INTERVAL, MPI_ARGS, THREADS.

Any arguments after "--" are passed directly to flow for all runs.
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
        --no-reuse-arm)
            REUSE_ARM=0
            shift
            ;;
        --default-reuse-setup)
            [[ $# -ge 2 ]] || die "--default-reuse-setup requires an integer"
            DEFAULT_REUSE_SETUP=$2
            shift 2
            ;;
        --default-reuse-interval)
            [[ $# -ge 2 ]] || die "--default-reuse-interval requires an integer"
            DEFAULT_REUSE_INTERVAL=$2
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
FLOW=$(abspath "$FLOW")
DECK=$(abspath "$DECK")
[[ -n "$MODEL" ]] && MODEL=$(abspath "$MODEL")
[[ -n "$AMG_MODEL" ]] && AMG_MODEL=$(abspath "$AMG_MODEL")
[[ -n "$ABORT_MODEL" ]] && ABORT_MODEL=$(abspath "$ABORT_MODEL")

ARMS=(default)
if [[ -n "$MODEL" ]]; then
    ARMS+=(neural)
    (( REUSE_ARM == 1 )) && ARMS+=(neural-reuse)
fi
[[ -n "$AMG_MODEL" ]] && ARMS+=(neural-amg)
[[ -n "$ABORT_MODEL" ]] && ARMS+=(neural-abort)
[[ -n "$AMG_MODEL" && -n "$ABORT_MODEL" ]] && ARMS+=(neural-amg+abort)

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
    local sim linsolve newton linear aborts dbg
    sim=$(grep "Simulation time:" "$log" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    linsolve=$(grep "Linear solve time:" "$log" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    newton=$(grep "Overall Newton Iterations" "$log" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)
    linear=$(grep "Overall Linear Iterations" "$log" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)
    dbg=$(ls "$dir"/*.DBG 2>/dev/null | head -1 || true)
    aborts=$( [[ -n "$dbg" ]] && grep -c "NeuralAbortPolicy: giving up" "$dbg" || true)
    printf "%-17s exit=%-3s wall=%-9s linear-solve=%-9s newton=%-7s linear=%-8s aborts=%s\n" \
        "$name" "$exit_code" "${sim:--}" "${linsolve:--}" "${newton:--}" "${linear:--}" "${aborts:--}"
}

summarize_all() {
    echo ""
    echo "Summary:"
    local arm
    for arm in "${ARMS[@]}"; do
        summarize_one "$arm" "$OUTDIR/$arm"
    done
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
    else
        args+=(--cpr-reuse-setup="$DEFAULT_REUSE_SETUP")
        args+=(--cpr-reuse-interval="$DEFAULT_REUSE_INTERVAL")
    fi
    [[ -n "$THREADS" ]] && args+=(--threads-per-process="$THREADS")
    args+=("${FLOW_ARGS[@]}")
    args+=("$DECK")

    (
        cd "$SCRIPT_DIR"
        # Clean policy environment, then set what this arm needs.
        unset OPM_NEURAL_CPR_WEIGHTS OPM_NEURAL_CPR_ACTIONS OPM_NEURAL_CPR_MIN_GAIN \
              OPM_NEURAL_CPR_CONFIDENCE OPM_NEURAL_CPR_FORBID_LABELS OPM_NEURAL_CPR_DISABLE \
              OPM_CPR_ABORT_MODEL OPM_CPR_ABORT_THRESHOLD OPM_CPR_SWEEP_LOG
        [[ -n "$THREADS" ]] && export OMP_NUM_THREADS="$THREADS"
        case "$mode" in
            default|neural-abort)
                export OPM_NEURAL_CPR_DISABLE=1
                ;;
            neural|neural-reuse)
                export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
                export OPM_NEURAL_CPR_CONFIDENCE="$CONFIDENCE"
                [[ -n "$MIN_GAIN" ]] && export OPM_NEURAL_CPR_MIN_GAIN="$MIN_GAIN"
                [[ -n "$FORBID_LABELS" ]] && export OPM_NEURAL_CPR_FORBID_LABELS="$FORBID_LABELS"
                ;;
            neural-amg|neural-amg+abort)
                export OPM_NEURAL_CPR_WEIGHTS="$AMG_MODEL"
                export OPM_NEURAL_CPR_ACTIONS=amg
                export OPM_NEURAL_CPR_MIN_GAIN="${MIN_GAIN:-0.05}"
                ;;
        esac
        if [[ "$mode" == "neural-abort" || "$mode" == "neural-amg+abort" ]]; then
            export OPM_CPR_ABORT_MODEL="$ABORT_MODEL"
            export OPM_CPR_ABORT_THRESHOLD="$ABORT_THRESHOLD"
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
    local name=$1
    local case_dir=$OUTDIR/$name
    mkdir -p "$case_dir"
    ( run_case "$name" "$name" ) &
    echo "$!" >"$case_dir/launcher.pid"
    echo "  $name: $case_dir (pid $(cat "$case_dir/launcher.pid"))"
}

echo "Neural CPR comparison"
echo "  flow        : $FLOW"
[[ -n "$MODEL" ]]       && echo "  model       : $MODEL (CPR variant policy)"
[[ -n "$AMG_MODEL" ]]   && echo "  amg model   : $AMG_MODEL (min gain ${MIN_GAIN:-0.05})"
[[ -n "$ABORT_MODEL" ]] && echo "  abort model : $ABORT_MODEL (threshold $ABORT_THRESHOLD)"
echo "  deck        : $DECK"
echo "  out         : $OUTDIR"
echo "  np          : $NPROCS${THREADS:+ x $THREADS threads}"
echo "  arms        : ${ARMS[*]}"
[[ -n "$MODEL" && -z "$MIN_GAIN" ]] && echo "  confidence  : $CONFIDENCE"
[[ -n "$FORBID_LABELS" ]] && echo "  forbid      : $FORBID_LABELS"
echo "  default reuse : setup=$DEFAULT_REUSE_SETUP interval=$DEFAULT_REUSE_INTERVAL"
[[ " ${ARMS[*]} " == *" neural-reuse "* ]] \
    && echo "  neural-reuse  : setup=$REUSE_SETUP interval=$REUSE_INTERVAL"

if (( BACKGROUND == 1 )); then
    echo ""
    echo "Launching in background:"
    for arm in "${ARMS[@]}"; do
        launch_case_bg "$arm"
    done
    echo ""
    echo "Watch logs with:"
    for arm in "${ARMS[@]}"; do
        echo "  tail -f $OUTDIR/$arm/flow.log"
    done
    exit 0
fi

for arm in "${ARMS[@]}"; do
    run_case "$arm" "$arm" || true
done
summarize_all
