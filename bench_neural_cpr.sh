#!/usr/bin/env bash
# bench_neural_cpr.sh — benchmark the neural CPR policies, and optionally
# regenerate the data and model they are built from.
#
# By default only Step 3 runs: the benchmark reuses the sweep data already in
# cpr_sweep_data/ and the model already at MODEL.  Steps 0–2 are the expensive
# ones (hours of flow runs) and are opt-in — see FRESH_SWEEP and SKIP_TRAIN
# under Usage below.
#
# ── Pipeline overview ─────────────────────────────────────────────────────────
#
#   Step 0 — regenerate sweep data            [SKIPPED by default; FRESH_SWEEP=1]
#     Runs all 24 CPR configurations against each deck twice (run1, run2) for
#     Norne + SPE9 (noise reduction via median), and once for SPE11A.
#     Each deck×config pair writes one CSV of per-Newton-step features + timing.
#     Tool: opm-simulators/doc/neural_cpr/run_sweep.py
#     On opm-simulators neural-cpr-control the sweep runs at Flow's CPR
#     settings, including setup reuse (4/30) and use_well_weights=false, so
#     label 12 (label 8 on decks without wells) is Flow's default solver.
#     Sweeps from older branches rebuilt the CPR setup on every solve.
#
#   Step 1 — label sweep data                 [SKIPPED by default; SKIP_TRAIN=0]
#     Groups rows by (deck, episode, attempt, dt, newton_iter), i.e. by solves
#     that several configs reached with the same timestep history, and picks
#     the config with the lowest median total_ms as the label.
#     Tool: opm-simulators/doc/neural_cpr/collect_sweep_labels.py
#
#   Step 2 — architecture search              [SKIPPED by default; SKIP_TRAIN=0]
#     Trains hidden = 8, 12, 16, 24, 32 in parallel; keeps the best val_acc.
#     Tool: opm-simulators/doc/neural_cpr/train_cpr_policy.py
#     Steps 1–2 train the CPR variant classifier ("neural-v3").  Flow's default
#     is the best or tied-best fixed variant on every deck tested so far, so
#     this policy has little headroom.
#
#   Step 3 — benchmark                        [the default: this is all that runs]
#     flow_default / baseline / neural-v3 / neural-reuse for each deck, plus the
#     neural policies of neural-cpr-control on Flow's default solver:
#       neural-amg        neural AMG policy: prolongation damping and coarsening
#                         target of the pressure AMG at each CPR setup rebuild
#       neural-abort      neural abort policy: gives up on a timestep attempt
#                         that is likely to be chopped anyway
#       neural-amg+abort  both
#     Their models are trained elsewhere (doc/neural_cpr/train_cpr_amg_policy.py,
#     train_abort_policy.py); set AMG_MODEL / ABORT_MODEL, or NEW_POLICIES=0.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#
#   ./bench_neural_cpr.sh                              # benchmark only, serial
#   NPROCS=32 FLOW_THREADS=2 ./bench_neural_cpr.sh     # MPI, 32 ranks x 2 threads
#
#   Opt in to the expensive steps (all skipped by default):
#   FRESH_SWEEP=1 ./bench_neural_cpr.sh                # also Step 0: regenerate cpr_sweep_data/
#   SKIP_TRAIN=0 ./bench_neural_cpr.sh                 # also Steps 1-2: label + retrain MODEL
#   FRESH_SWEEP=1 SKIP_TRAIN=0 ./bench_neural_cpr.sh   # the full pipeline, Steps 0-3
#
#   Inputs the default path reads (must already exist):
#   MODEL=/path/to/saved.model ./bench_neural_cpr.sh   # the policy model to benchmark
#                                                      # default: models/cpr_policy_retrained.model
#                                                      # TRAINING_CSV is only read when SKIP_TRAIN=0
#
#   Other options:
#   SPE11PYOPM_SOLVER_LABEL=11 ./bench_neural_cpr.sh   # pin SPE11 base label
#   TUNE_SPE11_REUSE=1 ./bench_neural_cpr.sh           # grid-search reuse-setup × reuse-interval for SPE11
#   BENCH_HELDOUT=1 ./bench_neural_cpr.sh              # also test Sleipner/SPE10 held-out decks
#   NEW_POLICIES=0 ./bench_neural_cpr.sh               # skip the neural AMG / abort arms
#   NEW_POLICY_ARMS="neural-amg" ./bench_neural_cpr.sh # only some of them
#   SPE11B_NPROCS=32 ...                               # SPE11B at 32 ranks (default 16)
#   MPI_ARGS="" ./bench_neural_cpr.sh                  # drop the default "--bind-to none"
#                                                      # (only ever used when NPROCS > 1)

set -euo pipefail

# When expanding CSV globs, ignore patterns that don't match any files
# so we don't pass literal '*.csv' strings to Python.
shopt -s nullglob

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Run from the script directory so relative paths are predictable
cd "$SCRIPT_DIR"

FLOW=$SCRIPT_DIR/build/opm-simulators/bin/flow
SCRIPTS=$SCRIPT_DIR/opm-simulators/doc/neural_cpr

# Set FRESH_SWEEP=1 to regenerate the sweep data in Step 0 (hours of flow runs);
# the default reuses the existing data in cpr_sweep_data/
FRESH_SWEEP=${FRESH_SWEEP:-0}
# Set SKIP_TRAIN=0 to run Steps 1-2 (label+train); the default benchmarks with
# the existing MODEL file
SKIP_TRAIN=${SKIP_TRAIN:-1}
# Set TUNE_SPE11_REUSE=1 to grid-search cpr-reuse-setup x cpr-reuse-interval for SPE11 neural-reuse
TUNE_SPE11_REUSE=${TUNE_SPE11_REUSE:-0}
# Set BENCH_HELDOUT=1 to benchmark decks not used in training.
BENCH_HELDOUT=${BENCH_HELDOUT:-0}
# Supported case names: spe10model1 spe10mod01 spe10model2 spe10mod02
HELDOUT_CASES=${HELDOUT_CASES:-"spe10model1"}

# MPI setup — NPROCS controls ranks per flow call in both sweep and benchmark
NPROCS=${NPROCS:-1}
# Extra mpirun flags for the benchmark runs; the sweep passes its own.  Reached
# only when NPROCS > 1, since set_flow_cmd prepends mpirun only then — a serial
# run never sees these.  Unset gives "--bind-to none"; MPI_ARGS="" passes none.
MPI_ARGS=${MPI_ARGS---bind-to none}
# Threads per rank, passed as --threads-per-process in serial runs too.
# Unset = Flow's own default.
FLOW_THREADS=${FLOW_THREADS:-}
[[ -n "$FLOW_THREADS" ]] && export OMP_NUM_THREADS="$FLOW_THREADS"

# set_flow_cmd N: FLOW_CMD for a benchmark run on N ranks
set_flow_cmd() {
    local np=$1
    FLOW_CMD=()
    if (( np > 1 )); then
        # Intentional shell splitting for user-provided MPI flags.
        # shellcheck disable=SC2206
        FLOW_CMD=(mpirun $MPI_ARGS -np "$np")
    fi
    FLOW_CMD+=("$FLOW")
    if [[ -n "$FLOW_THREADS" ]]; then
        FLOW_CMD+=(--threads-per-process="$FLOW_THREADS")
    fi
}
set_flow_cmd "$NPROCS"

# Sweep output directories.  The *_fresh dirs are written by Step 0, so they are
# only (re)generated when FRESH_SWEEP=1; by default they are read as they stand
# from a previous run.
SWEEP_DIR=$SCRIPT_DIR/cpr_sweep_data
SWEEP_RUN1=$SWEEP_DIR/sweep_run1_fresh
SWEEP_RUN2=$SWEEP_DIR/sweep_run2_fresh
SWEEP_SPE11=$SWEEP_DIR/sweep_spe11_fresh
SWEEP_SPE10MODEL1=$SWEEP_DIR/sweep_spe10model1_fresh
SWEEP_SPE10MOD01=$SWEEP_DIR/sweep_spe10mod01_fresh
# SWEEP_SPE10MODEL2=$SWEEP_DIR/sweep_spe10model2_fresh
# SWEEP_SLEIPNER=$SWEEP_DIR/sweep_sleipner_fresh


# Sweep CSVs used as training input (all 12-feature format)
#   sweep2/3    = 2 runs of SPE9+Norne
#   sweep_spe11 = SPE11A sweep data
SWEEP_CSVS=(
    $SCRIPT_DIR/cpr_sweep_data/sweep_run1_fresh/*.csv
    $SCRIPT_DIR/cpr_sweep_data/sweep_run2_fresh/*.csv
    $SCRIPT_DIR/cpr_sweep_data/sweep_spe11_fresh/*.csv
    # $SCRIPT_DIR/cpr_sweep_data/sweep5/*.csv
    # $SCRIPT_DIR/cpr_sweep_data/sweep7/*.csv
)


# Intermediate and output files
TRAINING_CSV=$SCRIPT_DIR/models/cpr_training_data.csv
MODEL=${MODEL:-$SCRIPT_DIR/models/cpr_policy_retrained.model}
ENCODER_PRETRAINED=$SCRIPT_DIR/models/cpr_encoder_pretrained.npz
OUTDIR=$SCRIPT_DIR/tmp/bench_neural_cpr

# Neural AMG and neural abort policies (Flow from opm-simulators neural-cpr-control),
# run on Flow's default solver and CPR reuse.  Defaults: the models evaluated in
# neural_cpr_snapshot_20261015 (both trained on SPE11A, Norne and SPE9).
NEW_POLICIES=${NEW_POLICIES:-1}
NEW_POLICY_ARMS=${NEW_POLICY_ARMS:-"neural-amg neural-abort neural-amg+abort"}
AMG_MODEL=${AMG_MODEL:-$SCRIPT_DIR/models/amg_policy/cpr_amg_policy.model}
ABORT_MODEL=${ABORT_MODEL:-$SCRIPT_DIR/models/abort_policy/abort_spe11a_norne_spe9.model}
AMG_MIN_GAIN=${AMG_MIN_GAIN:-0.05}
ABORT_THRESHOLD=${ABORT_THRESHOLD:-0.95}

# CPR variant labels the neural-v3 policy may not pick on SPE11: 16-23
# (trueimpesanalytic) need an oil phase, which SPE11 lacks.  Flow from
# neural-cpr-control blocks them itself (deck guards); this keeps older builds
# safe.  Do not forbid label 8: SPE11 has no wells, so the guards allow only
# the cpr labels, and label 8 (cpr, trueimpes, paroverilu0, ilu0) is then
# Flow's default solver.
SPE11_FORBID_LABELS=${SPE11_FORBID_LABELS:-16,17,18,19,20,21,22,23}

if (( NEW_POLICIES == 1 )); then
    for arm in $NEW_POLICY_ARMS; do
        case "$arm" in
            neural-amg|neural-abort|neural-amg+abort) ;;
            *) echo "ERROR: unknown arm in NEW_POLICY_ARMS: $arm" >&2; exit 1 ;;
        esac
        if [[ "$arm" == neural-amg* && ! -f "$AMG_MODEL" ]]; then
            echo "ERROR: AMG_MODEL not found: $AMG_MODEL (or set NEW_POLICIES=0)" >&2; exit 1
        fi
        if [[ "$arm" == *abort && ! -f "$ABORT_MODEL" ]]; then
            echo "ERROR: ABORT_MODEL not found: $ABORT_MODEL (or set NEW_POLICIES=0)" >&2; exit 1
        fi
    done
fi

# Decks (bikagit/decks, cloned by setup_bench_env.sh; override with DECKS_DIR)
DECKS_DIR=${DECKS_DIR:-$SCRIPT_DIR/decks}
NORNE_DECK=$DECKS_DIR/norne/NORNE_ATW2013.DATA
SPE9_DECK=$DECKS_DIR/spe9/SPE9.DATA
SPE11PYOPM_DECK=$DECKS_DIR/spe11pyopm/SPE11A.DATA
SPE11B_DECK=$DECKS_DIR/spe11b/SPE11B.DATA
SLEIPNER_DECK=$DECKS_DIR/sleipner/SLEIPNER_ORG.DATA
SPE10MODEL1_DECK=$DECKS_DIR/spe10model1/SPE10_MODEL1.DATA
SPE10MOD01_DECK=$DECKS_DIR/spe10/SPE10-MOD01-01.DATA
SPE10MODEL2_DECK=$DECKS_DIR/spe10model2/SPE10_MODEL2.DATA
SPE10MOD02_DECK=$DECKS_DIR/spe10/SPE10-MOD02-01.DATA
NORNE_MSW_DECK=$DECKS_DIR/norne/NORNE_ATW2013_1A_MSW.DATA
DROGON_HIST_DECK=$DECKS_DIR/drogon/model/DROGON_HIST.DATA
DROGON_PRED_DECK=$DECKS_DIR/drogon/model/DROGON_PRED.DATA

# SPE11 solver configs: the fresh sweep dir when Step 0 regenerates it,
# otherwise (the default) the configs kept in cpr_sweep_data/sweep_spe11/
if [[ $FRESH_SWEEP -eq 1 ]]; then
    SPE11PYOPM_CONFIGS_DIR=$SWEEP_SPE11/configs
else
    SPE11PYOPM_CONFIGS_DIR=$SWEEP_DIR/sweep_spe11/configs
fi
SPE11PYOPM_AVOID_LABEL23=${SPE11PYOPM_AVOID_LABEL23:-1}

# If the chosen configs dir is missing — the common case being the default
# FRESH_SWEEP=0 on a tree where only a freshly generated sweep was ever written
# — fall back to the fresh sweep configs instead of failing.
if [[ ! -d "$SPE11PYOPM_CONFIGS_DIR" && -d "$SWEEP_SPE11/configs" ]]; then
    echo "Warning: SPE11 configs dir $SPE11PYOPM_CONFIGS_DIR missing; falling back to $SWEEP_SPE11/configs"
    SPE11PYOPM_CONFIGS_DIR=$SWEEP_SPE11/configs
fi

mkdir -p "$SCRIPT_DIR/tmp"

# ── helpers ──────────────────────────────────────────────────────────────────

run_deck() {
    local tag=$1 deck=$2 outdir=$3
    shift 3
    echo "  → $tag"
    mkdir -p "$outdir"
    local status=0
    # Plain-flow arms must be upstream flow.  On binaries from before the
    # 2026-10-29 inertness fix, a missing model path switched on a silent
    # rule-based CPR schedule (log: "active (rule-based fallback)"), which
    # inflated every gain measured against them (Norne ~4 %, OMEGA-0 8-22 %).
    # Clear the policy environment and disable the policy explicitly.
    local -a envp=()
    if [[ "$tag" == "flow_default" || "$tag" == "baseline" ]]; then
        envp=(env -u OPM_NEURAL_CPR_WEIGHTS -u OPM_NEURAL_CPR_ACTIONS -u OPM_NEURAL_CPR_MIN_GAIN
              -u OPM_NEURAL_CPR_CONFIDENCE -u OPM_NEURAL_CPR_FORBID_LABELS -u OPM_NEURAL_CPR_RULE
              -u OPM_CPR_ABORT_MODEL -u OPM_CPR_ABORT_THRESHOLD -u OPM_CPR_SWEEP_LOG
              -u OPM_EW -u OPM_EW_GATE_MODEL -u OPM_EW_COST_MODEL -u OPM_EW_GATE_ONLINE
              OPM_NEURAL_CPR_DISABLE=1)
    fi
    if "${envp[@]}" "${FLOW_CMD[@]}" --output-dir="$outdir" "$@" "$deck" >"$outdir/flow.log" 2>&1; then
        status=0
    else
        status=$?
    fi
    echo "$status" >"$outdir/exit_code"
    if [[ "$tag" == "flow_default" || "$tag" == "baseline" ]] \
       && grep -q "rule-based fallback\|NeuralCprPolicy decision" "$outdir/flow.log" "$outdir"/*.DBG 2>/dev/null; then
        echo "  !! $tag ran with the CPR policy active - NOT an upstream baseline" | tee -a "$outdir/NOT_UPSTREAM"
    fi
}

label_desc() {
    local label=$1
    case "$label" in
        none)  echo "no fixed sweep label; Flow default linear solver" ;;
        ml)    echo "ML policy predicts labels 00-23 per Newton step" ;;
        *)
            local n=$((10#$label))
            local dec=$(( n / 8 )) cprw=$(( (n % 8) / 4 )) fine=$(( (n % 4) / 2 )) coarse=$(( n % 2 ))
            local wt; case $dec in 0) wt="quasiimpes";; 1) wt="trueimpes";; 2) wt="trueimpesanalytic";; *) wt="?";; esac
            local cpt; [[ $cprw -eq 1 ]] && cpt="cprw" || cpt="cpr"
            local ft;  [[ $fine  -eq 1 ]] && ft="dilu"  || ft="paroverilu0"
            local ct;  [[ $coarse -eq 1 ]] && ct="dilu"  || ct="ilu0"
            echo "label_$(printf '%02d' $n) = ${wt} + ${cpt} + ${ft} + ${ct}"
            ;;
    esac
}

policy_label_from_line() {
    local line=$1
    local dec=0 cprw=0 fine=0 coarse=0
    if [[ "$line" == *"label_"[0-9][0-9]* ]]; then
        local labeled=${line#*label_}
        printf "%02d" "$((10#${labeled:0:2}))"
        return
    fi
    if [[ "$line" == *"weight=trueimpesanalytic"* ]]; then
        dec=2
    elif [[ "$line" == *"weight=trueimpes"* ]]; then
        dec=1
    fi
    [[ "$line" == *"cprw weight="* ]] && cprw=1
    [[ "$line" == *"fine=dilu"* ]]    && fine=1
    [[ "$line" == *"coarse=dilu"* ]]  && coarse=1
    printf "%02d" "$((dec * 8 + cprw * 4 + fine * 2 + coarse))"
}

extract() {
    local log=$1 outdir exit_code
    outdir=$(dirname "$log")
    exit_code=$(cat "$outdir/exit_code" 2>/dev/null || echo 0)
    if [[ "$exit_code" != "0" ]] || grep -Eq "Assertion failed|Aborted|Segmentation fault" "$log" 2>/dev/null; then
        echo "FAIL FAIL FAIL"; return
    fi
    local newton linear wall
    newton=$(grep "Overall Newton Iterations" "$log" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    linear=$(grep "Overall Linear Iterations"  "$log" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    wall=$(grep "Simulation time:"             "$log" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "${newton:--} ${linear:--} ${wall:--}"
}

pct_delta() {
    local base=$1 new=$2
    if [[ "$base" == "-" || "$new" == "-" || "$base" == "FAIL" || "$new" == "FAIL" ]]; then echo "n/a"; return; fi
    awk "BEGIN { printf \"%+.1f%%\", ($new - $base) / $base * 100 }"
}

print_table() {
    # Usage: print_table "deck label" tag1 n1 l1 w1  [tag2 n2 l2 w2 ...]
    # First row is the reference (Δ = —); subsequent rows show Δ vs first row.
    local deck_label=$1; shift
    local ref_w first=1
    local b18 b10 b12
    b18=$(printf '─%.0s' {1..18}); b10=$(printf '─%.0s' {1..10}); b12=$(printf '─%.0s' {1..12})
    echo ""
    echo "┌${b18}─${b10}─${b10}─${b10}─${b12}┐"
    printf "│  %-62s│\n" "$deck_label"
    echo "├${b18}┬${b10}┬${b10}┬${b10}┬${b12}┤"
    # "Δ" and "—" take two and three bytes; printf pads by bytes.
    printf "│ %-16s │ %8s │ %8s │ %8s │ %11s │\n" "Run" "Newton" "Linear" "Wall (s)" "Δ wall"
    echo "├${b18}┼${b10}┼${b10}┼${b10}┼${b12}┤"
    while [[ $# -ge 4 ]]; do
        local tag=$1 rn=$2 rl=$3 rw=$4; shift 4
        if [[ $first -eq 1 ]]; then
            ref_w=$rw; first=0
            printf "│ %-16s │ %8s │ %8s │ %8s │ %12s │\n" "$tag" "$rn" "$rl" "$rw" "—"
        else
            printf "│ %-16s │ %8s │ %8s │ %8s │ %10s │\n" "$tag" "$rn" "$rl" "$rw" "$(pct_delta "$ref_w" "$rw")"
        fi
    done
    echo "└${b18}┴${b10}┴${b10}┴${b10}┴${b12}┘"
}

policy_check() {
    local dbg=$1
    local line
    line=$(grep "NeuralCprPolicy decision label_" "$dbg" 2>/dev/null | tail -1 || true)
    [[ -z "$line" ]] && line=$(grep "NeuralCprPolicy: .*weight=" "$dbg" 2>/dev/null | tail -1 || true)
    [[ -z "$line" ]] && line=$(grep "NeuralCprPolicy" "$dbg" 2>/dev/null | head -1 || true)
    if [[ -z "$line" ]]; then
        echo "  [!] No NeuralCprPolicy lines in DBG — model may not have loaded"
    elif [[ "$line" == *"weight="* || "$line" == *"decision label_"* ]]; then
        echo "  [✓] selected label_$(policy_label_from_line "$line"): $line"
    else
        echo "  [✓] $line"
    fi
}

policy_summary() {
    local dbg=$1
    [[ ! -f "$dbg" ]] && { echo "  Policy summary: DBG missing"; return; }
    awk '
        function remember(label) {
            labels[++n]=label; counts[label]++
            if (seen[label]++==0) unique++
            if (n>1 && label!=prev) switches++
            prev=label
        }
        /NeuralCprPolicy decision label_[0-9][0-9]/ {
            if (match($0, /label_[0-9][0-9]/)) {
                label = substr($0, RSTART + 6, 2) + 0
                remember(label)
                saw_decision=1
            }
            next
        }
        /NeuralCprPolicy: .*weight=/ {
            if (saw_decision) next
            dec=0; cprw=0; fine=0; coarse=0
            if ($0 ~ /weight=trueimpesanalytic/) dec=2
            else if ($0 ~ /weight=trueimpes/) dec=1
            if ($0 ~ /cprw weight=/) cprw=1
            if ($0 ~ /fine=dilu/)    fine=1
            if ($0 ~ /coarse=dilu/)  coarse=1
            label = dec*8 + cprw*4 + fine*2 + coarse
            remember(label)
        }
        END {
            if (n==0) { print "  Policy summary: no labeled decisions found"; exit }
            printf "  Policy summary: %d decisions, %d unique labels, %d switches\n", n, unique, switches
            for (rank=1; rank<=5; ++rank) {
                best=-1; best_count=-1
                for (label in counts)
                    if (!printed[label] && counts[label]>best_count) { best=label; best_count=counts[label] }
                if (best<0) break
                printed[best]=1
                printf "    label_%02d: %d decisions (%.1f%%)\n", best, best_count, 100.0*best_count/n
            }
        }
    ' "$dbg"
}

# ── neural AMG / abort arms ──────────────────────────────────────────────────

arm_dir() { local a=${1//-/_}; echo "${a//+/_}"; }   # neural-amg+abort → neural_amg_abort

run_new_policies() {
    # Usage: run_new_policies <out prefix> <deck> [flow args...]
    # Runs each arm of NEW_POLICY_ARMS with its own clean policy environment.
    (( NEW_POLICIES == 1 )) || return 0
    local prefix=$1 deck=$2 arm
    shift 2
    for arm in $NEW_POLICY_ARMS; do
        (
            unset OPM_NEURAL_CPR_WEIGHTS OPM_NEURAL_CPR_ACTIONS OPM_NEURAL_CPR_MIN_GAIN \
                  OPM_NEURAL_CPR_CONFIDENCE OPM_NEURAL_CPR_FORBID_LABELS OPM_NEURAL_CPR_DISABLE \
                  OPM_CPR_ABORT_MODEL OPM_CPR_ABORT_THRESHOLD OPM_CPR_SWEEP_LOG
            if [[ "$arm" == neural-amg* ]]; then
                export OPM_NEURAL_CPR_WEIGHTS="$AMG_MODEL" OPM_NEURAL_CPR_ACTIONS=amg
                export OPM_NEURAL_CPR_MIN_GAIN="$AMG_MIN_GAIN"
            else
                export OPM_NEURAL_CPR_DISABLE=1
            fi
            if [[ "$arm" == *abort ]]; then
                export OPM_CPR_ABORT_MODEL="$ABORT_MODEL" OPM_CPR_ABORT_THRESHOLD="$ABORT_THRESHOLD"
            fi
            run_deck "$arm" "$deck" "$OUTDIR/${prefix}_$(arm_dir "$arm")" "$@"
        )
    done
}

new_policy_rows() {
    # Usage: mapfile -t rows < <(new_policy_rows <out prefix>)
    # Prints print_table arguments (tag newton linear wall) for each arm.
    (( NEW_POLICIES == 1 )) || return 0
    local prefix=$1 arm n l w
    for arm in $NEW_POLICY_ARMS; do
        n=; l=; w=
        read -r n l w < <(extract "$OUTDIR/${prefix}_$(arm_dir "$arm")/flow.log") || true
        printf '%s\n' "$arm" "${n:--}" "${l:--}" "${w:--}"
    done
}

new_policy_check() {
    # Usage: new_policy_check <out prefix> <deck>
    # What the networks did: AMG setups changed, timestep attempts given up.
    (( NEW_POLICIES == 1 )) || return 0
    local prefix=$1 stem arm dbg note
    stem=$(basename "$2" .DATA)
    echo "  Neural policy check:"
    for arm in $NEW_POLICY_ARMS; do
        dbg="$OUTDIR/${prefix}_$(arm_dir "$arm")/$stem.DBG"
        if [[ ! -f "$dbg" ]]; then
            printf "    %-17s DBG missing\n" "$arm"
            continue
        fi
        note=
        if [[ "$arm" == neural-amg* ]]; then
            note=$(awk '/NeuralCprPolicy AMG decision/ {
                            n++; if ($0 !~ /damping=1\.0+ coarsenTarget=1200 /) moved++ }
                        END { printf "AMG changed at %d of %d setups", moved, n }' "$dbg")
        fi
        if [[ "$arm" == *abort ]]; then
            note+="${note:+; }gave up $(grep -c "NeuralAbortPolicy: giving up" "$dbg" || true) timestep attempts"
        fi
        printf "    %-17s %s\n" "$arm" "$note"
    done
}

heldout_case_info() {
    local case_name=$1
    HELDOUT_CASE=$case_name
    HELDOUT_TITLE=
    HELDOUT_DECK=
    HELDOUT_STEM=

    case "$case_name" in
        # sleipner)
        #     HELDOUT_TITLE="Sleipner"
        #     HELDOUT_DECK=$SLEIPNER_DECK
        #     HELDOUT_STEM="heldout_sleipner"
        #     ;;
        spe10model1)
            HELDOUT_TITLE="SPE10 model1"
            HELDOUT_DECK=$SPE10MODEL1_DECK
            HELDOUT_STEM="heldout_spe10model1"
            ;;
        spe10mod01)
            HELDOUT_TITLE="SPE10 MOD01-01"
            HELDOUT_DECK=$SPE10MOD01_DECK
            HELDOUT_STEM="heldout_spe10mod01"
            ;;
        # spe10model2)
        #     HELDOUT_TITLE="SPE10 model2"
        #     HELDOUT_DECK=$SPE10MODEL2_DECK
        #     HELDOUT_STEM="heldout_spe10model2"
        #     ;;
        spe10mod02)
            HELDOUT_TITLE="SPE10 MOD02-01"
            HELDOUT_DECK=$SPE10MOD02_DECK
            HELDOUT_STEM="heldout_spe10mod02"
            ;;
        *)
            echo "Warning: unknown held-out case '$case_name' (skipping)" >&2
            return 1
            ;;
    esac

    if [[ ! -f "$HELDOUT_DECK" ]]; then
        echo "Warning: held-out deck missing for '$case_name': $HELDOUT_DECK (skipping)" >&2
        return 1
    fi

    # if [[ "$case_name" == "sleipner" ]]; then
    #     local sleipner_grid=$DECKS_DIR/sleipner/Sleipner_Reference_Model_cleaned.grdecl
    #     if [[ ! -f "$sleipner_grid" && -f "${sleipner_grid}.gz" ]]; then
    #         echo "  Preparing Sleipner include: $(basename "$sleipner_grid")"
    #         gzip -dk "${sleipner_grid}.gz"
    #     fi
    #     if [[ ! -f "$sleipner_grid" ]]; then
    #         echo "Warning: Sleipner include missing: $sleipner_grid (skipping)" >&2
    #         return 1
    #     fi
    # fi
}

# ── Step 0 — regenerate sweeps ───────────────────────────────────────────────

run_sweep() {
    # Usage: run_sweep <tag> <out_dir> [--nprocs=N] --decks <deck...>
    # --nprocs=N overrides the global NPROCS for this sweep call only.
    local tag=$1 out=$2 nprocs=$NPROCS; shift 2
    if [[ $# -gt 0 && "$1" == --nprocs=* ]]; then
        nprocs=${1#--nprocs=}; shift
    fi
    echo "  ── $tag ──"
    python3 "$SCRIPTS/run_sweep.py" \
        --flow    "$FLOW" \
        --nprocs  "$nprocs" \
        --out     "$out" \
        "$@"
}

if [[ $FRESH_SWEEP -eq 1 ]]; then
    echo "========================================"
    echo "  Step 0 — regenerate sweep data"
    echo "  Norne + SPE9  : 2 independent runs  (→ sweep_run1_fresh, sweep_run2_fresh)"
    echo "  SPE11A        : 1 run               (→ sweep_spe11_fresh, $NPROCS ranks)"
    echo "  SPE11B        : 1 run               (→ sweep_spe11_fresh, ${SPE11B_NPROCS:-16} ranks — capped to avoid /dev/shm limit)"
    echo "  SPE10 model1  : 1 run               (→ sweep_spe10model1_fresh, skipped if deck missing)"
    echo "  SPE10 MOD01   : 1 run               (→ sweep_spe10mod01_fresh, skipped if deck missing)"
    # echo "  SPE10 model2  : 1 run               (→ sweep_spe10model2_fresh)"
    echo "  MPI ranks/job: $NPROCS"
    echo "========================================"

    run_sweep "Norne + SPE9 (run 1)" "$SWEEP_RUN1" \
        --decks "$NORNE_DECK" "$SPE9_DECK"

    run_sweep "Norne + SPE9 (run 2)" "$SWEEP_RUN2" \
        --decks "$NORNE_DECK" "$SPE9_DECK"

    run_sweep "SPE11A" "$SWEEP_SPE11" \
        --decks "$SPE11PYOPM_DECK"

    _spe11b_sweep_nprocs=${SPE11B_NPROCS:-16}
    (( _spe11b_sweep_nprocs > NPROCS )) && _spe11b_sweep_nprocs=$NPROCS
    run_sweep "SPE11B" "$SWEEP_SPE11" "--nprocs=${_spe11b_sweep_nprocs}" \
        --decks "$SPE11B_DECK"

    if [[ -f "$SPE10MODEL1_DECK" ]]; then
        run_sweep "SPE10 model1" "$SWEEP_SPE10MODEL1" \
            --decks "$SPE10MODEL1_DECK"
    else
        echo "  SPE10 model1 : SKIPPED (deck not found: $SPE10MODEL1_DECK)"
    fi

    if [[ -f "$SPE10MOD01_DECK" ]]; then
        run_sweep "SPE10 MOD01" "$SWEEP_SPE10MOD01" \
            --decks "$SPE10MOD01_DECK"
    else
        echo "  SPE10 MOD01  : SKIPPED (deck not found: $SPE10MOD01_DECK)"
    fi

    # run_sweep "SPE10 model2" "$SWEEP_SPE10MODEL2" \
    #     --decks "$SPE10MODEL2_DECK"

    # run_sweep "Sleipner" "$SWEEP_SLEIPNER" \
    #     --decks "$SLEIPNER_DECK"
else
    echo "========================================"
    echo "  Step 0 — SKIPPED (FRESH_SWEEP=0, the default; using SWEEP_CSVS)"
    for f in "${SWEEP_CSVS[@]}"; do echo "  $f"; done
    echo "========================================"
    missing=0
    for f in "${SWEEP_CSVS[@]}"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: missing sweep CSV: $f" >&2
            missing=1
        fi
    done
    [[ $missing -eq 1 ]] && exit 1
fi

# ── Step 1 — label sweep data ────────────────────────────────────────────────

echo ""
echo "========================================"
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo "  Step 1 — label sweep data"
    echo "  Output: $TRAINING_CSV"
    echo "========================================"

    # Build input CSV patterns depending on whether we just generated a fresh sweep
    if [[ $FRESH_SWEEP -eq 1 ]]; then
        INPUT_PATTERNS=("$SWEEP_RUN1"/*.csv "$SWEEP_RUN2"/*.csv "$SWEEP_SPE11"/*.csv \
                        "$SWEEP_SPE10MODEL1"/*.csv "$SWEEP_SPE10MOD01"/*.csv)
        echo "  Input : sweep_run1/2_fresh + sweep_spe11_fresh + sweep_spe10model1/mod01_fresh"
    else
        INPUT_PATTERNS=("${SWEEP_CSVS[@]}")
        echo "  Input : SWEEP_CSVS (existing local data)"
    fi

    # Expand patterns and keep only existing files (nullglob prevents literal patterns)
    EXISTING_SWEEP_CSVS=()
    for pat in "${INPUT_PATTERNS[@]}"; do
        for f in $pat; do
            [[ -f "$f" ]] || continue
            EXISTING_SWEEP_CSVS+=("$f")
        done
    done

    if [[ ${#EXISTING_SWEEP_CSVS[@]} -eq 0 ]]; then
        echo "Warning: no sweep CSVs found (patterns: ${INPUT_PATTERNS[*]}) — skipping label collection"
    else
        python3 "$SCRIPTS/collect_sweep_labels.py" \
            --csvs "${EXISTING_SWEEP_CSVS[@]}" \
            --out  "$TRAINING_CSV" \
            --min-configs 2
    fi
else
    echo "  Step 1 — SKIPPED (SKIP_TRAIN=1, the default; reusing $TRAINING_CSV)"
    echo "========================================"
    if [[ ! -f "$TRAINING_CSV" ]]; then
        echo "ERROR: $TRAINING_CSV missing — run once with SKIP_TRAIN=0 first" >&2
        exit 1
    fi
fi

# Auto-select best SPE11 baseline: most-winning label among safe labels (0-15).
# Override with SPE11PYOPM_SOLVER_LABEL=XX to pin a specific label across machines.
if [[ -z "${SPE11PYOPM_SOLVER_LABEL:-}" ]]; then
    SPE11PYOPM_SOLVER_LABEL=$(python3 -c "
import csv, collections, sys
try:
    rows = [r for r in csv.DictReader(open('$TRAINING_CSV')) if 'spe11' in r.get('deck','').lower()]
    safe = [r['label'] for r in rows if int(r['label']) < 16]
    pool = safe if safe else [r['label'] for r in rows]
    # Avoid labels 08 and 10: both unstable as baseline with cpr-reuse-setup=0 at high rank counts.
    avoid = {8, 10}
    if pool:
        cnt = collections.Counter(pool)
        best = None
        for lbl, _ in cnt.most_common():
            try:
                if int(lbl) in avoid:
                    continue
            except Exception:
                pass
            best = lbl
            break
        if best is None:
            best = '11'
    else:
        best = '11'
    print(f'{int(best):02d}')
except Exception as e:
    sys.stderr.write(f'Warning: SPE11 label auto-select failed: {e}\n')
    print('11')
" 2>/dev/null || echo "11")
fi
SPE11PYOPM_SOLVER="$SPE11PYOPM_CONFIGS_DIR/label_${SPE11PYOPM_SOLVER_LABEL}.json"
echo "  Auto-selected SPE11 baseline: label_${SPE11PYOPM_SOLVER_LABEL} ($(label_desc "$SPE11PYOPM_SOLVER_LABEL"))"

# ── Step 2 — architecture search ─────────────────────────────────────────────

n_features=$(python3 -c "
import csv
r = next(csv.DictReader(open('$TRAINING_CSV')))
print(len([k for k in r if k not in ('deck','best_config','label')]))" \
2>/dev/null || echo 12)

echo ""
echo "========================================"
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo "  Step 2 — pre-training encoder + architecture search"
    echo "  Candidates: hidden = 8, 12, 16, 24, 32"
    echo "========================================"

    # ── Phase 1: pre-train encoder on raw sweep rows (regression task) ──
    if [[ ${#EXISTING_SWEEP_CSVS[@]} -eq 0 ]]; then
        echo "  Phase 1: no sweep CSVs found — skipping pre-training"
    else
        echo "  Phase 1: pre-training encoder on raw sweep CSVs (${#EXISTING_SWEEP_CSVS[@]} files) ..."
        python3 "$SCRIPTS/pretrain_cpr_policy.py" \
            --csvs "${EXISTING_SWEEP_CSVS[@]}" \
            --out  "$ENCODER_PRETRAINED" \
            --hidden 24 --depth 1 \
            --epochs 30 --seed 42
        echo ""
    fi

    # ── Phase 2: architecture search with warm-started encoder ──
    best_hidden=32
    best_depth=1
    best_val=0.0
    best_arch_str="${n_features}→32→24"
    echo "  Input features detected: ${n_features}"
    echo "  Candidates: depth=1 h∈{8,12,16,24,32}  depth=2 h∈{16,24,32}"

    # Pass --pretrained only if the encoder file was produced
    pretrain_arg=()
    [[ -f "$ENCODER_PRETRAINED" ]] && pretrain_arg=(--pretrained "$ENCODER_PRETRAINED")

    # depth=1: all widths; depth=2: only larger widths (small h with 2 layers overfits)
    arch_specs=("8 1" "12 1" "16 1" "24 1" "32 1" "16 2" "24 2" "32 2")
    for spec in "${arch_specs[@]}"; do
        hidden=${spec%% *}; depth=${spec##* }
        (
            python3 "$SCRIPTS/train_cpr_policy.py" \
                --data       "$TRAINING_CSV" \
                --hidden     "$hidden" \
                --depth      "$depth" \
                --activation relu \
                --epochs     200 \
                --lr         0.001 \
                --out        "$SCRIPT_DIR/tmp/cpr_arch_h${hidden}_d${depth}.model" \
                --format     binary \
                "${pretrain_arg[@]}" \
                --seed       42 2>&1 \
            | grep "^Final" | grep -oE 'val acc: [0-9.]+' | grep -oE '[0-9.]+' \
            > "$SCRIPT_DIR/tmp/cpr_arch_h${hidden}_d${depth}.acc"
        ) &
    done
    wait

    for spec in "${arch_specs[@]}"; do
        hidden=${spec%% *}; depth=${spec##* }
        val_acc=$(cat "$SCRIPT_DIR/tmp/cpr_arch_h${hidden}_d${depth}.acc" 2>/dev/null || true)
        hidden_chain=$(printf '%s' "$hidden"; for (( i=1; i<depth; i++ )); do printf '→%s' "$hidden"; done)
        arch_label="${n_features}→${hidden_chain}→24"
        printf "  %-28s  val_acc=%s\n" "$arch_label" "${val_acc:--}"
        if awk "BEGIN { exit !(${val_acc:-0} > $best_val) }"; then
            best_val=${val_acc:-0}
            best_hidden=$hidden
            best_depth=$depth
            best_arch_str="$arch_label"
            cp "$SCRIPT_DIR/tmp/cpr_arch_h${hidden}_d${depth}.model" "$MODEL"
        fi
    done

    echo ""
    echo "  Best arch : ${best_arch_str}  (val_acc=${best_val})"
    echo "  Model written to: $MODEL"
else
    echo "  Step 2 — SKIPPED (SKIP_TRAIN=1, the default; using saved model)"
    echo "========================================"
    if [[ ! -f "$MODEL" ]]; then
        echo "ERROR: $MODEL missing — run once with SKIP_TRAIN=0 first" >&2
        exit 1
    fi
    best_hidden="saved"
    best_depth="saved"
    best_arch_str="saved"
    best_val="saved"
    echo "  Model: $MODEL ($(wc -c < "$MODEL") bytes)"
fi

# ── Step 3 — benchmark ───────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Step 3 — benchmark  (flow_default vs arch: ${best_arch_str})"
echo "  Model: $MODEL"
echo "  Run labels:"
echo "    Norne/SPE9/SPE10model1 flow_default : $(label_desc none)"
echo "    Norne/SPE9/SPE10model1 neural       : $(label_desc ml)"
echo "    SPE11 flow_default      : $(label_desc none)"
echo "    SPE11 baseline          : $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
echo "    SPE11 neural            : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels $SPE11_FORBID_LABELS forbidden"
if [[ $TUNE_SPE11_REUSE -eq 1 ]]; then
    echo "    SPE11 neural-reuse      : grid search setup={2,4,8} × interval={5,10,20} → best kept"
else
    echo "    SPE11 neural-reuse      : same neural guard; cpr-reuse-setup=4, cpr-reuse-interval=10"
fi
if [[ $BENCH_HELDOUT -eq 1 ]]; then
    echo "    Held-out decks          : $HELDOUT_CASES"
fi
if (( NEW_POLICIES == 1 )); then
    echo "    All decks, Flow's default solver and CPR reuse: $NEW_POLICY_ARMS"
    echo "      neural AMG model      : $AMG_MODEL (min gain $AMG_MIN_GAIN)"
    echo "      neural abort model    : $ABORT_MODEL (threshold $ABORT_THRESHOLD)"
fi
# MPI_ARGS only reaches mpirun when NPROCS > 1 (see set_flow_cmd), so don't
# report it on a serial run where no mpirun is spawned.
if (( NPROCS > 1 )); then
    echo "  MPI: $NPROCS ranks${FLOW_THREADS:+ x $FLOW_THREADS threads}${MPI_ARGS:+ (mpirun $MPI_ARGS)}"
else
    echo "  MPI: serial (1 rank, no mpirun)${FLOW_THREADS:+, $FLOW_THREADS threads}"
fi
echo "========================================"

echo ""
echo "── Norne ───────────────────────────────"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$NORNE_DECK" "$OUTDIR/norne_flow_default"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
run_deck "neural-v3"    "$NORNE_DECK" "$OUTDIR/norne_neural"
unset OPM_NEURAL_CPR_WEIGHTS
run_new_policies norne "$NORNE_DECK"

echo ""
echo "── Norne MSW (1A) ───────────────────────"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$NORNE_MSW_DECK" "$OUTDIR/norne_msw_flow_default"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
run_deck "neural-v3"    "$NORNE_MSW_DECK" "$OUTDIR/norne_msw_neural"
run_deck "neural-reuse" "$NORNE_MSW_DECK" "$OUTDIR/norne_msw_neural_reuse" \
    --cpr-reuse-setup=4 \
    --cpr-reuse-interval=10
unset OPM_NEURAL_CPR_WEIGHTS
run_new_policies norne_msw "$NORNE_MSW_DECK"

echo ""
echo "── Drogon HIST ─────────────────────────"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$DROGON_HIST_DECK" "$OUTDIR/drogon_hist_flow_default"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
run_deck "neural-v3"    "$DROGON_HIST_DECK" "$OUTDIR/drogon_hist_neural"
run_deck "neural-reuse" "$DROGON_HIST_DECK" "$OUTDIR/drogon_hist_neural_reuse" \
    --cpr-reuse-setup=4 \
    --cpr-reuse-interval=10
unset OPM_NEURAL_CPR_WEIGHTS
run_new_policies drogon_hist "$DROGON_HIST_DECK"

echo ""
echo "── Drogon PRED ─────────────────────────"
# PRED restarts from HIST step 82 and expects DROGON_HIST.UNRST next to the deck.
# Symlink the best available HIST output so PRED can find it.
DROGON_HIST_UNRST_LINK=$(dirname "$DROGON_PRED_DECK")/DROGON_HIST.UNRST
for hist_dir in "$OUTDIR/drogon_hist_flow_default" "$OUTDIR/drogon_hist_neural" "$OUTDIR/drogon_hist_neural_reuse"; do
    if [[ -f "$hist_dir/DROGON_HIST.UNRST" ]]; then
        ln -sf "$hist_dir/DROGON_HIST.UNRST" "$DROGON_HIST_UNRST_LINK"
        echo "  Linked HIST restart: $hist_dir/DROGON_HIST.UNRST"
        break
    fi
done
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$DROGON_PRED_DECK" "$OUTDIR/drogon_pred_flow_default"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
run_deck "neural-v3"    "$DROGON_PRED_DECK" "$OUTDIR/drogon_pred_neural"
run_deck "neural-reuse" "$DROGON_PRED_DECK" "$OUTDIR/drogon_pred_neural_reuse" \
    --cpr-reuse-setup=4 \
    --cpr-reuse-interval=10
unset OPM_NEURAL_CPR_WEIGHTS
run_new_policies drogon_pred "$DROGON_PRED_DECK"
rm -f "$DROGON_HIST_UNRST_LINK"

echo ""
echo "── SPE9 ────────────────────────────────"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$SPE9_DECK" "$OUTDIR/spe9_flow_default"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
run_deck "neural-v3"    "$SPE9_DECK" "$OUTDIR/spe9_neural"
unset OPM_NEURAL_CPR_WEIGHTS
run_new_policies spe9 "$SPE9_DECK"

echo ""
echo "── SPE10 model1 ─────────────────────────"
# SPE10model1 is a small 2-D deck; too many ranks cause solver failures.
# Cap with SPE10MODEL1_NPROCS (default 8) to stay within stable range.
_spe10m1_old_FLOW_CMD=("${FLOW_CMD[@]}")
SPE10MODEL1_NPROCS=${SPE10MODEL1_NPROCS:-8}
if (( SPE10MODEL1_NPROCS > NPROCS )); then SPE10MODEL1_NPROCS=$NPROCS; fi
set_flow_cmd "$SPE10MODEL1_NPROCS"
echo "  MPI ranks: $SPE10MODEL1_NPROCS"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$SPE10MODEL1_DECK" "$OUTDIR/spe10model1_flow_default"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
run_deck "neural-v3"    "$SPE10MODEL1_DECK" "$OUTDIR/spe10model1_neural"
unset OPM_NEURAL_CPR_WEIGHTS
run_new_policies spe10model1 "$SPE10MODEL1_DECK"
FLOW_CMD=("${_spe10m1_old_FLOW_CMD[@]}")

echo ""
echo "── SPE10 MOD01-01 ───────────────────────"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$SPE10MOD01_DECK" "$OUTDIR/spe10mod01_flow_default"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
run_deck "neural-v3"    "$SPE10MOD01_DECK" "$OUTDIR/spe10mod01_neural"
unset OPM_NEURAL_CPR_WEIGHTS
run_new_policies spe10mod01 "$SPE10MOD01_DECK"

# echo ""
# echo "── SPE10 model2 ─────────────────────────"
# unset OPM_NEURAL_CPR_WEIGHTS
# run_deck "flow_default" "$SPE10MODEL2_DECK" "$OUTDIR/spe10model2_flow_default"
# export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
# run_deck "neural-v3"    "$SPE10MODEL2_DECK" "$OUTDIR/spe10model2_neural"
# unset OPM_NEURAL_CPR_WEIGHTS

# echo ""
# echo "── Sleipner ─────────────────────────────"
# unset OPM_NEURAL_CPR_WEIGHTS
# run_deck "flow_default" "$SLEIPNER_DECK" "$OUTDIR/sleipner_flow_default"
# export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
# run_deck "neural-v3"    "$SLEIPNER_DECK" "$OUTDIR/sleipner_neural"
# unset OPM_NEURAL_CPR_WEIGHTS

echo ""
echo "── SPE11A (spe11pyopm) ─────────────────"
# SPE11 is sensitive to MPI rank counts; avoid too many ranks that lead to
# zero-cell processes. Temporarily reduce ranks for SPE11 runs if necessary.
old_FLOW_CMD=("${FLOW_CMD[@]}")
SPE11_NPROCS=${SPE11_NPROCS:-$NPROCS}
if (( SPE11_NPROCS > 48 )); then SPE11_NPROCS=48; fi
set_flow_cmd "$SPE11_NPROCS"
echo "  flow_default: no solver flags (cprw + trueimpes + paroverilu0 + ilu0-AMG, reuse-setup=4)"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$SPE11PYOPM_DECK" "$OUTDIR/spe11pyopm_flow_default"

# echo "  baseline fixed solver: $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
# run_deck "baseline"     "$SPE11PYOPM_DECK" "$OUTDIR/spe11pyopm_baseline" \
#     --linear-solver="$SPE11PYOPM_SOLVER" \
#     --cpr-reuse-setup=0

SPE11PYOPM_NEURAL_OUT="$OUTDIR/spe11pyopm_neural"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
export OPM_NEURAL_CPR_FORBID_LABELS="$SPE11_FORBID_LABELS"
echo "  neural base solver: $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels $SPE11_FORBID_LABELS forbidden"
run_deck "neural-v3"    "$SPE11PYOPM_DECK" "$SPE11PYOPM_NEURAL_OUT" \
    --linear-solver="$SPE11PYOPM_SOLVER" \
    --cpr-reuse-setup=0

best_reuse_setup=4
best_reuse_interval=10

if [[ $TUNE_SPE11_REUSE -eq 1 ]]; then
    echo "  neural-reuse tuning: grid search reuse-setup={2,4,8} × reuse-interval={5,10,20}"
    best_reuse_wall=""
    for rs in 2 4 8; do
        for ri in 5 10 20; do
            out="$OUTDIR/spe11pyopm_neural_reuse_s${rs}_i${ri}"
            run_deck "nr_s${rs}_i${ri}" "$SPE11PYOPM_DECK" "$out" \
                --linear-solver="$SPE11PYOPM_SOLVER" \
                --cpr-reuse-setup="$rs" \
                --cpr-reuse-interval="$ri"
            read -r _tn _tl tw < <(extract "$out/flow.log") || true
            if [[ "$tw" != "-" && "$tw" != "FAIL" ]]; then
                if [[ -z "$best_reuse_wall" ]] || awk "BEGIN { exit !($tw < $best_reuse_wall) }"; then
                    best_reuse_wall=$tw
                    best_reuse_setup=$rs
                    best_reuse_interval=$ri
                fi
            fi
        done
    done
    SPE11PYOPM_NEURAL_REUSE_OUT="$OUTDIR/spe11pyopm_neural_reuse_s${best_reuse_setup}_i${best_reuse_interval}"
    echo "  Best: reuse-setup=$best_reuse_setup reuse-interval=$best_reuse_interval (wall=${best_reuse_wall:-FAIL}s)"
else
    SPE11PYOPM_NEURAL_REUSE_OUT="$OUTDIR/spe11pyopm_neural_reuse"
    echo "  neural-reuse: same guard, with cpr-reuse-setup=${best_reuse_setup} and cpr-reuse-interval=${best_reuse_interval}"
    run_deck "neural-reuse" "$SPE11PYOPM_DECK" "$SPE11PYOPM_NEURAL_REUSE_OUT" \
        --linear-solver="$SPE11PYOPM_SOLVER" \
        --cpr-reuse-setup="$best_reuse_setup" \
        --cpr-reuse-interval="$best_reuse_interval"
fi
run_new_policies spe11pyopm "$SPE11PYOPM_DECK"

echo ""
echo "── SPE11B ──────────────────────────────"
# SPE11B is the 3-D CO2 benchmark with much larger per-rank messages than SPE11A (2-D).
# At high rank counts the shared-memory MPI transport exhausts /dev/shm (errno=EPERM).
# SPE11B_NPROCS caps the rank count; override with e.g. SPE11B_NPROCS=32 ./bench_neural_cpr.sh.
SPE11B_NPROCS=${SPE11B_NPROCS:-16}
if (( SPE11B_NPROCS > SPE11_NPROCS )); then SPE11B_NPROCS=$SPE11_NPROCS; fi
set_flow_cmd "$SPE11B_NPROCS"
echo "  flow_default: no solver flags (cprw + trueimpes + paroverilu0 + ilu0-AMG, reuse-setup=4)"
echo "  MPI ranks: $SPE11B_NPROCS (SPE11A used $SPE11_NPROCS)"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$SPE11B_DECK" "$OUTDIR/spe11b_flow_default"

echo "  baseline fixed solver: $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
run_deck "baseline"     "$SPE11B_DECK" "$OUTDIR/spe11b_baseline" \
    --linear-solver="$SPE11PYOPM_SOLVER" \
    --cpr-reuse-setup=0

SPE11B_NEURAL_OUT="$OUTDIR/spe11b_neural"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
# SPE11B neural-v3 crashes regardless of label when cpr-reuse-setup=0:
# the AMG is rebuilt every Newton step and becomes unstable late in the
# thermal CO2 simulation. Known limitation — neural-reuse works fine.
export OPM_NEURAL_CPR_FORBID_LABELS="$SPE11_FORBID_LABELS"
echo "  neural base solver: $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels $SPE11_FORBID_LABELS forbidden"
run_deck "neural-v3"    "$SPE11B_DECK" "$SPE11B_NEURAL_OUT" \
    --linear-solver="$SPE11PYOPM_SOLVER" \
    --cpr-reuse-setup=0

SPE11B_NEURAL_REUSE_OUT="$OUTDIR/spe11b_neural_reuse"
# Reuse: same forbid list (label_11 works fine with AMG reuse)
export OPM_NEURAL_CPR_FORBID_LABELS="$SPE11_FORBID_LABELS"
echo "  neural-reuse: same guard (labels $SPE11_FORBID_LABELS forbidden); cpr-reuse-setup=${best_reuse_setup} cpr-reuse-interval=${best_reuse_interval}"
run_deck "neural-reuse" "$SPE11B_DECK" "$SPE11B_NEURAL_REUSE_OUT" \
    --linear-solver="$SPE11PYOPM_SOLVER" \
    --cpr-reuse-setup="$best_reuse_setup" \
    --cpr-reuse-interval="$best_reuse_interval"
unset OPM_NEURAL_CPR_WEIGHTS
unset OPM_NEURAL_CPR_FORBID_LABELS
run_new_policies spe11b "$SPE11B_DECK"

# restore the previous FLOW_CMD (MPI ranks) for subsequent runs
FLOW_CMD=("${old_FLOW_CMD[@]}")

if [[ $BENCH_HELDOUT -eq 1 ]]; then
    echo ""
    echo "── Held-out generalization decks ───────"
    for heldout in $HELDOUT_CASES; do
        if ! heldout_case_info "$heldout"; then
            continue
        fi

        echo "  $HELDOUT_TITLE: $HELDOUT_DECK"
        unset OPM_NEURAL_CPR_WEIGHTS
        run_deck "${heldout}-flow_default" "$HELDOUT_DECK" "$OUTDIR/${HELDOUT_STEM}_flow_default"

        export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
        run_deck "${heldout}-neural-v3" "$HELDOUT_DECK" "$OUTDIR/${HELDOUT_STEM}_neural"
        run_deck "${heldout}-neural-reuse" "$HELDOUT_DECK" "$OUTDIR/${HELDOUT_STEM}_neural_reuse" \
            --cpr-reuse-setup=4 \
            --cpr-reuse-interval=10
        unset OPM_NEURAL_CPR_WEIGHTS
        run_new_policies "$HELDOUT_STEM" "$HELDOUT_DECK"
    done
fi

# ── results ──────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  RESULTS"
echo "========================================"

read -r nd_n nd_l nd_w < <(extract "$OUTDIR/norne_flow_default/flow.log") || true
read -r nn_n nn_l nn_w < <(extract "$OUTDIR/norne_neural/flow.log")        || true
mapfile -t new_rows < <(new_policy_rows norne)
print_table "Norne (NORNE_ATW2013)" \
    "flow_default" "${nd_n:--}" "${nd_l:--}" "${nd_w:--}" \
    "neural-v3"    "${nn_n:--}" "${nn_l:--}" "${nn_w:--}" \
    "${new_rows[@]}"
echo "  Policy check:"; policy_check "$OUTDIR/norne_neural/NORNE_ATW2013.DBG"
policy_summary "$OUTDIR/norne_neural/NORNE_ATW2013.DBG"
new_policy_check norne "$NORNE_DECK"

read -r sd_n sd_l sd_w < <(extract "$OUTDIR/spe9_flow_default/flow.log") || true
read -r sn_n sn_l sn_w < <(extract "$OUTDIR/spe9_neural/flow.log")       || true
mapfile -t new_rows < <(new_policy_rows spe9)
print_table "SPE9" \
    "flow_default" "${sd_n:--}" "${sd_l:--}" "${sd_w:--}" \
    "neural-v3"    "${sn_n:--}" "${sn_l:--}" "${sn_w:--}" \
    "${new_rows[@]}"
echo "  Policy check:"; policy_check "$OUTDIR/spe9_neural/SPE9.DBG"
policy_summary "$OUTDIR/spe9_neural/SPE9.DBG"
new_policy_check spe9 "$SPE9_DECK"

if [[ $TUNE_SPE11_REUSE -eq 1 ]]; then
    echo ""
    echo "  SPE11A reuse tuning grid (neural guard, labels 16-23 forbidden):"
    printf "  %-8s  %-10s  %8s  %8s  %8s  %10s\n" "setup" "interval" "Newton" "Linear" "Wall(s)" "Δ wall"
    printf "  %s\n" "──────────────────────────────────────────────────────────────"
    ref_tune_w=""
    for rs in 2 4 8; do
        for ri in 5 10 20; do
            out="$OUTDIR/spe11pyopm_neural_reuse_s${rs}_i${ri}"
            read -r tn tl tw < <(extract "$out/flow.log") || true
            [[ -z "$ref_tune_w" && "$tw" != "-" && "$tw" != "FAIL" ]] && ref_tune_w=$tw
            marker=""; [[ $rs -eq $best_reuse_setup && $ri -eq $best_reuse_interval ]] && marker=" ← best"
            printf "  %-8s  %-10s  %8s  %8s  %8s  %10s%s\n" \
                "$rs" "$ri" "${tn:--}" "${tl:--}" "${tw:--}" "$(pct_delta "$ref_tune_w" "$tw")" "$marker"
        done
    done
    echo ""
fi

read -r s10d_n s10d_l s10d_w < <(extract "$OUTDIR/spe10model1_flow_default/flow.log") || true
read -r s10n_n s10n_l s10n_w < <(extract "$OUTDIR/spe10model1_neural/flow.log")       || true
mapfile -t new_rows < <(new_policy_rows spe10model1)
print_table "SPE10 model1 (SPE10_MODEL1)" \
    "flow_default" "${s10d_n:--}" "${s10d_l:--}" "${s10d_w:--}" \
    "neural-v3"    "${s10n_n:--}" "${s10n_l:--}" "${s10n_w:--}" \
    "${new_rows[@]}"
echo "  Policy check:"; policy_check "$OUTDIR/spe10model1_neural/SPE10_MODEL1.DBG"
policy_summary "$OUTDIR/spe10model1_neural/SPE10_MODEL1.DBG"
new_policy_check spe10model1 "$SPE10MODEL1_DECK"

read -r s10m01d_n s10m01d_l s10m01d_w < <(extract "$OUTDIR/spe10mod01_flow_default/flow.log") || true
read -r s10m01n_n s10m01n_l s10m01n_w < <(extract "$OUTDIR/spe10mod01_neural/flow.log")       || true
mapfile -t new_rows < <(new_policy_rows spe10mod01)
print_table "SPE10 MOD01-01 (SPE10-MOD01-01)" \
    "flow_default" "${s10m01d_n:--}" "${s10m01d_l:--}" "${s10m01d_w:--}" \
    "neural-v3"    "${s10m01n_n:--}" "${s10m01n_l:--}" "${s10m01n_w:--}" \
    "${new_rows[@]}"
echo "  Policy check:"; policy_check "$OUTDIR/spe10mod01_neural/SPE10-MOD01-01.DBG"
policy_summary "$OUTDIR/spe10mod01_neural/SPE10-MOD01-01.DBG"
new_policy_check spe10mod01 "$SPE10MOD01_DECK"

# read -r s10m2d_n s10m2d_l s10m2d_w < <(extract "$OUTDIR/spe10model2_flow_default/flow.log") || true
# read -r s10m2n_n s10m2n_l s10m2n_w < <(extract "$OUTDIR/spe10model2_neural/flow.log")       || true
# print_table "SPE10 model2 (SPE10_MODEL2)" \
#     "flow_default" "${s10m2d_n:--}" "${s10m2d_l:--}" "${s10m2d_w:--}" \
#     "neural-v3"    "${s10m2n_n:--}" "${s10m2n_l:--}" "${s10m2n_w:--}"
# echo "  Policy check:"; policy_check "$OUTDIR/spe10model2_neural/SPE10_MODEL2.DBG"
# policy_summary "$OUTDIR/spe10model2_neural/SPE10_MODEL2.DBG"

# read -r slpd_n slpd_l slpd_w < <(extract "$OUTDIR/sleipner_flow_default/flow.log") || true
# read -r slpn_n slpn_l slpn_w < <(extract "$OUTDIR/sleipner_neural/flow.log")       || true
# print_table "Sleipner (SLEIPNER_ORG)" \
#     "flow_default" "${slpd_n:--}" "${slpd_l:--}" "${slpd_w:--}" \
#     "neural-v3"    "${slpn_n:--}" "${slpn_l:--}" "${slpn_w:--}"
# echo "  Policy check:"; policy_check "$OUTDIR/sleipner_neural/SLEIPNER_ORG.DBG"
# policy_summary "$OUTDIR/sleipner_neural/SLEIPNER_ORG.DBG"

read -r nmwd_n nmwd_l nmwd_w < <(extract "$OUTDIR/norne_msw_flow_default/flow.log") || true
read -r nmwn_n nmwn_l nmwn_w < <(extract "$OUTDIR/norne_msw_neural/flow.log")       || true
read -r nmwr_n nmwr_l nmwr_w < <(extract "$OUTDIR/norne_msw_neural_reuse/flow.log") || true
mapfile -t new_rows < <(new_policy_rows norne_msw)
print_table "Norne MSW 1A (NORNE_ATW2013_1A_MSW)" \
    "flow_default" "${nmwd_n:--}" "${nmwd_l:--}" "${nmwd_w:--}" \
    "neural-v3"    "${nmwn_n:--}" "${nmwn_l:--}" "${nmwn_w:--}" \
    "neural-reuse" "${nmwr_n:--}" "${nmwr_l:--}" "${nmwr_w:--}" \
    "${new_rows[@]}"
echo "  Policy check:"; policy_check "$OUTDIR/norne_msw_neural/NORNE_ATW2013_1A_MSW.DBG"
policy_summary "$OUTDIR/norne_msw_neural/NORNE_ATW2013_1A_MSW.DBG"
new_policy_check norne_msw "$NORNE_MSW_DECK"

read -r dghd_n dghd_l dghd_w < <(extract "$OUTDIR/drogon_hist_flow_default/flow.log") || true
read -r dghn_n dghn_l dghn_w < <(extract "$OUTDIR/drogon_hist_neural/flow.log")        || true
read -r dghr_n dghr_l dghr_w < <(extract "$OUTDIR/drogon_hist_neural_reuse/flow.log")  || true
mapfile -t new_rows < <(new_policy_rows drogon_hist)
print_table "Drogon HIST (DROGON_HIST)" \
    "flow_default" "${dghd_n:--}" "${dghd_l:--}" "${dghd_w:--}" \
    "neural-v3"    "${dghn_n:--}" "${dghn_l:--}" "${dghn_w:--}" \
    "neural-reuse" "${dghr_n:--}" "${dghr_l:--}" "${dghr_w:--}" \
    "${new_rows[@]}"
echo "  Policy check:"; policy_check "$OUTDIR/drogon_hist_neural/DROGON_HIST.DBG"
policy_summary "$OUTDIR/drogon_hist_neural/DROGON_HIST.DBG"
new_policy_check drogon_hist "$DROGON_HIST_DECK"

read -r dgpd_n dgpd_l dgpd_w < <(extract "$OUTDIR/drogon_pred_flow_default/flow.log") || true
read -r dgpn_n dgpn_l dgpn_w < <(extract "$OUTDIR/drogon_pred_neural/flow.log")        || true
read -r dgpr_n dgpr_l dgpr_w < <(extract "$OUTDIR/drogon_pred_neural_reuse/flow.log")  || true
mapfile -t new_rows < <(new_policy_rows drogon_pred)
print_table "Drogon PRED (DROGON_PRED)" \
    "flow_default" "${dgpd_n:--}" "${dgpd_l:--}" "${dgpd_w:--}" \
    "neural-v3"    "${dgpn_n:--}" "${dgpn_l:--}" "${dgpn_w:--}" \
    "neural-reuse" "${dgpr_n:--}" "${dgpr_l:--}" "${dgpr_w:--}" \
    "${new_rows[@]}"
echo "  Policy check:"; policy_check "$OUTDIR/drogon_pred_neural/DROGON_PRED.DBG"
policy_summary "$OUTDIR/drogon_pred_neural/DROGON_PRED.DBG"
new_policy_check drogon_pred "$DROGON_PRED_DECK"

read -r spd_n spd_l spd_w < <(extract "$OUTDIR/spe11pyopm_flow_default/flow.log") || true
read -r spb_n spb_l spb_w < <(extract "$OUTDIR/spe11pyopm_baseline/flow.log")     || true
read -r spn_n spn_l spn_w < <(extract "$SPE11PYOPM_NEURAL_OUT/flow.log")           || true
read -r spr_n spr_l spr_w < <(extract "$SPE11PYOPM_NEURAL_REUSE_OUT/flow.log")    || true
mapfile -t new_rows < <(new_policy_rows spe11pyopm)
print_table "SPE11A (spe11pyopm)" \
    "flow_default" "${spd_n:--}" "${spd_l:--}" "${spd_w:--}" \
    "baseline"     "${spb_n:--}" "${spb_l:--}" "${spb_w:--}" \
    "neural-v3"    "${spn_n:--}" "${spn_l:--}" "${spn_w:--}" \
    "neural-reuse" "${spr_n:--}" "${spr_l:--}" "${spr_w:--}" \
    "${new_rows[@]}"
echo "  Labels:"
echo "    flow_default : $(label_desc none)"
echo "    baseline     : $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
echo "    neural       : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels $SPE11_FORBID_LABELS forbidden"
echo "    neural-reuse : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels $SPE11_FORBID_LABELS forbidden; reuse-setup=${best_reuse_setup}; reuse-interval=${best_reuse_interval}"
echo "  Policy check:"; policy_check "$SPE11PYOPM_NEURAL_OUT/SPE11A.DBG"
policy_summary "$SPE11PYOPM_NEURAL_OUT/SPE11A.DBG"
echo "  Reuse policy check:"; policy_check "$SPE11PYOPM_NEURAL_REUSE_OUT/SPE11A.DBG"
policy_summary "$SPE11PYOPM_NEURAL_REUSE_OUT/SPE11A.DBG"
new_policy_check spe11pyopm "$SPE11PYOPM_DECK"

read -r spbd_n spbd_l spbd_w < <(extract "$OUTDIR/spe11b_flow_default/flow.log") || true
read -r spbb_n spbb_l spbb_w < <(extract "$OUTDIR/spe11b_baseline/flow.log")     || true
read -r spbn_n spbn_l spbn_w < <(extract "$SPE11B_NEURAL_OUT/flow.log")           || true
read -r spbr_n spbr_l spbr_w < <(extract "$SPE11B_NEURAL_REUSE_OUT/flow.log")    || true
mapfile -t new_rows < <(new_policy_rows spe11b)
print_table "SPE11B" \
    "flow_default" "${spbd_n:--}" "${spbd_l:--}" "${spbd_w:--}" \
    "baseline"     "${spbb_n:--}" "${spbb_l:--}" "${spbb_w:--}" \
    "neural-v3"    "${spbn_n:--}" "${spbn_l:--}" "${spbn_w:--}" \
    "neural-reuse" "${spbr_n:--}" "${spbr_l:--}" "${spbr_w:--}" \
    "${new_rows[@]}"
echo "  Labels:"
echo "    flow_default : $(label_desc none)"
echo "    baseline     : $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
echo "    neural       : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels $SPE11_FORBID_LABELS forbidden"
echo "    neural-reuse : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels $SPE11_FORBID_LABELS forbidden; reuse-setup=${best_reuse_setup}; reuse-interval=${best_reuse_interval}"
echo "  Policy check:"; policy_check "$SPE11B_NEURAL_OUT/SPE11B.DBG"
policy_summary "$SPE11B_NEURAL_OUT/SPE11B.DBG"
echo "  Reuse policy check:"; policy_check "$SPE11B_NEURAL_REUSE_OUT/SPE11B.DBG"
policy_summary "$SPE11B_NEURAL_REUSE_OUT/SPE11B.DBG"
new_policy_check spe11b "$SPE11B_DECK"

if [[ $BENCH_HELDOUT -eq 1 ]]; then
    for heldout in $HELDOUT_CASES; do
        if ! heldout_case_info "$heldout"; then
            continue
        fi

        read -r hb_n hb_l hb_w < <(extract "$OUTDIR/${HELDOUT_STEM}_flow_default/flow.log")  || true
        read -r hn_n hn_l hn_w < <(extract "$OUTDIR/${HELDOUT_STEM}_neural/flow.log")        || true
        read -r hr_n hr_l hr_w < <(extract "$OUTDIR/${HELDOUT_STEM}_neural_reuse/flow.log")  || true
        mapfile -t new_rows < <(new_policy_rows "$HELDOUT_STEM")
        print_table "$HELDOUT_TITLE — held-out generalization" \
            "flow_default" "${hb_n:--}" "${hb_l:--}" "${hb_w:--}" \
            "neural-v3"    "${hn_n:--}" "${hn_l:--}" "${hn_w:--}" \
            "neural-reuse" "${hr_n:--}" "${hr_l:--}" "${hr_w:--}" \
            "${new_rows[@]}"

        dbg="$OUTDIR/${HELDOUT_STEM}_neural/$(basename "$HELDOUT_DECK" .DATA).DBG"
        echo "  Policy check:"; policy_check "$dbg"
        policy_summary "$dbg"
        dbg="$OUTDIR/${HELDOUT_STEM}_neural_reuse/$(basename "$HELDOUT_DECK" .DATA).DBG"
        echo "  Reuse policy check:"; policy_check "$dbg"
        policy_summary "$dbg"
        new_policy_check "$HELDOUT_STEM" "$HELDOUT_DECK"
    done
fi
