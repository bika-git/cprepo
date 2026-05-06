#!/usr/bin/env bash
# bench_norne.sh — baseline vs v3 neural CPR benchmark for Norne, SPE9, and SPE11A
#
# ── How sweeps are generated ─────────────────────────────────────────────────
#
#   A "sweep" runs the full 3×2×2×2 = 24 CPR configurations against one or
#   more reservoir decks and records per-Newton-step solver features + timing
#   into one CSV per (deck, config) pair.
#
#   The tool is doc/neural_cpr/run_sweep.py.  Example invocation:
#
#     python3 run_sweep.py \
#         --flow  /path/to/flow \
#         --decks /path/to/SPE9.DATA /path/to/NORNE_ATW2013.DATA /path/to/SPE11A.DATA \
#         --out   $SCRIPT_DIR/cpr_sweep_data/sweep2
#
#   This runs flow once per (deck, config) in parallel (up to nCPU
#   workers).  Each run sets OPM_CPR_SWEEP_LOG so that OPM writes a CSV of
#   (config, episode, newton_iter, total_ms, features…) for every Newton step.
#   Output: <out>/<DeckStem>_<label>.csv  e.g. SPE9_07.csv, NORNE_ATW2013_23.csv
#
#   The sweep is then labelled with collect_sweep_labels.py:
#
#     python3 collect_sweep_labels.py \
#         --csvs /cpr_sweep_data/sweep2/*.csv /cpr_sweep_data/sweep3/*.csv \
#         --out  /tmp/training_data.csv
#
#   For each (deck, episode, newton_iter), collect_sweep_labels.py finds which
#   of the 24 configs had the lowest median total_ms and assigns that as the
#   label.  --min-configs 2 skips Newton steps where fewer than 2 configs
#   converged (too noisy to label reliably).
#
# ── Sweep directories (v3 training data) ─────────────────────────────────────
#
#   Both sweep2 and sweep3 ran the exact same decks (SPE9 + Norne) with the
#   exact same 24 configs.  They differ only in that they are independent
#   repeated runs:
#
#     sweep2  (run 1)  →  48 CSVs   SPE9_00-23  NORNE_ATW2013_00-23
#     sweep3  (run 2)  →  48 CSVs   SPE9_00-23  NORNE_ATW2013_00-23
#
#   Having two runs serves two purposes:
#     1. Timing noise reduction — wall-clock measurements on a workstation
#        have ~6% run-to-run variance.  collect_sweep_labels.py takes the
#        median across all matching (deck, episode, newton_iter) rows, so
#        labels based on both runs are more stable than labels from a single run.
#     2. More training samples — each run may cover slightly different Newton
#        steps (timestep chopping varies by config), giving the model a wider
#        view of the convergence trajectory.
#
#   CSV location:
#     $SCRIPT_DIR/cpr_sweep_data/sweep2/*.csv
#     $SCRIPT_DIR/cpr_sweep_data/sweep3/*.csv
#     $SCRIPT_DIR/cpr_sweep_data/sweep_spe11/*.csv
#
# ── Model summary ────────────────────────────────────────────────────────────
#
#   Model        : cpr_policy_retrained.model
#   Architecture : 19→32→24, ReLU, AdamW
#   Samples      : sweep2+3+spe11
#   Features     : 19 (added num_phases, dt_cut_count, prev2_linsolver_iters,
#                      condition_number_estimate, bhp_well_fraction)
#   Expected gain: SPE9 ~−5%  Norne ~−6% wall-time vs baseline
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FLOW=$SCRIPT_DIR/build/opm-simulators/bin/flow
SCRIPTS=$SCRIPT_DIR/opm-simulators/doc/neural_cpr

# Set NPROCS>1 to run flow under MPI: NPROCS=4 ./bench_norne.sh
NPROCS=${NPROCS:-1}
if (( NPROCS > 1 )); then
    FLOW_CMD=(mpirun -np "$NPROCS" "$FLOW")
else
    FLOW_CMD=("$FLOW")
fi

# # Sweep CSVs used as training input
# #   sweep2/3 = 2 runs of SPE9+Norne
# #   sweep_spe11 = SPE11A+SPE11B sweep data
# #   sweep5   = 3rd run of SPE9+Norne + well_density + nl_residual_trend (14 feat)
# #   sweep7   = 4th run of SPE9+Norne with all 19 features (new binary)
# SWEEP_CSVS=(
#     $SCRIPT_DIR/cpr_sweep_data/sweep2/*.csv
#     $SCRIPT_DIR/cpr_sweep_data/sweep3/*.csv
#     $SCRIPT_DIR/cpr_sweep_data/sweep_spe11/*.csv
#     # $SCRIPT_DIR/cpr_sweep_data/sweep5/*.csv
#     # $SCRIPT_DIR/cpr_sweep_data/sweep7/*.csv
# )

# Intermediate and output files for the training pipeline
TRAINING_CSV=$SCRIPT_DIR/tmp/cpr_training_data.csv
MODEL=$SCRIPT_DIR/tmp/cpr_policy_retrained.model

NORNE_DECK=$SCRIPT_DIR/opm-tests/norne/NORNE_ATW2013.DATA
SPE9_DECK=$SCRIPT_DIR/opm-tests/spe9/SPE9.DATA
SPE11PYOPM_DECK=$SCRIPT_DIR/opm-tests/spe11pyopm/SPE11A.DATA
SPE11PYOPM_CONFIGS_DIR=$SCRIPT_DIR/cpr_sweep_data/sweep_spe11/configs
# SPE11PYOPM_SOLVER_LABEL and SPE11PYOPM_SOLVER are derived from sweep data after Step 1
SPE11PYOPM_AVOID_LABEL23=${SPE11PYOPM_AVOID_LABEL23:-1}
# SPE11PYOPM_NEWTONMAXIT=50
# SPE11PYOPM_FLAGS=(
#     --relaxed-max-pv-fraction=0
#     --output-extra-convergence-info=steps,iterations
#     --enable-ecl-output=0
#     --full-time-step-initially=1
#     --tolerance-cnv-relaxed=1e-2
#     --tolerance-cnv=1e-2
#     --tolerance-mb=1e-5
#     --tolerance-mb-relaxed=1e-5
#     --newton-min-iterations=1
#     --newton-max-iterations="$SPE11PYOPM_NEWTONMAXIT"
# )
# SPE11PYOPM_ADAPT_FLAGS=(
#     "${SPE11PYOPM_FLAGS[@]}"
# )

OUTDIR=$SCRIPT_DIR/tmp/bench_neural_cpr

# ── helpers ─────────────────────────────────────────────────────────────────

run_deck() {
    local tag=$1 deck=$2 outdir=$3
    shift 3
    echo "  → $tag"
    mkdir -p "$outdir"
    local status=0
    if "${FLOW_CMD[@]}" --output-dir="$outdir" "$@" "$deck" >"$outdir/flow.log" 2>&1; then
        status=0
    else
        status=$?
    fi
    echo "$status" >"$outdir/exit_code"
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
            local ct;  [[ $coarse -eq 1 ]] && ct="dilu"  || ct="paroverilu0"
            echo "label_$(printf '%02d' $n) = ${wt} + ${cpt} + ${ft} + ${ct}"
            ;;
    esac
}

policy_label_from_line() {
    local line=$1
    local dec=0 cprw=0 fine=0 coarse=0

    if [[ "$line" == *"weight=trueimpesanalytic"* ]]; then
        dec=2
    elif [[ "$line" == *"weight=trueimpes"* ]]; then
        dec=1
    fi

    [[ "$line" == *"cprw weight="* ]] && cprw=1
    [[ "$line" == *"fine=dilu"* ]] && fine=1
    [[ "$line" == *"coarse=dilu"* ]] && coarse=1

    printf "%02d" "$((dec * 8 + cprw * 4 + fine * 2 + coarse))"
}

label23_seen() {
    local outdir=$1
    grep -Eq "NeuralCprPolicy: cprw weight=trueimpesanalytic fine=dilu coarse=dilu|Canonical phase 1 is not active|getTrueImpesAnalyticWeights" \
        "$outdir/flow.log" "$outdir"/*.DBG 2>/dev/null
}

extract() {
    local log=$1
    local outdir exit_code
    outdir=$(dirname "$log")
    exit_code=$(cat "$outdir/exit_code" 2>/dev/null || echo 0)
    if [[ "$exit_code" != "0" ]] || grep -Eq "Assertion failed|Aborted|Segmentation fault" "$log" 2>/dev/null; then
        echo "FAIL FAIL FAIL"
        return
    fi

    local newton linear wall
    newton=$(grep "Overall Newton Iterations" "$log" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    linear=$(grep "Overall Linear Iterations" "$log" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    wall=$(grep "Simulation time:" "$log" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "${newton:--} ${linear:--} ${wall:--}"
}

pct_delta() {
    # pct_delta base new  →  prints signed % (negative = improvement)
    local base=$1 new=$2
    if [[ "$base" == "-" || "$new" == "-" || "$base" == "FAIL" || "$new" == "FAIL" ]]; then echo "n/a"; return; fi
    awk "BEGIN { printf \"%+.1f%%\", ($new - $base) / $base * 100 }"
}

print_table() {
    # Usage: print_table "deck label" tag1 n1 l1 w1  [tag2 n2 l2 w2 ...]
    # First row is the reference (Δ = —); subsequent rows show Δ vs first row.
    local deck_label=$1; shift
    local ref_w first=1

    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    printf "│  %-59s│\n" "$deck_label"
    echo "├──────────────┬──────────┬──────────┬──────────┬────────────┤"
    printf "│ %-12s │ %8s │ %8s │ %8s │ %10s │\n" "Run" "Newton" "Linear" "Wall (s)" "Δ wall"
    echo "├──────────────┼──────────┼──────────┼──────────┼────────────┤"
    while [[ $# -ge 4 ]]; do
        local tag=$1 rn=$2 rl=$3 rw=$4; shift 4
        if [[ $first -eq 1 ]]; then
            ref_w=$rw; first=0
            printf "│ %-12s │ %8s │ %8s │ %8s │ %10s │\n" "$tag" "$rn" "$rl" "$rw" "—"
        else
            printf "│ %-12s │ %8s │ %8s │ %8s │ %10s │\n" "$tag" "$rn" "$rl" "$rw" "$(pct_delta "$ref_w" "$rw")"
        fi
    done
    echo "└──────────────┴──────────┴──────────┴──────────┴────────────┘"
}

policy_check() {
    local dbg=$1
    local line
    line=$(grep "NeuralCprPolicy: .*weight=" "$dbg" 2>/dev/null | tail -1 || true)
    if [[ -z "$line" ]]; then
        line=$(grep "NeuralCprPolicy" "$dbg" 2>/dev/null | head -1 || true)
    fi
    if [[ -z "$line" ]]; then
        echo "  [!] No NeuralCprPolicy lines in DBG — model may not have loaded"
    elif [[ "$line" == *"weight="* ]]; then
        echo "  [✓] selected label_$(policy_label_from_line "$line"): $line"
    else
        echo "  [✓] $line"
    fi
}

policy_summary() {
    local dbg=$1
    if [[ ! -f "$dbg" ]]; then
        echo "  Policy summary: DBG missing"
        return
    fi

    awk '
        /NeuralCprPolicy: .*weight=/ {
            dec = 0; cprw = 0; fine = 0; coarse = 0
            if ($0 ~ /weight=trueimpesanalytic/) dec = 2
            else if ($0 ~ /weight=trueimpes/) dec = 1
            if ($0 ~ /cprw weight=/) cprw = 1
            if ($0 ~ /fine=dilu/) fine = 1
            if ($0 ~ /coarse=dilu/) coarse = 1
            label = dec * 8 + cprw * 4 + fine * 2 + coarse
            labels[++n] = label
            counts[label]++
            if (seen[label]++ == 0) unique++
            if (n > 1 && label != prev) switches++
            prev = label
        }
        END {
            if (n == 0) {
                print "  Policy summary: no labeled decisions found"
                exit
            }
            printf "  Policy summary: %d decisions, %d unique labels, %d switches\n", n, unique, switches
            for (rank = 1; rank <= 5; ++rank) {
                best = -1
                best_count = -1
                for (label in counts) {
                    if (!printed[label] && counts[label] > best_count) {
                        best = label
                        best_count = counts[label]
                    }
                }
                if (best < 0) break
                printed[best] = 1
                printf "    label_%02d: %d decisions (%.1f%%)\n", best, best_count, 100.0 * best_count / n
            }
        }
    ' "$dbg"
}

# # ── training pipeline ────────────────────────────────────────────────────────
# #
# #   Step 1 — collect_sweep_labels.py
# #     Reads all sweep CSVs, groups rows by (deck, episode, newton_iter),
# #     picks the config with the lowest median total_ms as the label,
# #     and writes one row per labeled Newton step to TRAINING_CSV.
# #
# #   Step 2 — architecture search
# #     Trains several hidden-layer widths and picks the one with the best
# #     val accuracy.  All candidates share the same seed so the train/val
# #     split is identical and the comparison is fair.
# #     Candidates: 14→8→24  14→12→24  14→16→24  14→24→24  14→32→24
# #
# #   Step 3 — benchmark with the winning architecture.

# echo "========================================"
# echo "  Step 1 — label sweep data"
# echo "  Input CSVs: sweep2/*.csv + sweep3/*.csv + sweep_spe11/*.csv"
# echo "  Output    : $TRAINING_CSV"
# echo "========================================"
# python3 "$SCRIPTS/collect_sweep_labels.py" \
#     --csvs "${SWEEP_CSVS[@]}" \
#     --out  "$TRAINING_CSV" \
#     --min-configs 2

# Auto-select best SPE11 baseline from training data.
# Picks the label that wins the most Newton steps among safe labels (0-15);
# falls back to 0-23 if no safe label is present, or to 09 on error.
SPE11PYOPM_SOLVER_LABEL=$(python3 -c "
import csv, collections, sys
try:
    rows = [r for r in csv.DictReader(open('$TRAINING_CSV')) if 'spe11' in r.get('deck','').lower()]
    safe = [r['label'] for r in rows if int(r['label']) < 16]
    pool = safe if safe else [r['label'] for r in rows]
    best = collections.Counter(pool).most_common(1)[0][0] if pool else '9'
    print(f'{int(best):02d}')
except Exception as e:
    sys.stderr.write(f'Warning: SPE11 label auto-select failed: {e}\n')
    print('09')
" 2>/dev/null || echo "09")
SPE11PYOPM_SOLVER="$SPE11PYOPM_CONFIGS_DIR/label_${SPE11PYOPM_SOLVER_LABEL}.json"
echo "  Auto-selected SPE11 baseline: label_${SPE11PYOPM_SOLVER_LABEL} ($(label_desc "$SPE11PYOPM_SOLVER_LABEL"))"

echo ""
echo "========================================"
echo "  Step 2 — architecture search"
echo "  Candidates: hidden = 8, 12, 16, 24, 32"
echo "========================================"

best_hidden=32
best_val=0.0

n_features=$(python3 -c "import csv; r=next(csv.DictReader(open('$TRAINING_CSV'))); print(len([k for k in r if k not in ('deck','best_config','label')]))" 2>/dev/null || echo 19)
echo "  Input features detected: ${n_features}"

# Train all candidates in parallel — they share the same read-only CSV and
# use different output files, so there is no contention.
for hidden in 8 12 16 24 32; do
    (
        python3 "$SCRIPTS/train_cpr_policy.py" \
            --data       "$TRAINING_CSV" \
            --hidden     "$hidden" \
            --activation relu \
            --epochs     200 \
            --lr         0.001 \
            --out        "/tmp/cpr_arch_h${hidden}.model" \
            --format     binary \
            --seed       42 2>&1 \
        | grep "^Final" | grep -oE 'val acc: [0-9.]+' | grep -oE '[0-9.]+' \
        > "/tmp/cpr_arch_h${hidden}.acc"
    ) &
done
wait

for hidden in 8 12 16 24 32; do
    val_acc=$(cat "/tmp/cpr_arch_h${hidden}.acc" 2>/dev/null || true)
    printf "  %s→%d→24  val_acc=%s\n" "$n_features" "$hidden" "${val_acc:--}"
    if awk "BEGIN { exit !(${val_acc:-0} > $best_val) }"; then
        best_val=${val_acc:-0}
        best_hidden=$hidden
        cp "/tmp/cpr_arch_h${hidden}.model" "$MODEL"
    fi
done

echo ""
echo "  Best arch : ${n_features}→${best_hidden}→24  (val_acc=${best_val})"
echo "  Model written to: $MODEL"

# ── runs ────────────────────────────────────────────────────────────────────

echo "========================================"
echo "  Step 3 — benchmark  (flow_default vs best arch: ${n_features}→${best_hidden}→24)"
echo "  Model: $MODEL"
echo "  Run labels:"
echo "    Norne/SPE9 flow_default : $(label_desc none)"
echo "    Norne/SPE9 neural       : $(label_desc ml)"
echo "    SPE11 flow_default      : $(label_desc none)"
echo "    SPE11 baseline          : $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
echo "    SPE11 neural            : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels 16-23 forbidden (picks next best)"
echo "    SPE11 neural-reuse      : same neural guard; cpr-reuse-setup=4, cpr-reuse-interval=10"
echo "========================================"

echo ""
echo "── Norne ───────────────────────────────"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$NORNE_DECK" "$OUTDIR/norne_flow_default"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
run_deck "neural-v3"    "$NORNE_DECK" "$OUTDIR/norne_neural"
unset OPM_NEURAL_CPR_WEIGHTS

echo ""
echo "── SPE9 ────────────────────────────────"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$SPE9_DECK" "$OUTDIR/spe9_flow_default"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
run_deck "neural-v3"    "$SPE9_DECK" "$OUTDIR/spe9_neural"
unset OPM_NEURAL_CPR_WEIGHTS

echo ""
echo "── SPE11A (spe11pyopm) ─────────────────"
echo "  flow_default: no solver flags (cprw + trueimpes + paroverilu0 + ilu0-AMG, reuse-setup=4)"
unset OPM_NEURAL_CPR_WEIGHTS
run_deck "flow_default" "$SPE11PYOPM_DECK" "$OUTDIR/spe11pyopm_flow_default"

echo "  baseline fixed solver: $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
run_deck "baseline"     "$SPE11PYOPM_DECK" "$OUTDIR/spe11pyopm_baseline" \
    --linear-solver="$SPE11PYOPM_SOLVER" \
    --cpr-reuse-setup=0

SPE11PYOPM_NEURAL_OUT="$OUTDIR/spe11pyopm_neural"
export OPM_NEURAL_CPR_WEIGHTS="$MODEL"
export OPM_NEURAL_CPR_FORBID_LABELS=16,17,18,19,20,21,22,23
echo "  neural base solver: $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels 16-23 forbidden — policy picks next best label"
run_deck "neural-v3"    "$SPE11PYOPM_DECK" "$SPE11PYOPM_NEURAL_OUT" \
    --linear-solver="$SPE11PYOPM_SOLVER" \
    --cpr-reuse-setup=0

SPE11PYOPM_NEURAL_REUSE_OUT="$OUTDIR/spe11pyopm_neural_reuse"
echo "  neural-reuse: same guard, with cpr-reuse-setup=4 and cpr-reuse-interval=10"
run_deck "neural-reuse" "$SPE11PYOPM_DECK" "$SPE11PYOPM_NEURAL_REUSE_OUT" \
    --linear-solver="$SPE11PYOPM_SOLVER" \
    --cpr-reuse-setup=4 \
    --cpr-reuse-interval=10
unset OPM_NEURAL_CPR_WEIGHTS
unset OPM_NEURAL_CPR_FORBID_LABELS

# ── results ─────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  RESULTS"
echo "========================================"

read -r nd_n nd_l nd_w < <(extract "$OUTDIR/norne_flow_default/flow.log") || true
read -r nn_n nn_l nn_w < <(extract "$OUTDIR/norne_neural/flow.log") || true
print_table "Norne (NORNE_ATW2013) — in training set" \
    "flow_default" "${nd_n:--}" "${nd_l:--}" "${nd_w:--}" \
    "neural-v3"    "${nn_n:--}" "${nn_l:--}" "${nn_w:--}"
echo "  Policy check:"; policy_check "$OUTDIR/norne_neural/NORNE_ATW2013.DBG"

read -r sd_n sd_l sd_w < <(extract "$OUTDIR/spe9_flow_default/flow.log") || true
read -r sn_n sn_l sn_w < <(extract "$OUTDIR/spe9_neural/flow.log") || true
print_table "SPE9 — in training set" \
    "flow_default" "${sd_n:--}" "${sd_l:--}" "${sd_w:--}" \
    "neural-v3"    "${sn_n:--}" "${sn_l:--}" "${sn_w:--}"
echo "  Policy check:"; policy_check "$OUTDIR/spe9_neural/SPE9.DBG"

read -r spd_n spd_l spd_w < <(extract "$OUTDIR/spe11pyopm_flow_default/flow.log") || true
read -r spb_n spb_l spb_w < <(extract "$OUTDIR/spe11pyopm_baseline/flow.log") || true
read -r spn_n spn_l spn_w < <(extract "$SPE11PYOPM_NEURAL_OUT/flow.log") || true
read -r spr_n spr_l spr_w < <(extract "$SPE11PYOPM_NEURAL_REUSE_OUT/flow.log") || true
print_table "SPE11A (spe11pyopm) — in training set" \
    "flow_default" "${spd_n:--}" "${spd_l:--}" "${spd_w:--}" \
    "baseline"     "${spb_n:--}" "${spb_l:--}" "${spb_w:--}" \
    "neural-v3"    "${spn_n:--}" "${spn_l:--}" "${spn_w:--}" \
    "neural-reuse" "${spr_n:--}" "${spr_l:--}" "${spr_w:--}"
echo "  Labels:"
echo "    flow_default : $(label_desc none)"
echo "    baseline     : $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
echo "    neural       : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels 16-23 forbidden"
echo "    neural-reuse : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels 16-23 forbidden; cpr-reuse-setup=4; cpr-reuse-interval=10"
echo "  Policy check:"; policy_check "$SPE11PYOPM_NEURAL_OUT/SPE11A.DBG"
policy_summary "$SPE11PYOPM_NEURAL_OUT/SPE11A.DBG"
echo "  Reuse policy check:"; policy_check "$SPE11PYOPM_NEURAL_REUSE_OUT/SPE11A.DBG"
policy_summary "$SPE11PYOPM_NEURAL_REUSE_OUT/SPE11A.DBG"
