#!/usr/bin/env bash
# bench_norne.sh — full pipeline: sweep → label → train → benchmark
#
# ── Pipeline overview ─────────────────────────────────────────────────────────
#
#   Step 0 — regenerate sweep data
#     Runs all 24 CPR configurations against each deck twice (run1, run2) for
#     Norne + SPE9 (noise reduction via median), and once for SPE11A.
#     Each deck×config pair writes one CSV of per-Newton-step features + timing.
#     Tool: opm-simulators/doc/neural_cpr/run_sweep.py
#
#   Step 1 — label sweep data
#     Groups rows by (deck, episode, newton_iter), picks the config with the
#     lowest median total_ms as the label.
#     Tool: opm-simulators/doc/neural_cpr/collect_sweep_labels.py
#
#   Step 2 — architecture search
#     Trains hidden = 8, 12, 16, 24, 32 in parallel; keeps the best val_acc.
#     Tool: opm-simulators/doc/neural_cpr/train_cpr_policy.py
#
#   Step 3 — benchmark
#     flow_default / baseline / neural-v3 / neural-reuse for Norne, SPE9, SPE11A.
#
# Usage:
#   ./bench_norne.sh                              # serial, fresh sweep
#   NPROCS=8 OMPI_MCA_btl=self,sm ./bench_norne.sh   # MPI, fresh sweep
#   FRESH_SWEEP=0 ./bench_norne.sh                # skip sweep, reuse existing cpr_sweep_data/

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FLOW=$SCRIPT_DIR/build/opm-simulators/bin/flow
SCRIPTS=$SCRIPT_DIR/opm-simulators/doc/neural_cpr

# Set FRESH_SWEEP=0 to skip Step 0 and reuse existing data in cpr_sweep_data/
FRESH_SWEEP=${FRESH_SWEEP:-1}

# MPI setup — NPROCS controls ranks per flow call in both sweep and benchmark
NPROCS=${NPROCS:-1}
if (( NPROCS > 1 )); then
    FLOW_CMD=(mpirun -np "$NPROCS" "$FLOW")
else
    FLOW_CMD=("$FLOW")
fi

# Sweep output directories (freshly generated this run)
SWEEP_DIR=$SCRIPT_DIR/cpr_sweep_data
SWEEP_RUN1=$SWEEP_DIR/sweep_run1_fresh
SWEEP_RUN2=$SWEEP_DIR/sweep_run2_fresh
SWEEP_SPE11=$SWEEP_DIR/sweep_spe11_fresh


# Sweep CSVs used as training input (all 12-feature format)
#   sweep2/3    = 2 runs of SPE9+Norne
#   sweep_spe11 = SPE11A sweep data
SWEEP_CSVS=(
    $SCRIPT_DIR/cpr_sweep_data/sweep2/*.csv
    $SCRIPT_DIR/cpr_sweep_data/sweep3/*.csv
    $SCRIPT_DIR/cpr_sweep_data/sweep_spe11/*.csv
    # $SCRIPT_DIR/cpr_sweep_data/sweep5/*.csv
    # $SCRIPT_DIR/cpr_sweep_data/sweep7/*.csv
)


# Intermediate and output files
TRAINING_CSV=$SCRIPT_DIR/tmp/cpr_training_data.csv
MODEL=$SCRIPT_DIR/tmp/cpr_policy_retrained.model
OUTDIR=$SCRIPT_DIR/tmp/bench_neural_cpr

# Decks
NORNE_DECK=$SCRIPT_DIR/opm-tests/norne/NORNE_ATW2013.DATA
SPE9_DECK=$SCRIPT_DIR/opm-tests/spe9/SPE9.DATA
SPE11PYOPM_DECK=$SCRIPT_DIR/opm-tests/spe11pyopm/SPE11A.DATA

# SPE11 solver configs: fresh sweep dir when regenerating, existing dir otherwise
if [[ ${FRESH_SWEEP:-1} -eq 1 ]]; then
    SPE11PYOPM_CONFIGS_DIR=$SWEEP_SPE11/configs
else
    SPE11PYOPM_CONFIGS_DIR=$SWEEP_DIR/sweep_spe11/configs
fi
SPE11PYOPM_AVOID_LABEL23=${SPE11PYOPM_AVOID_LABEL23:-1}

mkdir -p "$SCRIPT_DIR/tmp"

# ── helpers ──────────────────────────────────────────────────────────────────

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
    [[ "$line" == *"weight=trueimpesanalytic"* ]] && dec=2 || \
    [[ "$line" == *"weight=trueimpes"* ]]         && dec=1
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
    [[ -z "$line" ]] && line=$(grep "NeuralCprPolicy" "$dbg" 2>/dev/null | head -1 || true)
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
    [[ ! -f "$dbg" ]] && { echo "  Policy summary: DBG missing"; return; }
    awk '
        /NeuralCprPolicy: .*weight=/ {
            dec=0; cprw=0; fine=0; coarse=0
            if ($0 ~ /weight=trueimpesanalytic/) dec=2
            else if ($0 ~ /weight=trueimpes/) dec=1
            if ($0 ~ /cprw weight=/) cprw=1
            if ($0 ~ /fine=dilu/)    fine=1
            if ($0 ~ /coarse=dilu/)  coarse=1
            label = dec*8 + cprw*4 + fine*2 + coarse
            labels[++n]=label; counts[label]++
            if (seen[label]++==0) unique++
            if (n>1 && label!=prev) switches++
            prev=label
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

# ── Step 0 — regenerate sweeps ───────────────────────────────────────────────

run_sweep() {
    local tag=$1 out=$2; shift 2
    echo "  ── $tag ──"
    python3 "$SCRIPTS/run_sweep.py" \
        --flow    "$FLOW" \
        --nprocs  "$NPROCS" \
        --out     "$out" \
        "$@"
}

if [[ $FRESH_SWEEP -eq 1 ]]; then
    echo "========================================"
    echo "  Step 0 — regenerate sweep data"
    echo "  Norne + SPE9 : 2 independent runs  (→ sweep_run1_fresh, sweep_run2_fresh)"
    echo "  SPE11A       : 1 run               (→ sweep_spe11_fresh)"
    echo "  MPI ranks/job: $NPROCS"
    echo "========================================"

    run_sweep "Norne + SPE9 (run 1)" "$SWEEP_RUN1" \
        --decks "$NORNE_DECK" "$SPE9_DECK"

    run_sweep "Norne + SPE9 (run 2)" "$SWEEP_RUN2" \
        --decks "$NORNE_DECK" "$SPE9_DECK"

    run_sweep "SPE11A" "$SWEEP_SPE11" \
        --decks "$SPE11PYOPM_DECK"
else
    echo "========================================"
    echo "  Step 0 — SKIPPED (FRESH_SWEEP=0, using SWEEP_CSVS)"
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
echo "  Step 1 — label sweep data"
echo "  Output: $TRAINING_CSV"
echo "========================================"

if [[ $FRESH_SWEEP -eq 1 ]]; then
    INPUT_CSVS=("$SWEEP_RUN1"/*.csv "$SWEEP_RUN2"/*.csv "$SWEEP_SPE11"/*.csv)
    echo "  Input : sweep_run1_fresh + sweep_run2_fresh + sweep_spe11_fresh (freshly generated)"
else
    INPUT_CSVS=("${SWEEP_CSVS[@]}")
    echo "  Input : SWEEP_CSVS (existing local data)"
fi

python3 "$SCRIPTS/collect_sweep_labels.py" \
    --csvs "${INPUT_CSVS[@]}" \
    --out  "$TRAINING_CSV" \
    --min-configs 2

# Auto-select best SPE11 baseline: most-winning label among safe labels (0-15)
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

# ── Step 2 — architecture search ─────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Step 2 — architecture search"
echo "  Candidates: hidden = 8, 12, 16, 24, 32"
echo "========================================"

best_hidden=32
best_val=0.0

n_features=$(python3 -c "
import csv
r = next(csv.DictReader(open('$TRAINING_CSV')))
print(len([k for k in r if k not in ('deck','best_config','label')]))" \
2>/dev/null || echo 12)
echo "  Input features detected: ${n_features}"

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

# ── Step 3 — benchmark ───────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Step 3 — benchmark  (flow_default vs best arch: ${n_features}→${best_hidden}→24)"
echo "  Model: $MODEL"
echo "  Run labels:"
echo "    Norne/SPE9 flow_default : $(label_desc none)"
echo "    Norne/SPE9 neural       : $(label_desc ml)"
echo "    SPE11 flow_default      : $(label_desc none)"
echo "    SPE11 baseline          : $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
echo "    SPE11 neural            : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels 16-23 forbidden"
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
echo "  neural base solver: $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels 16-23 forbidden"
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

# ── results ──────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  RESULTS"
echo "========================================"

read -r nd_n nd_l nd_w < <(extract "$OUTDIR/norne_flow_default/flow.log") || true
read -r nn_n nn_l nn_w < <(extract "$OUTDIR/norne_neural/flow.log")        || true
print_table "Norne (NORNE_ATW2013)" \
    "flow_default" "${nd_n:--}" "${nd_l:--}" "${nd_w:--}" \
    "neural-v3"    "${nn_n:--}" "${nn_l:--}" "${nn_w:--}"
echo "  Policy check:"; policy_check "$OUTDIR/norne_neural/NORNE_ATW2013.DBG"
policy_summary "$OUTDIR/norne_neural/NORNE_ATW2013.DBG"

read -r sd_n sd_l sd_w < <(extract "$OUTDIR/spe9_flow_default/flow.log") || true
read -r sn_n sn_l sn_w < <(extract "$OUTDIR/spe9_neural/flow.log")       || true
print_table "SPE9" \
    "flow_default" "${sd_n:--}" "${sd_l:--}" "${sd_w:--}" \
    "neural-v3"    "${sn_n:--}" "${sn_l:--}" "${sn_w:--}"
echo "  Policy check:"; policy_check "$OUTDIR/spe9_neural/SPE9.DBG"
policy_summary "$OUTDIR/spe9_neural/SPE9.DBG"

read -r spd_n spd_l spd_w < <(extract "$OUTDIR/spe11pyopm_flow_default/flow.log") || true
read -r spb_n spb_l spb_w < <(extract "$OUTDIR/spe11pyopm_baseline/flow.log")     || true
read -r spn_n spn_l spn_w < <(extract "$SPE11PYOPM_NEURAL_OUT/flow.log")           || true
read -r spr_n spr_l spr_w < <(extract "$SPE11PYOPM_NEURAL_REUSE_OUT/flow.log")    || true
print_table "SPE11A (spe11pyopm)" \
    "flow_default" "${spd_n:--}" "${spd_l:--}" "${spd_w:--}" \
    "baseline"     "${spb_n:--}" "${spb_l:--}" "${spb_w:--}" \
    "neural-v3"    "${spn_n:--}" "${spn_l:--}" "${spn_w:--}" \
    "neural-reuse" "${spr_n:--}" "${spr_l:--}" "${spr_w:--}"
echo "  Labels:"
echo "    flow_default : $(label_desc none)"
echo "    baseline     : $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); cpr-reuse-setup=0"
echo "    neural       : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels 16-23 forbidden"
echo "    neural-reuse : base $(label_desc "$SPE11PYOPM_SOLVER_LABEL"); labels 16-23 forbidden; reuse-setup=4; reuse-interval=10"
echo "  Policy check:"; policy_check "$SPE11PYOPM_NEURAL_OUT/SPE11A.DBG"
policy_summary "$SPE11PYOPM_NEURAL_OUT/SPE11A.DBG"
echo "  Reuse policy check:"; policy_check "$SPE11PYOPM_NEURAL_REUSE_OUT/SPE11A.DBG"
policy_summary "$SPE11PYOPM_NEURAL_REUSE_OUT/SPE11A.DBG"
