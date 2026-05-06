#!/usr/bin/env bash
set +e

OUT=/Users/macbookn/activopmwkspc/cpr/tmp_spe11A_only_restart
FLOW=/Users/macbookn/activopmwkspc/cpr/build/opm-simulators/bin/flow
DECK=/Users/macbookn/activopmwkspc/cpr/opm-tests/spe11pyopm/SPE11A.DATA
SOLVER=/Users/macbookn/activopmwkspc/cpr_sweep_data/sweep_spe11/configs/label_09.json
MODEL=/Users/macbookn/activopmwkspc/tmp/cpr_policy_retrained.model

mkdir -p "$OUT/baseline" "$OUT/neural"

date > "$OUT/run.log"
echo "baseline" >> "$OUT/run.log"
"$FLOW" \
    --output-dir="$OUT/baseline" \
    --linear-solver="$SOLVER" \
    --cpr-reuse-setup=0 \
    "$DECK" > "$OUT/baseline/flow.stdout" 2>&1
echo "$?" > "$OUT/baseline/exit_code"

echo "neural" >> "$OUT/run.log"
OPM_NEURAL_CPR_WEIGHTS="$MODEL" "$FLOW" \
    --output-dir="$OUT/neural" \
    --linear-solver="$SOLVER" \
    --cpr-reuse-setup=0 \
    "$DECK" > "$OUT/neural/flow.stdout" 2>&1
echo "$?" > "$OUT/neural/exit_code"

date >> "$OUT/run.log"
