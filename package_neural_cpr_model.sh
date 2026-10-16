#!/usr/bin/env bash
# package_neural_cpr_model.sh - create a portable bundle of Neural CPR models.
#
# Bundles the neural AMG and/or neural abort policy (and optionally a CPR
# variant policy) with the run and compare scripts, metadata and a README:
#   ./package_neural_cpr_model.sh \
#       --amg-model tmp/amg_policy/cpr_amg_policy.model \
#       --abort-model tmp/abort_policy/abort_spe11a_norne_spe9.model

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL=${MODEL:-}
AMG_MODEL=${AMG_MODEL:-}
ABORT_MODEL=${ABORT_MODEL:-}
NAME=${NAME:-}
OUTDIR=${OUTDIR:-}
CONFIDENCE=${OPM_NEURAL_CPR_CONFIDENCE:-0.6}
FORBID_LABELS=${OPM_NEURAL_CPR_FORBID_LABELS:-}
MIN_GAIN=${OPM_NEURAL_CPR_MIN_GAIN:-0.05}
ABORT_THRESHOLD=${OPM_CPR_ABORT_THRESHOLD:-0.95}
NPROCS_HINT=${NPROCS_HINT:-32}
THREADS_HINT=${THREADS_HINT:-2}
TRAINED_ON=${TRAINED_ON:-"SPE11A,Norne,SPE9"}
MAKE_TARBALL=1

usage() {
    cat <<'EOF'
Usage:
  package_neural_cpr_model.sh [--amg-model M] [--abort-model M] [--model M] [options]

Models (at least one):
  --amg-model PATH      Neural AMG policy.
  --abort-model PATH    Neural abort policy.
  --model PATH          CPR variant policy (24 configs).

Options:
  --name NAME           Bundle name. Default: the model's basename (one model) or
                        neural_cpr_<date> (several).
  --out DIR             Parent output directory. Default: ./dist.
  --min-gain G          Neural AMG margin for metadata and README. Default: 0.05.
  --abort-threshold P   Neural abort threshold. Default: 0.95.
  --confidence X        CPR variant policy confidence gate. Default: 0.6.
  --forbid-labels LIST  CPR variant labels to forbid. Default: none (Flow from
                        neural-cpr-control blocks trueimpesanalytic on decks without
                        oil itself).
  --np-hint N           MPI ranks for README examples. Default: 32.
  --threads-hint N      Threads per rank for README examples. Default: 2.
  --trained-on TEXT     Training decks note. Default: SPE11A,Norne,SPE9 (the decks
                        of the current neural AMG and abort models).
  --no-tar              Create directory only; skip .tar.gz.
  -h, --help            Show this help.

Environment overrides use the same names where practical:
  MODEL, AMG_MODEL, ABORT_MODEL, NAME, OUTDIR, OPM_NEURAL_CPR_CONFIDENCE,
  OPM_NEURAL_CPR_FORBID_LABELS, OPM_NEURAL_CPR_MIN_GAIN, OPM_CPR_ABORT_THRESHOLD,
  NPROCS_HINT, THREADS_HINT, TRAINED_ON.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

json_escape() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//$'\n'/\\n}
    printf '%s' "$value"
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
        --name)
            [[ $# -ge 2 ]] || die "--name requires a value"
            NAME=$2
            shift 2
            ;;
        --out)
            [[ $# -ge 2 ]] || die "--out requires a directory"
            OUTDIR=$2
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
        --np-hint)
            [[ $# -ge 2 ]] || die "--np-hint requires an integer"
            NPROCS_HINT=$2
            shift 2
            ;;
        --threads-hint)
            [[ $# -ge 2 ]] || die "--threads-hint requires an integer"
            THREADS_HINT=$2
            shift 2
            ;;
        --trained-on)
            [[ $# -ge 2 ]] || die "--trained-on requires text"
            TRAINED_ON=$2
            shift 2
            ;;
        --no-tar)
            MAKE_TARBALL=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL=$1
            else
                die "unexpected argument: $1"
            fi
            shift
            ;;
    esac
done

[[ -n "$MODEL$AMG_MODEL$ABORT_MODEL" ]] \
    || die "missing model; pass --amg-model, --abort-model and/or --model"
models=()
for m in "$AMG_MODEL" "$ABORT_MODEL" "$MODEL"; do
    [[ -z "$m" ]] && continue
    [[ -f "$m" ]] || die "model not found: $m"
    models+=("$m")
done
for f in run_neural_cpr.sh compare_neural_cpr.sh; do
    [[ -f "$SCRIPT_DIR/$f" ]] || die "missing $SCRIPT_DIR/$f"
done
[[ "$NPROCS_HINT" =~ ^[1-9][0-9]*$ ]] || die "--np-hint must be a positive integer"
[[ "$THREADS_HINT" =~ ^[1-9][0-9]*$ ]] || die "--threads-hint must be a positive integer"

amg_base=; abort_base=; model_base=
[[ -n "$AMG_MODEL" ]] && amg_base=$(basename "$AMG_MODEL")
[[ -n "$ABORT_MODEL" ]] && abort_base=$(basename "$ABORT_MODEL")
[[ -n "$MODEL" ]] && model_base=$(basename "$MODEL")
if [[ -n "$amg_base" && ( "$amg_base" == "$abort_base" || "$amg_base" == "$model_base" ) ]] \
   || [[ -n "$abort_base" && "$abort_base" == "$model_base" ]]; then
    die "the models must have different file names"
fi

if [[ -z "$NAME" ]]; then
    if (( ${#models[@]} == 1 )); then
        NAME=$(basename "${models[0]}" .model)
    else
        NAME=neural_cpr_$(date +%Y%m%d)
    fi
fi
OUTDIR=${OUTDIR:-$SCRIPT_DIR/dist}
BUNDLE_DIR=$OUTDIR/$NAME

mkdir -p "$BUNDLE_DIR"
for m in "${models[@]}"; do
    cp "$m" "$BUNDLE_DIR/$(basename "$m")"
done
for f in run_neural_cpr.sh compare_neural_cpr.sh; do
    cp "$SCRIPT_DIR/$f" "$BUNDLE_DIR/$f"
    chmod +x "$BUNDLE_DIR/$f"
done

# ── metadata ─────────────────────────────────────────────────────────────────

model_entry() {  # model_entry POLICY FILE [KEY=VALUE ...]: one entry of "models"
    local policy=$1 file=$2 env="" kv sha bytes
    shift 2
    for kv in "$@"; do
        env+="${env:+, }\"$(json_escape "${kv%%=*}")\": \"$(json_escape "${kv#*=}")\""
    done
    sha=$(sha256sum "$file" | awk '{print $1}')
    bytes=$(wc -c < "$file" | tr -d ' ')
    printf '    {"policy": "%s", "file": "%s", "sha256": "%s", "bytes": %s,\n     "env": {%s}}' \
        "$policy" "$(json_escape "$(basename "$file")")" "$sha" "$bytes" "$env"
}

entries=()
if [[ -n "$AMG_MODEL" ]]; then
    entries+=("$(model_entry neural-amg "$AMG_MODEL" \
        "OPM_NEURAL_CPR_WEIGHTS=$amg_base" OPM_NEURAL_CPR_ACTIONS=amg \
        "OPM_NEURAL_CPR_MIN_GAIN=$MIN_GAIN")")
fi
if [[ -n "$ABORT_MODEL" ]]; then
    entries+=("$(model_entry neural-abort "$ABORT_MODEL" \
        "OPM_CPR_ABORT_MODEL=$abort_base" "OPM_CPR_ABORT_THRESHOLD=$ABORT_THRESHOLD")")
fi
if [[ -n "$MODEL" ]]; then
    model_env=("OPM_NEURAL_CPR_WEIGHTS=$model_base" "OPM_NEURAL_CPR_CONFIDENCE=$CONFIDENCE")
    [[ -n "$FORBID_LABELS" ]] && model_env+=("OPM_NEURAL_CPR_FORBID_LABELS=$FORBID_LABELS")
    entries+=("$(model_entry cpr-variant "$MODEL" "${model_env[@]}")")
fi
nl=$'\n'
models_json=""
for e in "${entries[@]}"; do
    models_json+="${models_json:+,$nl}$e"
done

cat >"$BUNDLE_DIR/neural_cpr_model.json" <<EOF
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "flow": "opm-simulators branch neural-cpr-control",
  "trained_on": "$(json_escape "$TRAINED_ON")",
  "cpr_reuse": "flow defaults",
  "nprocs_hint": $NPROCS_HINT,
  "threads_hint": $THREADS_HINT,
  "models": [
$models_json
  ]
}
EOF

# ── README ───────────────────────────────────────────────────────────────────

if (( NPROCS_HINT > 1 )); then
    mpi_hint="--np $NPROCS_HINT --threads $THREADS_HINT --mpi-args \"--bind-to none\""
    flow_hint="mpirun -np $NPROCS_HINT --bind-to none flow --threads-per-process=$THREADS_HINT"
else
    mpi_hint="--threads $THREADS_HINT"
    flow_hint="flow --threads-per-process=$THREADS_HINT"
fi

model_rows=""; run_args=""; compare_args=""; env_lines=""
if [[ -n "$AMG_MODEL" ]]; then
    model_rows+="| Neural AMG | \`$amg_base\` | At each CPR setup rebuild: prolongation damping {1.0, 1.3, 1.6} and coarsening target {1200, 600, 300} of the pressure AMG. Keeps Flow's default (1.0, 1200) unless the predicted log-cost gain exceeds the margin ($MIN_GAIN). |$nl"
    run_args+=" --amg-model $amg_base"
    compare_args+=" --amg-model $amg_base"
    env_lines+="OPM_NEURAL_CPR_WEIGHTS=\$PWD/$amg_base OPM_NEURAL_CPR_ACTIONS=amg OPM_NEURAL_CPR_MIN_GAIN=$MIN_GAIN \\$nl"
fi
if [[ -n "$MODEL" ]]; then
    model_rows+="| CPR variant | \`$model_base\` | The CPR variant: pressure weights × cpr/cprw × fine and coarse smoother (24 configs). Cannot run together with the neural AMG policy. |$nl"
    compare_args+=" --model $model_base"
    if [[ -z "$AMG_MODEL" ]]; then
        run_args+=" --model $model_base"
        env_lines+="OPM_NEURAL_CPR_WEIGHTS=\$PWD/$model_base OPM_NEURAL_CPR_CONFIDENCE=$CONFIDENCE${FORBID_LABELS:+ OPM_NEURAL_CPR_FORBID_LABELS=$FORBID_LABELS} \\$nl"
    fi
fi
if [[ -n "$ABORT_MODEL" ]]; then
    model_rows+="| Neural abort | \`$abort_base\` | Before each CPR solve: the probability that the timestep attempt will be chopped anyway.  Above the threshold ($ABORT_THRESHOLD) the solve fails at once and Flow chops the timestep now. |$nl"
    run_args+=" --abort-model $abort_base"
    compare_args+=" --abort-model $abort_base"
    [[ -z "$AMG_MODEL$MODEL" ]] && env_lines+="OPM_NEURAL_CPR_DISABLE=1 \\$nl"
    env_lines+="OPM_CPR_ABORT_MODEL=\$PWD/$abort_base OPM_CPR_ABORT_THRESHOLD=$ABORT_THRESHOLD \\$nl"
fi

cat >"$BUNDLE_DIR/README.md" <<EOF
# Neural CPR policies for OPM Flow

Trained neural-network policies for the CPR linear solver of OPM Flow, with
helper scripts to run and compare them.  Flow and the decks are not included.
The models need a Flow built from opm-simulators branch \`neural-cpr-control\`;
other builds ignore the environment variables below.

## Models

| Policy | File | What the network decides at runtime |
|---|---|---|
${model_rows}
Trained on: $TRAINED_ON.  \`neural_cpr_model.json\` has the checksums and
the environment variables for each model.

Rules around the networks (not learned): Flow's default solver as the anchor;
no abort before an attempt's first solve or after three chops in a row; deck
guards (cprw only on decks with wells, cpr only without wells, no
trueimpesanalytic without oil).  CPR setup reuse stays at Flow's defaults.

## Run

\`\`\`bash
./run_neural_cpr.sh${run_args} \\
    --deck /path/to/CASE.DATA $mpi_hint
\`\`\`

If Flow is not at \`./build/opm-simulators/bin/flow\`, add \`--flow /path/to/flow\`.

## Compare with plain Flow

\`\`\`bash
./compare_neural_cpr.sh${compare_args} \\
    --deck /path/to/CASE.DATA $mpi_hint
\`\`\`

Runs plain Flow and one run per policy (and their combination) one after the
other, then prints wall time, linear solve time, Newton and linear iterations,
and how many timestep attempts the abort policy gave up.

## Direct Flow invocation

\`\`\`bash
${env_lines}$flow_hint /path/to/CASE.DATA
\`\`\`

Plain Flow for reference: \`OPM_NEURAL_CPR_DISABLE=1\` and no \`OPM_CPR_ABORT_MODEL\`.
EOF

if (( MAKE_TARBALL == 1 )); then
    tarball=$OUTDIR/$NAME.tar.gz
    tar -C "$OUTDIR" -czf "$tarball" "$NAME"
    echo "Bundle written:"
    echo "  directory : $BUNDLE_DIR"
    echo "  tarball   : $tarball"
else
    echo "Bundle written:"
    echo "  directory : $BUNDLE_DIR"
fi
