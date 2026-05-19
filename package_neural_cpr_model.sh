#!/usr/bin/env bash
# package_neural_cpr_model.sh - create a portable Neural CPR model bundle.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL=${MODEL:-}
NAME=${NAME:-}
OUTDIR=${OUTDIR:-}
CONFIDENCE=${OPM_NEURAL_CPR_CONFIDENCE:-0.6}
FORBID_LABELS=${OPM_NEURAL_CPR_FORBID_LABELS:-8,16,17,18,19,20,21,22,23}
REUSE_SETUP=${CPR_REUSE_SETUP:-4}
REUSE_INTERVAL=${CPR_REUSE_INTERVAL:-10}
SPE11_SOLVER_LABEL=${SPE11PYOPM_SOLVER_LABEL:-11}
NPROCS_HINT=${NPROCS_HINT:-8}
TRAINED_ON=${TRAINED_ON:-"Norne,SPE9,SPE10,SPE11A,SPE11B"}
MAKE_TARBALL=1

usage() {
    cat <<'EOF'
Usage:
  package_neural_cpr_model.sh --model MODEL.model [options]

Options:
  --model PATH          Trained Neural CPR .model file.
  --name NAME           Bundle name. Default: model basename without .model.
  --out DIR             Parent output directory. Default: ./dist.
  --confidence X        Default confidence gate written to metadata. Default: 0.6.
  --forbid-labels LIST  Default forbidden labels. Default: 8,16,17,18,19,20,21,22,23.
  --reuse-setup N       Default CPR reuse setup. Default: 4.
  --reuse-interval N    Default CPR reuse interval. Default: 10.
  --spe11-label N       SPE11 baseline label hint. Default: 11.
  --np-hint N           MPI rank hint for README examples. Default: 8.
  --trained-on TEXT     Comma-separated training data note.
  --no-tar              Create directory only; skip .tar.gz.
  -h, --help            Show this help.

Environment overrides use the same names where practical:
  MODEL, NAME, OUTDIR, OPM_NEURAL_CPR_CONFIDENCE, OPM_NEURAL_CPR_FORBID_LABELS,
  CPR_REUSE_SETUP, CPR_REUSE_INTERVAL, SPE11PYOPM_SOLVER_LABEL, NPROCS_HINT,
  TRAINED_ON.
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
        --spe11-label)
            [[ $# -ge 2 ]] || die "--spe11-label requires a label"
            SPE11_SOLVER_LABEL=$2
            shift 2
            ;;
        --np-hint)
            [[ $# -ge 2 ]] || die "--np-hint requires an integer"
            NPROCS_HINT=$2
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

[[ -n "$MODEL" ]] || die "missing model path; pass --model MODEL.model"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"
[[ -f "$SCRIPT_DIR/run_neural_cpr.sh" ]] || die "missing $SCRIPT_DIR/run_neural_cpr.sh"

model_abs=$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")
model_base=$(basename "$MODEL")
if [[ -z "$NAME" ]]; then
    NAME=${model_base%.model}
fi
OUTDIR=${OUTDIR:-$SCRIPT_DIR/dist}
BUNDLE_DIR=$OUTDIR/$NAME

mkdir -p "$BUNDLE_DIR"
cp "$model_abs" "$BUNDLE_DIR/$model_base"
cp "$SCRIPT_DIR/run_neural_cpr.sh" "$BUNDLE_DIR/run_neural_cpr.sh"
chmod +x "$BUNDLE_DIR/run_neural_cpr.sh"

model_sha=$(sha256sum "$model_abs" | awk '{print $1}')
model_bytes=$(wc -c < "$model_abs")
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat >"$BUNDLE_DIR/neural_cpr_model.json" <<EOF
{
  "model": "$(json_escape "$model_base")",
  "sha256": "$model_sha",
  "bytes": $model_bytes,
  "created_utc": "$created_utc",
  "confidence": $CONFIDENCE,
  "forbid_labels": "$(json_escape "$FORBID_LABELS")",
  "reuse_setup": $REUSE_SETUP,
  "reuse_interval": $REUSE_INTERVAL,
  "spe11_solver_label": "$(json_escape "$SPE11_SOLVER_LABEL")",
  "nprocs_hint": $NPROCS_HINT,
  "trained_on": "$(json_escape "$TRAINED_ON")"
}
EOF

cat >"$BUNDLE_DIR/README.md" <<EOF
# Neural CPR Model Bundle

This bundle contains a trained Neural CPR model and a small apply-only runner.
It does not include Flow or reservoir decks; use it from a machine with a Flow
binary built with the Neural CPR policy support.

## Files

- \`$model_base\`: trained policy weights.
- \`neural_cpr_model.json\`: runtime defaults and checksum.
- \`run_neural_cpr.sh\`: helper for running Flow with the model.

## Quick Start

\`\`\`bash
./run_neural_cpr.sh --model "$model_base" --deck /path/to/CASE.DATA --np $NPROCS_HINT
\`\`\`

If Flow is not at \`./build/opm-simulators/bin/flow\`, pass it explicitly:

\`\`\`bash
./run_neural_cpr.sh --flow /path/to/flow --model "$model_base" --deck /path/to/CASE.DATA --np $NPROCS_HINT
\`\`\`

For SPE11-style cases, use the safe label guard:

\`\`\`bash
./run_neural_cpr.sh --model "$model_base" --deck /path/to/SPE11.DATA --np $NPROCS_HINT --spe11-safe
\`\`\`

The equivalent direct Flow invocation is:

\`\`\`bash
OPM_NEURAL_CPR_WEIGHTS="$PWD/$model_base" \\
OPM_NEURAL_CPR_CONFIDENCE=$CONFIDENCE \\
OPM_NEURAL_CPR_FORBID_LABELS="$FORBID_LABELS" \\
flow --cpr-reuse-setup=$REUSE_SETUP --cpr-reuse-interval=$REUSE_INTERVAL /path/to/CASE.DATA
\`\`\`
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
