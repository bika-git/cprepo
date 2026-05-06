#!/usr/bin/env bash
# setup_bench_env.sh — clone and build everything needed to run bench_norne.sh
#
# Prerequisites: cmake, make, MPI (OpenMPI or MPICH), Python 3, git
#
# Usage:
#   cd /path/to/cprepo
#   bash setup_bench_env.sh
#
# What this does:
#   1. Clones Dune modules (v2.9.1) and OPM repos (custom neural-cpr branches)
#   2. Builds Dune via dunecontrol → opm-common → opm-grid → opm-simulators (flow)
#   3. Clones opm-tests (test decks for Norne, SPE9, SPE11A)
#   4. Generates the 24 CPR solver config JSONs required by bench_norne.sh
#
# After completion:
#   build/opm-simulators/bin/flow          — flow simulator
#   opm-tests/{norne,spe9,spe11pyopm}/     — reservoir decks
#   cpr_sweep_data/sweep_spe11/configs/    — label_00.json … label_23.json

set -euo pipefail

CURRENT_DIRECTORY="$PWD"

# ── configurable ──────────────────────────────────────────────────────────────
NJOBS=50
LIBTORCH_PREFIX=/home/AD.NORCERESEARCH.NO/bika/bika/workstation/opmstation/soft/libtorch
DUNE_VERSION=v2.11.0

DUNE_PREFIX_PATH="\
$CURRENT_DIRECTORY/dune-common/build-cmake;\
$CURRENT_DIRECTORY/dune-geometry/build-cmake;\
$CURRENT_DIRECTORY/dune-grid/build-cmake;\
$CURRENT_DIRECTORY/dune-istl/build-cmake"

OPM_PREFIX_PATH="\
$DUNE_PREFIX_PATH;\
$LIBTORCH_PREFIX;\
$CURRENT_DIRECTORY/build/opm-common;\
$CURRENT_DIRECTORY/build/opm-grid"

# ── 1. Clone repos ────────────────────────────────────────────────────────────

echo "========================================"
echo "  Step 1 — clone repositories"
echo "========================================"

for module in common geometry grid istl; do
    if [[ ! -d "$CURRENT_DIRECTORY/dune-$module" ]]; then
        git clone --branch "$DUNE_VERSION" \
            https://gitlab.dune-project.org/core/dune-$module.git \
            "$CURRENT_DIRECTORY/dune-$module"
    else
        echo "  dune-$module already present, skipping clone"
    fi
done

# opm-common and opm-grid: master branch from bikagit
for repo in common grid; do
    if [[ ! -d "$CURRENT_DIRECTORY/opm-$repo" ]]; then
        git clone https://github.com/bikagit/opm-$repo.git \
            "$CURRENT_DIRECTORY/opm-$repo"
    else
        echo "  opm-$repo already present, skipping clone"
    fi
done

# opm-simulators: neural-cpr-v3 branch (contains NeuralCprPolicy + doc/neural_cpr scripts)
if [[ ! -d "$CURRENT_DIRECTORY/opm-simulators" ]]; then
    git clone -b neural-cpr-v3 https://github.com/bikagit/opm-simulators.git \
        "$CURRENT_DIRECTORY/opm-simulators"
else
    echo "  opm-simulators already present — checking out neural-cpr-v3"
    git -C "$CURRENT_DIRECTORY/opm-simulators" fetch origin
    git -C "$CURRENT_DIRECTORY/opm-simulators" checkout neural-cpr-v3
fi

# opm-tests: tests-neural-cpr-v3 branch (Norne, SPE9, SPE11A decks)
if [[ ! -d "$CURRENT_DIRECTORY/opm-tests" ]]; then
    git clone -b tests-neural-cpr-v3 https://github.com/bikagit/opm-tests.git \
        "$CURRENT_DIRECTORY/opm-tests"
else
    echo "  opm-tests already present — checking out tests-neural-cpr-v3"
    git -C "$CURRENT_DIRECTORY/opm-tests" fetch origin
    git -C "$CURRENT_DIRECTORY/opm-tests" checkout tests-neural-cpr-v3
fi

# ── 2. Build Dune modules via dunecontrol ─────────────────────────────────────

echo ""
echo "========================================"
echo "  Step 2 — build Dune modules ($DUNE_VERSION)"
echo "========================================"

for module in common geometry grid istl; do
    "$CURRENT_DIRECTORY/dune-common/bin/dunecontrol" --only=dune-$module cmake
    "$CURRENT_DIRECTORY/dune-common/bin/dunecontrol" --only=dune-$module make -j"$NJOBS"
done

# ── 3. Build OPM stack ────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Step 3 — build OPM stack"
echo "========================================"

rm -rf build
mkdir build

for repo in common grid; do
    mkdir build/opm-$repo
    cd "$CURRENT_DIRECTORY/build/opm-$repo"
    cmake -DUSE_MPI=1 \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH="$OPM_PREFIX_PATH" \
          "$CURRENT_DIRECTORY/opm-$repo"
    make -j"$NJOBS"
    cd "$CURRENT_DIRECTORY"
done

mkdir build/opm-simulators
cd "$CURRENT_DIRECTORY/build/opm-simulators"
cmake -DUSE_MPI=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$OPM_PREFIX_PATH" \
      "$CURRENT_DIRECTORY/opm-simulators"
make -j"$NJOBS" flow
cd "$CURRENT_DIRECTORY"

# ── 4. Generate CPR solver config JSONs ───────────────────────────────────────
#
#   bench_norne.sh auto-selects the best SPE11 baseline from 24 configs stored
#   in cpr_sweep_data/sweep_spe11/configs/label_00.json … label_23.json.
#   Label encoding: label = dec*8 + cprw*4 + fine*2 + coarse
#     dec  : 0=quasiimpes  1=trueimpes  2=trueimpesanalytic
#     cprw : 0=cpr         1=cprw
#     fine : 0=paroverilu0 1=dilu       (fine smoother)
#     coarse: 0=ilu0       1=dilu       (AMG coarse smoother)

echo ""
echo "========================================"
echo "  Step 4 — generate CPR solver configs"
echo "========================================"

python3 - <<'PYEOF'
import json, os, pathlib

OUT = pathlib.Path(os.environ.get("CURRENT_DIRECTORY", ".")) / \
      "cpr_sweep_data/sweep_spe11/configs"
OUT.mkdir(parents=True, exist_ok=True)

WEIGHT = ["quasiimpes", "trueimpes", "trueimpesanalytic"]

def make_config(label):
    dec    = label // 8
    cprw   = (label % 8) // 4
    fine   = (label % 4) // 2
    coarse = label % 2

    prec = {
        "type":        "cprw" if cprw else "cpr",
        "weight_type": WEIGHT[dec],
        "pre_smooth":  0,
        "post_smooth": 1,
        "finesmoother": {
            "type":       "dilu" if fine else "paroverilu0",
            "relaxation": 1.0
        },
        "verbosity": 0,
        "coarsesolver": {
            "maxiter": 1, "tol": 0.1,
            "solver": "loopsolver", "verbosity": 0,
            "preconditioner": {
                "type": "amg",
                "alpha": 0.333333333333, "relaxation": 1.0,
                "iterations": 1, "coarsenTarget": 1200,
                "pre_smooth": 1, "post_smooth": 1,
                "beta": 0.0, "verbosity": 0, "maxlevel": 15,
                "skip_isolated": 0, "accumulate": 1,
                "prolongationdamping": 1.0,
                "maxdistance": 2, "maxconnectivity": 15,
                "maxaggsize": 6, "minaggsize": 4,
                "smoother": "dilu" if coarse else "ilu0"
            }
        }
    }
    if cprw:
        prec["use_well_weights"] = "true"
        prec["add_wells"]        = "true"

    return {"maxiter": 20, "tol": 0.005, "verbosity": 0,
            "solver": "bicgstab", "preconditioner": prec}

for i in range(24):
    path = OUT / f"label_{i:02d}.json"
    path.write_text(json.dumps(make_config(i), indent=2) + "\n")
    print(f"  wrote {path.name}")

print(f"  24 configs written to {OUT}")
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Setup complete"
echo "  flow binary : $CURRENT_DIRECTORY/build/opm-simulators/bin/flow"
echo "  test decks  : $CURRENT_DIRECTORY/opm-tests/{norne,spe9,spe11pyopm}/"
echo "  configs     : $CURRENT_DIRECTORY/cpr_sweep_data/sweep_spe11/configs/"
echo ""
echo "  Run the benchmark:"
echo "    NPROCS=8 OMPI_MCA_btl=self,sm ./bench_norne.sh"
echo "========================================"
