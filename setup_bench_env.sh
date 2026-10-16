#!/usr/bin/env bash
# setup_bench_env.sh — clone and build everything needed to run bench_neural_cpr.sh
#
# Prerequisites: cmake, make, MPI (OpenMPI or MPICH), Python 3, git
#
# Usage:
#   cd /path/to/cprepo
#   bash setup_bench_env.sh
#
# What this does:
#   1. Clones the Dune modules (DUNE_VERSION, v2.11.0), the OPM repos (opm-common
#      and opm-grid on OPM_RELEASE, release/2026.10; opm-simulators on
#      neural-cpr-control, which is based on that release) and opm-tests
#      (Norne, SPE9, SPE11A decks) — commented out below, the checkouts exist here
#   2. Builds the Dune modules via dunecontrol
#   3. Builds opm-common → opm-grid → opm-simulators (flow)
#   4. Generates the 24 CPR solver config JSONs required by bench_neural_cpr.sh
#
# After completion:
#   build/opm-simulators/bin/flow          — flow simulator
#   opm-tests/{norne,spe9,spe11pyopm}/     — reservoir decks
#   cpr_sweep_data/sweep_spe11/configs/    — label_00.json … label_23.json
#
# The neural AMG and neural abort models are not built here: they come from
# training (opm-simulators/doc/neural_cpr/) or from the snapshot,
#   tmp/amg_policy/cpr_amg_policy.model
#   tmp/abort_policy/abort_spe11a_norne_spe9.model

set -euo pipefail

CURRENT_DIRECTORY="$PWD"
export CURRENT_DIRECTORY

# ── configurable ──────────────────────────────────────────────────────────────
NJOBS=50
# Not needed by the neural CPR policies (they use opm-common's ML::NNModel);
# CMake skips the prefix if the path does not exist.
LIBTORCH_PREFIX=/home/AD.NORCERESEARCH.NO/bika/bika/workstation/opmstation/soft/libtorch
DUNE_VERSION=v2.11.0
# opm-common and opm-grid branch; neural-cpr-control is based on this release.
OPM_RELEASE=release/2026.10

DUNE_PREFIX_PATH="\
$CURRENT_DIRECTORY/dune-common/build-cmake;\
$CURRENT_DIRECTORY/dune-geometry/build-cmake;\
$CURRENT_DIRECTORY/dune-grid/build-cmake;\
$CURRENT_DIRECTORY/dune-istl/build-cmake"

# zoltan.h is in the source tree; Zoltan_config.h (generated) is in the build tree
ZOLTAN_INCLUDE_DIR="/usr/include/trilinos"
ZOLTAN_LIBRARY=/usr/lib/x86_64-linux-gnu/libtrilinos_zoltan.so

OPM_PREFIX_PATH="\
$DUNE_PREFIX_PATH;\
$LIBTORCH_PREFIX;\
$CURRENT_DIRECTORY/build/opm-common;\
$CURRENT_DIRECTORY/build/opm-grid"

# ── 1. Clone repos ────────────────────────────────────────────────────────────

# echo "========================================"
# echo "  Step 1 — clone repositories"
# echo "========================================"

# for module in common geometry grid istl; do
#     if [[ ! -d "$CURRENT_DIRECTORY/dune-$module" ]]; then
#         git clone --branch "$DUNE_VERSION" \
#             https://gitlab.dune-project.org/core/dune-$module.git \
#             "$CURRENT_DIRECTORY/dune-$module"
#     else
#         echo "  dune-$module already present, skipping clone"
#     fi
# done

# # opm-common and opm-grid: upstream OPM release (no local patches needed)
# for repo in common grid; do
#     if [[ ! -d "$CURRENT_DIRECTORY/opm-$repo" ]]; then
#         git clone --branch "$OPM_RELEASE" https://github.com/OPM/opm-$repo.git \
#             "$CURRENT_DIRECTORY/opm-$repo"
#     else
#         echo "  opm-$repo already present — checking out $OPM_RELEASE"
#         git -C "$CURRENT_DIRECTORY/opm-$repo" checkout "$OPM_RELEASE"
#     fi
# done

# # opm-simulators: neural-cpr-control branch (NeuralCprPolicy, neural AMG and
# # abort policies, deck guards, doc/neural_cpr scripts), branched from
# # neural-cpr-v3.  Not on GitHub yet: clone it from the snapshot's git bundle.
# OPM_SIM_SRC=https://github.com/bikagit/opm-simulators.git
# OPM_SIM_BUNDLE=$CURRENT_DIRECTORY/neural_cpr_snapshot_20261015/code/opm-simulators_neural-cpr-control.bundle
# [[ -f "$OPM_SIM_BUNDLE" ]] && OPM_SIM_SRC=$OPM_SIM_BUNDLE
# if [[ ! -d "$CURRENT_DIRECTORY/opm-simulators" ]]; then
#     git clone -b neural-cpr-control "$OPM_SIM_SRC" \
#         "$CURRENT_DIRECTORY/opm-simulators"
# else
#     echo "  opm-simulators already present — checking out neural-cpr-control"
#     git -C "$CURRENT_DIRECTORY/opm-simulators" checkout neural-cpr-control
# fi

# # opm-tests: tests-neural-cpr-v3 branch (Norne, SPE9, SPE11A decks)
# if [[ ! -d "$CURRENT_DIRECTORY/opm-tests" ]]; then
#     git clone -b tests-neural-cpr-v3 https://github.com/bikagit/opm-tests.git \
#         "$CURRENT_DIRECTORY/opm-tests"
# else
#     echo "  opm-tests already present — checking out tests-neural-cpr-v3"
#     git -C "$CURRENT_DIRECTORY/opm-tests" fetch origin
#     git -C "$CURRENT_DIRECTORY/opm-tests" checkout tests-neural-cpr-v3
# fi

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

mkdir build/opm-common
cd "$CURRENT_DIRECTORY/build/opm-common"
cmake -DUSE_MPI=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$OPM_PREFIX_PATH" \
      "$CURRENT_DIRECTORY/opm-common"
make -j"$NJOBS"
cd "$CURRENT_DIRECTORY"

mkdir build/opm-grid
cd "$CURRENT_DIRECTORY/build/opm-grid"
cmake -DUSE_MPI=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$OPM_PREFIX_PATH" \
      -DZOLTAN_INCLUDE_DIR="$ZOLTAN_INCLUDE_DIR" \
      -DZOLTAN_LIBRARY="$ZOLTAN_LIBRARY" \
      "$CURRENT_DIRECTORY/opm-grid"
make -j"$NJOBS"
cd "$CURRENT_DIRECTORY"

mkdir build/opm-simulators
cd "$CURRENT_DIRECTORY/build/opm-simulators"
cmake -DUSE_MPI=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$OPM_PREFIX_PATH" \
      -DZOLTAN_INCLUDE_DIR="$ZOLTAN_INCLUDE_DIR" \
      -DZOLTAN_LIBRARY="$ZOLTAN_LIBRARY" \
      -DUSE_OPENCL=OFF \
      "$CURRENT_DIRECTORY/opm-simulators"
make -j"$NJOBS" flow
cd "$CURRENT_DIRECTORY"

# ── 4. Generate CPR solver config JSONs ───────────────────────────────────────
#
#   bench_neural_cpr.sh auto-selects the best SPE11 baseline from 24 configs stored
#   in cpr_sweep_data/sweep_spe11/configs/label_00.json … label_23.json.
#   Label encoding: label = dec*8 + cprw*4 + fine*2 + coarse
#     dec  : 0=quasiimpes  1=trueimpes  2=trueimpesanalytic
#     cprw : 0=cpr         1=cprw
#     fine : 0=paroverilu0 1=dilu       (fine smoother)
#     coarse: 0=ilu0       1=dilu       (AMG coarse smoother)
#   The configs come from the sweep's own generator (run_sweep.py), so they
#   match the sweep data exactly: Flow's CPR settings (tol 0.005, maxiter 20,
#   use_well_weights=false for cprw), i.e. label 12 is Flow's default solver
#   (label 8 on decks without wells, where cpr and cprw coincide).

echo ""
echo "========================================"
echo "  Step 4 — generate CPR solver configs"
echo "========================================"

python3 -B - <<'PYEOF'
import os, pathlib, sys

ROOT = pathlib.Path(os.environ.get("CURRENT_DIRECTORY", "."))
sys.path.insert(0, str(ROOT / "opm-simulators/doc/neural_cpr"))
from run_sweep import write_configs

OUT = ROOT / "cpr_sweep_data/sweep_spe11/configs"
paths = write_configs(OUT)
for label in sorted(paths):
    print(f"  wrote {paths[label].name}")

print(f"  {len(paths)} configs written to {OUT}")
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Setup complete"
echo "  flow binary : $CURRENT_DIRECTORY/build/opm-simulators/bin/flow"
echo "  test decks  : $CURRENT_DIRECTORY/opm-tests/{norne,spe9,spe11pyopm}/"
echo "  configs     : $CURRENT_DIRECTORY/cpr_sweep_data/sweep_spe11/configs/"
echo ""
echo "  Run the benchmark (32 ranks x 2 threads, as for the published results):"
echo "    NPROCS=32 FLOW_THREADS=2 MPI_ARGS=\"--bind-to none\" ./bench_neural_cpr.sh"
echo ""
echo "  Neural AMG + neural abort against plain flow on one deck:"
echo "    ./compare_neural_cpr.sh --amg-model tmp/amg_policy/cpr_amg_policy.model \\"
echo "        --abort-model tmp/abort_policy/abort_spe11a_norne_spe9.model \\"
echo "        --deck opm-tests/spe9/SPE9.DATA --np 32 --threads 2 --mpi-args \"--bind-to none\""
echo "========================================"
