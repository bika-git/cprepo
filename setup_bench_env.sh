#!/usr/bin/env bash
# setup_bench_env.sh — clone and build everything needed to run bench_neural_cpr.sh
#
# Prerequisites: cmake, make, MPI (OpenMPI or MPICH), Python 3, git
#
# Usage (fresh machine):
#   git clone -b neural-cpr-control https://github.com/bika-git/cprepo.git
#   cd cprepo
#   bash setup_bench_env.sh
#
# Rerunning is safe: existing checkouts are left on their current branch (with
# a warning if it is not the expected one) and builds are incremental.
#
# Environment overrides:
#   NJOBS=N              parallel build jobs (default: nproc)
#   CLEAN_BUILD=1        delete build/ and rebuild the OPM stack from scratch
#   OPM_COMMON_COMMIT=, OPM_GRID_COMMIT=   follow release/2026.10 instead of
#                        the pinned commits the results were built with
#   ZOLTAN_INCLUDE_DIR, ZOLTAN_LIBRARY   Zoltan location (default: Ubuntu's
#                        libtrilinos-zoltan-dev paths)
#   LIBTORCH_PREFIX      optional, not needed by the neural CPR policies
#
# What this does:
#   1. Clones the Dune modules (DUNE_VERSION, v2.11.0), opm-common and opm-grid
#      (upstream OPM, OPM_RELEASE = release/2026.10), opm-simulators (bika-git,
#      neural-cpr-control, based on that release) and the decks (bika-git/decks:
#      Norne, SPE9, SPE11A/B, SPE10, Drogon and the spe11_decks collection)
#   2. Builds the Dune modules via dunecontrol
#   3. Builds opm-common → opm-grid → opm-simulators (flow)
#   4. Generates the 24 CPR solver config JSONs required by bench_neural_cpr.sh
#
# After completion:
#   build/opm-simulators/bin/flow          — flow simulator
#   decks/{norne,spe9,spe11pyopm,...}/     — reservoir decks
#   cpr_sweep_data/sweep_spe11/configs/    — label_00.json … label_23.json
#
# The neural AMG and neural abort models are not built here: they come from
# training (opm-simulators/doc/neural_cpr/) or from a package_neural_cpr_model.sh
# bundle; the scripts expect
#   models/amg_policy/cpr_amg_policy.model
#   models/abort_policy/abort_spe11a_norne_spe9.model

set -euo pipefail

CURRENT_DIRECTORY="$PWD"
export CURRENT_DIRECTORY

# ── configurable ──────────────────────────────────────────────────────────────
# Portable CPU count: nproc on Linux, sysctl on macOS
if [[ "$(uname -s)" == "Darwin" ]]; then
    _nproc=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
else
    _nproc=$(nproc 2>/dev/null || echo 4)
fi
NJOBS=${NJOBS:-$_nproc}
CLEAN_BUILD=${CLEAN_BUILD:-0}
# Not needed by the neural CPR policies (they use opm-common's ML::NNModel);
# CMake skips the prefix if the path does not exist.
LIBTORCH_PREFIX=${LIBTORCH_PREFIX:-}
DUNE_VERSION=v2.11.0
# opm-common and opm-grid branch; neural-cpr-control is based on this release.
OPM_RELEASE=release/2026.10
# Exact release commits the published results were built with (the release
# branch keeps moving, e.g. rc1 changed NNLayer instantiation and Zoltan
# partitioning).  Set OPM_COMMON_COMMIT= OPM_GRID_COMMIT= to follow the branch.
OPM_COMMON_COMMIT=${OPM_COMMON_COMMIT-b90f7aebfe601af0cdb104a1285cd140a011ced5}
OPM_GRID_COMMIT=${OPM_GRID_COMMIT-fa8b05067981f2ef824f5be45cc4975552f9d580}
OPM_SIM_BRANCH=neural-cpr-control
DECKS_BRANCH=neural-cpr-control

DUNE_PREFIX_PATH="\
$CURRENT_DIRECTORY/dune-common/build-cmake;\
$CURRENT_DIRECTORY/dune-geometry/build-cmake;\
$CURRENT_DIRECTORY/dune-grid/build-cmake;\
$CURRENT_DIRECTORY/dune-istl/build-cmake"

# ── Platform-specific defaults ────────────────────────────────────────────────
if [[ "$(uname -s)" == "Darwin" ]]; then
    # macOS: prefer Homebrew.  Detect prefix (Apple Silicon /opt/homebrew, Intel /usr/local).
    _brew_prefix=$(brew --prefix 2>/dev/null || echo /opt/homebrew)
    # Trilinos/Zoltan: brew install trilinos
    ZOLTAN_INCLUDE_DIR=${ZOLTAN_INCLUDE_DIR:-$_brew_prefix/include}
    _zoltan_lib=$_brew_prefix/lib/libtrilinos_zoltan.dylib
    # Older Homebrew trilinos installs the lib as libtrilinos_zoltan.dylib;
    # fall back to the versioned name if needed.
    if [[ ! -f "$_zoltan_lib" ]]; then
        _zoltan_lib=$(ls "$_brew_prefix/lib/libtrilinos_zoltan"*.dylib 2>/dev/null | head -1 || echo "$_zoltan_lib")
    fi
    ZOLTAN_LIBRARY=${ZOLTAN_LIBRARY:-$_zoltan_lib}
    # Extra CMake prefix path so find_package locates Homebrew MPI, Boost, etc.
    HOMEBREW_PREFIX_PATH=${HOMEBREW_PREFIX_PATH:-$_brew_prefix}
else
    # Linux: Ubuntu/Debian defaults
    ZOLTAN_INCLUDE_DIR=${ZOLTAN_INCLUDE_DIR:-/usr/include/trilinos}
    ZOLTAN_LIBRARY=${ZOLTAN_LIBRARY:-/usr/lib/x86_64-linux-gnu/libtrilinos_zoltan.so}
    HOMEBREW_PREFIX_PATH=${HOMEBREW_PREFIX_PATH:-}
fi

OPM_PREFIX_PATH="\
$DUNE_PREFIX_PATH;\
${LIBTORCH_PREFIX:+$LIBTORCH_PREFIX;}\
${HOMEBREW_PREFIX_PATH:+$HOMEBREW_PREFIX_PATH;}\
$CURRENT_DIRECTORY/build/opm-common;\
$CURRENT_DIRECTORY/build/opm-grid"

# ── 1. Clone repos ────────────────────────────────────────────────────────────

echo "========================================"
echo "  Step 1 — clone repositories"
echo "========================================"

# clone_repo DIR URL REF [COMMIT]: clone REF (branch or tag) into DIR and, if
# COMMIT is given, reset the branch to that commit.  An existing checkout is
# left alone, so local work is never switched away or overwritten; it only
# warns when the checkout is not on REF (or not at COMMIT).
clone_repo() {
    local dir=$CURRENT_DIRECTORY/$1 url=$2 ref=$3 commit=${4:-}
    if [[ ! -d "$dir" ]]; then
        git -c advice.detachedHead=false clone --branch "$ref" "$url" "$dir"
        if [[ -n "$commit" ]]; then
            git -C "$dir" reset --hard "$commit"
        fi
        return 0
    fi
    local current
    current=$(git -C "$dir" describe --tags --exact-match 2>/dev/null \
              || git -C "$dir" branch --show-current)
    if [[ "$current" != "$ref" ]]; then
        echo "  WARNING: $1 already present on '${current:-detached HEAD}', expected $ref;" \
             "leaving it as is (run: git -C $dir checkout $ref)"
    elif [[ -n "$commit" && "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
        echo "  WARNING: $1 on $ref but not at the pinned commit ${commit:0:9};" \
             "leaving it as is"
    else
        echo "  $1 already present on $ref${commit:+ at ${commit:0:9}}"
    fi
}

for module in common geometry grid istl; do
    clone_repo "dune-$module" "https://gitlab.dune-project.org/core/dune-$module.git" "$DUNE_VERSION"
done

# opm-common and opm-grid: upstream OPM release, no local patches needed.
clone_repo opm-common "https://github.com/OPM/opm-common.git" "$OPM_RELEASE" "$OPM_COMMON_COMMIT"
clone_repo opm-grid   "https://github.com/OPM/opm-grid.git"   "$OPM_RELEASE" "$OPM_GRID_COMMIT"

# opm-simulators: NeuralCprPolicy, neural AMG and abort policies, deck guards,
# doc/neural_cpr training scripts.
clone_repo opm-simulators "https://github.com/bika-git/opm-simulators.git" "$OPM_SIM_BRANCH"

# decks: Norne, SPE9, SPE11A/B, SPE10, Drogon, spe11_decks (replaces opm-tests).
clone_repo decks "https://github.com/bika-git/decks.git" "$DECKS_BRANCH"

for f in "$ZOLTAN_INCLUDE_DIR/zoltan.h" "$ZOLTAN_LIBRARY"; do
    [[ -e "$f" ]] || echo "  WARNING: $f not found; set ZOLTAN_INCLUDE_DIR/ZOLTAN_LIBRARY" \
                          "(Ubuntu: apt install libtrilinos-zoltan-dev," \
                          "macOS: brew install trilinos)"
done

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

if [[ "$CLEAN_BUILD" == 1 ]]; then
    echo "  CLEAN_BUILD=1: removing build/"
    rm -rf build
fi

mkdir -p build/opm-common
cd "$CURRENT_DIRECTORY/build/opm-common"
cmake -DUSE_MPI=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$OPM_PREFIX_PATH" \
      "$CURRENT_DIRECTORY/opm-common"
make -j"$NJOBS"
cd "$CURRENT_DIRECTORY"

mkdir -p build/opm-grid
cd "$CURRENT_DIRECTORY/build/opm-grid"
cmake -DUSE_MPI=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$OPM_PREFIX_PATH" \
      -DZOLTAN_INCLUDE_DIR="$ZOLTAN_INCLUDE_DIR" \
      -DZOLTAN_LIBRARY="$ZOLTAN_LIBRARY" \
      "$CURRENT_DIRECTORY/opm-grid"
make -j"$NJOBS"
cd "$CURRENT_DIRECTORY"

mkdir -p build/opm-simulators
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
echo "  decks       : $CURRENT_DIRECTORY/decks/"
echo "  configs     : $CURRENT_DIRECTORY/cpr_sweep_data/sweep_spe11/configs/"
echo ""
echo "  Run the benchmark (32 ranks x 2 threads, as for the published results):"
echo "    NPROCS=32 FLOW_THREADS=2 MPI_ARGS=\"--bind-to none\" ./bench_neural_cpr.sh"
echo ""
echo "  Neural AMG + neural abort against plain flow on one deck:"
echo "    ./compare_neural_cpr.sh --amg-model models/amg_policy/cpr_amg_policy.model \\"
echo "        --abort-model models/abort_policy/abort_spe11a_norne_spe9.model \\"
echo "        --deck decks/spe9/SPE9.DATA --np 32 --threads 2 --mpi-args \"--bind-to none\""
echo "========================================"
