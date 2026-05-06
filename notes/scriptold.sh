CURRENT_DIRECTORY="$PWD"

for module in common geometry grid istl
do   git clone https://gitlab.dune-project.org/core/dune-$module.git --branch v2.9.1
done
for module in common geometry grid istl
do   rm -f "$CURRENT_DIRECTORY/dune-$module/build-cmake/CMakeCache.txt"
     ./dune-common/bin/dunecontrol --only=dune-$module cmake -DCMAKE_BUILD_TYPE=Release -DDUNE_ENABLE_PYTHONBINDINGS=0 -DCMAKE_DISABLE_FIND_PACKAGE_SuiteSparse=true -DCMAKE_DISABLE_FIND_PACKAGE_UMFPACK=true
     ./dune-common/bin/dunecontrol --only=dune-$module make -j5
done

for repo in common grid simulators
do
    git clone https://github.com/OPM/opm-$repo.git
done


mkdir build

# for repo in common grid models
# do
#     mkdir build/opm-$repo
#     cd build/opm-$repo
#     cmake  -DWITH_NDEBUG=1 -DUSE_MPI=0  -DCMAKE_DISABLE_FIND_PACKAGE_SuperLU=true -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CURRENT_DIRECTORY/dune-common/build-cmake;$CURRENT_DIRECTORY/dune-grid/build-cmake;$CURRENT_DIRECTORY/dune-geometry/build-cmake;$CURRENT_DIRECTORY/dune-istl/build-cmake;$CURRENT_DIRECTORY/build/opm-common;/Users/macbookn/activopmwkspc/soft/Trilinos" $CURRENT_DIRECTORY/opm-$repo
#     make -j1 opm$repo
#     cd ../..
# done
#
# mkdir build/opm-simulators
# cd build/opm-simulators
# cmake -DUSE_MPI=0 -DWITH_NDEBUG=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CURRENT_DIRECTORY/dune-common/build-cmake;$CURRENT_DIRECTORY/dune-grid/build-cmake;$CURRENT_DIRECTORY/dune-geometry/build-cmake;$CURRENT_DIRECTORY/dune-istl/build-cmake;$CURRENT_DIRECTORY/build/opm-common;$CURRENT_DIRECTORY/build/opm-grid;/Users/macbookn/activopmwkspc/soft/Trilinos" $CURRENT_DIRECTORY/opm-simulators
# make -j1 flow
# cd ../..


DUNE_PREFIX="$CURRENT_DIRECTORY/dune-common/build-cmake;$CURRENT_DIRECTORY/dune-grid/build-cmake;$CURRENT_DIRECTORY/dune-geometry/build-cmake;$CURRENT_DIRECTORY/dune-istl/build-cmake"
ZOLTAN_LIB="$CURRENT_DIRECTORY/Trilinos/build/packages/zoltan/src/libzoltan.dylib"
ZOLTAN_INC="$CURRENT_DIRECTORY/Trilinos/packages/zoltan/src/include"
TRILINOS_INC="$CURRENT_DIRECTORY/Trilinos/build/include"
INSTALL_PREFIX="$CURRENT_DIRECTORY/build/ThirdParty"

mkdir -p "$INSTALL_PREFIX"

for repo in common grid
do
    rm -f "$CURRENT_DIRECTORY/build/opm-$repo/CMakeCache.txt"
    mkdir -p "$CURRENT_DIRECTORY/build/opm-$repo"
    cd "$CURRENT_DIRECTORY/build/opm-$repo"
    cmake -DUSE_MPI=1 -DCMAKE_DISABLE_FIND_PACKAGE_SuperLU=true -DCMAKE_BUILD_TYPE=Release \
        -DSuiteSparse_ROOT=/opt/homebrew/opt/suite-sparse \
        -DCMAKE_CXX_FLAGS="-I/opt/homebrew/opt/suite-sparse/include/suitesparse" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX;$DUNE_PREFIX;$CURRENT_DIRECTORY/build/opm-common;$CURRENT_DIRECTORY/build/opm-grid" \
        -DZOLTAN_LIBRARY="$ZOLTAN_LIB" \
        -DZOLTAN_INCLUDE_DIR="$ZOLTAN_INC;$TRILINOS_INC" \
        "$CURRENT_DIRECTORY/opm-$repo"
    make -j4 libopm${repo} && make install
    cd "$CURRENT_DIRECTORY"
done

rm -f "$CURRENT_DIRECTORY/build/opm-simulators/CMakeCache.txt"
mkdir -p "$CURRENT_DIRECTORY/build/opm-simulators"
cd "$CURRENT_DIRECTORY/build/opm-simulators"
cmake -DUSE_MPI=1 -DUSE_GPU_BRIDGE=0 -DCMAKE_BUILD_TYPE=Release \
    -DSuiteSparse_ROOT=/opt/homebrew/opt/suite-sparse \
        -DUMFPACK_INCLUDE_DIR=/opt/homebrew/opt/suite-sparse/include/suitesparse \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX;$DUNE_PREFIX;$CURRENT_DIRECTORY/build/opm-common;$CURRENT_DIRECTORY/build/opm-grid" \
    -DZOLTAN_LIBRARY="$ZOLTAN_LIB" \
    -DZOLTAN_INCLUDE_DIR="$ZOLTAN_INC;$TRILINOS_INC" \
    "$CURRENT_DIRECTORY/opm-simulators"
make -j4 flow
cd "$CURRENT_DIRECTORY"
