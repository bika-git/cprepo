CURRENT_DIRECTORY="$PWD"

# Trilinos already built and installed to build/ThirdParty

for module in common geometry grid istl
do 
    cd "$CURRENT_DIRECTORY/dune-$module"
    git checkout v2.11.0
    mkdir -p build && cd build
    cmake -DDUNE_ENABLE_PYTHONBINDINGS=0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CURRENT_DIRECTORY/build/ThirdParty" "$CURRENT_DIRECTORY/dune-$module"
    make -j8 && make install
    cd "$CURRENT_DIRECTORY"
done



# CURRENT_DIRECTORY="$PWD"


# for repo in common grid simulators
# do  git clone https://github.com/bikagit/opm-$repo.git || true
# done

mkdir -p build

for repo in common grid simulators
do
    mkdir -p "$CURRENT_DIRECTORY/build/opm-$repo"
    cd "$CURRENT_DIRECTORY/build/opm-$repo"
    cmake -DUSE_GPU_BRIDGE=0 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$CURRENT_DIRECTORY/build/ThirdParty" \
        -DCMAKE_PREFIX_PATH="$CURRENT_DIRECTORY/build/ThirdParty" \
        "$CURRENT_DIRECTORY/opm-$repo"
    if [[ $repo == simulators ]]; then
        make -j8 flow
    else
        make -j8 libopm$repo && make -j8 install
    fi
    cd "$CURRENT_DIRECTORY"
done
