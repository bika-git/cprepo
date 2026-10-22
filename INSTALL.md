# Installing on a New Machine

This guide sets up everything needed to run the neural CPR work on a fresh
Linux machine: Flow with the neural CPR policies, the decks and the trained
models. One script does the work; you only need the system packages and
GitHub access first.

## 1. Install the system packages

On Ubuntu:

```bash
sudo apt install build-essential cmake git openmpi-bin libopenmpi-dev \
  libboost-all-dev libsuitesparse-dev libblas-dev liblapack-dev \
  libmetis-dev libparmetis-dev libtrilinos-zoltan-dev libfmt-dev python3-numpy
```

These are what the build uses: a C++ compiler with OpenMP (tested with
GCC 14), CMake, MPI, Boost, SuiteSparse, BLAS/LAPACK, METIS/ParMETIS, Zoltan
and fmt. Python needs only `numpy` beyond the standard library.

On other distributions, install the equivalent packages. If Zoltan is not in
`/usr/include/trilinos` and `/usr/lib/x86_64-linux-gnu`, point the setup to it:

```bash
export ZOLTAN_INCLUDE_DIR=/path/to/zoltan/include
export ZOLTAN_LIBRARY=/path/to/libzoltan.so
```

## 2. Log in to GitHub

`bikagit/cprepo` needs a GitHub login. Everything else the setup downloads is
public.

With GitHub CLI:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
```

Without it, create a personal access token at
<https://github.com/settings/tokens> with read access to `bikagit/cprepo`, and
paste it when git asks for a password.

## 3. Clone and run the setup

```bash
git clone -b neural-cpr-control https://github.com/bikagit/cprepo.git
cd cprepo
bash setup_bench_env.sh
```

The script:

1. Clones the source code and the decks into `cprepo/`:

   | Folder | Source | Version |
   |---|---|---|
   | `dune-common`, `dune-geometry`, `dune-grid`, `dune-istl` | dune-project.org | `v2.11.0` |
   | `opm-common`, `opm-grid` | OPM on GitHub | `release/2026.10`, pinned to the commits the results were built with |
   | `opm-simulators` | `bikagit/opm-simulators` | branch `neural-cpr-control` (the neural CPR policies) |
   | `decks` | `bikagit/decks` | branch `main` |

2. Builds the Dune modules.
3. Builds opm-common, opm-grid and Flow into `build/`.
4. Writes the 24 CPR solver configurations used by the benchmark.

A full build takes a while: it compiles the whole OPM stack.

**Rerunning is safe.** Repositories that are already there are left as they
are; the script only warns if one is on a different branch than expected. The
build continues where it stopped.

**Options.** Set these before running the script:

| Variable | Effect |
|---|---|
| `NJOBS=N` | Number of parallel build jobs. Default: all cores. Lower it on a shared machine. |
| `CLEAN_BUILD=1` | Delete `build/` and build everything from scratch. |
| `OPM_COMMON_COMMIT= OPM_GRID_COMMIT=` | Use the latest `release/2026.10` instead of the pinned commits. Results may then differ slightly. |
| `ZOLTAN_INCLUDE_DIR`, `ZOLTAN_LIBRARY` | Zoltan location (see step 1). |

Example: `NJOBS=16 bash setup_bench_env.sh`.

## 4. What you have afterwards

| Path | What it is |
|---|---|
| `build/opm-simulators/bin/flow` | Flow, with the neural CPR policies |
| `decks/` | The decks: Norne, SPE9, SPE10, SPE11A/B/C, Drogon, Smeaheia (see `decks/README.md`) |
| `models/amg_policy/cpr_amg_policy.model` | Neural AMG policy: chooses the pressure-AMG settings at each CPR setup |
| `models/abort_policy/abort_spe11a_norne_spe9.model` | Neural abort policy: gives up early on timestep attempts that will fail anyway |

Both models were trained on SPE11A, Norne and SPE9.

## 5. Check that it works

Compare plain Flow with the two neural policies on SPE9 (a small deck; each
run takes seconds):

```bash
./compare_neural_cpr.sh \
  --amg-model models/amg_policy/cpr_amg_policy.model \
  --abort-model models/abort_policy/abort_spe11a_norne_spe9.model \
  --deck decks/spe9/SPE9.DATA --np 8
```

It runs four cases one after the other and ends with a summary of wall time,
linear solve time and iteration counts. Every run should show `exit=0`.

Next steps:

- `NEURAL_CPR_DEPLOY.md`: applying the models to your own decks.
- `COMPARE_NEURAL_CPR_README.md`: the comparison script in detail, with
  reference results.
- `bench_neural_cpr.sh`: the full sweep, train and benchmark pipeline. On a
  32-core machine: `NPROCS=32 FLOW_THREADS=2 MPI_ARGS="--bind-to none" ./bench_neural_cpr.sh`.

## Known limitations

- **Some deck files are compressed** to stay under GitHub's 100 MB file limit.
  Unpack them with `gunzip -k` before running those decks:
  - SPE11C r2 and r3: `decks/spe11_decks/benchmark/spe11c/r{2,3}_*/GRID.INC.gz`
  - Smeaheia: `decks/simulation_models_SMEAHEIA/.../Statoil_Feasibility_sim_model_with_depletion_KROSS_INJ_SECTOR_20.DATA.gz`
- **SPE11C r4 has no grid file** (it is 4.2 GB). `decks/README.md` explains
  how to regenerate it.
- **SPE10 model 1 is missing.** `bench_neural_cpr.sh` lists it, but the deck
  is not available. That one benchmark case fails and is logged, and the rest
  of the run continues.
- **Flow from this build is not plain Flow by default.** Without any
  settings, it uses a fixed-rule CPR policy. For true release behaviour, set
  `OPM_NEURAL_CPR_DISABLE=1`. The comparison and benchmark scripts already do
  this for their `default` runs.
