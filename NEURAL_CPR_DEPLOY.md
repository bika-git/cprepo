# Deploying Trained Neural CPR Models

Use `bench_neural_cpr.sh` for the full research pipeline: sweep, label, train, and
benchmark. Use `run_neural_cpr.sh` when you already have trained `.model` files
and want to apply them to a Flow deck on this or another machine.

All of this needs a `flow` binary built from the opm-simulators branch
`neural-cpr-control`.

## The Three Neural Policies

| Policy | Flag | What the model decides |
|---|---|---|
| Neural AMG | `--amg-model` | At each CPR setup rebuild: prolongation damping {1.0, 1.3, 1.6} and coarsening target {1200, 600, 300} of the pressure AMG. Keeps Flow's default (1.0, 1200) unless the predicted log-cost gain exceeds `--min-gain` (default 0.05). |
| Neural abort | `--abort-model` | Before each CPR solve: the probability that the timestep attempt will be chopped anyway. Above `--abort-threshold` (default 0.95) the solve fails at once and Flow chops the timestep now. |
| Neural CPR variant | `--model` | The CPR variant: pressure weights × cpr/cprw × fine and coarse smoother (24 configs). |

Current models:

```text
tmp/amg_policy/cpr_amg_policy.model                 neural AMG,   trained on SPE11A, Norne, SPE9
tmp/abort_policy/abort_spe11a_norne_spe9.model      neural abort, trained on SPE11A, Norne, SPE9
```

`--model` and `--amg-model` both use `OPM_NEURAL_CPR_WEIGHTS`, so a single run
can use only one of them. The neural abort policy combines with either one, or
with plain Flow.

## Apply Models

The neural AMG and neural abort policies together:

```bash
./run_neural_cpr.sh \
  --amg-model tmp/amg_policy/cpr_amg_policy.model \
  --abort-model tmp/abort_policy/abort_spe11a_norne_spe9.model \
  --deck CASE.DATA \
  --np 32 --threads 2 --mpi-args "--bind-to none"
```

`--np 32 --threads 2 --mpi-args "--bind-to none"` is the benchmark setup on
cno-0008. Use what fits your machine.

If the Flow binary is elsewhere, add `--flow /path/to/flow`.

The neural CPR variant policy alone (positional form):

```bash
./run_neural_cpr.sh cpr_policy.model CASE.DATA --np 8
```

### CPR Reuse

By default the wrapper adds no reuse flags, so Flow uses its own defaults
(setup 4, interval 30). To set them explicitly:

```bash
./run_neural_cpr.sh --amg-model amg.model --deck CASE.DATA \
  --reuse --reuse-setup 4 --reuse-interval 10
```

### SPE11 Decks

When the deck has no oil, Flow from `neural-cpr-control` itself blocks the
trueimpesanalytic labels (16–23) for the neural CPR variant policy.
`--spe11-safe` sets the same thing explicitly, for older builds:

```bash
OPM_NEURAL_CPR_FORBID_LABELS=16,17,18,19,20,21,22,23
```

Do not forbid label 8 on these decks. When a deck has no wells, the guards
already block the cprw labels, so label 8 is Flow's default solver.

## Package for Another Machine

```bash
./package_neural_cpr_model.sh \
  --amg-model tmp/amg_policy/cpr_amg_policy.model \
  --abort-model tmp/abort_policy/abort_spe11a_norne_spe9.model \
  --name neural-cpr-v2
```

This writes:

```text
dist/neural-cpr-v2/
dist/neural-cpr-v2.tar.gz
```

The bundle contains:

- the models
- `neural_cpr_model.json`, with checksums and settings
- a README with ready-to-run commands
- `run_neural_cpr.sh` and `compare_neural_cpr.sh`

`--np-hint` and `--threads-hint` (default 32 and 2) set the rank and thread
counts used in the README examples. `--trained-on` sets the training-deck note.
Use `--no-tar` to skip the tarball.

The receiving machine still needs a compatible `flow` binary.

## Compare Against Plain Flow

`compare_neural_cpr.sh` runs plain Flow plus one arm for each policy you pass.
With `--amg-model` and `--abort-model` that means four arms: `default`,
`neural-amg`, `neural-abort`, and `neural-amg+abort`.

```bash
./compare_neural_cpr.sh \
  --amg-model tmp/amg_policy/cpr_amg_policy.model \
  --abort-model tmp/abort_policy/abort_spe11a_norne_spe9.model \
  --deck /path/to/SPE11A.DATA \
  --np 32 --threads 2 --mpi-args "--bind-to none"
```

For the arms, the output layout and reference results, see
`COMPARE_NEURAL_CPR_README.md`.

## Direct Runtime Contract

The wrapper is only a convenience. Flow reads the policies from environment
variables:

```bash
# Neural AMG policy
OPM_NEURAL_CPR_WEIGHTS=/path/to/cpr_amg_policy.model \
OPM_NEURAL_CPR_ACTIONS=amg \
OPM_NEURAL_CPR_MIN_GAIN=0.05 \
# Neural abort policy
OPM_CPR_ABORT_MODEL=/path/to/abort.model \
OPM_CPR_ABORT_THRESHOLD=0.95 \
flow /path/to/CASE.DATA
```

The neural CPR variant policy instead uses `OPM_NEURAL_CPR_WEIGHTS=model` and
`OPM_NEURAL_CPR_CONFIDENCE=0.6`, plus `OPM_NEURAL_CPR_FORBID_LABELS` if needed.

To run only the neural abort policy, set `OPM_NEURAL_CPR_DISABLE=1` so that the
CPR choice stays Flow's default. If `OPM_NEURAL_CPR_WEIGHTS` is unset, Flow falls
back to its non-ML policy. If `OPM_CPR_ABORT_MODEL` is unset, Flow never aborts
early.
