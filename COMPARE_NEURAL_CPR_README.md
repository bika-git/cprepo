# Compare Neural CPR

`compare_neural_cpr.sh` runs the same deck with plain Flow, then once more for
each neural policy you pass:

```text
default           plain Flow, policies disabled                   always
neural            neural CPR variant policy, Flow's reuse defaults --model
neural-reuse      neural CPR variant policy + explicit reuse flags --model (skip with --no-reuse-arm)
neural-amg        neural AMG policy                               --amg-model
neural-abort      neural abort policy on the default solver       --abort-model
neural-amg+abort  neural AMG and neural abort together            --amg-model and --abort-model
```

What the models decide:

- **neural AMG**: the pressure-AMG prolongation damping and coarsening target,
  chosen at each CPR setup rebuild.
- **neural abort**: whether to give up on a timestep attempt that will be
  chopped anyway.
- **neural CPR variant**: one of 24 CPR configurations.

`NEURAL_CPR_DEPLOY.md` has the details.

Give this script to someone on another machine who wants to test trained models
against their local Flow build.

## Requirements

- A `flow` binary built from the opm-simulators branch `neural-cpr-control`.
- At least one trained `.model`.
- A deck, for example `SPE11A.DATA`.
- MPI, if you run with `--np` greater than `1`.

## Example: Neural AMG and Neural Abort

From this repository:

```bash
./compare_neural_cpr.sh \
  --amg-model models/amg_policy/cpr_amg_policy.model \
  --abort-model models/abort_policy/abort_spe11a_norne_spe9.model \
  --deck /path/to/SPE11A.DATA \
  --np 32 --threads 2 --mpi-args "--bind-to none"
```

On another machine, add `--flow /path/to/flow` and use the model paths from the
bundle.

## Output Layout

The script creates an output root like:

```text
tmp/compare_neural_cpr/SPE11A_YYYYMMDD_HHMMSS/
```

It has one directory per arm, for example:

```text
default/
neural-amg/
neural-abort/
neural-amg+abort/
```

Each run directory contains Flow's output files (`*.PRT`, `*.DBG`, `*.UNRST`,
...) and:

```text
flow.log       Flow stdout/stderr
launcher.log   small wrapper log
exit_code      process exit code; 0 means success
```

To watch a run:

```bash
tail -f tmp/compare_neural_cpr/SPE11A_*/neural-amg+abort/flow.log
```

## Foreground vs Background

Without `--background`, the arms run one after another and the script prints a
summary with these fields for each arm:

- exit code
- wall time
- linear solve time
- Newton and linear iterations
- number of neural aborts (`NeuralAbortPolicy: giving up` lines in the `.DBG`)

`--background` starts all arms at once and returns immediately. The runs then
compete for cores, so use it only for smoke tests, not for timing.

## Reference Results (SPE11A, 2026-10-16)

Each set was run in the foreground with 8 ranks × 2 threads and the current
neural AMG and neural abort models. Wasted is the share of work spent on
timestep attempts that were chopped.

Full SPE11A deck (31034 cells), `tmp/compare_neural_cpr/SPE11A_20261016_131346`:

| Arm | Wall | Linear solve | Newton its (wasted) | Linear its (wasted) | Aborts |
|---|---|---|---|---|---|
| default | 302.2 s | 179.2 s | 14053 (31.8%) | 75885 (37.1%) | 0 |
| neural-amg | 287.8 s (−5%) | 157.3 s | 13320 (32.8%) | 71181 (39.4%) | 0 |
| neural-abort | 262.8 s (−13%) | 155.4 s | 12458 (25.5%) | 62973 (26.6%) | 197 |
| neural-amg+abort | **250.9 s (−17%)** | 140.6 s | 12455 (30.0%) | 60647 (31.3%) | 241 |

Small spe11a example deck (3916 cells), `tmp/compare_neural_cpr/SPE11A_20261016_133605`:

| Arm | Wall | Linear solve | Newton its (wasted) | Linear its (wasted) | Aborts |
|---|---|---|---|---|---|
| default | 21.1 s | 16.2 s | 2813 (60.7%) | 12675 (58.9%) | 0 |
| neural-amg | 13.0 s (−38%) | 9.1 s | 2637 (58.5%) | 13662 (59.6%) | 0 |
| neural-abort | 27.5 s (+30%) | 20.8 s | 3374 (30.1%) | 17958 (32.2%) | 114 |
| neural-amg+abort | **10.1 s (−52%)** | 7.0 s | 2039 (50.1%) | 10413 (50.4%) | 61 |

On the small deck, the neural abort policy alone is slower than plain Flow. It
cuts the wasted share, but the earlier chops lead to more Newton and linear
iterations in total. Combined with the neural AMG policy, it gives the best
result on both decks.

These are single runs at 8 ranks. The benchmark numbers use 32 ranks ×
2 threads with `--bind-to none`.

## Useful Options

```text
--np N                       MPI ranks (default 1)
--threads N                  threads per rank (--threads-per-process, OMP_NUM_THREADS)
--mpi-args "ARGS"            extra mpirun args before -np
--min-gain G                 neural AMG margin on predicted log-cost gain (default 0.05)
--abort-threshold P          neural abort chop-probability threshold (default 0.95)
--confidence X               neural CPR variant confidence gate (default 0.6)
--forbid-labels LIST         neural CPR variant labels to forbid
--spe11-safe                 forbid labels 16-23 (trueimpesanalytic); automatic on
                             decks without oil in neural-cpr-control builds
--reuse-setup N              neural-reuse arm (default 4)
--reuse-interval N           neural-reuse arm (default 10)
--no-reuse-arm               skip the neural-reuse arm
--default-reuse-setup N      all other arms (default 4, Flow's)
--default-reuse-interval N   all other arms (default 30, Flow's)
--out DIR                    output root
-- FLOW_OPTIONS...           passed to flow for every arm
```

If Open MPI's shared-memory transport fails, try TCP:

```bash
OMPI_MCA_btl=tcp,self ./compare_neural_cpr.sh \
  --amg-model amg.model --abort-model abort.model \
  --deck SPE11A.DATA --np 8
```
