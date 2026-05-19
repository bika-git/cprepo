# Compare Neural CPR

`compare_neural_cpr.sh` runs the same deck three ways:

```text
default       Flow without OPM_NEURAL_CPR_WEIGHTS
neural        Flow with a trained .model, no CPR reuse flags
neural-reuse  Flow with the same .model plus CPR reuse flags
```

This is the simple script to give someone on another machine when they want to
test a trained Neural CPR model against their local Flow build.

## Requirements

- A `flow` binary built with Neural CPR support.
- A trained `.model` file.
- A deck, for example `SPE11B.DATA`.
- MPI if running with `--np` greater than `1`.

## SPE11B Example

From this repository:

```bash
./compare_neural_cpr.sh \
  --model tmp/cpr_policy_retrained_20260518_054314.model \
  --deck opm-tests/spe11b/SPE11B.DATA \
  --np 1 \
  --spe11-safe \
  --background
```

On another machine:

```bash
./compare_neural_cpr.sh \
  --flow /path/to/flow \
  --model /path/to/cpr_policy_retrained_20260518_054314.model \
  --deck /path/to/SPE11B.DATA \
  --np 1 \
  --spe11-safe \
  --background
```

`--spe11-safe` sets:

```bash
OPM_NEURAL_CPR_FORBID_LABELS=8,16,17,18,19,20,21,22,23
```

## Output Layout

The script creates an output root like:

```text
tmp/compare_neural_cpr/SPE11B_YYYYMMDD_HHMMSS/
```

Inside it:

```text
default/
neural/
neural-reuse/
```

Each run directory contains:

```text
flow.log       Flow stdout/stderr
launcher.log   small wrapper log
exit_code      process exit code, 0 means success
```

For SPE11B, watch progress with:

```bash
tail -f tmp/compare_neural_cpr/SPE11B_*/neural-reuse/flow.log
```

## Foreground Mode

Without `--background`, the script runs the three cases sequentially and prints
a final summary:

```bash
./compare_neural_cpr.sh \
  --model model.model \
  --deck CASE.DATA \
  --np 8
```

## Useful Options

```text
--reuse-setup N       default 4
--reuse-interval N    default 10
--confidence X        default 0.6
--forbid-labels LIST  custom forbidden labels
--mpi-args "ARGS"     extra mpirun args before -np
```

If Open MPI shared-memory transport fails, try TCP:

```bash
OMPI_MCA_btl=tcp,self ./compare_neural_cpr.sh \
  --model model.model \
  --deck SPE11B.DATA \
  --np 1 \
  --spe11-safe \
  --background
```
