# Deploying a Trained Neural CPR Model

Use `bench_norne.sh` for the full research pipeline: sweep, label, train, and
benchmark. Use `run_neural_cpr.sh` when you already have a trained `.model` and
want to apply it to a Flow deck on this or another machine.

## Apply a Model

```bash
./run_neural_cpr.sh --model tmp/cpr_policy_retrained.model --deck opm-tests/norne/NORNE_ATW2013.DATA --np 8
```

If the Flow binary is elsewhere:

```bash
./run_neural_cpr.sh \
  --flow /path/to/flow \
  --model /path/to/cpr_policy.model \
  --deck /path/to/CASE.DATA \
  --np 8
```

By default the wrapper enables CPR reuse:

```bash
--cpr-reuse-setup=4 --cpr-reuse-interval=10
```

Disable that for experiments with:

```bash
./run_neural_cpr.sh --model model.model --deck CASE.DATA --no-reuse
```

For SPE11-like cases, use the conservative guard used by the benchmark script:

```bash
./run_neural_cpr.sh --model model.model --deck SPE11.DATA --spe11-safe
```

That sets:

```bash
OPM_NEURAL_CPR_FORBID_LABELS=8,16,17,18,19,20,21,22,23
```

## Package for Another Machine

Create a bundle:

```bash
./package_neural_cpr_model.sh --model tmp/cpr_policy_retrained.model --name neural-cpr-v1
```

This writes:

```text
dist/neural-cpr-v1/
dist/neural-cpr-v1.tar.gz
```

The bundle contains the model, a checksum/metadata JSON file, a README, and the
apply-only runner. The receiving machine still needs a compatible `flow` binary
built with Neural CPR support.

## Compare Default vs Neural vs Neural-Reuse

Use `compare_neural_cpr.sh` when you want one command to run all three variants:

```bash
./compare_neural_cpr.sh \
  --model tmp/cpr_policy_retrained.model \
  --deck opm-tests/spe11b/SPE11B.DATA \
  --np 1 \
  --spe11-safe
```

On another machine:

```bash
./compare_neural_cpr.sh \
  --flow /path/to/flow \
  --model /path/to/cpr_policy.model \
  --deck /path/to/SPE11B.DATA \
  --np 1 \
  --spe11-safe
```

To launch the three runs and return immediately:

```bash
./compare_neural_cpr.sh \
  --flow /path/to/flow \
  --model /path/to/cpr_policy.model \
  --deck /path/to/SPE11B.DATA \
  --np 1 \
  --spe11-safe \
  --background
```

The output root contains:

```text
default/
neural/
neural-reuse/
```

Each subdirectory has `flow.log`, `launcher.log`, and `exit_code`.

## Direct Runtime Contract

The wrapper is just convenience. The minimal runtime invocation is:

```bash
OPM_NEURAL_CPR_WEIGHTS=/path/to/model.model \
OPM_NEURAL_CPR_CONFIDENCE=0.6 \
flow --cpr-reuse-setup=4 --cpr-reuse-interval=10 /path/to/CASE.DATA
```

If `OPM_NEURAL_CPR_WEIGHTS` is unset, Flow falls back to the non-ML policy.
