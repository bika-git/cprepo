# Neural AMG policy models

Each model must be run with a Flow built from the matching `opm-simulators`
branch (https://github.com/bika-git/opm-simulators). A different build computes
the inputs differently or maps them to other slots, and feeds the model wrong
values without warning.

| Model | Inputs | `opm-simulators` branch | Commit |
|---|---|---|---|
| `cpr_amg_policy.model` | 12 | `neural-cpr-control` | |
| `cpr_amg_policy_6feat_newton_x3.model` | 6, no diagonal-dominance input | `neural-amg-6feat` | `6e6814dc7` |
| `cpr_amg_policy_7feat_allrows_newton_x3.model` | 7, `diag_dominance` over all owned rows of all ranks | `neural-amg-7feat-allrows` | `15dd586ad` |

On OPM master the same commits are on `neural-cpr-master` (for
`cpr_amg_policy.model`) and `neural-amg-7feat-allrows-master` (for the
seven-input model, where `9c1a83754` is the commit `15dd586ad` of the table).
`setup_bench_env.sh` on this branch checks out `neural-amg-7feat-allrows-master`.
For the six-input model, switch `opm-simulators` to `neural-amg-6feat` (OPM
release/2026.10) and rebuild Flow before running.

The six- and seven-input files are byte-identical copies of
`doc/neural_cpr/models/` on their branches; `doc/neural_cpr/models/README.md`
there describes the inputs, the training data (Norne, SPE11A, SPE9) and how to
reproduce each file.

## Running

```bash
./run_neural_cpr.sh --amg-model models/amg_policy/<model> --deck CASE.DATA \
    --np 32 --threads 2 --mpi-args "--bind-to none"
```

## Results

Wall time against Flow's default, 32 ranks x 2 threads, one run per arm,
2026-11-13/14, Flow built from each branch at the commit above. Only Norne,
SPE11A and SPE9 were in the training data. Single runs carry a few percent of
timing noise.

| Deck | Six inputs | Seven inputs |
|---|---|---|
| SPE11A (trained on) | -19 % | -26 % |
| SPE11B | -8 % | -13 % |
| Drogon HIST | no change in time; Krylov iterations +44 % | no change in time; Krylov iterations +17 % |
| Smeaheia | +25 % | +18 % |
| Sleipner | +26 % | +13 % |

Both models are slower than Flow's default on Smeaheia and Sleipner. The
12-input model is slower than the default there too (Sleipner: 2026-10-18 on a
`neural-cpr-control` build, +4 %; Smeaheia: 2026-11-14 on the `neural-amg-6feat`
build, +21 %).
