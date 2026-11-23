# Neural CPR cost models

These models predict the cost of each of the 24 CPR variants for the solve
about to be set up. Flow runs them in anchored mode: it stays on Flow's default
solver (label 12 on decks with wells, label 8 on decks without) and switches to
another variant only when that variant is predicted to be more than
`OPM_NEURAL_CPR_MIN_GAIN` cheaper in log-cost (0.05 ≈ 5 %). After a failed
solve, Flow goes back to the default. This differs from the CPR variant
classifier (`models/cpr_policy_retrained.model`), which follows its top label
whenever its confidence is at least 0.6.

| Model | Inputs | Training target |
|---|---|---|
| `cpr_cost_tau1.model` | 12 | log-cost of the solve (denoised) |
| `cpr_cost_newton21.model` | 21 | the same, plus log(solves of the variant's run / solves of the default run) around the report step, so a variant is also charged for the extra Newton iterations and timestep cuts it causes |

Both were trained on 2026-11-16/17 from the `cpr_sweep_data` sweeps
`sweep_run1_fresh`, `sweep_run2_fresh`, `sweep_spe11_fresh` and
`sweep_spe10mod01_fresh` (Norne, SPE9, SPE10 model 1, SPE11A, SPE11B), run at
Flow's CPR settings (setup reuse 4/30). Each model is an ensemble of 5 networks
with tau = 1.

Run them on a Flow built from `opm-simulators` branch `neural-amg-7feat-allrows`
(OPM release/2026.10) or `neural-amg-7feat-allrows-master` (the same commits on
OPM master). Running a cost model together with a neural AMG model
(`OPM_NEURAL_CPR_AMG_WEIGHTS`) needs the `ISTLSolver.hpp` change on that
branch. `cpr_cost_newton21.model` was trained with `--newton-cost` and
`--n-features` in `doc/neural_cpr/train_cpr_cost.py`, which are also on that
branch.

## Reproducing

From `opm-simulators/doc/neural_cpr`, with `$CSVS` set to the non-empty `*.csv`
files of the four sweeps above:

```bash
# cpr_cost_tau1.model
python3 train_cpr_cost.py --csvs $CSVS --merge-cprw-decks SPE11A SPE11B --denoise \
    --ensemble 5 --balance-decks --tau 1 --margins 0.0 0.02 0.05 0.1 0.2 \
    --baseline 12 --out cpr_cost_tau1.model

# cpr_cost_newton21.model: the same, plus
    --newton-cost --n-features 21
```

## Running

```bash
export OPM_NEURAL_CPR_WEIGHTS=models/cost_policy/cpr_cost_tau1.model
export OPM_NEURAL_CPR_MIN_GAIN=0.05
# optional, neural AMG model on top:
export OPM_NEURAL_CPR_AMG_WEIGHTS=models/amg_policy/cpr_amg_policy_7feat_allrows_newton_x3.model
export OPM_NEURAL_CPR_AMG_MIN_GAIN=0.05
mpirun --bind-to none -np 32 build/opm-simulators/bin/flow --threads-per-process=2 CASE.DATA
```

## Results

Simulation time against Flow's default, 32 ranks x 2 threads, 2026-11-16/17,
mean of 3 runs (newton21: 1 run). Norne, SPE9, SPE10 model 1, SPE11A and SPE11B
were in the training data; Norne MSW and Drogon were not. AMG is
`cpr_amg_policy_7feat_allrows_newton_x3.model`; abort is
`abort_policy/abort_spe11a_norne_spe9.model` at threshold 0.95.

| Arm | Norne | Norne MSW | Drogon HIST | SPE9 | SPE10 | SPE11A | SPE11B | Sum |
|---|---|---|---|---|---|---|---|---|
| tau1 | +5 % | -7 % | +1 % | +1 % | +1 % | -2 % | -6 % | -3 % |
| AMG alone | -3 % | -14 % | -4 % | -16 % | -7 % | -23 % | -12 % | -15 % |
| tau1 + AMG | -1 % | -29 % | +6 % | -8 % | -3 % | -25 % | -7 % | -14 % |
| tau1 + AMG + abort | -3 % | -21 % | +10 % | -17 % | -10 % | -31 % | -11 % | -17 % |
| newton21 | -2 % | +5 % | -6 % | +9 % | +3 % | +1 % | -2 % | 0 % |

tau1 never leaves the default on SPE11A, so its -2 % there is timing noise.
On SPE11B it switches to label 0 four times per run, which saves about 3 % of
the Newton iterations; the run-to-run spread there is ±11 s on about 175 s.
On Drogon it uses label 4 (quasi-IMPES) for most decisions, which adds about
20 % to the linear iterations without saving time.
