# Neural AMG cost models from the feature-reduction study (2026-11-05/06)

Variants of `../amg_policy/cpr_amg_policy.model` (policy B, 12 inputs) trained
with fewer input features. All are 5-member ensembles trained with
`opm-simulators/doc/neural_cpr/train_cpr_amg_policy.py` on the AMG sweep
`boiler/cprepo/tmp/sweep_amg_20261015/*_pd*_ct*.csv` (Norne, SPE11A, SPE9),
branch `neural-amg-7feat` of opm-simulators. Run with
`OPM_NEURAL_CPR_ACTIONS=amg OPM_NEURAL_CPR_MIN_GAIN=0.05` and, for the
no-structure models, `OPM_NEURAL_CPR_AMG_FEATURES=<layout>`.

Slots: 0 dt_days, 1 dt_ratio, 2 nl_residual_norm, 3 nl_residual_reduce,
4 nl_iteration, 5 nnz_per_row, 6 diag_dominance (clip x/4), 7 prev_linsolver_iters,
8 prev_solve_failed, 9 time_elapsed_frac, 10 num_cells_log, 11 block_size,
19 log10(diag_dominance)/10.

| file | layout | target | seed |
|---|---|---|---|
| `cpr_amg_policy_5feat.model` | 1,2,3,6,10 | absolute | 42 |
| `cpr_amg_policy_7feat.model` | 0,1,2,3,4,7,10 | absolute | 42 |
| `cpr_amg_policy_8feat.model` | 0,1,2,5,6,7,10,11 | absolute | 42 |
| `cpr_amg_policy_9feat.model` | 0,2,3,5,6,7,9,10,11 | absolute | 42 |
| `cpr_amg_policy_10feat{,_s100,_s200}.model` | 0,1,2,3,4,5,6,7,10,11 | absolute | 42 / 100 / 200 |
| `cpr_amg_policy_12feat_s100.model`, `_s200` | 0-11 | absolute | 100 / 200 (seed-100 crashes SPE11B) |
| `cpr_amg_policy_7nostruct.model` | 0,1,2,3,4,6,7 | absolute | 42 |
| `cpr_amg_policy_7nostruct_logdiag{,_s100,_s200}.model` | 0,1,2,3,4,19,7 | absolute | 42 / 100 / 200 |
| `cpr_amg_policy_7nostruct_logdiag_rel{,_s100,_s200}.model` | 0,1,2,3,4,19,7 | `--relative` | 42 / 100 / 200 |
| `cpr_amg_policy_7nostruct_logdiag_rel_x3.model` | 0,1,2,3,4,19,7 | `--relative` | 42+100+200 packed, 15 members |

The last row is the state-only model that matches the deployed 12-feature
model in flow wall time (Norne / SPE11A / SPE11B: 59.3 / 144.6 / 170.8 s vs
59 / 135 / 178 s). Rebuild instructions, benchmark scripts and results:
`workstation/NEURAL_AMG_7FEAT_REBUILD.md`.
