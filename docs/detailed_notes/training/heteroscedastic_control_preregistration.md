# Heteroscedastic Control Arms: Pre-registration

> Fixed on 30 September 2026, before any heteroscedastic chain was launched. The analysis
> below is run as written, whatever the outcome.

Extracted from `networks/heteroscedastic.py`, `configs/baselines/` and
`scripts/analysis/seed_level.py`.

## Question

In the 2x2 ablation the EKF covariance input raises parking success only when the actor
is the evidential NIG head. At `anchor_deployment`, pooled over seeds 42, 123 and 7 and
600 episodes per arm:

| Actor head | No covariance | Covariance | Covariance effect |
|---|---|---|---|
| Standard Gaussian | `vanilla_ppo` 23.8% | `input_uncertainty` 25.2% | +1.3 pp |
| Evidential NIG | `output_uncertainty` 19.0% | `full_method` 45.8% | +26.8 pp |

The two heads differ in more than one respect. Nu receives no gradient and stays at its
prior of about 0.313, so in the RL path the evidential head acts as a tanh-squashed
Gaussian whose variance depends on the state, whereas the standard head learns a single
state-independent `log_std`. The hypothesis is that it is the state-dependent action
variance, not the NIG parameterisation, that lets the policy use the covariance input.

## Design

A third head, `policy_type: heteroscedastic`, is the evidential head with the NIG
parameterisation and its prior anchor removed. It keeps the same backbone, critic, mean
bias, initial action variance (0.398), sampling clamp `[aleatoric_floor, 1.0]`,
squashing, entropy, PPO hyperparameters and `evidential_overrides` entropy schedule. The
contract is tabulated in [networks/README.md](../../../uncertainty_rl/networks/README.md).

| Arm | `policy_type` | `include_covariance` |
|---|---|---|
| `vanilla_ppo` | `standard` | false |
| `input_uncertainty` | `standard` | true |
| `heteroscedastic` | `heteroscedastic` | false |
| `heteroscedastic_input` | `heteroscedastic` | true |
| `output_uncertainty` | `evidential` | false |
| `full_method` | `evidential` | true |

## Run protocol

- Both new arms are trained for seeds 42, 123 and 7 through curriculum stages 1 to 6,
  resume-chained, with
  `ARMS_OVERRIDE="heteroscedastic heteroscedastic_input" make run-seed-leg`. Each chain
  is 10,000,000 policy decisions.
- Stage 6 of each chain is evaluated without the SafetyWrapper (`NO_SAFETY=1`), under
  every condition in `configs/eval_config.yaml`, 200 episodes each, `PER_STEP_CAP=440`,
  deterministic actions, exactly as for the existing arms.
- `train/aleatoric_uncertainty`, `train/entropy_loss` and the rolling success rate are
  monitored. A std that sits at the floor (0.2) or the cap (1.0) for most of a stage is
  reported.

## Primary hypothesis (H1g)

The covariance input raises parking success for the heteroscedastic head.

- **Primary contrast:** success(`heteroscedastic_input`) - success(`heteroscedastic`) at
  `anchor_deployment`.
- **Decision rule:** H1g is supported if the 95% two-level bootstrap interval of the
  primary contrast excludes zero. For each arm, three seeds are drawn with replacement,
  then the episodes of each drawn seed. The arms are resampled independently, over
  10,000 resamples.
- The same contrast is reported for `anchor_empty` and `gnss_degrade_one_way`, together
  with the exact one-sided seed-level permutation test over the C(6, 3) = 20
  arrangements (smallest attainable p = 0.05).

## Secondary estimates

These are reported with two-level intervals whatever the outcome:

1. Interaction with the standard head: the heteroscedastic contrast minus the standard
   contrast.
2. The evidential contrast minus the heteroscedastic contrast.
3. success(`full_method`) - success(`heteroscedastic_input`).
4. success(`heteroscedastic`) - success(`vanilla_ppo`), the effect of state-dependent
   variance without the covariance input.
5. The per-seed Spearman correlation between `mean_brake_cmd` and `ekf_std_pos_mean_m`
   within each varying condition, averaged over the three conditions, and the
   success-conditional final position error and decision count.

## Interpretation fixed in advance

- If H1g is supported and the interval of estimate 3 includes zero, the covariance
  benefit is attributed to state-dependent action variance under this implementation,
  not to the NIG parameterisation.
- If H1g is not supported, the benefit requires the NIG mechanism, meaning its
  parameterisation together with its prior anchor.
- Three seeds cannot establish equivalence. An interval that includes zero is reported
  as "no detectable difference", never as "equal".

## Implementation

`make analyse-seed-level STAGE=6 BOOTSTRAP_SEED=20260927` computes every quantity above
and writes it under `outputs/raw_derived/seed_level/`. As a regression check, the
pooled success rates, contrasts and intervals of the four original arms must be
identical to the existing bundle under the same bootstrap seed. The heteroscedastic
contrast is appended last in `_CONTRAST_PAIRS`, so the original pairs keep their
bootstrap draws.
