# evidential_nig_initialisation

Extracted from `uncertainty_rl/networks/evidential_policy.py` (`EvidentialLayer.__init__`).
The inline comment block was too long for a method docstring; the derivation lives here.

## NIG hyperprior initialisation rationale

`EvidentialLayer` outputs four parameters per action dimension: gamma, nu, alpha, beta.
The linear layer that produces them is initialised so that at step 0 the NIG prior is
already in a stable, well-defined region rather than landing near pathological boundaries.

Default Kaiming initialisation produces near-zero nu (undefined precision) or alpha near
1.0 (infinite variance in the NIG). Both make the evidential loss undefined at the start
of training and can cause gradient explosion before the first update.

### Bias layout

The linear layer has shape `(input_dim, output_dim * 4)`. The bias vector is laid out
as four contiguous blocks of `output_dim` elements: `[gamma | nu | alpha | beta]`.

### Target values and derivation

| Parameter | Bias value | Activation | Result at init | Interpretation |
|-----------|-----------|-----------|----------------|----------------|
| gamma | 0.0 | none | 0.0 | Zero mean action at init |
| nu | 0.9 | softplus + 1e-6 | ~0.97 | Reasonable initial precision |
| alpha | 0.9 | softplus + 1.0 | ~1.97 | Well-defined finite variance (alpha > 1 required) |
| beta | 0.0 | softplus + 1e-6 | ~0.69 | Moderate scale |

`softplus(0.9) = log(1 + exp(0.9)) ~ 1.243 - 0.35 ~ 0.941` (more precisely, ~0.9413).
So:
- nu bias 0.9 => softplus(0.9) + 1e-6 ~ 0.9413
- alpha bias 0.9 => softplus(0.9) + 1.0 ~ 1.9413
- beta bias 0.0 => softplus(0.0) + 1e-6 = log(2) + 1e-6 ~ 0.6931

The NIG log-penalty prior targets in `EvidentialPPO.train` use these values:
`nu_prior = 1.24`, `alpha_prior = 2.24` (rounded to two decimal places from the above).

### Weight scaling

All weights are multiplied by 0.01 so outputs at step 0 are dominated by the biases
rather than random input projections. This prevents large gamma variance from masking
the bias-driven prior at initialisation.

### Orthogonal initialisation interaction

When `ortho_init=True`, SB3's `init_weights` zeroes all biases. The NIG bias values
must be reapplied after the `init_weights` loop in `EvidentialActorCriticPolicy._build()`.
See the `with th.no_grad()` block after the `module_gains` loop.

## See also

- `uncertainty_rl/networks/evidential_policy.py` - EvidentialLayer.__init__
- `uncertainty_rl/networks/sb3_integration.py` - EvidentialPPO.train (nu_prior, alpha_prior)
- Amini et al. 2020 "Deep Evidential Regression" for the NIG formulation
