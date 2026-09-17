# evidential_nig_initialisation

Extracted from `uncertainty_rl/networks/evidential_policy.py` (`EvidentialLayer.__init__`).

Section 3.5.1 of the dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical
for the NIG head and its initialisation, covering the softplus-with-offsets constraints,
why the alpha offset is 1.5 rather than 1.0, the transfer of control from weights to
biases, the per-axis gamma biases, the sub-1 nu prior, and the re-application of the
biases after SB3's orthogonal-initialisation pass. Figure 3.9 there illustrates it.

This note records the bias layout the code depends on, which the dissertation does not
enumerate index by index.

## Bias layout

The head emits the four Normal-Inverse-Gamma parameters of deep evidential
regression [1] per action dimension. The linear layer has shape
`(input_dim, output_dim * 4)`, and its bias vector is four contiguous blocks of
`output_dim` elements, `[gamma | nu | alpha | beta]`, with `forward()` splitting the
output on the same boundaries.

| Block | Bias | Activation | Value at init |
|-------|------|-----------|---------------|
| gamma (steer) | 0.0 | none | 0.0, bipolar |
| gamma (throttle) | 0.5 | none | 0.5, default-on |
| gamma (brake) | -1.0 | none | -1.0, default-off |
| nu | -1.0 | softplus + 1e-6 | ~0.313 |
| alpha | 0.9 | softplus + 1.5 | ~2.741 |
| beta | 0.0 | softplus + 1e-6 | ~0.693 |

The per-axis gamma block applies when `output_dim == 3`; any other width falls back to
a uniform zero bias for the test harness. All weights are scaled by 0.01 so the biases,
not random input projections, dominate the output at step 0.

The log-penalty prior targets in `EvidentialPPO.train` must track this table:
`nu_prior = 0.313` and `alpha_prior = 2.741`, the latter carrying the layer's 1.5
offset.

## See also

- `uncertainty_rl/networks/evidential_policy.py` - `EvidentialLayer.__init__`, `forward`
- `uncertainty_rl/networks/sb3_integration.py` - `EvidentialPPO.train`, and the
  `with th.no_grad()` block after the `module_gains` loop in
  `EvidentialActorCriticPolicy._build()` that re-applies the biases

## References

[1] A. Amini, W. Schwarting, A. Soleimany, and D. Rus, "Deep evidential regression," in
    *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 14927-14937.
