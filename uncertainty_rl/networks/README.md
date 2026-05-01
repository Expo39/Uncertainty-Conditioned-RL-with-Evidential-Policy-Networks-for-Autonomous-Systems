# networks/

Core novel component - evidential deep learning policy networks for uncertainty-aware action selection.

## Module: `evidential_policy.py`

### Classes

| Class | Purpose |
|-------|---------|
| `EvidentialLayer` | Output layer producing Normal-Inverse-Gamma (NIG) parameters (gamma, nu, alpha, beta) per action dimension. **NIG parameters are clamped to max 100.0 to prevent divergence during RL training.** |
| `EvidentialPolicyNetwork` | Full actor network: feature extraction backbone + evidential output head |
| `UncertaintyConditionedActor` | Dual-encoder variant that processes state and uncertainty components through separate pathways before merging |

### Key Interface

```python
action, uncertainty_dict = network.get_action(state_tensor)
# uncertainty_dict keys: epistemic, aleatoric, total, gamma, nu, alpha, beta
```

### Uncertainty Formulae

- **Aleatoric** (data uncertainty, E[sigma^2]): `beta / (alpha - 1)`
- **Epistemic** (model uncertainty, Var[mu]): `beta / (nu * (alpha - 1))`

### NIG Parameter Clamping

During RL training, unbounded growth of nu, alpha, beta causes the regularisation term to explode. We enforce **upper bounds of 100.0** on all three parameters in `EvidentialLayer.forward()`. This differs from supervised regression (Amini et al. 2020), where ground-truth targets exist. In RL, `|actions - gamma|` is just policy noise, not prediction error, so clamping prevents the reg loss from diverging.

### Constraints

- Evidential learning applies to the **actor only** (never the critic).
- `softplus + offset` on nu, alpha, beta ensures finite variance and positivity.
- NIG parameters clamped to max 100.0 for RL stability.
- LayerNorm used (not BatchNorm).
- `EvidentialPolicyNetwork` is a standalone test harness only - NOT used in the RL training pipeline.
- `compute_evidential_loss` is a supervised regression loss for unit tests only - NOT the training loss.

## Module: `sb3_integration.py`

### Classes

| Class | Purpose |
|-------|---------|
| `EvidentialDistribution` | SB3 `Distribution` subclass: Gaussian approximation of NIG predictive. `std = sqrt(aleatoric)` only - epistemic uncertainty is not added to action noise. Caches NIG params for the regularisation loss. |
| `EvidentialActorCriticPolicy` | SB3 `ActorCriticPolicy` subclass with evidential actor head and standard critic. Supports flat MLP (`use_uncertainty_conditioning=False`) and dual-encoder (`True`) modes. |
| `EvidentialPPO` | SB3 `PPO` subclass adding evidential regularisation to the PPO loss. Uses **prior-anchoring log-penalty** (not Amini et al. 2020 supervised term). Linearly anneals `lambda_reg` from 0 over `lambda_reg_warmup_steps`. |

### Evidential Regularisation Loss

**Prior-anchoring log-penalty** (RL-stable):
```
reg = mean(log(nu / nu_prior + 1)) + mean(log(alpha / alpha_prior + 1))
```

Replaces the Amini et al. 2020 supervised term `|actions - gamma| * (2*nu + alpha)`, which is ill-defined in RL:
- Supervised term assumes ground-truth action targets (don't exist in RL)
- In RL, `|actions - gamma|` is just policy sampling noise, not prediction error
- Unbounded growth with lack of signal - regularisation loss explodes

Log-penalty approach:
- Bounded (finite for all nu, alpha values)
- Penalises drift from initialisation priors without action coupling
- Works well with NIG clamping (max 100.0)

### Actor Modes

| `use_uncertainty_conditioning` | Actor | Obs split |
|-------------------------------|-------|-----------|
| `False` | Flat MLP extractor + `EvidentialLayer` | Full obs -> MLP latent |
| `True` | `UncertaintyConditionedActor` (dual-encoder) | `obs[:1]` = vyaw (VEHICLE_STATE_DIM), `obs[1:4]` = covariance (COVARIANCE_FEATURES_DIM) |

### Sampling std

```
std = sqrt(beta / (alpha - 1))  # aleatoric std only
```

Epistemic uncertainty (`beta / (nu * (alpha - 1))`) quantifies model uncertainty over `gamma` and is NOT added to action noise.

### TensorBoard Logs (evidential-specific)

- `train/evidential_reg_loss` - prior-anchoring regularisation term: `mean(log(nu/nu_prior+1)) + mean(log(alpha/alpha_prior+1))`
- `train/epistemic_uncertainty` - mean `beta / (nu * (alpha - 1))` per update
- `train/aleatoric_uncertainty` - mean `beta / (alpha - 1)` per update
- `train/lambda_reg` - current annealed regularisation weight

### Python Logging

Both modules use `logging.getLogger("uncertainty_rl.networks.*")`. Level is controlled by `debug: true/false` in `configs/train_config.yaml` (or `eval_config.yaml`), which sets `logging.basicConfig` in the respective entry-point `main()`.
