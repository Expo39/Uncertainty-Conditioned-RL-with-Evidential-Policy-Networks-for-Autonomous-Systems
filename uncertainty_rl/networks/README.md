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

During evaluation and deployment, use `EvidentialActorCriticPolicy.get_action_with_uncertainty()`:

```python
action, uncertainty_dict = policy.get_action_with_uncertainty(obs_tensor)
# uncertainty_dict keys: epistemic, aleatoric, total, gamma, nu, alpha, beta
```

`EvidentialPolicyNetwork.get_action()` provides the same interface but is a standalone test harness only - not used in the RL training pipeline.

### Uncertainty Formulae

- **Aleatoric** (data uncertainty, E[sigma^2]): `beta / (alpha - 1)`
- **Epistemic** (model uncertainty, Var[mu]): `beta / (nu * (alpha - 1))`

### NIG Parameter Clamping

During RL training, unbounded growth of nu, alpha, beta destabilises both the NLL loss and the prior-anchoring log-penalty regularisation used by `EvidentialPPO`. Upper bounds of **100.0** are enforced on all three parameters in `EvidentialLayer.forward()`.

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
| `EvidentialPPO` | SB3 `PPO` subclass adding evidential regularisation to the PPO loss. Uses **prior-anchoring log-penalty**. Linearly anneals `lambda_reg` from 0 over `lambda_reg_warmup_steps`. |

### Evidential Regularisation Loss

**Prior-anchoring log-penalty** (RL-stable):
```
reg = mean(log(nu / nu_prior + 1)) + mean(log(alpha / alpha_prior + 1))
```

`nu_prior = 1.24`, `alpha_prior = 2.24` (hardcoded constants in `EvidentialPPO.train()`).

Replaces `|actions - gamma| * (2*nu + alpha)`, which is ill-defined in RL:
- Supervised term assumes ground-truth action targets (don't exist in RL)
- In RL, `|actions - gamma|` is just policy sampling noise, not prediction error
- Unbounded growth with lack of signal - regularisation loss explodes

Log-penalty approach:
- Bounded (finite for all nu, alpha values)
- Penalises drift from initialisation priors without action coupling
- Works well with NIG clamping (max 100.0)

### Actor Modes

| `use_uncertainty_conditioning` | Actor | Obs input |
|-------------------------------|-------|-----------|
| `False` | Flat MLP extractor + `EvidentialLayer` | Full obs vector |
| `True` | `UncertaintyConditionedActor` (dual-encoder) | `obs[:1]` (vyaw) + `obs[1:4]` (std_x/y/yaw) only - indices 4+ are not used |

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
