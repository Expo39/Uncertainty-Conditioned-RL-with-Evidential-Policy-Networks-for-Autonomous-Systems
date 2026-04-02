# networks/

Core novel component - evidential deep learning policy networks for uncertainty-aware action selection.

## Module: `evidential_policy.py`

### Classes

| Class | Purpose |
|-------|---------|
| `EvidentialLayer` | Output layer producing Normal-Inverse-Gamma (NIG) parameters (gamma, nu, alpha, beta) per action dimension |
| `EvidentialPolicyNetwork` | Full actor network: feature extraction backbone + evidential output head |
| `UncertaintyConditionedActor` | Dual-encoder variant that processes state and uncertainty components through separate pathways before merging |

### Key Interface

```python
action, uncertainty_dict = network.get_action(state_tensor)
# uncertainty_dict keys: epistemic, aleatoric, total, gamma, nu, alpha, beta
```

### Uncertainty Formulae

- **Epistemic** (model uncertainty): `beta / (alpha - 1)`
- **Aleatoric** (data uncertainty): `beta / (nu * (alpha - 1))`

### Constraints

- Evidential learning applies to the **actor only** (never the critic).
- `softplus + offset` on nu, alpha, beta ensures finite variance.
- LayerNorm used (not BatchNorm).
- `EvidentialPolicyNetwork` is a standalone test harness only -- NOT used in the RL training pipeline.
- `compute_evidential_loss` is a supervised regression loss for unit tests only -- NOT the training loss.

## Module: `sb3_integration.py`

### Classes

| Class | Purpose |
|-------|---------|
| `EvidentialDistribution` | SB3 `Distribution` subclass: Gaussian approximation of NIG predictive. `std = sqrt(aleatoric)` only -- epistemic uncertainty is not added to action noise. Caches NIG params for the regularisation loss. |
| `EvidentialActorCriticPolicy` | SB3 `ActorCriticPolicy` subclass with evidential actor head and standard critic. Supports flat MLP (`use_uncertainty_conditioning=False`) and dual-encoder (`True`) modes. |
| `EvidentialPPO` | SB3 `PPO` subclass adding evidential regularisation to the PPO loss. Linearly anneals `lambda_reg` from 0 over `lambda_reg_warmup_steps`. |

### Actor Modes

| `use_uncertainty_conditioning` | Actor | Obs split |
|-------------------------------|-------|-----------|
| `False` | Flat MLP extractor + `EvidentialLayer` | Full obs -> MLP latent |
| `True` | `UncertaintyConditionedActor` (dual-encoder) | `obs[:6]` = vehicle state, `obs[6:15]` = covariance features |

### Sampling std

```
std = sqrt(beta / (nu * (alpha - 1)))  # aleatoric only
```

Epistemic uncertainty (`beta / (alpha - 1)`) quantifies model uncertainty over `gamma` and is NOT added to action noise.

### TensorBoard Logs (evidential-specific)

- `train/evidential_reg_loss` -- regularisation term `mean(|actions - gamma| * (2*nu + alpha))`
- `train/epistemic_uncertainty` -- mean `beta / (alpha - 1)` per update
- `train/aleatoric_uncertainty` -- mean `beta / (nu * (alpha - 1))` per update
- `train/lambda_reg` -- current annealed regularisation weight

### Python Logging

Both modules use `logging.getLogger("uncertainty_rl.networks.*")`. Level is controlled by `debug: true/false` in `configs/train_config.yaml` (or `eval_config.yaml`), which sets `logging.basicConfig` in the respective entry-point `main()`.
