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
