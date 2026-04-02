"""
@file test_evidential_policy.py
@brief Tests for evidential policy networks.

Validates NIG parameter constraints, uncertainty decomposition, loss computation,
and interface contracts for EvidentialLayer, EvidentialPolicyNetwork, and
UncertaintyConditionedActor.
"""

import pytest

# Skip entire module if torch is not available (CI without training deps)
torch = pytest.importorskip("torch")

from uncertainty_rl.networks import (  # noqa: E402
    EvidentialLayer,
    EvidentialPolicyNetwork,
    UncertaintyConditionedActor,
)
from uncertainty_rl.utils.constants import ACTION_DIM  # noqa: E402

# Use smaller dims for fast tests
STATE_DIM = 15
HIDDEN_DIMS = [64, 64]
BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# EvidentialLayer
# ---------------------------------------------------------------------------


class TestEvidentialLayer:
    """
    @class TestEvidentialLayer
    @brief Tests for the EvidentialLayer output head.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """
        @brief Create a fresh layer for each test.
        """
        self.layer = EvidentialLayer(input_dim=64, output_dim=ACTION_DIM)

    def test_output_shapes(self) -> None:
        """
        @brief Verify all four NIG parameters have correct shapes.
        """
        x = torch.randn(BATCH_SIZE, 64)
        gamma, nu, alpha, beta = self.layer(x)

        assert gamma.shape == (BATCH_SIZE, ACTION_DIM)
        assert nu.shape == (BATCH_SIZE, ACTION_DIM)
        assert alpha.shape == (BATCH_SIZE, ACTION_DIM)
        assert beta.shape == (BATCH_SIZE, ACTION_DIM)

    def test_nu_is_positive(self) -> None:
        """
        @brief Nu (precision) must be strictly positive.
        """
        x = torch.randn(BATCH_SIZE, 64)
        _, nu, _, _ = self.layer(x)
        assert (nu > 0).all(), "Nu must be positive everywhere"

    def test_alpha_greater_than_one(self) -> None:
        """
        @brief Alpha (shape) must be > 1 to ensure finite variance.
        """
        x = torch.randn(BATCH_SIZE, 64)
        _, _, alpha, _ = self.layer(x)
        assert (alpha > 1.0).all(), "Alpha must be > 1 for finite variance"

    def test_beta_is_positive(self) -> None:
        """
        @brief Beta (rate) must be strictly positive.
        """
        x = torch.randn(BATCH_SIZE, 64)
        _, _, _, beta = self.layer(x)
        assert (beta > 0).all(), "Beta must be positive everywhere"

    def test_single_sample(self) -> None:
        """
        @brief Layer works with batch size 1.
        """
        x = torch.randn(1, 64)
        gamma, nu, alpha, beta = self.layer(x)
        assert gamma.shape == (1, ACTION_DIM)

    def test_nig_clamp_nu_upper_bound(self) -> None:
        """
        @brief Nu (precision) must be clamped to max 100.0 even with extreme inputs.
        """
        # Create a large input that would produce unbounded nu without clamping
        x = torch.full((BATCH_SIZE, 64), 100.0)
        _, nu, _, _ = self.layer(x)
        assert (nu <= 100.0).all(), f"Nu exceeded upper bound: max={nu.max()}"

    def test_nig_clamp_alpha_upper_bound(self) -> None:
        """
        @brief Alpha (shape) must be clamped to max 100.0 even with extreme inputs.
        """
        x = torch.full((BATCH_SIZE, 64), 100.0)
        _, _, alpha, _ = self.layer(x)
        assert (alpha <= 100.0).all(), f"Alpha exceeded upper bound: max={alpha.max()}"

    def test_nig_clamp_beta_upper_bound(self) -> None:
        """
        @brief Beta (rate) must be clamped to max 100.0 even with extreme inputs.
        """
        x = torch.full((BATCH_SIZE, 64), 100.0)
        _, _, _, beta = self.layer(x)
        assert (beta <= 100.0).all(), f"Beta exceeded upper bound: max={beta.max()}"


# ---------------------------------------------------------------------------
# EvidentialPolicyNetwork
# ---------------------------------------------------------------------------


class TestEvidentialPolicyNetwork:
    """
    @class TestEvidentialPolicyNetwork
    @brief Tests for the full evidential actor network.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """
        @brief Create a fresh network for each test.
        """
        self.net = EvidentialPolicyNetwork(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=HIDDEN_DIMS,
            activation="relu",
        )

    def test_forward_shapes(self, state_batch: torch.Tensor) -> None:
        """
        @brief Forward pass returns four tensors with correct shapes.
        """
        gamma, nu, alpha, beta = self.net(state_batch)
        assert gamma.shape == (BATCH_SIZE, ACTION_DIM)
        assert nu.shape == (BATCH_SIZE, ACTION_DIM)
        assert alpha.shape == (BATCH_SIZE, ACTION_DIM)
        assert beta.shape == (BATCH_SIZE, ACTION_DIM)

    def test_get_action_returns_correct_interface(
        self, state_batch: torch.Tensor
    ) -> None:
        """
        @brief get_action must return (action, uncertainty_dict) with required keys.
        """
        action, unc = self.net.get_action(state_batch, deterministic=False)

        assert action.shape == (BATCH_SIZE, ACTION_DIM)

        required_keys = {
            "epistemic",
            "aleatoric",
            "total",
            "gamma",
            "nu",
            "alpha",
            "beta",
        }
        assert required_keys.issubset(
            unc.keys()
        ), f"Missing keys: {required_keys - unc.keys()}"

    def test_deterministic_action_is_gamma(self, single_state: torch.Tensor) -> None:
        """
        @brief In deterministic mode the action must equal gamma (the NIG mean).
        """
        action, unc = self.net.get_action(single_state, deterministic=True)
        torch.testing.assert_close(action, unc["gamma"])

    def test_uncertainty_decomposition(self, state_batch: torch.Tensor) -> None:
        """
        @brief Total uncertainty must equal epistemic + aleatoric.
        """
        _, unc = self.net.get_action(state_batch, deterministic=False)
        expected_total = unc["epistemic"] + unc["aleatoric"]
        torch.testing.assert_close(unc["total"], expected_total)

    def test_epistemic_formula(self, state_batch: torch.Tensor) -> None:
        """
        @brief Epistemic uncertainty must be beta / (alpha - 1).
        """
        _, unc = self.net.get_action(state_batch, deterministic=False)
        expected = unc["beta"] / (unc["alpha"] - 1)
        torch.testing.assert_close(unc["epistemic"], expected)

    def test_aleatoric_formula(self, state_batch: torch.Tensor) -> None:
        """
        @brief Aleatoric uncertainty must be beta / (nu * (alpha - 1)).
        """
        _, unc = self.net.get_action(state_batch, deterministic=False)
        expected = unc["beta"] / (unc["nu"] * (unc["alpha"] - 1))
        torch.testing.assert_close(unc["aleatoric"], expected)

    def test_uncertainties_are_positive(self, state_batch: torch.Tensor) -> None:
        """
        @brief All uncertainty values must be positive.
        """
        _, unc = self.net.get_action(state_batch, deterministic=False)
        assert (unc["epistemic"] > 0).all()
        assert (unc["aleatoric"] > 0).all()
        assert (unc["total"] > 0).all()

    def test_gradient_flows(self, state_batch: torch.Tensor) -> None:
        """
        @brief Gradients must flow through the network for training.
        """
        action, _ = self.net.get_action(state_batch, deterministic=True)
        loss = action.sum()
        loss.backward()

        for param in self.net.parameters():
            assert param.grad is not None, "Gradient missing on a parameter"

    def test_different_activations(self) -> None:
        """
        @brief Network should work with all supported activation functions.
        """
        for activation in ["relu", "tanh", "elu", "leaky_relu"]:
            net = EvidentialPolicyNetwork(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                hidden_dims=HIDDEN_DIMS,
                activation=activation,
            )
            x = torch.randn(2, STATE_DIM)
            gamma, nu, alpha, beta = net(x)
            assert gamma.shape == (2, ACTION_DIM)


# ---------------------------------------------------------------------------
# Evidential loss
# ---------------------------------------------------------------------------


class TestEvidentialLoss:
    """
    @class TestEvidentialLoss
    @brief Tests for the evidential regression loss computation.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """
        @brief Create network and run a forward pass to get NIG params.
        """
        self.net = EvidentialPolicyNetwork(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        self.gamma, self.nu, self.alpha, self.beta = self.net(state)
        self.target = torch.randn(BATCH_SIZE, ACTION_DIM)

    def test_loss_dict_keys(self) -> None:
        """
        @brief Loss dict must contain 'loss', 'nll', and 'regularisation'.
        """
        result = self.net.compute_evidential_loss(
            self.gamma, self.nu, self.alpha, self.beta, self.target
        )
        assert set(result.keys()) == {"loss", "nll", "regularisation"}

    def test_loss_is_scalar(self) -> None:
        """
        @brief Each loss component must be a scalar tensor.
        """
        result = self.net.compute_evidential_loss(
            self.gamma, self.nu, self.alpha, self.beta, self.target
        )
        for key, val in result.items():
            assert val.dim() == 0, f"{key} should be scalar, got shape {val.shape}"

    def test_loss_is_finite(self) -> None:
        """
        @brief Loss values must be finite (no NaN or Inf).
        """
        result = self.net.compute_evidential_loss(
            self.gamma, self.nu, self.alpha, self.beta, self.target
        )
        for key, val in result.items():
            assert torch.isfinite(val), f"{key} is not finite: {val}"

    def test_zero_lambda_removes_regularisation(self) -> None:
        """
        @brief With lambda_reg=0, loss should equal NLL.
        """
        result = self.net.compute_evidential_loss(
            self.gamma,
            self.nu,
            self.alpha,
            self.beta,
            self.target,
            lambda_reg=0.0,
        )
        torch.testing.assert_close(result["loss"], result["nll"])

    def test_loss_differentiable(self) -> None:
        """
        @brief Loss must be differentiable for backpropagation.
        """
        result = self.net.compute_evidential_loss(
            self.gamma, self.nu, self.alpha, self.beta, self.target
        )
        result["loss"].backward()
        for param in self.net.parameters():
            assert param.grad is not None


# ---------------------------------------------------------------------------
# UncertaintyConditionedActor
# ---------------------------------------------------------------------------


class TestUncertaintyConditionedActor:
    """
    @class TestUncertaintyConditionedActor
    @brief Tests for the dual-encoder actor network.
    """

    def test_forward_shapes(self) -> None:
        """
        @brief Dual-encoder output shapes must match action dim.
        """
        # State without uncertainty: first 6 dims
        # Uncertainty features: remaining 9 dims
        actor = UncertaintyConditionedActor(
            state_dim=6,
            uncertainty_dim=9,
            action_dim=ACTION_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        state = torch.randn(BATCH_SIZE, 6)
        uncertainty = torch.randn(BATCH_SIZE, 9)

        gamma, nu, alpha, beta = actor(state, uncertainty)
        assert gamma.shape == (BATCH_SIZE, ACTION_DIM)
        assert (nu > 0).all()
        assert (alpha > 1.0).all()
        assert (beta > 0).all()
