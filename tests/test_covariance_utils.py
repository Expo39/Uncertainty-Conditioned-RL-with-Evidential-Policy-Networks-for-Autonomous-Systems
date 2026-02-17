"""
@file test_covariance_utils.py
@brief Tests for EKF covariance matrix utility functions.

Validates feature extraction from 3x3 and 6x6 covariance matrices,
matrix validation, and edge cases (zero, identity, non-symmetric).
"""

import numpy as np
import pytest

from uncertainty_rl.utils import (
    extract_2d_covariance_features,
    get_covariance_dimension,
    validate_covariance_matrix,
)

# ---------------------------------------------------------------------------
# extract_2d_covariance_features
# ---------------------------------------------------------------------------


class TestExtract2DCovarianceFeatures:
    """
    @class TestExtract2DCovarianceFeatures
    @brief Tests for the 2D covariance feature extractor.
    """

    def test_output_shape_3x3(self) -> None:
        """
        @brief 3x3 input should produce 9-element feature vector.
        """
        cov = np.eye(3) * 0.1
        features = extract_2d_covariance_features(cov)
        assert features.shape == (9,)

    def test_output_shape_6x6(self) -> None:
        """
        @brief 6x6 input should produce 9-element feature vector.
        """
        cov = np.eye(6) * 0.1
        features = extract_2d_covariance_features(cov)
        assert features.shape == (9,)

    def test_identity_3x3_values(self) -> None:
        """
        @brief Identity matrix should give std=1 and zero off-diagonals.
        """
        cov = np.eye(3)
        features = extract_2d_covariance_features(cov)

        # [std_x, std_y, std_yaw, cov_xx, cov_yy, cov_yawyaw,
        #  cov_xy, cov_xyaw, cov_yyaw]
        np.testing.assert_approx_equal(features[0], 1.0)  # std_x
        np.testing.assert_approx_equal(features[1], 1.0)  # std_y
        np.testing.assert_approx_equal(features[2], 1.0)  # std_yaw
        np.testing.assert_approx_equal(features[3], 1.0)  # cov_xx
        np.testing.assert_approx_equal(features[4], 1.0)  # cov_yy
        np.testing.assert_approx_equal(features[5], 1.0)  # cov_yawyaw
        np.testing.assert_approx_equal(features[6], 0.0)  # cov_xy
        np.testing.assert_approx_equal(features[7], 0.0)  # cov_xyaw
        np.testing.assert_approx_equal(features[8], 0.0)  # cov_yyaw

    def test_scaled_diagonal_3x3(self) -> None:
        """
        @brief Diagonal matrix with known variances should give correct stds.
        """
        cov = np.diag([0.04, 0.09, 0.16])
        features = extract_2d_covariance_features(cov)

        np.testing.assert_approx_equal(features[0], 0.2)  # std_x = sqrt(0.04)
        np.testing.assert_approx_equal(features[1], 0.3)  # std_y = sqrt(0.09)
        np.testing.assert_approx_equal(features[2], 0.4)  # std_yaw = sqrt(0.16)

    def test_6x6_extracts_correct_indices(self) -> None:
        """
        @brief 6x6 input should extract x(0), y(1), yaw(5) sub-matrix.
        """
        cov = np.zeros((6, 6))
        # Set x, y, yaw variances at indices 0, 1, 5
        cov[0, 0] = 0.25
        cov[1, 1] = 0.36
        cov[5, 5] = 0.01
        # Set a cross-correlation between x and yaw
        cov[0, 5] = 0.05
        cov[5, 0] = 0.05

        features = extract_2d_covariance_features(cov)

        np.testing.assert_approx_equal(features[0], 0.5)  # std_x = sqrt(0.25)
        np.testing.assert_approx_equal(features[1], 0.6)  # std_y = sqrt(0.36)
        np.testing.assert_approx_equal(features[2], 0.1)  # std_yaw = sqrt(0.01)
        np.testing.assert_approx_equal(features[7], 0.05)  # cov_xyaw

    def test_off_diagonal_preserved(self) -> None:
        """
        @brief Off-diagonal covariance terms should appear in features.
        """
        cov = np.eye(3) * 0.1
        cov[0, 1] = 0.02
        cov[1, 0] = 0.02  # Keep symmetric
        cov[0, 2] = 0.03
        cov[2, 0] = 0.03

        features = extract_2d_covariance_features(cov)

        np.testing.assert_approx_equal(features[6], 0.02)  # cov_xy
        np.testing.assert_approx_equal(features[7], 0.03)  # cov_xyaw

    def test_rejects_wrong_shape(self) -> None:
        """
        @brief Non-3x3 / non-6x6 matrices should raise ValueError.
        """
        with pytest.raises(ValueError, match="Expected 3"):
            extract_2d_covariance_features(np.eye(4))

        with pytest.raises(ValueError, match="Expected 3"):
            extract_2d_covariance_features(np.eye(2))

    def test_output_dtype_is_float(self) -> None:
        """
        @brief Output should be a float array.
        """
        cov = np.eye(3, dtype=np.float32) * 0.1
        features = extract_2d_covariance_features(cov)
        assert np.issubdtype(features.dtype, np.floating)


# ---------------------------------------------------------------------------
# validate_covariance_matrix
# ---------------------------------------------------------------------------


class TestValidateCovarianceMatrix:
    """
    @class TestValidateCovarianceMatrix
    @brief Tests for covariance matrix validation.
    """

    def test_identity_is_valid(self) -> None:
        """
        @brief Identity matrix is a valid covariance matrix.
        """
        assert validate_covariance_matrix(np.eye(3)) is True

    def test_scaled_identity_is_valid(self) -> None:
        """
        @brief Scaled identity matrix is valid.
        """
        assert validate_covariance_matrix(np.eye(3) * 0.5) is True

    def test_zero_matrix_is_valid(self) -> None:
        """
        @brief Zero matrix is positive semi-definite (all eigenvalues = 0).
        """
        assert validate_covariance_matrix(np.zeros((3, 3))) is True

    def test_non_symmetric_is_invalid(self) -> None:
        """
        @brief Asymmetric matrix is not a valid covariance matrix.
        """
        cov = np.eye(3)
        cov[0, 1] = 0.5
        # cov[1, 0] left as 0 -- not symmetric
        assert validate_covariance_matrix(cov) is False

    def test_negative_eigenvalue_is_invalid(self) -> None:
        """
        @brief Matrix with negative eigenvalue is not valid.
        """
        # Diagonal with a negative entry -> negative eigenvalue
        cov = np.diag([1.0, 1.0, -0.5])
        assert validate_covariance_matrix(cov) is False

    def test_non_square_is_invalid(self) -> None:
        """
        @brief Non-square matrix is not valid.
        """
        assert validate_covariance_matrix(np.ones((3, 4))) is False

    def test_realistic_ekf_covariance(self) -> None:
        """
        @brief A realistic EKF covariance matrix should be valid.
        """
        # Typical EKF output: small variances with mild correlations
        cov = np.array(
            [
                [0.04, 0.005, 0.001],
                [0.005, 0.09, 0.002],
                [0.001, 0.002, 0.01],
            ]
        )
        assert validate_covariance_matrix(cov) is True

    def test_6x6_valid(self) -> None:
        """
        @brief 6x6 identity is valid.
        """
        assert validate_covariance_matrix(np.eye(6)) is True


# ---------------------------------------------------------------------------
# get_covariance_dimension
# ---------------------------------------------------------------------------


class TestGetCovarianceDimension:
    """
    @class TestGetCovarianceDimension
    @brief Tests for the covariance dimension helper.
    """

    def test_returns_nine(self) -> None:
        """
        @brief Must return 9 (matching COVARIANCE_FEATURES_DIM).
        """
        assert get_covariance_dimension() == 9

    def test_matches_feature_vector_length(self) -> None:
        """
        @brief Dimension should match actual output of extract_2d_covariance_features.
        """
        features = extract_2d_covariance_features(np.eye(3))
        assert len(features) == get_covariance_dimension()
