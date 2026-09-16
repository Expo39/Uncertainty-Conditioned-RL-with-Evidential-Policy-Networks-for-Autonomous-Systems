"""
@file covariance_utils.py
@brief Shared utilities for EKF covariance matrix extraction.
"""

from typing import List, cast

import numpy as np

from uncertainty_rl.utils.constants import COVARIANCE_FEATURES_DIM

# Indices into a 6x6 pose covariance diagonal for [x, y, yaw]
_COV6_INDICES = np.array([0, 1, 5])


def extract_2d_covariance_features(cov_matrix: np.ndarray) -> np.ndarray:
    """
    @brief Extract 2D covariance features from covariance matrix.

    @param cov_matrix: Covariance matrix (3x3 for [x, y, yaw] or 6x6 for
                       full 3D pose).
    @return: 1D array of 3 covariance features [std_x, std_y, std_yaw].
    @raises ValueError: If matrix shape is not 3x3 or 6x6.
    """
    diag = cov_matrix.diagonal()
    if cov_matrix.shape == (6, 6):
        result = np.sqrt(diag[_COV6_INDICES]).astype(np.float64)
        return cast(np.ndarray, result)
    if cov_matrix.shape == (3, 3):
        result = np.sqrt(diag).astype(np.float64)
        return cast(np.ndarray, result)
    raise ValueError(
        f"Expected 3x3 or 6x6 covariance matrix, got shape {cov_matrix.shape}"
    )


def get_covariance_dimension() -> int:
    """
    @brief Get the dimensionality of extracted covariance features.
    @return: Number of covariance features (COVARIANCE_FEATURES_DIM = 3).
    """
    return COVARIANCE_FEATURES_DIM


def validate_covariance_matrix(cov_matrix: np.ndarray) -> bool:
    """
    @brief Validate that covariance matrix is positive semi-definite and symmetric.

    @param cov_matrix: Covariance matrix to validate.
    @return: True if valid, False otherwise.

    @note PSD is tested via the symmetric eigenvalue decomposition rather than
          Cholesky factorisation. Cholesky requires strict positive-definiteness
          and so rejects valid PSD matrices with zero eigenvalues (e.g. an EKF
          reporting zero covariance before any measurement update).
    """
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        return False
    if not np.allclose(cov_matrix, cov_matrix.T):
        return False
    # eigvalsh is for symmetric matrices; symmetry is already confirmed above.
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    # Small negative tolerance absorbs floating-point round-off in eigvalsh.
    return bool(np.all(eigenvalues >= -1e-9))


def make_diagonal_covariance(diag: List[float]) -> List[float]:
    """
    @brief Build a flat 36-element ROS covariance array from a 6-element diagonal.

    @note ROS nav_msgs/Odometry pose.covariance is a row-major 6x6 matrix stored
          as a flat list of 36 floats.

    @param diag: Six diagonal variance values [var_x, var_y, var_z,
                 var_roll, var_pitch, var_yaw].
    @return Flat list of 36 floats (row-major 6x6, zeros off-diagonal).
    """
    cov_arr = np.zeros(36)
    cov_arr[::7] = diag
    return [float(x) for x in cov_arr.tolist()]
