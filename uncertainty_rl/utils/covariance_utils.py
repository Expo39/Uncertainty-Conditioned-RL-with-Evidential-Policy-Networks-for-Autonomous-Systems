"""
@file covariance_utils.py
@brief Shared utilities for EKF covariance matrix extraction.

This module provides common functions for extracting uncertainty features
from EKF covariance matrices for use in RL state representations.
"""

from typing import List

import numpy as np

from uncertainty_rl.utils.constants import COVARIANCE_FEATURES_DIM


def extract_2d_covariance_features(cov_matrix: np.ndarray) -> np.ndarray:
    """
    @brief Extract 2D covariance features from covariance matrix.

    @param cov_matrix: Covariance matrix (3x3 for [x, y, yaw] or 6x6 for
                       full 3D pose).
    @return: 1D array of 3 covariance features [std_x, std_y, std_yaw].
    @raises ValueError: If matrix shape is not 3x3 or 6x6.
    """
    if cov_matrix.shape == (6, 6):
        # Full 6D pose covariance [x, y, z, roll, pitch, yaw]
        # Extract diagonal variances for x=0, y=1, yaw=5
        var_x = cov_matrix[0, 0]
        var_y = cov_matrix[1, 1]
        var_yaw = cov_matrix[5, 5]
    elif cov_matrix.shape == (3, 3):
        var_x = cov_matrix[0, 0]
        var_y = cov_matrix[1, 1]
        var_yaw = cov_matrix[2, 2]
    else:
        raise ValueError(
            f"Expected 3x3 or 6x6 covariance matrix, got shape {cov_matrix.shape}"
        )

    return np.array([np.sqrt(var_x), np.sqrt(var_y), np.sqrt(var_yaw)])


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

    A valid covariance matrix must be:
    1. Symmetric: C = C^T
    2. Positive semi-definite: all eigenvalues >= 0
    """
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        return False

    # Check symmetry
    if not np.allclose(cov_matrix, cov_matrix.T):
        return False

    # Check positive semi-definiteness via eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigenvalues < -1e-10):  # Small negative tolerance for numerical errors
        return False

    return True


def make_diagonal_covariance(diag: List[float]) -> List[float]:
    """
    @brief Build a flat 36-element ROS covariance array from a 6-element diagonal.

    ROS nav_msgs/Odometry pose.covariance is a row-major 6x6 matrix stored as
    a flat list of 36 floats. This helper constructs that array from the six
    diagonal variance values, leaving all off-diagonal elements as zero.

    @param diag: Six diagonal variance values [var_x, var_y, var_z,
                 var_roll, var_pitch, var_yaw].
    @return Flat list of 36 floats (row-major 6x6, zeros off-diagonal).
    """
    cov: List[float] = [0.0] * 36
    for i, v in enumerate(diag):
        cov[i * 7] = v
    return cov
