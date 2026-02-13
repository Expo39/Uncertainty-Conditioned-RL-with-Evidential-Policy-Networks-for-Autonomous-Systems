"""
@file covariance_utils.py
@brief Shared utilities for SLAM covariance matrix extraction.

This module provides common functions for extracting uncertainty features
from SLAM covariance matrices for use in RL state representations.
"""

import numpy as np
from typing import Tuple


def extract_2d_covariance_features(cov_matrix: np.ndarray) -> np.ndarray:
    """
    @brief Extract 2D covariance features from covariance matrix.
    
    Extracts [std_x, std_y, std_yaw, cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw]
    for use as RL state input. This provides both marginal uncertainties
    (standard deviations) and correlation structure (covariances).
    
    @param cov_matrix: Covariance matrix (3x3 for [x, y, yaw] or 6x6 for full 3D pose).
    @return: 1D array of 9 covariance features.
    @raises ValueError: If matrix shape is not 3x3 or 6x6.
    
    Example:
        >>> cov_6d = np.eye(6) * 0.1  # 10cm std on all axes
        >>> features = extract_2d_covariance_features(cov_6d)
        >>> assert features.shape == (9,)
        >>> assert np.isclose(features[0], np.sqrt(0.1))  # std_x
    """
    # Extract 2D pose covariance (x, y, yaw)
    if cov_matrix.shape == (6, 6):
        # Full 6D pose covariance [x, y, z, roll, pitch, yaw]
        # Extract indices: x=0, y=1, yaw=5
        cov_2d = np.array([
            [cov_matrix[0, 0], cov_matrix[0, 1], cov_matrix[0, 5]],
            [cov_matrix[1, 0], cov_matrix[1, 1], cov_matrix[1, 5]],
            [cov_matrix[5, 0], cov_matrix[5, 1], cov_matrix[5, 5]],
        ])
    elif cov_matrix.shape == (3, 3):
        # Already 2D pose covariance [x, y, yaw]
        cov_2d = cov_matrix
    else:
        raise ValueError(
            f"Expected 3x3 or 6x6 covariance matrix, got shape {cov_matrix.shape}"
        )
    
    # Extract standard deviations (marginal uncertainties)
    std_x = np.sqrt(cov_2d[0, 0])
    std_y = np.sqrt(cov_2d[1, 1])
    std_yaw = np.sqrt(cov_2d[2, 2])
    
    # Extract full covariance terms (includes correlations)
    cov_xx = cov_2d[0, 0]
    cov_yy = cov_2d[1, 1]
    cov_yawyaw = cov_2d[2, 2]
    cov_xy = cov_2d[0, 1]  # Correlation between x and y
    cov_xyaw = cov_2d[0, 2]  # Correlation between x and yaw
    cov_yyaw = cov_2d[1, 2]  # Correlation between y and yaw

    return np.array([std_x, std_y, std_yaw, cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw])


def get_covariance_dimension() -> int:
    """
    @brief Get the dimensionality of extracted covariance features.
    @return: Number of covariance features (9 for 2D case).
    
    This is useful for defining observation space dimensions in Gymnasium environments.
    """
    return 9


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
