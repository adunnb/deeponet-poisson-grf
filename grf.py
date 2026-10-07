"""Gaussian random field forcing curves.

The forcing functions are draws from a zero-mean GRF with squared-exponential
covariance K(x, y) = sigma^2 exp(-(x - y)^2 / (2 l^2)).
"""
import numpy as np


def covariance_matrix(grid, length_scale, sigma):
    distance = grid[:, None] - grid[None, :]
    return sigma**2 * np.exp(-distance**2 / (2 * length_scale**2))


def cholesky_factor(grid, length_scale, sigma, jitter):
    """Lower-triangular L with L L^T = K + jitter I.

    The tiny diagonal jitter stabilizes the factorization.
    """
    K = covariance_matrix(grid, length_scale, sigma)
    return np.linalg.cholesky(K + jitter * np.eye(grid.size))


def sample_grf(L, n_functions, seed):
    """Draw n_functions forcing curves, one per row, in float64."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((L.shape[0], n_functions))
    return (L @ z).T  # (functions, grid points)


def sensor_indices(n_grid, n_sensors):
    """Evenly spaced branch sensors that lie on the solution grid."""
    if (n_grid - 1) % (n_sensors - 1) != 0:
        raise ValueError("Sensors must lie on the solution grid.")
    return np.arange(0, n_grid, (n_grid - 1) // (n_sensors - 1))

