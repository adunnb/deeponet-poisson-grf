import numpy as np
import pytest

from grf import cholesky_factor, sample_grf, sensor_indices


def test_samples_are_reproducible_and_shaped():
    grid = np.linspace(0, 1, 65)
    L = cholesky_factor(grid, 0.2, 1.0, 1e-10)
    a, b = sample_grf(L, 10, seed=5), sample_grf(L, 10, seed=5)
    assert a.shape == (10, 65)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, sample_grf(L, 10, seed=6))


def test_sample_covariance_matches_kernel():
    grid = np.linspace(0, 1, 17)
    sigma, length_scale = 1.5, 0.3
    samples = sample_grf(cholesky_factor(grid, length_scale, sigma, 1e-10), 20000, seed=0)
    kernel = sigma**2 * np.exp(-(grid[:, None] - grid[None, :])**2 / (2 * length_scale**2))
    np.testing.assert_allclose(np.cov(samples, rowvar=False), kernel, atol=0.1)


def test_sensor_indices_lie_on_grid():
    indices = sensor_indices(257, 65)
    assert len(indices) == 65 and indices[0] == 0 and indices[-1] == 256
    with pytest.raises(ValueError):
        sensor_indices(257, 64)
