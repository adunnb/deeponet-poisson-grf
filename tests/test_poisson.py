import numpy as np
import pytest

from grf import cholesky_factor, sample_grf
from poisson import FactoredPoissonSolver, solve_poisson


def test_constant_forcing_is_exact():
    # -u'' = 1 has exact solution x(1 - x)/2, which the scheme reproduces exactly.
    grid = np.linspace(0, 1, 257)
    np.testing.assert_allclose(solve_poisson(np.ones_like(grid), grid),
                               grid * (1 - grid) / 2, atol=1e-10, rtol=0)


def test_second_order_convergence():
    errors = []
    for n_points in (65, 129, 257):
        grid = np.linspace(0, 1, n_points)
        forcing = np.sin(np.pi * grid)
        errors.append(np.max(np.abs(solve_poisson(forcing, grid) - forcing / np.pi**2)))
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    assert np.all((ratios > 3.9) & (ratios < 4.1))


def test_grf_batch_satisfies_discrete_equation():
    grid = np.linspace(0, 1, 257)
    forcing = sample_grf(cholesky_factor(grid, 0.2, 1.0, 1e-10), 16, seed=0)
    solution = solve_poisson(forcing, grid)
    h = grid[1] - grid[0]
    residual = (-solution[:, :-2] + 2 * solution[:, 1:-1] - solution[:, 2:]) / h**2 - forcing[:, 1:-1]
    assert np.max(np.abs(residual)) < 1e-8 * max(1.0, np.max(np.abs(forcing)))
    assert np.all(solution[:, [0, -1]] == 0)


def test_single_curve_matches_batch():
    grid = np.linspace(0, 1, 65)
    forcing = sample_grf(cholesky_factor(grid, 0.2, 1.0, 1e-10), 3, seed=1)
    np.testing.assert_array_equal(solve_poisson(forcing[1], grid), solve_poisson(forcing, grid)[1])


def test_factored_solver_matches_direct_solve():
    grid = np.linspace(0, 1, 129)
    forcing = sample_grf(cholesky_factor(grid, 0.2, 1.0, 1e-10), 8, seed=2)
    np.testing.assert_allclose(FactoredPoissonSolver(grid)(forcing), solve_poisson(forcing, grid),
                               rtol=1e-8, atol=1e-10)


def test_rejects_nonuniform_grid():
    grid = np.array([0.0, 0.1, 0.5, 1.0])
    with pytest.raises(ValueError):
        solve_poisson(np.ones(4), grid)
