"""Finite-difference reference solver for -u''(x) = f(x), u(0) = u(1) = 0."""
import numpy as np


def stiffness_matrix(n_interior, h):
    """Second-order centered difference operator on the interior grid points."""
    return (2 * np.eye(n_interior) - np.eye(n_interior, k=1)
            - np.eye(n_interior, k=-1)) / h**2


def solve_poisson(forcing, grid):
    """Solve on a uniform grid; endpoint values of u are fixed at zero.

    Accepts one curve of shape (grid points,) or a batch of shape
    (functions, grid points); all right-hand sides are solved together.
    The operator is assembled and factored on every call.
    """
    forcing = np.asarray(forcing, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("Use a one-dimensional grid with at least three points.")
    h = grid[1] - grid[0]
    if h <= 0 or not np.allclose(np.diff(grid), h):
        raise ValueError("The finite-difference solver requires an increasing uniform grid.")
    if forcing.ndim not in (1, 2) or forcing.shape[-1] != grid.size:
        raise ValueError("Forcing must have shape (grid points,) or (functions, grid points).")
    one_curve = forcing.ndim == 1
    forcing = np.atleast_2d(forcing)
    A = stiffness_matrix(grid.size - 2, h)
    solution = np.zeros_like(forcing)
    solution[:, 1:-1] = np.linalg.solve(A, forcing[:, 1:-1].T).T
    return solution[0] if one_curve else solution


class FactoredPoissonSolver:
    """Factor the fixed operator once, then apply it to new forcing batches.

    This is what a production solver would do on a fixed grid. The factorization
    here is an explicit inverse of the (n_grid - 2) x (n_grid - 2) operator.
    """

    def __init__(self, grid):
        grid = np.asarray(grid, dtype=np.float64)
        self.inverse_operator = np.linalg.inv(stiffness_matrix(grid.size - 2, grid[1] - grid[0]))

    def __call__(self, forcing):
        solution = np.zeros_like(forcing)
        solution[:, 1:-1] = forcing[:, 1:-1] @ self.inverse_operator.T
        return solution
