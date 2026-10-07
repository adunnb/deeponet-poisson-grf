"""Inference cost of the trained DeepONet versus the finite-difference solver."""
import time

import numpy as np
import torch

from model import synchronize
from poisson import FactoredPoissonSolver, solve_poisson, stiffness_matrix


def median_seconds(run, device, n_repeats):
    """Median wall-clock seconds per call, after one untimed warm-up call."""
    run()  # Warm-up: allocator, BLAS threads, and any lazy initialization.
    samples = []
    for _ in range(n_repeats):
        synchronize(device)
        start = time.perf_counter()
        run()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def run_benchmark(model, query_x, output_scale, inputs, forcing, grid, *,
                  batch_sizes, n_repeats, device):
    """Time the network and both solver baselines at each batch size.

    inputs: normalized branch inputs, (functions, sensors).
    forcing: the same functions on the full grid, (functions, grid points).
    Baselines: "rebuild" assembles and factors the operator on every call
    (solve_poisson as used for the data); "reuse" factors it once.
    """
    forcing = np.ascontiguousarray(forcing)
    inputs = inputs.contiguous()

    stiffness = stiffness_matrix(grid.size - 2, grid[1] - grid[0])
    factorization_seconds = median_seconds(lambda: np.linalg.inv(stiffness), device, n_repeats)
    solve_reuse = FactoredPoissonSolver(grid)

    def solve_rebuild(batch):
        return solve_poisson(batch, grid)

    def predict_network(batch):
        with torch.no_grad():
            return model(batch, query_x) * output_scale

    # The fast baseline must solve the same problem before it is timed against.
    np.testing.assert_allclose(solve_reuse(forcing[:8]), solve_rebuild(forcing[:8]),
                               rtol=1e-8, atol=1e-10)

    model.eval()
    rows = []
    for batch in batch_sizes:
        forcing_batch = np.ascontiguousarray(forcing[:batch])
        inputs_batch = inputs[:batch].contiguous()
        row = {
            "batch": batch,
            "network": median_seconds(lambda b=inputs_batch: predict_network(b), device, n_repeats),
            "rebuild": median_seconds(lambda b=forcing_batch: solve_rebuild(b), device, n_repeats),
            "reuse": median_seconds(lambda b=forcing_batch: solve_reuse(b), device, n_repeats),
        }
        per_function = {key: 1e6 * row[key] / batch for key in ("network", "rebuild", "reuse")}
        row.update({f"{key}_microseconds_per_function": value for key, value in per_function.items()})
        row["speedup_vs_rebuild"] = per_function["rebuild"] / per_function["network"]
        row["speedup_vs_reuse"] = per_function["reuse"] / per_function["network"]
        rows.append(row)
    return {"operator_factorization_seconds": factorization_seconds, "rows": rows}


def print_table(rows):
    header = (f"{'batch':>6} | {'network':>12} | {'FD rebuild':>12} | {'FD reuse':>12} | "
              f"{'vs rebuild':>10} | {'vs reuse':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        us = {key: row[f"{key}_microseconds_per_function"] for key in ("network", "rebuild", "reuse")}
        print(f"{row['batch']:>6} | {us['network']:>9.1f} us | {us['rebuild']:>9.1f} us | "
              f"{us['reuse']:>9.1f} us | {row['speedup_vs_rebuild']:>9.2f}x | {row['speedup_vs_reuse']:>8.2f}x")
    print("\nTimes are per forcing function; speedups above 1 favor the network.")


def break_even_solves(training_seconds, row):
    """Solves needed for training to pay for itself versus the rebuild baseline."""
    saving = (row["rebuild"] - row["network"]) / row["batch"]
    return training_seconds / saving if saving > 0 else float("inf")
