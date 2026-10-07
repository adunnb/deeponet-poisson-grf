"""DeepONet architecture and training loop."""
import math
import time

import torch
from torch import nn


def make_mlp(input_dim, width, latent_dim):
    network = nn.Sequential(
        nn.Linear(input_dim, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, latent_dim),
    )
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return network


class DeepONet(nn.Module):
    """Branch network on sensor values, trunk network on the query coordinate."""

    def __init__(self, n_sensors, width, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.branch = make_mlp(n_sensors, width, latent_dim)
        self.trunk = make_mlp(1, width, latent_dim)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, function_values, x):
        branch_features = self.branch(function_values)
        trunk_features = self.trunk(2 * x - 1)
        return branch_features @ trunk_features.T / math.sqrt(self.latent_dim) + self.bias


def interpolate_targets(values, query_x):
    """Linear interpolation of grid solutions at queries in [0, 1].

    values: (batch, n_grid); query_x: (queries, 1).
    The training loop supplies in-domain queries, including both endpoints.
    """
    n_grid = values.shape[1]
    positions = query_x.reshape(-1) * (n_grid - 1)
    left = positions.floor().long().clamp(0, n_grid - 2)
    weight = (positions - left.to(positions.dtype)).unsqueeze(0)
    return (1 - weight) * values[:, left] + weight * values[:, left + 1]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train(model, train_inputs, train_targets, output_scale,
          validation_inputs, validation_x, validation_targets, *,
          n_steps, batch_size, n_queries, validate_every, batch_seed, device):
    """Adam with cosine learning-rate decay and validation checkpoint selection.

    Each step samples a batch of functions and fresh query coordinates
    (both endpoints included), and fits the linearly interpolated reference.
    The checkpoint with the lowest validation relative L2 error is restored
    into `model` before returning.
    """
    n_train = train_inputs.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=1e-5)
    batch_generator = torch.Generator().manual_seed(batch_seed)
    train_history, validation_history = [], []
    best_validation_error = float("inf")
    best_state, best_step = None, None

    synchronize(device)
    start_time = time.perf_counter()
    model.train()
    for step in range(1, n_steps + 1):
        indices = torch.randint(n_train, (batch_size,), generator=batch_generator).to(device)
        # New query locations each step, with both endpoints included.
        x = torch.cat([
            torch.tensor([[0.0], [1.0]]),
            torch.rand(n_queries - 2, 1, generator=batch_generator),
        ]).to(device)
        targets = interpolate_targets(train_targets[indices], x) / output_scale

        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_inputs[indices], x)
        loss = (predictions - targets).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}.")
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_history.append(loss.detach().item())

        if step % validate_every == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                predictions = model(validation_inputs, validation_x) * output_scale
                error = torch.linalg.vector_norm(predictions - validation_targets)
                error /= torch.linalg.vector_norm(validation_targets)
            validation_error = error.item()
            validation_history.append((step, validation_error))
            if validation_error < best_validation_error:
                best_validation_error = validation_error
                best_step = step
                best_state = {name: tensor.detach().cpu().clone()
                              for name, tensor in model.state_dict().items()}
            model.train()
            if step % 2000 == 0 or step == n_steps:
                print(f"Step {step:5d} | MSE {loss.item():.3e} | validation relative L2 {validation_error:.3%}")

    synchronize(device)
    training_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    model.eval()
    return {
        "train_history": train_history,
        "validation_history": validation_history,
        "best_step": best_step,
        "best_validation_error": best_validation_error,
        "training_seconds": training_seconds,
    }
