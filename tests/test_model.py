import torch

from model import DeepONet, interpolate_targets


def test_output_shape():
    model = DeepONet(n_sensors=65, width=16, latent_dim=8)
    out = model(torch.randn(5, 65), torch.rand(7, 1))
    assert out.shape == (5, 7)


def test_interpolation_reproduces_grid_values():
    values = torch.randn(3, 33)
    grid = torch.linspace(0, 1, 33).reshape(-1, 1)
    torch.testing.assert_close(interpolate_targets(values, grid), values, rtol=1e-5, atol=1e-6)


def test_interpolation_is_linear_between_nodes():
    values = torch.tensor([[0.0, 2.0, 4.0]])  # nodes at x = 0, 0.5, 1
    x = torch.tensor([[0.25], [0.75]])
    torch.testing.assert_close(interpolate_targets(values, x), torch.tensor([[1.0, 3.0]]))
