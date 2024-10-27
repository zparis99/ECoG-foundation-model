import torch

from metrics import pearson_correlation


def test_get_signal_correlations():
    signal_a = torch.ones(16, 5, 40, 8, 8)

    # Flip sign for every time step.
    for i in range(signal_a.shape[2]):
        signal_a[:, :, i, :, :] *= -(1**i) * i

    signal_b = signal_a * -1

    corr_matrix = pearson_correlation(signal_a, signal_b)

    assert corr_matrix.detach().numpy().shape == (5, 8, 8)
    assert torch.isclose(corr_matrix, -torch.ones_like(corr_matrix)).all()
