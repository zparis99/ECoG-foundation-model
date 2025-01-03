import torch

from metrics import pearson_correlation


def test_get_signal_correlations():
    signal_a = torch.ones(16)

    # Flip sign for every time step.
    for i in range(signal_a.shape[0]):
        signal_a[i] *= -(1**i) * i

    signal_b = signal_a * -1

    corr = pearson_correlation(signal_a, signal_b)

    assert corr.detach().numpy().shape == ()
    assert torch.isclose(corr, torch.tensor(-1.0))
