import torch.nn.functional as F


def pearson_correlation(x1, x2):
    """Compute pearson correlation between x1 and x2.

    Args:
        x1 (Tensor): shape [N, C, T, H, W]
        x2 (Tensor): shape [N, C, T, H, W]
    """
    correlations = F.cosine_similarity(
        x1 - x1.mean(dim=2, keepdims=True), x2 - x2.mean(dim=2, keepdims=True), dim=2
    )
    return correlations.mean(dim=0)
