import torch.nn.functional as F


def pearson_correlation(x1, x2):
    """Compute pearson correlation between x1 and x2.

    Args:
        x1 (Tensor): shape [N]
        x2 (Tensor): shape [N]
    """
    return F.cosine_similarity(
        x1 - x1.mean(),
        x2 - x2.mean(),
        dim=0,
    )
