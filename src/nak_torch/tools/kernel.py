import torch
from typing import Optional
from jaxtyping import Float
from torch import Tensor


def sqexp_kernel_matrix(
        particles_leaf: Float[Tensor, "batch d"],
        sigma2: float,
        p_leaf2: Optional[Float[Tensor, "batch d"]] = None
):
    if p_leaf2 is None:
        p_leaf2 = particles_leaf
    diff = particles_leaf.unsqueeze(1) - p_leaf2.unsqueeze(0)  # (N, N, d)

    return torch.exp(- (diff ** 2).sum(dim=-1) / sigma2)


def sqexp_kernel_elem(x: Float[Tensor, " d"], y: Float[Tensor, " d"], sigma_sq: float = 1.0) -> Float:
    assert x.shape == y.shape and y.ndim == 1
    ret = torch.exp(- (x - y).square_().sum() / (2*sigma_sq))
    return ret
