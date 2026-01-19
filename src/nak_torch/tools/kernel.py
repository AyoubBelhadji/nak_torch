import torch
from typing import Optional

def gaussian_kernel_matrix(particles_leaf: torch.Tensor, sigma2: float, p_leaf2: Optional[torch.Tensor] = None):
    if p_leaf2 is None:
        p_leaf2 = particles_leaf
    diff = particles_leaf.unsqueeze(1) - p_leaf2.unsqueeze(0)  # (N, N, d)

    return torch.exp(- (diff ** 2).sum(dim=-1) / sigma2)
