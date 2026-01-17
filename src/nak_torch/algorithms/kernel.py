import torch

def gaussian_kernel_matrix(particles_leaf, sigma2):
    diff = particles_leaf.unsqueeze(1) - \
                particles_leaf.unsqueeze(0)  # (N, N, d)

    return torch.exp(- (diff ** 2).sum(dim=-1) / sigma2)
