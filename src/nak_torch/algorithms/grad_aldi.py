import torch
from typing import Optional
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.types import BatchGradLogDensity, BatchPtType
import warnings
from tqdm import tqdm
import numpy as np
from nak_torch.tools.util import sym_sqrtm

@torch.compile
def grad_aldi_step(
    particles: BatchPtType,
    grad_log_dens: BatchPtType,
    rng: torch.Generator,
) -> tuple[BatchPtType, BatchPtType]:

    batch, dim = particles.shape
    particles_mean = particles.mean(dim=0, keepdim=True)
    particles_diff = particles - particles_mean
    particles_cov = (particles_diff.T @ particles_diff) / batch
    # -C(U) ∇Φ(u^i)--- note that Φ = -log p
    term1 = grad_log_dens @ particles_cov.T
    # (D+1)/N (u^i - m(U))
    term2 = particles_diff.mul_((dim + 1) / batch)
    # Get noise
    particles_sqrt_cov = sym_sqrtm(2 * particles_cov)
    # sqrt(2) comes from noise
    particles_noise_iid = torch.normal(
        0.0, 1.0, size=particles.shape, generator=rng
    )
    particles_noise = particles_noise_iid @ particles_sqrt_cov
    drift_term = term1.add_(term2)
    return drift_term, particles_noise


def grad_aldi(
    log_density,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    rng: Optional[torch.Generator] = None,
    keep_all: bool = True,
    is_density_vectorized: bool = False,
    grad_log_density: Optional[BatchGradLogDensity] = None,
    **unused_kwargs
):
    if len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if rng is None:
        rng = torch.default_generator
    if seed is not None:
        rng.manual_seed(seed)

    if grad_log_density is None:
        if is_density_vectorized:
            def grad_log_p_(pts: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
                pts_cl = pts.clone().requires_grad_()
                return torch.autograd.grad(log_density(pts_cl).sum(), pts_cl)[0]
            grad_log_p = grad_log_p_
        else:
            grad_log_p = torch.vmap(torch.func.grad(log_density))
    else:
        grad_log_p = grad_log_density

    particles: Tensor
    if init_particles is None:
        if bounds is None:
            particles = torch.normal(0.0, 1.0, (n_particles, dim), generator=rng, device=device)
        else:
            particles = torch.empty((n_particles, dim), device=device).uniform_(*bounds)
    else:
        particles = torch.as_tensor(init_particles, device=device).clone()

    if keep_all:
        trajectories = torch.empty(
            (n_steps, *particles.shape), dtype=particles.dtype)
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    sqrt_lr = torch.sqrt(torch.tensor(lr))

    for idx in tqdm(range(n_steps)):
        grad_log_dens_eval = grad_log_p(particles)
        with torch.no_grad():
            particles_diff, particles_noise = grad_aldi_step(particles, grad_log_dens_eval, rng)
            particles_diff.mul_(lr)
            particles_noise.mul_(sqrt_lr)
            particles.add_(particles_diff).add_(particles_noise)
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx].copy_(particles)

    if keep_all:
        return trajectories, bounds
    else:
        return particles.unsqueeze_(0), bounds
