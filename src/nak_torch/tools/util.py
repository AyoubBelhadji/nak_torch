import torch
from torch import Tensor
from jaxtyping import Float
from typing import Optional, Callable
from .types import BatchPtType
import numpy as np
import inspect

def sym_sqrtm(A: Float[Tensor, "n n"], use_inv: bool = False):
    e, v = torch.linalg.eigh(A)
    if use_inv:
        return torch.einsum("ij,j,kj->ik", v, torch.reciprocal_(e.sqrt_()), v)
    else:
        return torch.einsum("ij,j,kj->ik", v, e.sqrt_(), v)

def get_keywords(fcn: Callable):
    sig = inspect.signature(fcn)
    return [p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY]

def initialize_particles(
        n_particles: int,
        dim: int,
        init_particles: Optional[Tensor | np.ndarray],
        device: Optional[torch.device],
        bounds: Optional[tuple[float, float]],
        rng: Optional[torch.Generator] = None
) -> BatchPtType:
    rng = torch.default_generator
    if init_particles is None:
        if bounds is None:
            return torch.randn((n_particles, dim), device=device)
        else:
            return torch.empty((n_particles, dim), device=device).uniform_(*bounds, generator=rng)
    if init_particles.shape != (n_particles, dim):
        raise ValueError("Unexpected dimensions of init particles: got {}, expected {}".format(
            init_particles.shape, (n_particles, dim)
        ))
    if device is not None and init_particles.device != torch.device(device):
        raise ValueError("Unexpected device for init_particles: got {}, expected {}".format(
            init_particles.device, torch.device(device)
        ))
    return torch.as_tensor(init_particles, device=device).clone()

def batched_grad_log_density_factory(
        log_density: Callable,
        is_log_density_batched: bool,
        grad_log_density: Optional[Callable],
):
    if grad_log_density is None:
        if is_log_density_batched:
            def grad_log_p_(pts: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
                pts_cl = pts.clone().requires_grad_()
                return torch.autograd.grad(log_density(pts_cl).sum(), pts_cl)[0]
            return grad_log_p_
        else:
            return torch.vmap(torch.func.grad(log_density))
    else:
        return grad_log_density
