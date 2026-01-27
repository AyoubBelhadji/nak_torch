#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import warnings
import numpy as np
import torch
from typing import Optional, Callable
from tqdm import tqdm
from nak_torch.tools.kernel import sqexp_kernel_elem
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.types import KernelFunction, BatchGradLogDensity, BatchPtType
from nak_torch.tools.util import batched_grad_log_density_factory, initialize_particles


def create_svgd_step(
    kernel_elem: KernelFunction,
    grad_log_p: BatchGradLogDensity,
    kernel_length_scale: float
) -> Callable[[BatchPtType], BatchPtType]:
    kernel_grad_val = torch.func.grad_and_value(
        lambda x, y: kernel_elem(x, y, kernel_length_scale), argnums=1
    )
    kernel_grad_val_vec = torch.vmap(
        torch.vmap(kernel_grad_val, in_dims=(None, 0)),
        in_dims=(0, None)
    )

    def svgd_step_dir(points: BatchPtType):
        # kg[i,j,k] = grad_2(k) k(x_i, x_j), k[j,i] = k(x_i, x_j)
        k_grad, k_eval = kernel_grad_val_vec(points, points)
        # lpg[i,k] = grad(k) log_p(x_i)
        log_p_grad_ev = grad_log_p(points)
        term_1 = k_eval.T @ log_p_grad_ev
        term_2 = k_grad.sum(1)
        return (term_1 + term_2) / points.shape[0]

    return svgd_step_dir


def svgd(
    log_density,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    kernel_length_scale: float = 1.0,
    kernel_elem: KernelFunction = sqexp_kernel_elem,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    is_log_density_batched: bool = False,
    grad_log_density: Optional[BatchGradLogDensity] = None,
    **unused_kwargs
):
    if len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)

    particles = initialize_particles(
        n_particles, dim, init_particles, device, bounds
    )

    if keep_all:
        trajectories = torch.empty(
            (n_steps, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    grad_log_p = batched_grad_log_density_factory(log_density, is_log_density_batched, grad_log_density)
    step_fcn = create_svgd_step(kernel_elem, grad_log_p, kernel_length_scale)

    for idx in tqdm(range(n_steps)):
        particles_diff = step_fcn(particles)
        with torch.no_grad():
            particles = particles + lr * particles_diff
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)
