#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import warnings
import numpy as np
import torch
from typing import Optional
from tqdm import tqdm
from nak_torch.tools.kernel import sqexp_kernel_elem
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.types import KernelFunction, BatchGradLogDensity


def create_svgd_step(
    kernel_elem: KernelFunction,
    grad_log_p: BatchGradLogDensity,
    bandwidth: float
):
    kernel_grad_val = torch.func.grad_and_value(
        lambda x, y: kernel_elem(x, y, bandwidth), argnums=1
    )
    kernel_grad_val_vec = torch.vmap(
        torch.vmap(kernel_grad_val, in_dims=(None, 0)),
        in_dims=(0, None)
    )

    def svgd_step_dir(points: Float[Tensor, "batch d"]):
        # kg[i,j,k] = grad_2(k) k(x_i, x_j), k[j,i] = k(x_i, x_j)
        k_grad, k_eval = kernel_grad_val_vec(points, points)
        # lpg[i,k] = grad(k) log_p(x_i)
        log_p_grad_ev = grad_log_p(points)
        term_1 = torch.einsum("jd,ji->id", log_p_grad_ev, k_eval)
        term_2 = k_grad.sum(1)
        return (term_1 + term_2) / points.shape[0]

    return torch.compile(svgd_step_dir)


def svgd(
    log_density,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    kernel_bandwidth: float = 1.0,
    kernel_elem: Optional[KernelFunction] = None,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    is_objective_vectorized: bool = False,
    grad_log_density: Optional[BatchGradLogDensity] = None,
    **unused_kwargs
):
    if len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)

    particles: Tensor
    if init_particles is None:
        if bounds is None:
            particles = torch.randn((n_particles, dim), device=device)
        else:
            particles = torch.empty((n_particles, dim), device=device).uniform_(bounds[0], bounds[1])
    else:
        particles = torch.as_tensor(init_particles, device=device).clone()
    if keep_all:
        trajectories = torch.empty(
            (n_steps, *particles.shape), dtype=particles.dtype)
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    grad_log_p: BatchGradLogDensity
    kernel_fcn: KernelFunction = sqexp_kernel_elem if kernel_elem is None else kernel_elem
    if grad_log_density is None:
        if is_objective_vectorized:
            def grad_log_p_(pts: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
                pts_cl = pts.clone().requires_grad_()
                return torch.autograd.grad(log_density(pts_cl).sum(), pts_cl)[0]
            grad_log_p = grad_log_p_
        else:
            grad_log_p = torch.vmap(torch.func.grad(log_density))
    else:
        grad_log_p = grad_log_density

    step_fcn = create_svgd_step(kernel_fcn, grad_log_p, kernel_bandwidth)

    for idx in tqdm(range(n_steps)):
        particles_diff = step_fcn(particles)
        with torch.no_grad():
            particles = (1.0 - lr) * particles + lr * particles_diff
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx].copy_(particles)

    if keep_all:
        return trajectories, bounds
    else:
        return particles.unsqueeze_(0), bounds
