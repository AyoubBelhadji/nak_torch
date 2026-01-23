#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import warnings
import numpy as np
import torch
from nak_torch.tools.util import initialize_particles, get_keywords
from .msip_map import msip_map, MSIPEstimator, MSIPFredholm
from typing import Optional
from tqdm import tqdm
from nak_torch.tools.types import BatchLogDensityGradVal, BatchPtType, BatchType, LogDensity, BatchLogDensity


def process_msip_density(
    log_density: LogDensity | BatchLogDensity | MSIPEstimator,
    *_,
    is_log_density_batched: bool = False,
    kernel_length_scale: float = 0.0,
    gradient_decay: float = 1.0,
    **__
) -> MSIPEstimator:
    if isinstance(log_density, MSIPEstimator):
        return log_density
    sigma_sq = kernel_length_scale**2
    log_density_grad_val: BatchLogDensityGradVal
    if is_log_density_batched:
        def log_density_grad_val_(particles: BatchPtType) -> tuple[BatchPtType, BatchType]:
            particles_copy = particles.clone().requires_grad_(True)
            log_dens_eval = log_density(particles_copy)
            grad_log_dens_eval, = torch.autograd.grad(
                log_dens_eval.sum(), particles_copy
            )
            return grad_log_dens_eval, log_dens_eval
        log_density_grad_val = log_density_grad_val_
    else:
        log_density_grad_val = torch.vmap(
            torch.func.grad_and_value(log_density))
    return MSIPFredholm(sigma_sq, gradient_decay, log_density_grad_val)


msip_map_used_keys = get_keywords(msip_map) + \
    get_keywords(process_msip_density)


def msip(
    log_density: LogDensity | BatchLogDensity | MSIPEstimator,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    noise: float = 0.05,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    **msip_kwargs
):
    r"""
        TODO: Document
    """
    unused_kwargs = {
        k: v for (k, v) in msip_kwargs.items() if k not in msip_map_used_keys
    }

    if len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs: {}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)

    msip_estimator = process_msip_density(log_density, **msip_kwargs)
    particles = initialize_particles(
        n_particles, dim, init_particles, device, bounds
    )

    if keep_all:
        trajectories = torch.empty(
            (n_steps+1, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
        traj_log_wts = torch.empty(
            (n_steps, particles.shape[0]), device=device, dtype=particles.dtype
        )
        traj_log_wts[0].fill_(0.)
    else:
        trajectories = torch.empty(())
        traj_log_wts = torch.empty(())

    log_wts = torch.empty(())

    for idx in tqdm(range(n_steps)):
        msip_estimator_out = msip_estimator.get_v_evals(particles)

        particles_diff, log_wts = msip_map(
            msip_estimator_out,
            particles,
            output_idx=None,
            **msip_kwargs
        )

        with torch.no_grad():
            particles = (1.0 - lr) * particles + lr * particles_diff
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx+1].copy_(particles)
            torch.softmax(log_wts, 0, out=traj_log_wts[idx])

    if keep_all:
        return trajectories.detach(), traj_log_wts.detach()
    else:
        return particles.unsqueeze_(0).detach(), log_wts.unsqueeze(0).detach()
