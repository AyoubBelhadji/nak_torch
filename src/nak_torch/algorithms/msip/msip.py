#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import warnings
from typing import Optional

from tqdm import tqdm
import numpy as np
import torch

from nak_torch.tools.kernel import default_kernel_matrix
from nak_torch.tools.util import initialize_particles, get_keywords
from .msip_map import MSIPEstimatorOutput, msip_map, get_msip_wts
from .estimators import MSIPEstimator, MSIPFredholm

from nak_torch.tools.types import LogDensity, BatchLogDensity, \
    BatchLogDensityGradVal, BatchPtType, BatchType, MatSelfKernelFunction


def process_msip_density(
    log_density: LogDensity | BatchLogDensity | MSIPEstimator,
    *_,
    is_log_density_batched: bool = False,
    gradient_decay: float = 1.0,
    **__
) -> MSIPEstimator:
    if isinstance(log_density, MSIPEstimator):
        return log_density
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
    return MSIPFredholm(gradient_decay, log_density_grad_val)


msip_map_used_keys = get_keywords(msip_map) + \
    get_keywords(process_msip_density)


def msip(
    log_density: LogDensity | BatchLogDensity | MSIPEstimator,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    kernel_length_scale: float,
    noise: float = 0.05,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    get_kernel_matrix: Optional[MatSelfKernelFunction] = None,
    kernel_diag_infl: float = 0.0,
    **msip_kwargs
):
    r"""
        TODO: Document
    """

    if n_steps < 0:
        raise ValueError("Expected positive number of steps.")

    unused_kwargs = {
        k: v for (k, v) in msip_kwargs.items() if k not in msip_map_used_keys
    }

    if len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs: {}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)
    if get_kernel_matrix is None:
        get_kernel_matrix = default_kernel_matrix

    msip_estimator = process_msip_density(log_density, **msip_kwargs)
    particles = initialize_particles(
        n_particles, dim, init_particles, device, bounds
    )

    if keep_all:
        trajectories = torch.empty(
            (n_steps+1, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
        traj_wts = torch.empty(
            (n_steps, particles.shape[0]), device=device, dtype=particles.dtype
        )
    else:
        trajectories = torch.empty(())
        traj_wts = torch.empty(
            (1, particles.shape[0]), device=device, dtype=particles.dtype
        )

    msip_estimator_out: MSIPEstimatorOutput
    for idx in tqdm(range(n_steps + 1)):
        kernel_matrix = get_kernel_matrix(particles, kernel_length_scale)
        kernel_matrix[
            torch.arange(n_particles), torch.arange(n_particles)
        ] += kernel_diag_infl

        msip_estimator_out = msip_estimator.get_v_evals(
            particles, kernel_length_scale
        )

        particle_wts = get_msip_wts(
            particles, msip_estimator_out,
            kernel_matrix
        )

        if idx < n_steps:
            if kernel_diag_infl > 0:
                kernel_matrix_inverse = torch.linalg.inv(kernel_matrix)
            else:
                kernel_matrix_inverse = torch.linalg.pinv(kernel_matrix)

            particles_diff = msip_map(
                msip_estimator_out,
                particles,
                kernel_matrix_inverse,
                output_idx=None,
            )

            with torch.no_grad():
                particles = (1.0 - lr) * particles + lr * particles_diff
                if bounds is not None:
                    particles.clamp_(bounds[0], bounds[1])
            if keep_all:
                trajectories[idx+1].copy_(particles)
                traj_wts[idx].copy_(particle_wts)

    if not keep_all:
        trajectories = particles.unsqueeze_(0)

    return trajectories.detach(), traj_wts.detach()
