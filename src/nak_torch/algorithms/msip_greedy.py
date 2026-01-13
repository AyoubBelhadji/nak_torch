#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles_greedy
# Ayoub Belhadji
# 05/12/2025

import numpy as np
import torch
import copy
from tqdm import tqdm
from typing import Optional


def recursive_weighted_average_alpha_v(
        y: torch.Tensor,
        alpha: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        log_v: Optional[torch.Tensor] = None,
        eps: float = 1e-18
):
    r"""
    Compute a stable weighted average $\sum v_i a_i y_i / \sum v_i a_i$ using log-weights
    y: (N, d)   the containing the vectors $y_i$
    alpha: (N,) the array of arbitrary weights
    v: (N,) or log_v: (N,) the array of postive weights
    """
    N, d = y.shape
    if alpha.ndim != 1 or N != alpha.shape[0]:
        raise ValueError(f"Invalid alpha dimensions {alpha.shape}")

    y = torch.as_tensor(y)
    alpha = torch.as_tensor(alpha)
    device = y.device
    dtype = y.dtype
    N, d = y.shape

    # Check the 'mode' of weighting:
    # 1) v is given
    # 2) log_v is given
    # 3) v nor log_v is given

    if v is not None:
        v = torch.as_tensor(v, device=device, dtype=dtype)
        if (v <= 0).any():
            raise ValueError("v must be positive.")
        log_v = torch.log(v)
    elif log_v is not None:
        log_v = torch.as_tensor(log_v, device=device, dtype=dtype)
    else:
        v = torch.ones_like(alpha)
        log_v = torch.zeros_like(alpha)
        # raise ValueError("Either v or log_v must be provided.")
    if not (v is None or (v.ndim == 1 and v.shape[0] == N)) or \
            not (log_v.ndim == 1 and log_v.shape[0] == N):
        raise ValueError("Invalid dimensions")

    # Look for the non-vanishing a_i != 0, and restrict the average to those
    nonzero_alpha_mask = (alpha != 0)
    if nonzero_alpha_mask.sum() == 0:
        raise ValueError("All alpha are zero.")
    y = y[nonzero_alpha_mask]
    alpha = alpha[nonzero_alpha_mask]
    log_v = log_v[nonzero_alpha_mask]

    # Compute log |w_i|:= log |a_i| + log |v_i|  and sign of the a_i

    log_abs_alpha = torch.log(alpha.abs())
    logw = log_abs_alpha + log_v
    sign = alpha.sign()

    # Look for max log |w_i|
    z_max = logw.max()

    # Calculate stable weights
    exp_scaled = torch.exp(logw - z_max)

    # Calculate the denominator
    weighted_signs = sign * exp_scaled
    denominator = weighted_signs.sum()

    if denominator.abs() < eps:
        raise ValueError("Sum of weights too close to zero.")

    # Calculate the numerator
    numerator = (weighted_signs.unsqueeze(1) * y).sum(dim=0)

    # Calculate the ratio
    weighted_average = numerator / denominator

    return weighted_average


def msip_map(
        objective_function,
        particles: torch.Tensor,
        kernel_bandwidth: float = 1.0,
        bandwidth_factor: float = 0.5,
        bounds: tuple[float, float] = (-torch.inf, torch.inf),
        projection: bool = True,
        gradient_informed: bool = True
):
    """
    Compute the full MSIP map T(y) for all particles at once.
    Returns t_arr with shape (N, d).
    """
    # Make sure this is a leaf with grad

    lower_bounds = torch.as_tensor(
        bounds[0], dtype=particles[0].dtype,
        device=particles[0].device
    )

    upper_bounds = torch.as_tensor(
        bounds[1], dtype=particles[0].dtype,
        device=particles[0].device
    )

    particles_leaf = particles.detach().clone()
    particles_leaf.requires_grad_(True)

    fitness = objective_function(particles_leaf)   # shape (N,)
    grads = None
    if gradient_informed:
        grads, = torch.autograd.grad(fitness.sum(), particles_leaf)

    with torch.no_grad():
        diff = particles_leaf.unsqueeze(1) - \
            particles_leaf.unsqueeze(0)  # (N, N, d)

        sigma2 = kernel_bandwidth ** 2
        # (N, N)
        kernel_matrix = torch.exp(- (diff ** 2).sum(dim=-1) / sigma2)

        K_minus_one = torch.linalg.inv(kernel_matrix)

        N, d = particles_leaf.shape
        t_list = []
        for i in range(N):
            alpha_i = K_minus_one[i, :]  # (N,)

            t1 = recursive_weighted_average_alpha_v(
                particles, alpha_i, log_v=fitness
            )
            if gradient_informed:
                t2 = recursive_weighted_average_alpha_v(
                    grads, alpha_i, log_v=fitness
                )
                if projection:
                    t_list.append(torch.clamp(
                        t1 + bandwidth_factor * bandwidth_factor * sigma2 * t2,
                        min=lower_bounds, max=upper_bounds
                    ))
                else:
                    t_list.append(
                        t1 + bandwidth_factor *
                        bandwidth_factor * sigma2 * t2
                    )
            else:
                if projection:
                    t_list.append(torch.clamp(
                        t1, min=lower_bounds, max=upper_bounds
                    ))
                else:
                    t_list.append(t1)
        t_arr = torch.stack(t_list, dim=0)  # (N, d)

    return t_arr


def update_one_particle(
        objective_function,
        particles: torch.Tensor,
        idx: int,
        lr: float = 0.1,
        kernel_bandwidth: float = 1.0,
        bandwidth_factor: float = 0.5,
        inner_tol: float = 1e-4,
        max_inner_steps: int = 50,
        bounds: tuple[float, float] = (-torch.inf, torch.inf),
        projection: bool = True,
        gradient_informed: bool = True
):
    """
    Coordinate-wise MSIP update:
    - All particles are kept fixed except particle `idx`
    - For particle `idx`, we iterate until the MSIP update is small
      or max_inner_steps is reached.
    Mutates `particles` in-place and returns it.
    """

    new_list_particles = [particles.detach().cpu().numpy()]
    for _ in range(max_inner_steps):
        # Compute full MSIP map given current particles

        t_arr = msip_map(
            objective_function, particles,
            kernel_bandwidth=kernel_bandwidth, bounds=bounds,
            projection=projection, gradient_informed=gradient_informed
        )

        with torch.no_grad():
            old_pos = particles[idx].clone()
            new_pos = (1.0 - lr) * old_pos + lr * t_arr[idx]

            move_norm = (new_pos - old_pos).norm()
            particles[idx].copy_(new_pos)
            new_list_particles.append(
                particles.detach().cpu().numpy().copy()
            )

        if move_norm.isnan():
            print('nan')

        if move_norm.item() < inner_tol:
            break

    return new_list_particles


def msip_greedy(
    objective_function,
    n_particles: int = 50,
    # now interpreted as "epochs" (passes over all particles)
    n_steps: int = 10,
    dim: int = 2,
    bounds: tuple[float, float] = (-5, 5),
    gradient_informed: bool = True,
    projection: bool = True,
    lr: float = 0.1,
    noise: float = 0.05,          # currently unused, kept for compatibility
    kernel_bandwidth: float = 1.0,
    bandwidth_factor: float = 0.5,
    inner_tol: float = 1e-4,      # equilibrium tolerance for a particle
    max_inner_steps: int = 50,  # max inner iterations per particle
    seed: Optional[int] = None,
    device: str = "cpu",
    init_particles: Optional[torch.Tensor | np.ndarray] = None
):

    if seed is not None:
        torch.manual_seed(seed)

    if init_particles is None:
        # Init particles
        particles = (bounds[1] - bounds[0]) * \
            torch.rand((n_particles, dim), device=device) + bounds[0]
    elif isinstance(init_particles, np.ndarray):
        particles = torch.tensor(init_particles, device=device)
    else:
        particles = init_particles.clone()

    trajectories = [particles.detach().cpu().numpy().copy()]

    # Outer loop: epochs
    for _ in tqdm(range(n_steps)):
        # Loop over particles, one at a time
        for i in range(n_particles):
            new_list_particles = update_one_particle(
                objective_function,
                particles,
                idx=i,
                lr=lr,
                kernel_bandwidth=kernel_bandwidth,
                bandwidth_factor=bandwidth_factor,
                inner_tol=inner_tol,
                max_inner_steps=max_inner_steps, bounds=bounds, projection=projection,
                gradient_informed=gradient_informed
            )
            # If you want a very fine-grained trajectory, record after each particle:
            # trajectories.append(particles.detach().cpu().numpy().copy())
            trajectories = trajectories + new_list_particles

        # If you prefer only one snapshot per epoch, move the append here instead:
        # trajectories.append(particles.detach().cpu().numpy().copy())

    return np.array(trajectories), bounds
