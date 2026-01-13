#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles_greedy
# Ayoub Belhadji
# 05/12/2025

import numpy as np
import torch
import copy
from tqdm import tqdm


def recursive_weighted_average_alpha_v(
        y, alpha, v=None, log_v=None, eps=1e-18
):
    r"""
    Compute a stable weighted average $\sum v_i a_i y_i / \sum v_i a_i$ using log-weights
    y: (N, d)   the containing the vectors $y_i$
    alpha: (N,) the array of arbitrary weights
    v: (N,) or log_v: (N,) the array of postive weights
    """
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
        objective_function, particles,
        kernel_bandwidth=1.0, bandwidth_factor=0.5,
        bounds=[0], projection=True,
        gradient_informed=True
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
    # print(particles)
    particles_leaf.requires_grad_(True)

    fitness = objective_function(particles_leaf)   # shape (N,)
    if gradient_informed:
        grads, = torch.autograd.grad(fitness.sum(), particles_leaf)
    # f_times_y = fitness.unsqueeze(-1) * particles_leaf

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


def update_one_particle(objective_function,
                        particles,
                        idx,
                        lr=0.1,
                        kernel_bandwidth=1.0,
                        bandwidth_factor=0.5,
                        inner_tol=1e-4,
                        max_inner_steps=50,
                        bounds=[0],
                        projection=True,
                        gradient_informed=True):
    """
    Coordinate-wise MSIP update:
    - All particles are kept fixed except particle `idx`
    - For particle `idx`, we iterate until the MSIP update is small
      or max_inner_steps is reached.
    Mutates `particles` in-place and returns it.
    """

    new_list_particles = [copy.deepcopy(particles)]
    for _ in range(max_inner_steps):
        # Compute full MSIP map given current particles

        t_arr = msip_map(objective_function, particles,
                         kernel_bandwidth=kernel_bandwidth, bounds=bounds,
                         projection=projection, gradient_informed=gradient_informed)

        with torch.no_grad():
            old_pos = particles[idx].clone()
            new_pos = (1.0 - lr) * old_pos + lr * t_arr[idx]

            move_norm = (new_pos - old_pos).norm()
            particles[idx].copy_(new_pos)
            new_list_particles.append(copy.deepcopy(
                particles.detach().cpu().numpy()))

        if move_norm.isnan():
            print('nan')
        if move_norm.item() < inner_tol:
            break

    return new_list_particles


def msip_greedy(
    objective_function,
    n_particles=50,
    # now interpreted as "epochs" (passes over all particles)
    n_steps=10,
    dim=2,
    bounds=[-5, 5],
    gradient_informed=True,
    projection=True,
    lr=0.1,
    noise=0.05,          # currently unused, kept for compatibility
    kernel_bandwidth=1.0,
    bandwidth_factor=0.5,
    inner_tol=1e-4,      # equilibrium tolerance for a particle
    max_inner_steps=50,  # max inner iterations per particle
    seed=None,
    device="cpu"
):

    if seed is not None:
        torch.manual_seed(seed)

    # Init particles
    particles = (bounds[1] - bounds[0]) * \
        torch.rand((n_particles, dim), device=device) + bounds[0]
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
