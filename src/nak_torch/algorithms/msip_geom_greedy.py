#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles_greedy
# Ayoub Belhadji
# 05/12/2025

import numpy as np
import torch
import copy


def recursive_weighted_average_alpha_v(y, alpha, v=None, log_v=None, eps=1e-18):
    r"""
    Compute a stable weighted average \sum v_i a_i y_i / \sum v_i a_i using log-weights
    y: (N, d)   the containing the vectors y_i
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
    # 3) neither v nor log_v is given
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


def msip_map(objective_function, particles, kernel_bandwidth=1.0):
    """
    Compute the full MSIP map T(y) for all particles at once.
    Returns t_arr with shape (N, d).
    """
    # Make sure this is a leaf with grad
    particles_leaf = particles.detach().clone()
    particles_leaf.requires_grad_(True)

    fitness = objective_function(particles_leaf)   # shape (N,)
    grads, = torch.autograd.grad(fitness.sum(), particles_leaf)
    f_times_y = fitness.unsqueeze(-1) * particles_leaf

    with torch.no_grad():
        diff = particles_leaf.unsqueeze(1) - particles_leaf.unsqueeze(0)  # (N, N, d)
        sigma2 = kernel_bandwidth ** 2
        kernel_matrix = torch.exp(- (diff ** 2).sum(dim=-1) / sigma2)     # (N, N)

        K_minus_one = torch.linalg.inv(kernel_matrix)

        N, d = particles_leaf.shape
        t_list = []
        for i in range(N):
            alpha_i = K_minus_one[i, :]  # (N,)

            t1_1 = recursive_weighted_average_alpha_v(f_times_y, alpha_i)
            t2_1 = recursive_weighted_average_alpha_v(grads, alpha_i)
            t2_2 = recursive_weighted_average_alpha_v(fitness.unsqueeze(-1), alpha_i)

            t2 = t2_1 / t2_2
            t1 = t1_1 / t2_2

            t_list.append(t1 + sigma2 * t2)

        t_arr = torch.stack(t_list, dim=0)  # (N, d)

    return t_arr


def _geometric_safe_step(
    particles,
    idx,
    target,
    base_lr,
    lr_max,
    min_separation,
    min_lr=1e-6,
    shrink_factor=0.5,
):
    """
    Purely geometric line-search on the segment from x_i to target.
    We look for the largest step lambda in (0, min(base_lr, lr_max)] such that
    the new point is at least 'min_separation' away from all other particles.
    No objective or gradient evaluations are used.
    """
    with torch.no_grad():
        x_i = particles[idx]
        direction = target - x_i
        dir_norm = direction.norm()

        # If direction is (numerically) zero, no move
        if dir_norm < 1e-12:
            return x_i.clone(), 0.0, False

        # Normalize direction once; we will scale by effective step.
        # However, to stay consistent with your convex combination form
        # we interpret "step" as the convex weight in [0,1].
        direction_unit = direction / dir_norm

        # Effective step is in [0, min(base_lr, lr_max)]
        step = float(min(base_lr, lr_max))
        min_sep_sq = float(min_separation ** 2)

        # Build the set of "other" particles
        if particles.shape[0] > 1:
            others = torch.cat([particles[:idx], particles[idx+1:]], dim=0)
        else:
            # Only one particle: trivially safe
            new_pos = x_i + step * direction
            return new_pos, step, True

        moved = False
        while step >= min_lr:
            cand = x_i + step * direction  # keep your convex formulation
            diff = cand.unsqueeze(0) - others
            dist_sq = (diff ** 2).sum(dim=1)
            if (dist_sq >= min_sep_sq).all():
                # Geometrically safe endpoint
                moved = True
                return cand, step, moved
            # Otherwise shrink the step
            step *= shrink_factor

        # Could not find a safe step above min_lr -> skip move
        return x_i.clone(), 0.0, False


def update_one_particle(
    objective_function,
    particles,
    idx,
    lr=0.1,
    kernel_bandwidth=1.0,
    inner_tol=1e-4,
    max_inner_steps=50,
    min_separation=0.2,
    lr_max=0.5,
    min_lr=1e-6,
    shrink_factor=0.5,
):
    """
    Coordinate-wise MSIP update with geometric safety:
    - All particles are kept fixed except particle `idx`.
    - For particle `idx`, we iterate until the MSIP update is small (equilibrium)
      or max_inner_steps is reached.
    - The step size along the MSIP direction is adapted geometrically to ensure
      the updated particle remains at least `min_separation` away from all others.
      If no such step >= min_lr exists, we skip the update for this particle.
    - The effective step is also capped by `lr_max`.

    Returns a list of snapshots of the particle system.
    """
    new_list_particles = [copy.deepcopy(particles)]

    for _ in range(max_inner_steps):
        # Compute full MSIP map given current particles
        t_arr = msip_map(objective_function, particles, kernel_bandwidth=kernel_bandwidth)

        with torch.no_grad():
            old_pos = particles[idx].clone()
            target = t_arr[idx]

            # Purely geometric adaptive step:
            new_pos, eff_lr, moved = _geometric_safe_step(
                particles=particles,
                idx=idx,
                target=target,
                base_lr=lr,
                lr_max=lr_max,
                min_separation=min_separation,
                min_lr=min_lr,
                shrink_factor=shrink_factor,
            )

            move_norm = (new_pos - old_pos).norm()

            if moved:
                particles[idx].copy_(new_pos)
                new_list_particles.append(copy.deepcopy(particles.detach().cpu().numpy()))
            else:
                # Could not move without violating separation; consider this an equilibrium
                print(f"Particle {idx}: geometric blocking after {_} inner steps.")
                break

        if move_norm.item() < inner_tol:
            print(f"Particle {idx}: inner convergence at step {_}.")
            break

    return new_list_particles


def msip_geom_greedy(
    objective_function,
    n_particles=50,
    n_steps=10,          # now interpreted as "epochs" (passes over all particles)
    dim=2,
    bounds=(-5, 5),
    lr=0.1,
    noise=0.05,          # currently unused, kept for compatibility
    kernel_bandwidth=1.0,
    inner_tol=1e-4,      # equilibrium tolerance for a particle
    max_inner_steps=50,  # max inner iterations per particle
    min_separation=0.2,  # NEW: minimal allowed distance between particles
    lr_max=0.5,          # NEW: global cap on effective learning rate
    min_lr=1e-6,
    shrink_factor=0.5,
    seed=None,
    device="cpu",
):
    if seed is not None:
        torch.manual_seed(seed)

    # Init particles
    particles = (bounds[1] - bounds[0]) * torch.rand((n_particles, dim), device=device) + 1.5 * bounds[0]
    trajectories = [particles.detach().cpu().numpy().copy()]

    # Outer loop: epochs
    for _ in range(n_steps):
        # Loop over particles, one at a time
        for i in range(n_particles):
            new_list_particles = update_one_particle(
                objective_function,
                particles,
                idx=i,
                lr=lr,
                kernel_bandwidth=kernel_bandwidth,
                inner_tol=inner_tol,
                max_inner_steps=max_inner_steps,
                min_separation=min_separation,
                lr_max=lr_max,
                min_lr=min_lr,
                shrink_factor=shrink_factor,
            )
            trajectories = trajectories + new_list_particles

    return np.array(trajectories), bounds

