#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles_greedy
# Ayoub Belhadji
# 05/12/2025

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional
from .msip_map import msip_map


def update_one_particle(
        objective_function,
        particles: torch.Tensor,
        idx: int,
        lr: float = 0.1,
        inner_tol: float = 1e-4,
        max_inner_steps: int = 50,
        # kernel_bandwidth: float = 1.0,
        # bandwidth_factor: float = 0.5,
        # bounds: tuple[float, float] = (-torch.inf, torch.inf),
        # projection: bool = True,
        # gradient_informed: bool = True,
        # diag_infl: float = 0.0,
        **msip_kwargs
):
    """
    Coordinate-wise MSIP update:
    - All particles are kept fixed except particle `idx`
    - For particle `idx`, we iterate until the MSIP update is small
      or max_inner_steps is reached.
    Mutates `particles` in-place and returns it.
    """

    new_list_particles = []
    for _ in range(max_inner_steps):
        # Compute full MSIP map given current particles

        t_arr = msip_map(
            objective_function,
            particles,
            # kernel_bandwidth,
            # bandwidth_factor,
            # bounds,
            # projection,
            # gradient_informed,
            # diag_infl,
            output_idx=idx,
            **msip_kwargs
        )

        with torch.no_grad():
            old_pos = particles[idx]
            new_pos = (1.0 - lr) * old_pos + lr * t_arr

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
    lr: float = 0.1,
    noise: float = 0.05,          # currently unused, kept for compatibility
    inner_tol: float = 1e-4,      # equilibrium tolerance for a particle
    seed: Optional[int] = None,
    max_inner_steps: int = 50,  # max inner iterations per particle
    device: str | torch.device = "cpu",
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    **msip_kwargs
    # gradient_informed: bool = True,
    # projection: bool = True,
    # bounds: tuple[float, float] = (-5, 5),
    # kernel_bandwidth: float = 1.0,
    # bandwidth_factor: float = 0.5,
    # diag_infl: float = 0.0
):

    if seed is not None:
        torch.manual_seed(seed)

    bounds = msip_kwargs.get('bounds', None)
    if init_particles is None:
        if bounds is None:
            particles = torch.randn((n_particles, dim), device=device)
        else:
            # Init particles
            particles = (bounds[1] - bounds[0]) * \
                torch.rand((n_particles, dim), device=device) + bounds[0]
    elif isinstance(init_particles, np.ndarray):
        particles = torch.tensor(init_particles, device=device)
    else:
        particles = init_particles.clone()

    trajectories = [particles.detach().cpu().numpy().copy()]

    # Outer loop: epochs
    pbar = tqdm(total=n_steps*n_particles)
    for _ in range(n_steps):
        # Loop over particles, one at a time
        for i in range(n_particles):
            new_list_particles = update_one_particle(
                objective_function,
                particles,
                idx=i,
                lr=lr,
                inner_tol=inner_tol,
                max_inner_steps=max_inner_steps,
                # bounds=bounds,
                # projection=projection,
                # kernel_bandwidth=kernel_bandwidth,
                # bandwidth_factor=bandwidth_factor,
                # gradient_informed=gradient_informed,
                # diag_infl=diag_infl
                **msip_kwargs
            )
            # If you want a very fine-grained trajectory, record after each particle:
            # trajectories.append(particles.detach().cpu().numpy().copy())
            trajectories = trajectories + new_list_particles
            pbar.update()

        # If you prefer only one snapshot per epoch, move the append here instead:
        # trajectories.append(particles.detach().cpu().numpy().copy())

    return torch.tensor(trajectories), bounds
