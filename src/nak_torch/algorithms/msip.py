#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import numpy as np
import torch
from .msip_map import msip_map
from typing import Optional

def msip(objective_function,
    n_particles=50,
    n_steps=100,
    dim=2,
    lr=0.1,
    noise=0.05,
    seed=None,
    device="cpu",
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    # kernel_bandwidth = 1.0,
    # bandwidth_factor = 0.5,
    # bounds=(-5, 5),
    # projection = True,
    # gradient_informed = True,
    **msip_kwargs
):
    if seed is not None:
        torch.manual_seed(seed)
    bounds = msip_kwargs.get("bounds", None)
    if init_particles is None:
        if bounds is None:
            particles = torch.randn((n_particles, dim), device=device)
        else:
            particles = (bounds[1] - bounds[0]) * torch.rand((n_particles, dim), device=device) + 1.5*bounds[0]
    else:
        particles = torch.tensor(init_particles, device=device)

    trajectories = [particles.detach().cpu().numpy().copy()]

    for _ in range(n_steps):
        particles_diff = msip_map(
            objective_function,
            particles,
            output_idx=None,
            **msip_kwargs
        )

        with torch.no_grad():
            particles = (1.0 - lr) * particles + lr * particles_diff

        trajectories.append(particles.detach().cpu().numpy().copy())

    return np.array(trajectories), bounds
