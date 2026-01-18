#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import numpy as np
import torch
from .msip_map import msip_map
from typing import Optional
from tqdm import tqdm

def msip(objective_function,
    n_particles=50,
    n_steps=100,
    dim=2,
    lr=0.1,
    noise=0.05,
    seed=None,
    device="cpu",
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    keep_all: bool = True,
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
        particles = torch.as_tensor(init_particles, device=device)
    if keep_all:
        trajectories = [particles.detach().cpu().numpy().copy()]
    else:
        trajectories = []

    progress_bar = tqdm(range(n_particles*n_steps))

    for _ in range(n_steps):
        particles_diff = msip_map(
            objective_function,
            particles,
            output_idx=None,
            progress_bar=progress_bar,
            **msip_kwargs
        )

        with torch.no_grad():
            particles = (1.0 - lr) * particles + lr * particles_diff
        if keep_all:
            trajectories.append(particles.detach().cpu().numpy().copy())
    if keep_all:
        return np.array(trajectories), bounds
    else:
        return particles.unsqueeze_(0), bounds
