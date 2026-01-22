#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import numpy as np
import torch

from nak_torch.tools.util import initialize_particles
from .msip_map import msip_map
from typing import Optional
from tqdm import tqdm


def msip(
    log_density,
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

    progress_bar = tqdm(range(n_particles*n_steps))

    for idx in range(n_steps):
        particles_diff = msip_map(
            log_density,
            particles,
            output_idx=None,
            progress_bar=progress_bar,
            bounds=bounds,
            **msip_kwargs
        )

        with torch.no_grad():
            particles = (1.0 - lr) * particles + lr * particles_diff
        if keep_all:
            trajectories[idx+1].copy_(particles)

    return trajectories.detach_() if keep_all else particles.unsqueeze_(0)
