#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of consensus-based optimization (CBO)
# TBD
# Ayoub Belhadji
# 05/12/2025

import torch



# CBO optimization (to be tested later)
def update_particles_cbo(objective_function, particles, lr=0.1, noise=0.05):
    fitness = objective_function(particles)
    best_idx = torch.argmin(fitness)
    best_particle = particles[best_idx]

    noise_term = noise * torch.randn_like(particles)
    particles += lr * (best_particle - particles) + noise_term
    return particles



