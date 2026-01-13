#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains a typical 2D mixture of Gaussians with isotropic
# covariance matrices
# Ayoub Belhadji
# 05/12/2025


import torch



def mixture_of_gaussians(x):
    centers = torch.tensor([
        [-7, -7],
        [4, -4],
        [-4, 4],
        [4, 4],
        [0, 0]
    ], dtype=x.dtype, device=x.device)
    sigmas = 1.2 * torch.ones(5, device=x.device)
    weights = torch.tensor([0.4, 0.2, 0.2, 0.1, 0.1], dtype=x.dtype, device=x.device)  # Tuned weights

    densities = torch.stack([
        w * torch.exp(-((x - c)**2).sum(dim=-1) / (2 * s**2)) for c, s, w in zip(centers, sigmas, weights)
    ], dim=-1)
    return torch.log(densities.sum(dim=-1))



