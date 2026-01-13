#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import numpy as np
import torch



def recursive_weighted_average_alpha_v(y, alpha, v=None, log_v=None, eps=1e-18):
    """
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
        #raise ValueError("Either v or log_v must be provided.")

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




def update_particles_ni(
    objective_function,
    particles,
    t,
    lr=0.1,
    kernel_bandwidth=1.0,
    noise_level=0.01,
    noise_injection=None,   # <-- new argument: a map for noise
):
    # Make sure this is a leaf with grad
    particles = particles.detach().clone()
    particles.requires_grad_(True)

    fitness = objective_function(particles)  # shape (N,)
    # If fitness is vector, you usually do sum() to get grads w.r.t. all particles
    grads, = torch.autograd.grad(fitness.sum(), particles)
    # Be aware that this is the gradient of the log(p), which is the grad of V
    
    expf_times_y = torch.exp(fitness.unsqueeze(-1)) * particles

    # From here on, think of 'fitness' and 'grads' as just arrays
    with torch.no_grad():
        diff = particles.unsqueeze(1) - particles.unsqueeze(0)
        sigma2 = kernel_bandwidth ** 2
        kernel_matrix = torch.exp(- (diff ** 2).sum(dim=-1) / sigma2)
        #print(kernel_matrix)

        K_minus_one = torch.linalg.inv(kernel_matrix)
        #print(K_minus_one)

        N, d = particles.shape
        t_list = []
        for i in range(N):
            alpha_i = K_minus_one[i, :]

            # all these calls now use 'plain' tensors
            # t1 = recursive_weighted_average_alpha_v(
            #     particles, alpha_i, log_v=torch.log(fitness)
            # )

            #t1_1 = recursive_weighted_average_alpha_v(expf_times_y, alpha_i)
            #t2_1 = recursive_weighted_average_alpha_v(grads, alpha_i)
            #t2_2 = recursive_weighted_average_alpha_v(fitness.unsqueeze(-1), alpha_i)
            #t2 = t2_1 / t2_2
            t1 = recursive_weighted_average_alpha_v(particles, alpha_i, log_v=fitness)
            t2 = recursive_weighted_average_alpha_v(grads, alpha_i, log_v=fitness)
            #     
            #t1_1 / t2_2
            #print(t2)
            t_list.append(t1 + sigma2 * t2)

        t_arr = torch.stack(t_list, dim=0)

        # deterministic transport step
        particles_det = (1 - lr) * particles + lr * t_arr

        # optional noise injection map
        if noise_injection is not None:
            # expect: noise_injection(particles, t, noise_level) -> new_particles
            particles_new = noise_injection(particles_det, t, noise_level)
        else:
            particles_new = particles_det

    return particles_new

def gaussian_noise_injection(particles, t, noise_level):
    return particles + noise_level * torch.randn_like(particles)



def msip_ni(objective_function, 
    n_particles=50,
    n_steps=100,
    dim=2,
    bounds=(-5, 5),
    lr=0.1,
    noise_level_0=0.05,
    kernel_bandwidth = 1.0,
    seed=None,
    device="cpu"
):
    if seed is not None:
        torch.manual_seed(seed)

    particles = (bounds[1] - bounds[0]) * torch.rand((n_particles, dim), device=device) + bounds[0]
    trajectories = [particles.detach().cpu().numpy().copy()]

    for t in range(n_steps):
        noise_level = noise_level_0/(t+1)
        particles = update_particles_ni(objective_function, particles, 0,lr,
                                        kernel_bandwidth,
                                        noise_level, gaussian_noise_injection)
        trajectories.append(particles.detach().cpu().numpy().copy())

    return np.array(trajectories), bounds
