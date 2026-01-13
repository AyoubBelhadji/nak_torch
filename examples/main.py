#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from tools import animate_trajectories_box
from functions import mixture_of_gaussians
from functions import himmelblau

from algorithms import msip
from algorithms import msip_ni
from algorithms import msip_greedy

from datetime import datetime






# Define the density 




if __name__ == "__main__":
    save_gif = True
    now_stamp = datetime.now()
    #algorithm_name = "msip_ni"
    algorithm_name = "msip_ni"
    #function_name = "mixture_of_gaussians"
    #objective_function = mixture_of_gaussians
    
    function_name = "himmelblau"
    objective_function = himmelblau(50.0)

    #trajectories, bounds = msip_greedy(objective_function,
    #    n_particles=5, n_steps=5000, dim=2, lr=0.2, noise_level_0=0.001,kernel_bandwidth = 1.2, seed=2, device="cpu"
    #)
    
    
    # trajectories, bounds = msip_greedy(objective_function, 
    #          n_particles=10,
    #          n_steps=1,          # now interpreted as "epochs" (passes over all particles)
    #          dim=2,
    #          bounds=(-5, 5),
    #          lr=0.15,
    #          noise=0.05,          # currently unused, kept for compatibility
    #          kernel_bandwidth=0.7,
    #          inner_tol=1e-3,      # equilibrium tolerance for a particle
    #          max_inner_steps=1000,  # max inner iterations per particle
    #          seed=None,
    #          device="cpu")
    
    trajectories, bounds = msip_greedy(objective_function, 
             n_particles=10,
             n_steps=50,          # now interpreted as "epochs" (passes over all particles)
             dim=2,
             bounds=(-8, 8),
             projection = True,
             lr=0.25,
             noise=0.05,          # currently unused, kept for compatibility
             kernel_bandwidth=0.9,
             inner_tol=1e-4,      # equilibrium tolerance for a particle
             max_inner_steps=1000,  # max inner iterations per particle
             seed=None,
             device="cpu")
    
    # trajectories, bounds = msip_ni(objective_function, 
    #          n_particles=10,
    #          n_steps=1000,          # now interpreted as "epochs" (passes over all particles)
    #          dim=2,
    #          bounds=(-5, 5),
    #          lr=0.15,
    #          noise_level_0=1,          # currently unused, kept for compatibility
    #          kernel_bandwidth=0.45,
    #          seed=None,
    #          device="cpu")
    
    #print(trajectories)
    if save_gif:
        bounds = [-15,15]
        animate_trajectories_box(objective_function, trajectories, 50, bounds, save_path="results/gif/"+algorithm_name+"_"+function_name+"_particles_"+str(now_stamp)+".gif")

