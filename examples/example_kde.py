#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from tools import animate_trajectories_box
from functions import mixture_of_gaussians
from functions import himmelblau
from functions import gaussian_kde

from algorithms import msip
from algorithms import msip_ni
from algorithms import msip_greedy


from datetime import datetime






if __name__ == "__main__":
    save_gif = True
    now_stamp = datetime.now()
    algorithm_name = "msip_greedy" # Other choices: msip, msip_gs, msip_ni
    function_name = "gaussian_kde" # Other choices: mixture_of_gaussians, himmelblau
    
    
    
    sigma = 0.3
    npz_file = np.load('datasets/two_bananas.npz')
    dataset = 10*npz_file['X'].T - np.mean(10*npz_file['X'].T)
    objective_function = gaussian_kde(sigma,dataset)



    
    trajectories, _ = msip_greedy(objective_function, 
             n_particles=25,
             n_steps=40,          # "epochs" (passes over all particles)
             dim=2,
             bounds=(-4, 4),   # [a,b]^d
             gradient_informed = True,
             projection = True,
             lr=0.8,
             noise=0.05,          # currently unused
             kernel_bandwidth=0.3,
             bandwidth_factor = 0.001,
             inner_tol=1e-6,      # equilibrium tolerance for a particle
             max_inner_steps=5,  # max inner iterations per particle
             seed=0,
             device="cpu")
    

    
    #print(trajectories)
    if save_gif:
        bounds = [-4,4]
        animate_trajectories_box(objective_function, trajectories, 50, bounds, save_path="results/gif/"+algorithm_name+"_"+function_name+"_particles_"+str(now_stamp)+".gif")

