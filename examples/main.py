#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nak_torch.tools import animate_trajectories_box
from nak_torch.functions import himmelblau
from nak_torch.algorithms import msip_greedy
from datetime import datetime

if __name__ == "__main__":
    save_gif = True
    now_stamp = datetime.now()
    algorithm_name = "msip_ni"
    function_name = "himmelblau"
    objective_function = himmelblau(50.0)

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

    if save_gif:
        bounds = [-15,15]
        animate_trajectories_box(objective_function, trajectories, 50, bounds, save_path="results/gif/"+algorithm_name+"_"+function_name+"_particles_"+str(now_stamp)+".gif")

