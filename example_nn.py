#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from tools import animate_trajectories_box
from functions import mixture_of_gaussians
from functions import himmelblau
from functions import loss_nn_dataset,plot_2D_classification_with_dataset_from_theta

from algorithms import msip
from algorithms import msip_ni
from algorithms import msip_greedy

from datetime import datetime






# Define the density 



def eval_function_trajectories(the_objective_function,the_trajectories):
    T,M,d = the_trajectories.shape
    eval_tensor = torch.zeros((T,M))
    for t in range(T):
        for m in range(M):
            eval_tensor[t,m] = -the_objective_function(the_trajectories[t,m,:])
    return eval_tensor


def plot_eval_tensor(the_eval_tensor):
    fig = plt.figure()
    T,M = the_eval_tensor.shape
    np_eval_tensor = the_eval_tensor.detach().numpy()
    for m in range(M):
        plt.plot(list(range(T)), np_eval_tensor[:,m])
    
    plt.show()
    
def plot_eval_best_tensor(the_eval_tensor):
    fig = plt.figure()
    T,M = the_eval_tensor.shape
    np_eval_tensor = the_eval_tensor.detach().numpy()
    np_eval_min_tensor = np.min(np_eval_tensor,1)
    plt.plot(list(range(T)), np_eval_min_tensor)
    
    plt.show()
    
def plot_2D_classification_tensor(the_trajectories,dataset_name,m,t):
    plot_2D_classification_with_dataset_from_theta(the_trajectories[t,m,:],dataset_name, [-2,2], 50, device="cpu")
    


if __name__ == "__main__":
    save_gif = False
    now_stamp = datetime.now()
    #algorithm_name = "msip_ni"
    algorithm_name = "msip"
    #function_name = "mixture_of_gaussians"
    #objective_function = mixture_of_gaussians
    
    function_name = "loss_nn_dataset"
    objective_function = loss_nn_dataset('two_bananas', beta = 2.0, lambda2=0.01, device="cpu")

    M = 5
    trajectories, bounds = msip_greedy(objective_function, 
             n_particles=M,
             n_steps=300,          # now interpreted as "epochs" (passes over all particles)
             dim=60,
             bounds=(-2.5, 2.5),
             projection = True,
             lr=0.5,
             noise=0.05,          # currently unused, kept for compatibility
             kernel_bandwidth=5.0,
             bandwidth_factor = 0.001,#0.001,
             inner_tol=1e-4,      # equilibrium tolerance for a particle
             max_inner_steps=15,  # max inner iterations per particle
             seed=None,
             device="cpu")
    
    eval_tensor = eval_function_trajectories(objective_function,trajectories)
    plot_eval_tensor(eval_tensor)
    plot_eval_best_tensor(eval_tensor)
    T,_,_ = trajectories.shape
    
    for m in range(M):
        t_m = np.argmin(eval_tensor[:,m].detach().numpy())
    
        plot_2D_classification_tensor(trajectories,'two_bananas',m,t_m)

    
    if save_gif:
        bounds = [-5,5]
        animate_trajectories_box(objective_function, trajectories, bounds, save_path="results/gif/"+algorithm_name+"_"+function_name+"_particles_"+str(now_stamp)+".gif")

