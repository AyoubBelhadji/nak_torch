 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
#from torch.nn.utils import vector_to_parameters
import matplotlib.pyplot as plt
from viz_tools import animate_trajectories_box
from functions import loss_nn_dataset
#from nak_torch.algorithms import msip_greedy

from nak_torch.algorithms import grad_aldi, eks, gradfree_aldi, cbs, msip, kfrflow
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientInformed, MSIPQuadGradientFree
from torch import nn
import random
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from datetime import datetime
import torch.nn.functional as F



#import numpy as np
#import torch
#import torch.nn.functional as F
from torch.func import functional_call
from collections import OrderedDict

from typing import Dict, List, Tuple
from ucimlrepo import fetch_ucirepo




# Neural network model
#, prior_std: float

class bnn(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int):
        super().__init__()
        self.d_in = d_in
        self.hidden_dim = hidden_dim
        #self.prior_std = prior_std
        
        self.fc1 = nn.Linear(d_in,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)
        #self.log_noise = nn.Parameter(torch.tensor(0.0))
        
        
    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        # xb: [B, d_in]-> mu: [B]
        h = F.relu(self.fc1(xb))
        mu = self.fc2(h).squeeze(-1)
        
        return mu
        


class sigma_pi(nn.Module):
    def __init__(self,d,M,N,C,activation):
        super().__init__()
        self.vs = nn.Parameter(torch.randn(d,M,N,C) / math.sqrt(d))
        self.alphas = nn.Parameter(torch.randn(M,N,C))
        self.betas = nn.Parameter(torch.randn(M,N,C))
        self.bias = nn.Parameter(torch.zeros(M,N,C))
        self.ws = nn.Parameter(torch.randn(N,C))
        self.activation = activation
        self.d = d
        self.C = C
        self.M = M
        self.N = N
    def forward(self, xb):
        if self.activation == 'ReLU':
            nn_eval = torch.zeros(xb.shape[0],self.C)
            #print(np.linalg.norm(self.ws.detach().numpy()))
            for c in list(range(self.C)):
                for n in list(range(self.N)):
                    tmp_var = 1
                    for m in list(range(self.M)):
                        #print("xb.shape         =", xb.shape)
                        #print("vs_slice.shape   =", self.vs[:, m, n, c].shape)
                        #print("bias.shape   =", self.bias.shape)
                        tmp_var = tmp_var*(self.alphas[m,n,c]+ self.betas[m,n,c]*nn.ReLU()(xb @ self.vs[:,m,n,c] + self.bias[m,n,c]))
                    nn_eval[:,c] += self.ws[n,c]*tmp_var
        elif self.activation == 'Tanh':
            nn_eval = torch.zeros(xb.shape[0],self.C)
            #print(np.linalg.norm(self.ws.detach().numpy()))
            for c in list(range(self.C)):
                for n in list(range(self.N)):
                    tmp_var = 1
                    for m in list(range(self.M)):
                        #print("xb.shape         =", xb.shape)
                        #print("vs_slice.shape   =", self.vs[:, m, n, c].shape)
                        #print("bias.shape   =", self.bias.shape)
                        tmp_var = tmp_var*(self.alphas[m,n,c]+ self.betas[m,n,c]*nn.Tanh()(xb @ self.vs[:,m,n,c] + self.bias[m,n,c]))
                    nn_eval[:,c] += self.ws[n,c]*tmp_var
        return nn_eval



# Dataset loading

def load_uci_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a few standard UCI regression datasets via ucimlrepo.

    Supported names:
        - boston
        - concrete
        - energy
        - yacht
        - wine
    """

    name = name.lower()
    
    if name == "boston":
        ds = fetch_ucirepo(id=531)
        X = ds.data.features.to_numpy(dtype=np.float64)
        X = np.delete(X, 11, axis=1)
        y = ds.data.targets.to_numpy(dtype=np.float64).reshape(-1)


    if name == "concrete":
        # Concrete Compressive Strength
        ds = fetch_ucirepo(id=165)
        X = ds.data.features.to_numpy(dtype=np.float64)
        y = ds.data.targets.to_numpy(dtype=np.float64).reshape(-1)

    elif name == "energy":
        # Energy Efficiency
        ds = fetch_ucirepo(id=242)
        X = ds.data.features.to_numpy(dtype=np.float64)
        y_all = ds.data.targets.to_numpy(dtype=np.float64)
        # Use heating load as target (common choice)
        y = y_all[:, 0].reshape(-1)

    elif name == "yacht":
        # Yacht Hydrodynamics
        ds = fetch_ucirepo(id=243)
        X = ds.data.features.to_numpy(dtype=np.float64)
        y = ds.data.targets.to_numpy(dtype=np.float64).reshape(-1)

    elif name == "wine":
        # Wine Quality (red wine)
        ds = fetch_ucirepo(id=186)
        X = ds.data.features.to_numpy(dtype=np.float64)
        y = ds.data.targets.to_numpy(dtype=np.float64).reshape(-1)

    else:
        raise ValueError(
            f"Unsupported dataset '{name}'. "
            "Use one of: boston, concrete, energy, yacht, wine."
        )

    # Remove rows with missing values if any
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    return torch.from_numpy(X), torch.from_numpy(y)




def loss_nn_dataset(dataset_name, nn_params, beta=1.0, lambda2=0.01, device="cpu"):
    
    # Define the loss
    #loss_func = F.soft_margin_loss
    loss_func = F.mse_loss
    log_prior_lambda = lambda2
    
    # Load dataset
    X,Y = load_uci_dataset(dataset_name)
    #X = (X - X.mean(0)) / X.std(0)
    #Y = (Y - Y.mean()) / Y.std()
    
    d = X.shape[1]
    N = X.shape[0]
    nn_params['d_in']=d
    print(d)
    print(N)
    # Call the neural network 
    model = bnn(**nn_params).to(device)

    # Save parameter metadata for unpacking theta
    param_info = []
    for name, p in model.named_parameters():
        param_info.append((name, p.shape, p.numel()))

    buffer_dict = OrderedDict(model.named_buffers())

    total_numel = sum(numel for _, _, numel in param_info)

    def theta_to_param_dict(theta_1d: torch.Tensor):
        """Convert flat theta of shape (d,) into an OrderedDict of named parameters."""
        if theta_1d.ndim != 1:
            raise ValueError(f"theta_1d must be 1D, got shape {theta_1d.shape}")
        if theta_1d.numel() != total_numel:
            raise ValueError(
                f"theta_1d has {theta_1d.numel()} entries, expected {total_numel}"
            )

        out = OrderedDict()
        idx = 0
        for name, shape, numel in param_info:
            out[name] = theta_1d[idx:idx + numel].view(shape)
            idx += numel
        return out

    def loss_for_single_theta(theta_1d: torch.Tensor) -> torch.Tensor:
        """theta_1d: (d,) -> scalar loss"""
        param_dict = theta_to_param_dict(theta_1d)

        pred = functional_call(model, (param_dict, buffer_dict), (X,)).squeeze(-1)
        data_loss = -loss_func(pred, Y)*N
        reg = -log_prior_lambda * (theta_1d ** 2).sum()
        return (data_loss + reg) / beta

    def objective_function(theta: torch.Tensor):
        """
        theta:
          - shape (d,)   -> scalar tensor
          - shape (N,d)  -> tensor of shape (N,)
        """
        if theta.ndim == 1:
            return loss_for_single_theta(theta)
        # elif theta.ndim == 2:
        #     return torch.vmap(loss_for_single_theta)(theta)
        elif theta.ndim == 2:
            return torch.stack(
                [loss_for_single_theta(theta[i]) for i in range(theta.shape[0])]
            )
        else:
            raise ValueError(f"theta must be 1D or 2D, got shape {theta.shape}")

    return objective_function




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



def nn_param_counter(model):
    count = 0
    for name, p in model.named_parameters():
        count = count + p.numel()
    return count

if __name__ == "__main__":
    save_gif = False
    now_stamp = datetime.now()
    #algorithm_name = "msip_ni"
    algorithm_name = "msip"
    #function_name = "mixture_of_gaussians"
    #objective_function = mixture_of_gaussians

    function_name = "loss_nn_dataset"
    
    # nn_params = {
    #     'd': 1,
    #     'M': 2,
    #     'N': 100,
    #     'C': 1,
    #     'activation': 'ReLU'
    #     }
    
    nn_params = {
        'd_in': 1,
        'hidden_dim': 200,
        }
    
    
    dataset_name = 'concrete'
    objective_function = loss_nn_dataset(dataset_name,nn_params, beta = 1.0, lambda2=0.0, device="cpu")

    
    

    #post_log_dens_batch = torch.vmap(objective_function)
    post_log_dens_grad_val = torch.func.grad_and_value(objective_function)
    model = bnn(**nn_params)
    
    n_particles = 30
    dimension = nn_param_counter(model)
    n_steps = 1000
    
    init_particles = torch.randn((n_particles, dimension)) 

    kernel_length_scale = 2.5
    bounds = (-100., 100.)
    gradient_decay = 1.0
    lr_msip = 1e-3
    kernel_diag_infl = 1e-8
    msip_fredholm = MSIPFredholm(
        gradient_decay,
        post_log_dens_grad_val
    )
    
    trajectories_msip, traj_wts_msip = msip(
        objective_function, n_particles, n_steps, dim=dimension,
        lr=lr_msip, init_particles=init_particles,
        kernel_length_scale=kernel_length_scale,
        is_log_density_batched=True,
        kernel_diag_infl=kernel_diag_infl,
        bounds=bounds,
        gradient_decay=gradient_decay,
        keep_all=True,
        compile_step=False,
        verbose=True
    )



    eval_tensor = eval_function_trajectories(objective_function,trajectories_msip)
    plot_eval_tensor(torch.log(eval_tensor))
    plot_eval_best_tensor(torch.log(eval_tensor))
    T,_,_ = trajectories_msip.shape

    for m in range(n_particles):
        t_m = np.argmin(eval_tensor[:,m].detach().numpy())

        #plot_2D_classification_tensor(trajectories_msip,'two_bananas',m,t_m)


    if save_gif:
        fpath = f"results/gif/{algorithm_name}_{function_name}_particles_{now_stamp}.gif"
        animate_trajectories_box(
            objective_function,
            trajectories_msip, 50, bounds,
            save_path=fpath
        )

