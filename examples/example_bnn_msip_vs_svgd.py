import torch
import numpy as np
import matplotlib.pyplot as plt
from viz_tools import animate_trajectories_box
from nak_torch.algorithms import grad_aldi, eks, gradfree_aldi, cbs, msip, kfrflow
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientInformed, MSIPQuadGradientFree
from torch import nn
import random
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from datetime import datetime
import torch.nn.functional as F
from torch.func import functional_call
from collections import OrderedDict


# ══════════════════════════════════════════════════════════════════════════════
# BNN model
# ══════════════════════════════════════════════════════════════════════════════



class bnn(nn.Module):
    """
    Standard MLP for binary classification.
    Output is a raw logit — apply sigmoid externally for probabilities.

    Parameters
    ----------
    d_in       : int   input dimension
    hidden_dim : int   width of each hidden layer
    n_layers   : int   number of hidden layers (default 1)
    """
    def __init__(self, d_in: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        layers = []
        in_dim = d_in
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        # xb : (B, d_in)  →  (B,)
        return self.net(xb).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading with train-test splitting 
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_name, train_ratio=0.8, seed=0):
    """
    Loads  datasets/{dataset_name}.npz  (keys X: (d,N), Y: (1,N))
    and splits into train / test.

    Returns X_train, Y_train, X_test, Y_test as torch.double tensors
    with shapes (N, d) and (N,).
    """
    data  = np.load(f"datasets/{dataset_name}.npz")
    X_all = torch.from_numpy(data["X"].T).double()    # (N_total, d)
    Y_all = torch.from_numpy(data["Y"].T).double().squeeze()  # (N_total,)

    N_total = X_all.shape[0]
    rng     = np.random.RandomState(seed)
    idx     = rng.permutation(N_total)
    n_train = int(N_total * train_ratio)

    i_tr, i_te = idx[:n_train], idx[n_train:]
    print(f"  Dataset : {dataset_name}  |  "
          f"train = {len(i_tr)}  test = {len(i_te)}")

    return X_all[i_tr], Y_all[i_tr], X_all[i_te], Y_all[i_te]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — OBJECTIVE FUNCTION FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(X, Y, model_class='bnn',
                   hidden_dim=10, n_layers=1,
                   beta=1.0, lambda2=0.01):
    """
    Returns a log-posterior callable  theta -> scalar  (or (N,) if batched).

    Parameters
    ----------
    X, Y         : torch.double tensors  (N, d) and (N,)
    model_class  : 'sigma_pi'  or  'bnn'
    hidden_dim   : int   hidden width
    n_layers     : int   depth  (bnn only; sigma_pi always uses M=1)
    beta         : float temperature
    lambda2      : float L2 regularisation weight
    """
    d = X.shape[1]

    if model_class == 'bnn':
        model = bnn(d_in=d, hidden_dim=hidden_dim, n_layers=n_layers).double()
    else:
        raise ValueError(f"Unknown model_class: {model_class!r}")

    param_info  = [(name, p.shape, p.numel())
                   for name, p in model.named_parameters()]
    buffer_dict = OrderedDict(model.named_buffers())
    total_numel = sum(nu for _, _, nu in param_info)

    print(f"  Model   : {model_class}  |  "
          f"hidden_dim = {hidden_dim}  n_layers = {n_layers}  |  "
          f"dim(theta) = {total_numel}")

    def theta_to_param_dict(theta_1d):
        out, i = OrderedDict(), 0
        for name, shape, numel in param_info:
            out[name] = theta_1d[i:i + numel].view(shape)
            i += numel
        return out

    def loss_for_single_theta(theta_1d):
        param_dict = theta_to_param_dict(theta_1d)
        pred       = functional_call(model, (param_dict, buffer_dict), (X,))
        pred       = pred.squeeze()
        data_loss  = F.soft_margin_loss(pred, Y)
        reg        = lambda2 * (theta_1d ** 2).sum()
        return -(data_loss + reg) / beta

    def objective_function(theta):
        if theta.ndim == 1:
            return loss_for_single_theta(theta)
        elif theta.ndim == 2:
            return torch.stack([loss_for_single_theta(theta[i])
                                for i in range(theta.shape[0])])
        else:
            raise ValueError(f"theta must be 1D or 2D, got {theta.shape}")

    # attach metadata so callers can read them
    objective_function.total_numel = total_numel
    objective_function.model_class = model_class
    objective_function.model       = model
    objective_function.param_info  = param_info
    objective_function.buffer_dict = buffer_dict

    return objective_function


# ══════════════════════════════════════════════════════════════════════════════
# SVGD
# ══════════════════════════════════════════════════════════════════════════════

def svgd_step(particles, log_prob_grad_fn, lr, bandwidth=None):
    N, D  = particles.shape
    diff  = particles.unsqueeze(0) - particles.unsqueeze(1)   # (N, N, D)
    sq    = (diff ** 2).sum(-1)                                # (N, N)
    h     = sq.median() / (2 * math.log(N + 1) + 1e-6) \
            if bandwidth is None \
            else torch.tensor(bandwidth, dtype=particles.dtype)
    K     = torch.exp(-sq / h)                                 # (N, N)
    dK    = -2 / h * diff * K.unsqueeze(-1)                    # (N, N, D)
    scores = log_prob_grad_fn(particles)
    phi    = (K.unsqueeze(-1) * scores.unsqueeze(0)).mean(1) + dK.mean(1)
    return particles + lr * phi


def svgd(objective_function, n_particles, n_steps, dim,
         lr=1e-2, init_particles=None, bandwidth=None,
         keep_all=True, verbose=False):
    particles = (torch.randn(n_particles, dim).double()
                 if init_particles is None
                 else init_particles.clone().double())

    def log_prob_grad_fn(P):
        P     = P.double().detach().requires_grad_(True)
        objective_function(P).sum().backward()
        return P.grad.detach()

    all_p = []
    for t in range(n_steps):
        if keep_all:
            all_p.append(particles.clone())
        particles = svgd_step(particles, log_prob_grad_fn, lr, bandwidth)
        if verbose and (t + 1) % 100 == 0:
            print(f"  [SVGD] step {t+1:4d}/{n_steps}  "
                  f"mean log-post = {objective_function(particles).mean():.4f}")

    all_p.append(particles.clone())
    trajectories = torch.stack(all_p, dim=0) if keep_all else particles.unsqueeze(0)
    weights = torch.ones(trajectories.shape[0], n_particles) / n_particles
    return trajectories, weights


# ══════════════════════════════════════════════════════════════════════════════
# Tools for handling grids
# ══════════════════════════════════════════════════════════════════════════════

def _make_grid(bounds, M_res):
    a, b   = bounds
    xs, ys = np.linspace(a, b, M_res), np.linspace(a, b, M_res)
    Xg, Yg = np.meshgrid(xs, ys)
    grid   = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], 1),
                          dtype=torch.double)
    return Xg, Yg, grid


def _get_ensemble_probs(traj, obj_fn, grid):
    """Returns (N_particles, N_grid) numpy array of sigmoid probabilities."""
    particles = traj[-1]
    model     = obj_fn.model
    probs     = []
    for i in range(len(particles)):
        vector_to_parameters(particles[i].double(), model.parameters())
        with torch.no_grad():
            probs.append(torch.sigmoid(model(grid).squeeze()).numpy())
    return np.stack(probs, axis=0)


def _scatter(ax, X, Y):
    x = X.numpy() 
    y = Y.numpy() 
    ax.scatter(x[y > 0, 0], x[y > 0, 1], c='gold',   s=25, zorder=5)
    ax.scatter(x[y < 0, 0], x[y < 0, 1], c='tomato', s=25, zorder=5)



# ══════════════════════════════════════════════════════════════════════════════
# Visualization tools
# ══════════════════════════════════════════════════════════════════════════════

def plot_boundaries(trajectories_dict, objective_fns_dict,
                      X_tr, Y_tr, bounds=[-0.1, 1.1], M_res=60):
    Xg, Yg, grid = _make_grid(bounds, M_res)
    M = len(trajectories_dict)
    fig, axes = plt.subplots(1, M, figsize=(6*M, 5))
    if M == 1: axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        probs = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        for pi in probs:
            ax.contour(Xg, Yg, np.sign(pi - 0.5).reshape(M_res, M_res),
                       levels=[0], colors=['steelblue'], linewidths=0.6, alpha=0.3)
        p_bar = probs.mean(0).reshape(M_res, M_res)
        ax.contour(Xg, Yg, p_bar, levels=[0.5], colors=['black'], linewidths=2.0)
        _scatter(ax, X_tr, Y_tr)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlim(*bounds); ax.set_ylim(*bounds)
        ax.set_xlabel('x1'); ax.set_ylabel('x2')

    plt.suptitle("Decision boundary (final iteration)", fontsize=14)
    plt.savefig("svgd_vs_msip_boundaries.pdf")
    plt.tight_layout(); plt.show()



def plot_mean_prediction(trajectories_dict, objective_fns_dict,
                          X_tr, Y_tr, bounds=[-0.1, 1.1], M_res=60):
    Xg, Yg, grid = _make_grid(bounds, M_res)
    M = len(trajectories_dict)
    fig, axes = plt.subplots(1, M, figsize=(6*M, 5))
    if M == 1: axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        probs = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        p_bar = probs.mean(0).reshape(M_res, M_res)
        im = ax.imshow(p_bar, extent=[*bounds, *bounds], origin='lower',
                       cmap='RdBu_r', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='p̄(y=+1 | x)')
        ax.contour(Xg, Yg, p_bar, levels=[0.5], colors=['white'], linewidths=2.0)
        _scatter(ax, X_tr, Y_tr)
        ax.set_title(f"{name}", fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('x1'); ax.set_ylabel('x2')

    plt.suptitle("Mean predictive probability", fontsize=14)
    plt.savefig("svgd_vs_msip_mean_predictive_p.pdf")
    plt.tight_layout(); plt.show()


def plot_diversity_curve(trajectories_dict, subsample=10):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, traj in trajectories_dict.items():
        T, N, D = traj.shape
        steps, divs = [], []
        for t in range(0, T, subsample):
            P   = traj[t]
            sq  = ((P.unsqueeze(0) - P.unsqueeze(1)) ** 2).sum(-1)
            idx = torch.triu_indices(N, N, offset=1)
            divs.append(sq[idx[0], idx[1]].sqrt().min().item())
            steps.append(t)
        ax.plot(steps, divs, lw=2, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Smallest pairwise distance")
    ax.set_title("Particle diversity over training")
    plt.savefig("svgd_vs_msip_div.pdf")
    ax.legend(); plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluations on the dataset
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(trajectories_dict, objective_fns_dict, X, Y, split_name="test"):
    y_true = (Y.numpy() > 0)
    grid   = X.double()

    print(f"\n{'─'*75}")
    print(f"  Evaluation on {split_name} set  ({len(X)} points)")
    print(f"{'─'*75}")
    print(f"  {'Method':<12} {'Acc':>6} {'Confidence':>12} {'DAMV':>10}")
    print(f"{'─'*75}")

    results = {}
    for name, traj in trajectories_dict.items():
        probs      = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        p_bar      = probs.mean(0)
        acc        = ((p_bar > 0.5) == y_true).mean()
        confidence = np.abs(p_bar - 0.5).mean() * 2
        damv       = traj[-1].var(dim=0).mean().item()

        print(f"  {name:<12} {acc:>6.3f} {confidence:>12.3f} {damv:>10.3f}")
        results[name] = dict(p_bar=p_bar, accuracy=acc,
                             confidence=confidence, damv=damv)

    print(f"{'─'*75}\n")
    return results

def eval_function_trajectories(obj_fn, trajectories):
    T, M, d = trajectories.shape
    out = torch.zeros(T, M)
    for t in range(T):
        for m in range(M):
            out[t, m] = -obj_fn(trajectories[t, m, :])
    return out


def plot_eval_tensor(eval_tensor):
    plt.figure()
    for m in range(eval_tensor.shape[1]):
        plt.plot(eval_tensor[:, m].detach().numpy())
    plt.show()


def plot_eval_best_tensor(eval_tensor):
    plt.figure()
    plt.plot(eval_tensor.detach().numpy().min(1))
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Config ────────────────────────────────────────────────────────────────
    DATASET      = 'two_bananas'
    MODEL_CLASS  = 'bnn'
    HIDDEN_DIM   = 5         
    N_LAYERS     = 1          
    N_TRAIN      = 0.8        # train ratio
    N_PARTICLES  = 250
    N_STEPS      = 500
    BETA         = 1.0        # beta in x-> exp(-beta^{-1}V(x))
    LAMBDA2      = 0.0005     # lambda in prior;
                              # lambda close to 0 means weak prior
    LR_SVGD      = 100e-2
    LR_MSIP      = 100e-2
    SIGMA        = 0.25

    # Data loading
    X_train, Y_train, X_test, Y_test = load_dataset(
        DATASET, train_ratio=N_TRAIN, seed=0)

    # Objective loading
    obj_msip = make_objective(X_train, Y_train, model_class=MODEL_CLASS,
                              hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                              beta=BETA, lambda2=LAMBDA2)
    obj_svgd = make_objective(X_train, Y_train, model_class=MODEL_CLASS,
                              hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                              beta=BETA, lambda2=LAMBDA2)

    dimension = obj_msip.total_numel

    # Shared inititializtion
    init_particles = 5*torch.randn(N_PARTICLES, dimension).double()

    # Run MSIP
    post_log_dens_grad_val = torch.func.grad_and_value(obj_msip)
    msip_fredholm          = MSIPFredholm(1.0, post_log_dens_grad_val)


    trajectories_msip, _ = msip(
        obj_msip, N_PARTICLES, N_STEPS, dim=dimension,
        lr=LR_MSIP, init_particles=init_particles,
        kernel_length_scale=SIGMA, is_log_density_batched=True,
        kernel_diag_infl=1e-8, bounds=(-100., 100.),
        gradient_decay=1.0, keep_all=True,
        compile_step=False, verbose=True,
    )

    # Run SVGD 
    trajectories_svgd, _ = svgd(
        obj_svgd, N_PARTICLES, N_STEPS, dim=dimension,
        lr=LR_SVGD, init_particles=init_particles, bandwidth=SIGMA,
        keep_all=True, verbose=True,
    )

    # ── Bundle ────────────────────────────────────────────────────────────────
    trajectories_dict  = {"MSIP": trajectories_msip, "SVGD": trajectories_svgd}
    objective_fns_dict = {"MSIP": obj_msip,          "SVGD": obj_svgd}

    # Optimization diagnostics 
    eval_tensor_msip = eval_function_trajectories(obj_msip, trajectories_msip)
    plot_eval_tensor(eval_tensor_msip)
    plot_eval_best_tensor(eval_tensor_msip)

    eval_tensor_svgd = eval_function_trajectories(obj_svgd, trajectories_svgd)
    plot_eval_tensor(eval_tensor_svgd)
    plot_eval_best_tensor(eval_tensor_svgd)

    # Visualization
    plot_boundaries(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_mean_prediction(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_diversity_curve(trajectories_dict)

    # Evaluation on a dataset
    evaluate(trajectories_dict, objective_fns_dict, X_train, Y_train, "train")
    evaluate(trajectories_dict, objective_fns_dict, X_test,  Y_test,  "test")