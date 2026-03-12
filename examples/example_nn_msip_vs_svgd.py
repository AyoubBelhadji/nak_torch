import torch
import numpy as np
import matplotlib.pyplot as plt
from viz_tools import animate_trajectories_box
from functions import loss_nn_dataset
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


# ─────────────────────────────────────────────────────────────────────────────
# YOUR EXISTING CODE (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

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
            for c in list(range(self.C)):
                for n in list(range(self.N)):
                    tmp_var = 1
                    for m in list(range(self.M)):
                        tmp_var = tmp_var*(self.alphas[m,n,c]+ self.betas[m,n,c]*nn.ReLU()(xb @ self.vs[:,m,n,c] + self.bias[m,n,c]))
                    nn_eval[:,c] += self.ws[n,c]*tmp_var
        elif self.activation == 'Tanh':
            nn_eval = torch.zeros(xb.shape[0],self.C)
            for c in list(range(self.C)):
                for n in list(range(self.N)):
                    tmp_var = 1
                    for m in list(range(self.M)):
                        tmp_var = tmp_var*(self.alphas[m,n,c]+ self.betas[m,n,c]*nn.Tanh()(xb @ self.vs[:,m,n,c] + self.bias[m,n,c]))
                    nn_eval[:,c] += self.ws[n,c]*tmp_var
        return nn_eval


def loss_nn_dataset(dataset_name, beta=1.0, lambda2=0.01, device="cpu"):
    data = np.load(f"datasets/{dataset_name}.npz")
    X = torch.from_numpy(data["X"].T).double().to(device)
    Y = torch.from_numpy(data["Y"].T).double().to(device)
    loss_func = F.soft_margin_loss
    model = sigma_pi(2, 1, 10, 1, 'ReLU').to(device)
    param_info = []
    for name, p in model.named_parameters():
        param_info.append((name, p.shape, p.numel()))
    buffer_dict = OrderedDict(model.named_buffers())
    total_numel = sum(numel for _, _, numel in param_info)

    def theta_to_param_dict(theta_1d):
        if theta_1d.ndim != 1:
            raise ValueError(f"theta_1d must be 1D, got shape {theta_1d.shape}")
        if theta_1d.numel() != total_numel:
            raise ValueError(f"theta_1d has {theta_1d.numel()} entries, expected {total_numel}")
        out = OrderedDict()
        idx = 0
        for name, shape, numel in param_info:
            out[name] = theta_1d[idx:idx + numel].view(shape)
            idx += numel
        return out

    def loss_for_single_theta(theta_1d):
        param_dict = theta_to_param_dict(theta_1d)
        pred = functional_call(model, (param_dict, buffer_dict), (X,))
        data_loss = loss_func(pred, Y)
        reg = lambda2 * (theta_1d ** 2).sum()
        return -(data_loss + reg) / beta

    def objective_function(theta):
        if theta.ndim == 1:
            return loss_for_single_theta(theta)
        elif theta.ndim == 2:
            return torch.stack([loss_for_single_theta(theta[i]) for i in range(theta.shape[0])])
        else:
            raise ValueError(f"theta must be 1D or 2D, got shape {theta.shape}")

    return objective_function


def eval_function_trajectories(the_objective_function, the_trajectories):
    T, M, d = the_trajectories.shape
    eval_tensor = torch.zeros((T, M))
    for t in range(T):
        for m in range(M):
            eval_tensor[t, m] = -the_objective_function(the_trajectories[t, m, :])
    return eval_tensor


def plot_eval_tensor(the_eval_tensor):
    fig = plt.figure()
    T, M = the_eval_tensor.shape
    np_eval_tensor = the_eval_tensor.detach().numpy()
    for m in range(M):
        plt.plot(list(range(T)), np_eval_tensor[:, m])
    plt.show()


def plot_eval_best_tensor(the_eval_tensor):
    fig = plt.figure()
    T, M = the_eval_tensor.shape
    np_eval_tensor = the_eval_tensor.detach().numpy()
    np_eval_min_tensor = np.min(np_eval_tensor, 1)
    plt.plot(list(range(T)), np_eval_min_tensor)
    plt.show()


def plot_2D_classification_tensor(the_trajectories, dataset_name, m, t):
    plot_2D_classification_with_dataset_from_theta(the_trajectories[t, m, :], dataset_name, [-2, 2], 50, device="cpu")


def plot_2D_classification_with_dataset_from_theta(theta, dataset_name, bounds, M_res, device="cpu"):
    a = bounds[0]
    b = bounds[1]
    M = M_res
    data = np.load(f"datasets/{dataset_name}.npz")
    x_train = torch.from_numpy(data["X"]).float()
    y_train = torch.from_numpy(data["Y"]).float().view(-1)
    model = sigma_pi(2, 1, 10, 1, 'ReLU').to(device).float()
    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    vector_to_parameters(theta_t, model.parameters())
    x = np.linspace(a, b, M)
    y = np.linspace(a, b, M)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((M, M))
    for m_1 in range(M):
        for m_2 in range(M):
            z = torch.from_numpy(np.array((X[m_1, m_2], Y[m_1, m_2]))).float()
            z = torch.sign(model.forward(z)[0])
            Z[m_1, m_2] = z
    plt.figure(figsize=(10, 8))
    plt.imshow(Z, extent=[a, b, a, b], origin='lower')
    plt.colorbar(label='Z')
    plt.plot(x_train[0, y_train.flatten() > 0], x_train[1, y_train.flatten() > 0], 'b.')
    plt.plot(x_train[0, y_train.flatten() < 0], x_train[1, y_train.flatten() < 0], 'r.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SVGD  (drop-in addition)
# ─────────────────────────────────────────────────────────────────────────────

def svgd_step(particles, log_prob_grad_fn, lr, bandwidth=None):
    """
    One SVGD step.

    Parameters
    ----------
    particles       : (N, D) tensor
    log_prob_grad_fn: callable  particles -> (N, D) gradients of log-posterior
    lr              : float
    bandwidth       : float or None  (None = median heuristic)

    Returns
    -------
    updated particles : (N, D) tensor
    """
    N, D = particles.shape

    # ── RBF kernel + grad ────────────────────────────────────────────────────
    diff  = particles.unsqueeze(0) - particles.unsqueeze(1)   # (N, N, D)
    sq    = (diff ** 2).sum(-1)                                # (N, N)

    if bandwidth is None:
        # median heuristic  (same as your framework.py)
        h = sq.median() / (2 * math.log(N + 1) + 1e-6)
    else:
        h = torch.tensor(bandwidth, dtype=particles.dtype)

    K  =  torch.exp(-sq / h)                                  # (N, N)
    dK = -2 / h * diff * K.unsqueeze(-1)                      # (N, N, D)
    #    dK[i, j, :]  =  grad_{x_i} k(x_i, x_j)

    # ── score  ∇_w log p(w | D) for every particle ───────────────────────────
    scores = log_prob_grad_fn(particles)                       # (N, D)

    # ── SVGD update  φ(x_i) = 1/N Σ_j [ k(x_j, x_i) s_j + ∇_{x_j} k(x_j, x_i) ]
    #   K[j, i] * scores[j]  summed over j  → (N, D)
    phi = (K.unsqueeze(-1) * scores.unsqueeze(0)).mean(1) \
        + dK.mean(1)                                           # (N, D)

    return particles + lr * phi


def svgd(objective_function, n_particles, n_steps, dim,
         lr=1e-2, init_particles=None, bandwidth=None,
         keep_all=True, verbose=False):
    """
    Run SVGD to sample from exp(objective_function).
    Mirrors the calling convention of msip() so you can swap them.

    Parameters
    ----------
    objective_function : callable  (D,) or (N,D) -> scalar or (N,) tensor
                         Should return log-posterior values (higher = better).
                         This is exactly what loss_nn_dataset() returns.
    n_particles        : int
    n_steps            : int
    dim                : int   weight-space dimension
    lr                 : float learning rate
    init_particles     : (N, D) tensor or None  (random N(0,1) if None)
    bandwidth          : float or None  (median heuristic if None)
    keep_all           : bool  if True return (n_steps, N, D), else (1, N, D)
    verbose            : bool

    Returns
    -------
    trajectories : (T, N, D) tensor   T = n_steps if keep_all else 1
    weights      : (T, N)   tensor    uniform (SVGD has no importance weights)
    """
    if init_particles is None:
        particles = torch.randn(n_particles, dim).double()
    else:
        particles = init_particles.clone().double()   # always Float32

    particles = particles.detach().requires_grad_(False)

    # ── gradient oracle: grad log p(w|D) for all particles ───────────────────
    def log_prob_grad_fn(P):
        """P: (N, D) -> (N, D)  score matrix"""
        P = P.double().detach().requires_grad_(True)
        # evaluate log-posterior for each particle
        log_p = objective_function(P)          # (N,)
        # sum and backprop to get all gradients in one pass
        log_p.sum().backward()
        return P.grad.detach()

    all_particles = []

    for t in range(n_steps):
        if keep_all:
            all_particles.append(particles.clone())

        particles = svgd_step(particles, log_prob_grad_fn, lr, bandwidth)

        if verbose and (t + 1) % 100 == 0:
            mean_lp = objective_function(particles).mean().item()
            print(f"  [SVGD] step {t+1:4d}/{n_steps}  mean log-post = {mean_lp:.4f}")

    all_particles.append(particles.clone())   # always keep final

    if keep_all:
        trajectories = torch.stack(all_particles, dim=0)       # (T, N, D)
    else:
        trajectories = particles.unsqueeze(0)                  # (1, N, D)

    weights = torch.ones(trajectories.shape[0], n_particles) / n_particles

    return trajectories, weights


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def plot_boundary_fan(trajectories_dict, dataset_name, t=-1,
                      bounds=[-0.1, 1.1], M_res=60):
    """
    Overlay all particle decision boundaries at step t for every method.
    One panel per method, same layout as your existing plots.

    Parameters
    ----------
    trajectories_dict : dict  {"MSIP": traj_msip, "SVGD": traj_svgd, ...}
    t                 : int   which time step (-1 = final)
    """
    data    = np.load(f"datasets/{dataset_name}.npz")
    x_train = torch.from_numpy(data["X"]).float()
    y_train = torch.from_numpy(data["Y"]).float().view(-1)

    a, b = bounds
    xs   = np.linspace(a, b, M_res)
    ys   = np.linspace(a, b, M_res)
    Xg, Yg = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1),
                        dtype=torch.float32)

    n_methods = len(trajectories_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        # traj shape: (T, N, D)
        particles_t = traj[t]                  # (N, D)
        N = particles_t.shape[0]

        model = sigma_pi(2, 1, 10, 1, 'ReLU').float()

        # plot individual boundaries faintly
        for i in range(N):
            vector_to_parameters(particles_t[i].float(), model.parameters())
            with torch.no_grad():
                Z = torch.sign(model(grid)).reshape(M_res, M_res).numpy()
            ax.contour(Xg, Yg, Z, levels=[0], colors=['steelblue'],
                       linewidths=0.6, alpha=0.3)

        # plot mean boundary
        mean_p = particles_t.mean(0)
        vector_to_parameters(mean_p.float(), model.parameters())
        with torch.no_grad():
            Z_mean = torch.sign(model(grid)).reshape(M_res, M_res).numpy()
        ax.contour(Xg, Yg, Z_mean, levels=[0], colors=['white'],
                   linewidths=2.0)

        # data
        ax.scatter(x_train[0, y_train > 0].numpy(),
                   x_train[1, y_train > 0].numpy(), c='gold',  s=25, zorder=5)
        ax.scatter(x_train[0, y_train < 0].numpy(),
                   x_train[1, y_train < 0].numpy(), c='tomato', s=25, zorder=5)

        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlim(a, b); ax.set_ylim(a, b)
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

    plt.suptitle(f"Decision boundaries (last iteration)", fontsize=14)
    plt.tight_layout()
    plt.savefig("svgd_vs_msip_boundaries.pdf")
    plt.show()


def plot_mutual_information(trajectories_dict, dataset_name,
                             bounds=[-0.1, 1.1], M_res=60):
    """
    Side-by-side MI heatmaps for each method.
    MI = H[p_bar] - mean_i H[p_i]   (epistemic uncertainty)
    Uses sigmoid(network_output) as p_i.
    """
    data    = np.load(f"datasets/{dataset_name}.npz")
    x_train = torch.from_numpy(data["X"]).float()
    y_train = torch.from_numpy(data["Y"]).float().view(-1)

    a, b = bounds
    xs   = np.linspace(a, b, M_res)
    ys   = np.linspace(a, b, M_res)
    Xg, Yg = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1),
                        dtype=torch.float32)

    def bH(p):
        """Binary entropy, safe."""
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    n_methods = len(trajectories_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        particles_t = traj[-1]                 # (N, D) — final particles
        N = particles_t.shape[0]
        model = sigma_pi(2, 1, 10, 1, 'ReLU').float()

        # p_i for every particle: shape (N, M_res²)
        all_probs = []
        for i in range(N):
            vector_to_parameters(particles_t[i].float(), model.parameters())
            with torch.no_grad():
                logits = model(grid).squeeze()        # (M_res²,)
                pi     = torch.sigmoid(logits).numpy()
            all_probs.append(pi)

        all_probs = np.stack(all_probs, axis=0)       # (N, M_res²)
        p_bar = all_probs.mean(0)                     # (M_res²,)

        H_total    = bH(p_bar)                        # H[\bar p]
        H_per_part = bH(all_probs).mean(0)            # E_i[H[p_i]]
        MI         = H_total - H_per_part             # epistemic uncertainty

        MI_map = MI.reshape(M_res, M_res)

        im = ax.imshow(MI_map, extent=[a, b, a, b], origin='lower',
                       cmap='viridis', vmin=0)
        plt.colorbar(im, ax=ax, label='MI  (nats)')

        # data overlay
        ax.scatter(x_train[0, y_train > 0].numpy(),
                   x_train[1, y_train > 0].numpy(), c='gold',   s=25, zorder=5)
        ax.scatter(x_train[0, y_train < 0].numpy(),
                   x_train[1, y_train < 0].numpy(), c='tomato', s=25, zorder=5)

        mean_mi = MI_map.mean()
        ax.set_title(f"{name}  —  mean MI = {mean_mi:.3f}", fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

    plt.suptitle("Epistemic uncertainty  H[p̄] − E[H[pᵢ]]", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_diversity_curve(trajectories_dict, subsample=10):
    """
    Avg pairwise distance between particles over training iterations.
    One curve per method.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, traj in trajectories_dict.items():
        T, N, D = traj.shape
        steps = list(range(0, T, subsample))
        divs  = []
        for t in steps:
            P    = traj[t]                               # (N, D)
            diff = P.unsqueeze(0) - P.unsqueeze(1)       # (N, N, D)
            sq   = (diff ** 2).sum(-1)                   # (N, N)
            idx  = torch.triu_indices(N, N, offset=1)
            divs.append(sq[idx[0], idx[1]].sqrt().mean().item())

        ax.plot(steps, divs, lw=2, label=name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg pairwise  ‖wᵢ − wⱼ‖₂")
    ax.set_title("Particle diversity over training")
    ax.legend()
    plt.tight_layout()
    plt.savefig("svgd_vs_msip_diversity.pdf")
    plt.show()


def plot_mean_prediction(trajectories_dict, dataset_name,
                          bounds=[-0.1, 1.1], M_res=60):
    """
    Side-by-side mean prediction heatmaps  p̄(x) = 1/N Σ sigmoid(f_w^i(x))
    Tells you: where does the ensemble think class = +1 ?
    """
    data    = np.load(f"datasets/{dataset_name}.npz")
    x_train = torch.from_numpy(data["X"]).float()
    y_train = torch.from_numpy(data["Y"]).float().view(-1)

    a, b = bounds
    xs   = np.linspace(a, b, M_res)
    ys   = np.linspace(a, b, M_res)
    Xg, Yg = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1),
                        dtype=torch.double)

    n_methods = len(trajectories_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        particles_t = traj[-1]                 # (N, D) — final particles
        N = particles_t.shape[0]
        model = sigma_pi(2, 1, 10, 1, 'ReLU').double()

        # p_i for every particle
        all_probs = []
        for i in range(N):
            vector_to_parameters(particles_t[i].double(), model.parameters())
            with torch.no_grad():
                logits = model(grid).squeeze()
                pi     = torch.sigmoid(logits).numpy()
            all_probs.append(pi)

        all_probs = np.stack(all_probs, axis=0)   # (N, M_res²)
        p_bar     = all_probs.mean(0)             # (M_res²,)
        p_bar_map = p_bar.reshape(M_res, M_res)

        im = ax.imshow(p_bar_map, extent=[a, b, a, b], origin='lower',
                       cmap='RdBu_r', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='p̄(y=+1 | x)')

        # mean boundary contour
        ax.contour(Xg, Yg, p_bar_map, levels=[0.5],
                   colors=['white'], linewidths=2.0)

        # data
        ax.scatter(x_train[0, y_train > 0].numpy(),
                   x_train[1, y_train > 0].numpy(), c='gold',   s=25, zorder=5)
        ax.scatter(x_train[0, y_train < 0].numpy(),
                   x_train[1, y_train < 0].numpy(), c='tomato', s=25, zorder=5)

        ax.set_title(f"{name}", fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

    plt.suptitle("Mean predictive probability", fontsize=14)
    plt.tight_layout()
    plt.savefig("svgd_vs_msip_mean_predictive_p.pdf")
    plt.show()

def compute_mean_prediction_accuracy(trajectories_dict, dataset_name):
    """
    For each method, computes the mean prediction p̄(x) on every test point
    and reports accuracy and mean confidence.

    Expects a file  datasets/{dataset_name}_test.npz  with keys X, Y.
    Falls back to the training set if no test set is found.
    """
    # load test data
    try:
        data = np.load(f"datasets/{dataset_name}_test.npz")
        split = "test"
    except FileNotFoundError:
        data = np.load(f"datasets/{dataset_name}.npz")
        split = "train (no test set found)"

    X_test = torch.from_numpy(data["X"].T).double()   # (N_test, 2)
    Y_test = torch.from_numpy(data["Y"].T).double()   # (N_test, 1)
    y_true = (Y_test.squeeze() > 0).numpy()           # bool, +1 -> True

    print(f"\n{'─'*50}")
    print(f"  Mean prediction accuracy  [{split}]")
    print(f"{'─'*50}")

    results = {}
    for name, traj in trajectories_dict.items():
        particles_t = traj[-1]                 # (N_particles, D)
        N = particles_t.shape[0]
        model = sigma_pi(2, 1, 10, 1, 'ReLU').double()

        # p_i for every particle  →  (N_particles, N_test)
        all_probs = []
        for i in range(N):
            vector_to_parameters(particles_t[i].double(), model.parameters())
            with torch.no_grad():
                logits = model(X_test).squeeze()
                pi     = torch.sigmoid(logits).numpy()
            all_probs.append(pi)

        all_probs  = np.stack(all_probs, axis=0)  # (N_particles, N_test)
        p_bar      = all_probs.mean(0)            # (N_test,)
        y_pred     = p_bar > 0.5

        accuracy        = (y_pred == y_true).mean()
        mean_confidence = np.abs(p_bar - 0.5).mean() * 2   # 0 = random, 1 = certain

        print(f"  {name:10s}  |  acc = {accuracy:.3f}  |  mean confidence = {mean_confidence:.3f}")
        results[name] = dict(p_bar=p_bar, accuracy=accuracy,
                             mean_confidence=mean_confidence)

    print(f"{'─'*50}\n")
    return results



# def generate_test_set(dataset_name, n_test=200, noise=0.1, seed=99):
#     """
#     Generates an in-distribution test set from the same two-moons
#     generative process as the training data and saves it as
#         datasets/{dataset_name}_test.npz

#     Parameters
#     ----------
#     dataset_name : str
#     n_test       : int    number of test points
#     noise        : float  same noise level as training
#     seed         : int

#     Returns
#     -------
#     X_test : (n_test, 2) numpy array
#     Y_test : (n_test,)   numpy array  values in {-1, +1}
#     """
#     from sklearn.datasets import make_moons

#     X, y = make_moons(n_samples=n_test, noise=noise, random_state=seed)
#     Y    = np.where(y == 1, 1.0, -1.0)   # convert 0/1 → -1/+1 to match training

#     np.savez(f"datasets/{dataset_name}_test.npz", X=X.T, Y=Y.reshape(1, -1))
#     print(f"  Saved  {dataset_name}_test.npz  ({n_test} points)")

#     return X, Y

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    save_gif      = False
    now_stamp     = datetime.now()
    dataset_name  = 'two_bananas'

    objective_function     = loss_nn_dataset(dataset_name, beta=1.0,
                                             lambda2=0.00001, device="cpu")
    post_log_dens_grad_val = torch.func.grad_and_value(objective_function)

    n_particles = 50
    dimension   = 60
    n_steps     = 1000

    # ── shared initialisation (same start for fair comparison) ───────────────
    init_particles = torch.randn((n_particles, dimension)).double()

    # ── MSIP ─────────────────────────────────────────────────────────────────
    kernel_length_scale = 1.0
    bounds              = (-100., 100.)
    gradient_decay      = 1.0
    lr_msip             = 100e-2
    kernel_diag_infl    = 1e-8

    msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val)

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
        verbose=True,
    )

    # ── SVGD ─────────────────────────────────────────────────────────────────
    trajectories_svgd, _ = svgd(
        objective_function, n_particles, n_steps, dim=dimension,
        lr=1e-1,                     # tune this if needed
        init_particles=init_particles, # same init as MSIP
        bandwidth=kernel_length_scale,
        keep_all=True,
        verbose=True,
    )

    # ── bundle for comparison plots ───────────────────────────────────────────
    trajectories_dict = {
        "MSIP": trajectories_msip,
        "SVGD": trajectories_svgd,
    }

    # ── existing per-particle plots (MSIP) ───────────────────────────────────
    eval_tensor = eval_function_trajectories(objective_function, trajectories_msip)
    plot_eval_tensor(eval_tensor)
    plot_eval_best_tensor(eval_tensor)
    T, _, _ = trajectories_msip.shape
    #for m in range(n_particles):
    #    t_m = np.argmin(eval_tensor[:, m].detach().numpy())
    #    plot_2D_classification_tensor(trajectories_msip, dataset_name, m, t_m)

    # ── NEW comparison plots ──────────────────────────────────────────────────
    plot_boundary_fan(trajectories_dict, dataset_name)
    plot_mutual_information(trajectories_dict, dataset_name)
    plot_diversity_curve(trajectories_dict)
    plot_mean_prediction(trajectories_dict, dataset_name)
    compute_mean_prediction_accuracy(trajectories_dict, dataset_name)
    
    if save_gif:
        fpath = f"results/gif/msip_{dataset_name}_{now_stamp}.gif"
        animate_trajectories_box(objective_function, trajectories_msip,
                                 50, bounds, save_path=fpath)