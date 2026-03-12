# bnn_particles_uci.py
#
# Compare two interacting-particle methods for Bayesian neural network regression:
#   1) SVGD
#   2) SPOS-like (SVGD drift + Langevin noise)
#
# Benchmark protocol:
#   - 90/10 train/test split
#   - repeated random trials
#   - report mean ± std for RMSE and test log-likelihood
#
# Notes:
#   - This is a compact, research-friendly implementation.
#   - It uses a one-hidden-layer BNN with Gaussian prior and Gaussian observation noise.
#   - The "particle" is the full flattened parameter vector of the network + log noise.
#
# Usage examples:
#   python bnn_particles_uci.py --dataset concrete
#   python bnn_particles_uci.py --dataset energy
#   python bnn_particles_uci.py --dataset wine
#
# Optional dependency for dataset loading:
#   pip install ucimlrepo
#
# For exact reproduction of old BNN papers, you would usually need:
#   - the exact same splits
#   - exact preprocessing
#   - exact hyperparameters per dataset
#   - exact prior/noise parameterization
#

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Config
# ----------------------------

@dataclass
class ExperimentConfig:
    hidden_dim: int = 50
    n_particles: int = 20
    n_epochs: int = 300
    batch_size: int = 100
    lr: float = 1e-3
    prior_std: float = 1.0
    init_std: float = 0.1
    test_size: float = 0.1
    n_trials: int = 20
    seed: int = 0
    device: str = "cpu"
    method: str = "svgd"  # or "spos"
    beta: float = 1.0     # temperature for SPOS-like noise
    log_noise_init: float = -1.0
    median_heuristic_eps: float = 1e-8
    dtype: torch.dtype = torch.float32


# ----------------------------
# Dataset loading
# ----------------------------

def load_uci_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a few standard UCI regression datasets via ucimlrepo.

    Supported names:
        - concrete
        - energy
        - yacht
        - wine
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "Please install ucimlrepo first: pip install ucimlrepo"
        ) from exc

    name = name.lower()

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
            "Use one of: concrete, energy, yacht, wine."
        )

    # Remove rows with missing values if any
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    return X, y


# ----------------------------
# Flatten/unflatten utilities
# ----------------------------

class OneHiddenBNNShape:
    """
    One-hidden-layer scalar-output network:
        x -> Linear(d_in, h) -> ReLU -> Linear(h, 1)
    plus a scalar log_noise.

    Flattened parameter vector layout:
        W1: [d_in, h]
        b1: [h]
        W2: [h, 1]
        b2: [1]
        log_noise: [1]
    """

    def __init__(self, d_in: int, hidden_dim: int):
        self.d_in = d_in
        self.hidden_dim = hidden_dim

        self.n_w1 = d_in * hidden_dim
        self.n_b1 = hidden_dim
        self.n_w2 = hidden_dim
        self.n_b2 = 1
        self.n_log_noise = 1

        self.total_dim = (
            self.n_w1 + self.n_b1 + self.n_w2 + self.n_b2 + self.n_log_noise
        )

    def unpack(self, theta: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        theta: [..., total_dim]
        returns:
            W1 [..., d_in, h]
            b1 [..., h]
            W2 [..., h, 1]
            b2 [..., 1]
            log_noise [..., 1]
        """
        idx = 0

        W1 = theta[..., idx: idx + self.n_w1]
        W1 = W1.reshape(*theta.shape[:-1], self.d_in, self.hidden_dim)
        idx += self.n_w1

        b1 = theta[..., idx: idx + self.n_b1]
        idx += self.n_b1

        W2 = theta[..., idx: idx + self.n_w2]
        W2 = W2.reshape(*theta.shape[:-1], self.hidden_dim, 1)
        idx += self.n_w2

        b2 = theta[..., idx: idx + self.n_b2]
        idx += self.n_b2

        log_noise = theta[..., idx: idx + self.n_log_noise]
        idx += self.n_log_noise

        return W1, b1, W2, b2, log_noise


def initialize_particles(
    n_particles: int,
    shape: OneHiddenBNNShape,
    cfg: ExperimentConfig,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    theta = cfg.init_std * torch.randn(
        n_particles, shape.total_dim, generator=generator, device=device, dtype=cfg.dtype,
    )
    theta[:, -1] = cfg.log_noise_init
    return theta


# ----------------------------
# BNN forward and log posterior
# ----------------------------

def forward_particles(
    theta: torch.Tensor,
    x: torch.Tensor,
    shape: OneHiddenBNNShape,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    theta: [P, D]
    x: [B, d_in]
    returns:
        mu: [P, B]
        log_noise: [P, 1]
    """
    W1, b1, W2, b2, log_noise = shape.unpack(theta)

    # Hidden layer
    # x: [B, d_in]
    # W1: [P, d_in, h]
    # -> hidden_pre: [P, B, h]
    hidden_pre = torch.einsum("bd,pdh->pbh", x, W1) + b1[:, None, :]
    hidden = F.relu(hidden_pre)

    # Output
    # W2: [P, h, 1] -> squeeze to [P, h] for clean einsum
    # b2: [P, 1] -> squeeze to [P] then unsqueeze for broadcast
    mu = torch.einsum("pbh,ph->pb", hidden, W2.squeeze(-1))  # [P, B]
    mu = mu + b2.squeeze(-1).unsqueeze(-1)                   # [P, B] + [P, 1] -> [P, B]

    return mu, log_noise


def minibatch_log_posterior(theta, x_batch, y_batch, n_train, shape, cfg):
    P = theta.shape[0]
    B = x_batch.shape[0]

    mu, log_noise = forward_particles(theta, x_batch, shape)  # mu [P, B]

    sigma2 = torch.exp(2 * log_noise).view(P, 1)              # [P, 1]

    y = y_batch.view(1, B)                                    # [1, B]

    residual2 = (y - mu) ** 2                                 # [P, B]

    loglik = -0.5 * (
        residual2 / sigma2 +
        torch.log(2 * math.pi * sigma2)
    )                                                          # [P, B]

    loglik_per_particle = loglik.sum(1)                        # [P]
    loglik_per_particle = (n_train / B) * loglik_per_particle  # [P]

    weight_part = theta[:, :-1]
    logprior_weights = -0.5 * weight_part.pow(2).sum(1)        # [P]
    logprior_log_noise = -0.5 * theta[:, -1] ** 2              # [P]

    return loglik_per_particle + logprior_weights + logprior_log_noise




def score_function(
    theta: torch.Tensor,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    n_train: int,
    shape: OneHiddenBNNShape,
    cfg: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        logp: [P]
        grad_logp: [P, D]
    """
    theta = theta.clone().detach().requires_grad_(True)
    logp = minibatch_log_posterior(theta, x_batch, y_batch, n_train, shape, cfg)
    grad = torch.autograd.grad(logp.sum(), theta)[0]
    return logp.detach(), grad.detach()


# ----------------------------
# Kernel and particle updates
# ----------------------------

def pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    # x: [P, D]
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    d2 = x2 + x2.T - 2.0 * x @ x.T
    return d2.clamp_min(0.0)


def rbf_kernel_and_grad(theta: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        K: [P, P]
        grad_K_sum: [P, D], where grad_K_sum[i] = sum_j ∇_{theta_j} K(theta_j, theta_i)

    With RBF kernel:
        K_ij = exp(-||theta_i-theta_j||^2 / h)

    The SVGD repulsive term is:
        sum_j ∇_{theta_j} K(theta_j, theta_i)
    """
    P, D = theta.shape
    d2 = pairwise_sq_dists(theta).detach()

    # Median heuristic
    h = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0, device=theta.device)
    h = h / math.log(P + 1.0)
    h = h.clamp_min(eps)

    K = torch.exp(-d2 / h)  # [P, P]

    # For each i, sum_j ∇_{theta_j} K(theta_j, theta_i)
    # ∇_{theta_j} K_ji = -(2/h) K_ji (theta_j - theta_i)
    diff = theta.unsqueeze(1) - theta.unsqueeze(0)  # [P, P, D], diff[j, i] = theta_j - theta_i
    grad_K = -(2.0 / h) * K.unsqueeze(-1) * diff    # [P, P, D]
    grad_K_sum = grad_K.sum(dim=0)                  # [P, D], sum over j

    return K, grad_K_sum


def svgd_direction(
    theta: torch.Tensor,
    grad_logp: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    SVGD direction:
        phi_i = (1/P) sum_j [ K_ji grad_logp_j + ∇_{theta_j} K_ji ]
    """
    K, grad_K_sum = rbf_kernel_and_grad(theta, eps=eps)
    P = theta.shape[0]
    phi = (K @ grad_logp + grad_K_sum) / P
    return phi


def update_particles(
    theta: torch.Tensor,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    n_train: int,
    shape: OneHiddenBNNShape,
    cfg: ExperimentConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    _, grad_logp = score_function(theta, x_batch, y_batch, n_train, shape, cfg)
    phi = svgd_direction(theta, grad_logp, eps=cfg.median_heuristic_eps)

    if cfg.method == "svgd":
        theta = theta + cfg.lr * phi

    elif cfg.method == "spos":
        noise = torch.randn(
            theta.shape, generator=generator, device=theta.device, dtype=theta.dtype
        )
        theta = theta + cfg.lr * phi + math.sqrt(2.0 * cfg.lr / cfg.beta) * noise

    else:
        raise ValueError(f"Unknown method '{cfg.method}'")

    return theta.detach()


# ----------------------------
# Prediction and metrics
# ----------------------------

@torch.no_grad()
def predictive_moments(
    theta: torch.Tensor,
    x: torch.Tensor,
    shape: OneHiddenBNNShape,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mixture-of-Gaussians predictive moments under equally weighted particles.

    Returns:
        mean: [N]
        var:  [N]
    """
    mu, log_noise = forward_particles(theta, x, shape)   # mu: [P, N]
    sigma2 = torch.exp(2.0 * log_noise)                  # [P, 1]

    mean = mu.mean(dim=0)                                # [N]
    second_moment = (mu ** 2 + sigma2).mean(dim=0)       # [N]
    var = (second_moment - mean ** 2).clamp_min(1e-8)    # [N]

    return mean, var


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def avg_test_loglik(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    # Gaussian approximation to predictive mixture through first two moments
    return float(np.mean(-0.5 * (np.log(2.0 * np.pi * var) + (y_true - mean) ** 2 / var)))


# ----------------------------
# Training and evaluation
# ----------------------------

def make_batches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    n = X.shape[0]
    perm = torch.randperm(n, generator=generator, device=X.device)
    batches = []
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        batches.append((X[idx], y[idx]))
    return batches


def run_single_trial(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ExperimentConfig,
    trial_seed: int,
) -> Dict[str, float]:
    device = torch.device(cfg.device)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=trial_seed
    )

    # Standardize using train split only
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test_std = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    X_train_t = torch.tensor(X_train, dtype=cfg.dtype, device=device)
    X_test_t = torch.tensor(X_test, dtype=cfg.dtype, device=device)
    y_train_t = torch.tensor(y_train, dtype=cfg.dtype, device=device)

    n_train, d_in = X_train.shape
    shape = OneHiddenBNNShape(d_in=d_in, hidden_dim=cfg.hidden_dim)

    g = torch.Generator(device=device)
    g.manual_seed(trial_seed)

    theta = initialize_particles(
        n_particles=cfg.n_particles,
        shape=shape,
        cfg=cfg,
        generator=g,
        device=device,
    )

    t0 = time.time()

    for _ in range(cfg.n_epochs):
        batches = make_batches(X_train_t, y_train_t, cfg.batch_size, g)
        for x_batch, y_batch in batches:
            theta = update_particles(
                theta=theta,
                x_batch=x_batch,
                y_batch=y_batch,
                n_train=n_train,
                shape=shape,
                cfg=cfg,
                generator=g,
            )

    elapsed = time.time() - t0

    # Predict in standardized space
    mean_std_t, var_std_t = predictive_moments(theta, X_test_t, shape)
    mean_std = mean_std_t.cpu().numpy()
    var_std = var_std_t.cpu().numpy()

    # Convert back to original y scale
    y_mean = float(y_scaler.mean_[0])
    y_scale = float(y_scaler.scale_[0])

    mean = y_mean + y_scale * mean_std
    var = (y_scale ** 2) * var_std

    rmse_val = rmse(y_test, mean)
    ll_val = avg_test_loglik(y_test, mean, var)

    return {
        "rmse": rmse_val,
        "ll": ll_val,
        "time_sec": elapsed,
    }


def summarize(vals: List[float]) -> str:
    vals = np.asarray(vals, dtype=np.float64)
    return f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"


def run_benchmark(dataset_name: str, cfg: ExperimentConfig) -> Dict[str, List[float]]:
    X, y = load_uci_dataset(dataset_name)

    if dataset_name == "year":
        # not loaded here, but if you add it later, large datasets usually use bigger batches
        cfg.batch_size = max(cfg.batch_size, 1000)

    results = {"rmse": [], "ll": [], "time_sec": []}

    for trial in range(cfg.n_trials):
        trial_seed = cfg.seed + trial
        out = run_single_trial(X, y, cfg, trial_seed)
        for k in results:
            results[k].append(out[k])

        print(
            f"[{cfg.method.upper()}] trial {trial+1:02d}/{cfg.n_trials} | "
            f"RMSE={out['rmse']:.4f} | LL={out['ll']:.4f} | time={out['time_sec']:.2f}s"
        )

    return results


def print_comparison_table(name: str, res_a: Dict[str, List[float]], label_a: str,
                           res_b: Dict[str, List[float]], label_b: str) -> None:
    print("\n" + "=" * 78)
    print(f"Dataset: {name}")
    print("=" * 78)
    print(f"{'Metric':<16}{label_a:<28}{label_b:<28}")
    print("-" * 78)
    print(f"{'Avg. Test RMSE':<16}{summarize(res_a['rmse']):<28}{summarize(res_b['rmse']):<28}")
    print(f"{'Avg. Test LL':<16}{summarize(res_a['ll']):<28}{summarize(res_b['ll']):<28}")
    print(f"{'Avg. Time (s)':<16}{summarize(res_a['time_sec']):<28}{summarize(res_b['time_sec']):<28}")
    print("=" * 78 + "\n")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="concrete",
                        choices=["concrete", "energy", "yacht", "wine"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--n_particles", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_cfg = ExperimentConfig(
        hidden_dim=args.hidden_dim,
        n_particles=args.n_particles,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_trials=args.n_trials,
        seed=args.seed,
        device=args.device,
    )

    cfg_svgd = ExperimentConfig(**{**base_cfg.__dict__, "method": "svgd"})
    cfg_spos = ExperimentConfig(**{**base_cfg.__dict__, "method": "spos", "beta": 1.0})

    print(f"Running on dataset='{args.dataset}' with device='{args.device}'")
    print("Method A: SVGD")
    res_svgd = run_benchmark(args.dataset, cfg_svgd)

    print("Method B: SPOS-like")
    res_spos = run_benchmark(args.dataset, cfg_spos)

    print_comparison_table(
        name=args.dataset,
        res_a=res_svgd,
        label_a="SVGD",
        res_b=res_spos,
        label_b="SPOS-like",
    )


if __name__ == "__main__":
    main()