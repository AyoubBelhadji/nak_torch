# %%
import math
import torch
from nak_torch.functions import aristoff_bangerth as ab, build_aristoff_bangerth
from nak_torch.algorithms import msip
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
torch.set_default_device("cpu")

# # %%
# N, N_obs = 32, 13
# solve_args = ab.build_forward_solver_args(N, N_obs, torch.float32)
# thetas = torch.as_tensor(ab.theta_true, device=solve_args[0].device).repeat(11,1)

# # %%
# hi_res_obs = ab.build_forward_solver_args(N, 128, torch.float32)[0]

# # %%
# out = ab.forward_solver(thetas, N, *solve_args)
# obs = (hi_res_obs @ out.T).reshape(128, 128, -1).permute(2, 0, 1)

# %%
log_p = build_aristoff_bangerth(use_compiled=True, dtype=torch.float64)
log_th = torch.randn(3, 64, requires_grad=True, dtype=torch.float64)
test_out = log_p(log_th)
test_eval = torch.autograd.grad(test_out.sum(), log_th)

# %%
n_particles = 500
torch.manual_seed(1)
init_particles = 2 * torch.randn((n_particles, 64),
                                 dtype=torch.float64, device='cpu')  # Sample from prior
kernel_bandwidth = 0.5

trajectories, _ = msip(
    log_p,
    n_particles=n_particles,
    n_steps=25,  # "epochs" (passes over all particles)
    dim=64,
    bounds=(-8, 8),   # [a,b]^d
    gradient_informed=True,
    projection=True,
    lr=1e-1,
    # noise=0.05,          # currently unused
    init_particles=init_particles,
    kernel_bandwidth=kernel_bandwidth,
    bandwidth_factor=0.75,
    # inner_tol=1e-6,      # equilibrium tolerance for a particle
    # max_inner_steps=1,  # max inner iterations per particle
    seed=0,
    diag_infl=1e-10,
    device="cpu",
    keep_all=False
)

# %%
side_len = min(6, int(math.floor(math.sqrt(n_particles))))
pts = trajectories[-1][:side_len**2]# - init_particles[:side_len**2]
fig = plt.figure(figsize=(9, 6), layout='constrained')
gs = fig.add_gridspec(side_len, side_len + 2)
vabs = max(pts.min().abs(), pts.max().abs())
plt_kwargs = {'vmin': -vabs, 'vmax': vabs, 'extent': (0, 8, 0, 8)}

for i in range(side_len):
    for j in range(side_len):
        ax = fig.add_subplot(gs[i, j])
        # ax.set_axis_off()
        ax.set_aspect('equal')
        t = ax.matshow(pts[i*side_len + j].reshape(8, 8), **plt_kwargs)
        # ax.vlines(jnp.arange(1,8), -0.1, 8.1, color='w', lw=0.75)
        # ax.hlines(jnp.arange(1,8), -0.1, 8.1, color='w', lw=0.75)
        ax.minorticks_on()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_minor_locator(ticker.MultipleLocator())
        ax.yaxis.set_minor_locator(ticker.MultipleLocator())
        ax.grid(which="both", linewidth=1.5, color="w")
        ax.tick_params(which="minor", length=0)
ax_cb = fig.add_subplot(gs[:-2, -2:])
ax_cb.set_title(r"Scale of $\log\theta$", y=0.6)
cax_cb = ax_cb.inset_axes((0.1, 0.45, 0.8, 0.1))
ax_cb.axis('off')
fig.colorbar(t, cax=cax_cb, orientation='horizontal')
ax_true = fig.add_subplot(gs[-2:, -2:])
ax_true.set_aspect('equal')
ax_true.matshow(ab.theta_true.log().reshape(8, 8), **plt_kwargs)
ax_true.minorticks_on()
ax_true.set_xticks([])
ax_true.set_yticks([])
ax_true.xaxis.set_minor_locator(ticker.MultipleLocator())
ax_true.yaxis.set_minor_locator(ticker.MultipleLocator())
ax_true.set_title(r"True $\theta$")
ax_true.grid(which="both", linewidth=1.5, color="w")
ax_true.tick_params(which="minor", length=0)
plt.show()

# %%
pts

# %%
np.linalg.norm(trajectories[4] - trajectories[3])

# %%
pts = trajectories[1000]
im = plt.imshow(torch.mean(torch.tensor(pts), 0).reshape(8, 8), vmin=-3, vmax=3)
plt.colorbar(im)

# %%
theta_true = torch.as_tensor(ab.theta_true, dtype=init_particles.dtype)
T_idx = 750
thetas = torch.exp(torch.tensor(trajectories[T_idx]))
N_solver, N_viz = 32, 128
solve_args = ab.build_forward_solver_args(N_solver, N_viz, dtype=thetas.dtype)
H_obs = solve_args[0]
u_true = ab.forward_solver(theta_true, N_solver, *solve_args)
u_solve = ab.forward_solver(thetas, N_solver, *solve_args)
# u_viz = H_obs
z_true = (H_obs @ u_true)
z_viz = (u_solve @ H_obs.T)

# %%
# viz_grid = torch.linspace(0, 1, N_viz)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sample_idx = 5
z_viz_mean = z_viz.mean(0)
axs[0].imshow(z_viz_mean.reshape(N_viz, N_viz).detach().numpy())
axs[0].set_title("Posterior predictive mean")
axs[1].imshow(z_true.reshape(N_viz, N_viz).detach().numpy())
axs[1].set_title("True simulation")
plt.show()

# %%
