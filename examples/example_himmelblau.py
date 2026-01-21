# %%
from nak_torch.tools import animate_trajectories_box
from nak_torch.functions import himmelblau
from nak_torch.algorithms import msip_greedy
from datetime import datetime

save_gif = True
now_stamp = datetime.now()
algorithm_name = "msip_ni"
function_name = "himmelblau"
log_density = himmelblau(50.0)

trajectories = msip_greedy(
    log_density,
    n_particles=10,
    n_steps=50,          # now interpreted as "epochs" (passes over all particles)
    dim=2,
    bounds=(-8, 8),
    lr=0.25,
    noise=0.05,          # currently unused, kept for compatibility
    kernel_bandwidth=0.9,
    inner_tol=1e-4,      # equilibrium tolerance for a particle
    max_inner_steps=1000,  # max inner iterations per particle
    seed=None,
    device="cpu"
)

if save_gif:
    bounds = [-15,15]
    fpath = f"results/gif/{algorithm_name}_{function_name}_particles_{now_stamp}.gif"
    animate_trajectories_box(
        log_density, trajectories, 50, bounds,
        save_path=fpath
    )


# %%
