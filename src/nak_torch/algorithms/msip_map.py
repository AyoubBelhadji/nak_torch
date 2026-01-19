
import torch
from typing import Optional, Callable
from .average import recursive_weighted_average_alpha_v
from .kernel import sqexp_kernel_matrix
from tqdm import tqdm

def msip_map(
        objective_function,
        particles: torch.Tensor,
        kernel_bandwidth: float = 1.0,
        bandwidth_factor: float = 0.5,
        bounds: tuple[float, float] = (-torch.inf, torch.inf),
        projection: bool = True,
        gradient_informed: bool = True,
        diag_infl: float = 0.0,
        output_idx: Optional[int] = None,
        get_kernel_matrix: Callable[[torch.Tensor, float], torch.Tensor] = sqexp_kernel_matrix,
        progress_bar: Optional[tqdm] = None
):
    """
    Compute the full MSIP map T(y) for all particles at once.
    Returns t_arr with shape (N, d).
    """
    n_particles = particles.shape[0]
    # Make sure this is a leaf with grad
    lower_bounds = torch.as_tensor(
        bounds[0], dtype=particles[0].dtype,
        device=particles[0].device
    )

    upper_bounds = torch.as_tensor(
        bounds[1], dtype=particles[0].dtype,
        device=particles[0].device
    )

    particles_leaf = particles.detach().clone()
    particles_leaf.requires_grad_(True)

    fitness = objective_function(particles_leaf)   # shape (N,)
    grads = None
    if gradient_informed:
        grads, = torch.autograd.grad(fitness.sum(), particles_leaf)

    with torch.no_grad():
        # (N, N)
        sigma2 = kernel_bandwidth ** 2
        kernel_matrix = get_kernel_matrix(particles_leaf, sigma2)

        if diag_infl > 0:
            kernel_matrix[torch.arange(n_particles), torch.arange(n_particles)] += diag_infl

        if diag_infl > 0.: # We know it's invertible if diag is inflated
            K_minus_one = torch.linalg.inv(kernel_matrix)
        else:
            K_minus_one = torch.linalg.pinv(kernel_matrix)
        N, d = particles_leaf.shape
        def apply_msip_once(out: torch.Tensor, i: int):
            alpha_i = K_minus_one[i, :]  # (N,)

            t1 = recursive_weighted_average_alpha_v(
                particles, alpha_i, log_v=fitness
            )
            if gradient_informed:
                assert grads is not None
                t2 = recursive_weighted_average_alpha_v(
                    grads, alpha_i, log_v=fitness
                )
                if projection:
                    torch.clamp(
                        t1 + bandwidth_factor * bandwidth_factor * sigma2 * t2,
                        min=lower_bounds, max=upper_bounds, out=out
                    )
                else:
                    torch.add(t1, bandwidth_factor * bandwidth_factor * sigma2 * t2, out=out)
            else:
                if projection:
                    torch.clamp(
                        t1, min=lower_bounds, max=upper_bounds, out=out
                    )
                else:
                    out.copy_(t1)
        if output_idx is None:
            t_arr = torch.empty_like(particles)
            for idx in range(N):
                apply_msip_once(t_arr[idx], idx)
                if progress_bar is not None:
                    progress_bar.update()
        else:
            t_arr = torch.empty_like(particles[0])
            apply_msip_once(t_arr, output_idx)

    return t_arr
