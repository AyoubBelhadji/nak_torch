import torch
from torch import Tensor
from jaxtyping import Float

@torch.compile
def recursive_weighted_average_alpha_v(
        y: Float[Tensor, "batch dim"],
        alpha: Float[Tensor, " batch"],
        log_v: Float[Tensor, " batch"],
        eps: float = 1e-18
) -> tuple[Float[Tensor, "batch dim"], Float[Tensor, " batch"]]:
    r"""
    Compute a stable weighted average
        $$z_j = \sum_i v_i a_i y_ij / \sum_i v_i a_i$$
    using log-weights
    y: (N, d)   the containing the vectors $y_i$
    alpha: (N,) the array of arbitrary weights
    v: (N,) or log_v: (N,) the array of postive weights

    Returns:
    z: (d,) see above
    w: (1,) defined by $\sum_i v_i a_i
    """
    N, d = y.shape
    if alpha.ndim != 1 or N != alpha.shape[0]:
        raise ValueError(f"Invalid alpha dimensions {alpha.shape}")

    y = torch.as_tensor(y)
    alpha = torch.as_tensor(alpha)
    N, d = y.shape

    # Check the 'mode' of weighting:
    # 1) v is given
    # 2) log_v is given
    # 3) v nor log_v is given

    # Look for the non-vanishing a_i != 0, and restrict the average to those
    # nonzero_alpha_mask = (alpha != 0)
    # if nonzero_alpha_mask.sum() == 0:
    #     raise ValueError("All alpha are zero.")
    # y = y[nonzero_alpha_mask]
    # alpha = alpha[nonzero_alpha_mask]
    # log_v = log_v[nonzero_alpha_mask]

    # Compute log |w_i|:= log |a_i| + log |v_i|  and sign of the a_i
    log_abs_alpha = torch.log(alpha.abs())
    logw = log_abs_alpha + log_v
    sign = alpha.sign()

    # Look for max log |w_i|
    z_max = logw.max()

    # Calculate stable weights
    exp_scaled = torch.exp(logw - z_max)

    # Calculate the denominator
    weighted_signs = sign * exp_scaled
    denominator = weighted_signs.sum()

    # if denominator.abs() < eps:
    #     raise ValueError("Sum of weights too close to zero.")

    # Calculate the numerator
    numerator = weighted_signs @ y

    # Calculate the ratio
    weighted_average = numerator / denominator

    return weighted_average, denominator.log()
