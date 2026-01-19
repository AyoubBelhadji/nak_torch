# import jax
# import jax.numpy as jnp
# from typing import Callable, Tuple, Any, Protocol
# from jaxtyping import Float, Array
# from nak.kernel import VectorMSKernel
# from nak.target import KernelizedTarget
# from nak.util import CentroidT, WeightT, PointT, QuantizationState
import torch
from jaxtyping import Float
from .types import KernelType, VecGradLogDensity, GradLogDensity, MatSelfKernelType
from torch import Tensor
__all__ = [
    "build_stein_kernel",
]

def build_kernel_diffs(kernel_fcn: KernelType, bandwidth: float):
    def process_kernel_fcn(x,y):
        ret = kernel_fcn(x,y,bandwidth)
        return ret, ret

    kernel_grad = torch.func.grad(process_kernel_fcn, has_aux=True, argnums=0)

    def process_kernel_grad(x, y):
        grad, aux = kernel_grad(x, y)
        return grad, (grad, aux)

    kernel_jac = torch.func.jacrev(process_kernel_grad, has_aux=True, argnums=1)

    def process_kernel_jac(x, y):
        jac, (grad, ev) = kernel_jac(x, y)
        return torch.trace(jac), grad, ev

    kernel_diffs = torch.vmap(
        torch.vmap(process_kernel_jac, in_dims = (0, None), out_dims = 0),
        in_dims = (None, 0), out_dims=1
    )
    return kernel_diffs

def build_stein_kernel(
        grad_log_p: GradLogDensity | VecGradLogDensity,
        kernel_fcn: KernelType,
        bandwidth: float,
        is_grad_vectorized: bool = False
) -> MatSelfKernelType:
    kernel_diffs = build_kernel_diffs(kernel_fcn, bandwidth)
    grad_log_p_v = grad_log_p if is_grad_vectorized else torch.vmap(grad_log_p)
    def stein_kernel(pts: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch batch"]:
        grad_log_p_eval = grad_log_p_v(pts)
        trace_kernel, grad1_kernel, eval_kernel = kernel_diffs(pts, pts)
        grad_term_1 = torch.einsum("ijd,jd->ij", grad1_kernel, grad_log_p_eval)
        grad_term_2 = grad_term_1.T
        ev_term = torch.einsum("ij,id,jd->ij", eval_kernel, grad_log_p_eval, grad_log_p_eval)
        return trace_kernel + grad_term_1 + grad_term_2 + ev_term
    return stein_kernel
