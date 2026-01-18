# import jax
# import jax.numpy as jnp
# from typing import Callable, Tuple, Any, Protocol
# from jaxtyping import Float, Array
# from nak.kernel import VectorMSKernel
# from nak.target import KernelizedTarget
# from nak.util import CentroidT, WeightT, PointT, QuantizationState
import torch
from torchtyping import TensorType
from typing import Callable
KernelType = Callable[[TensorType[" d"], TensorType[" d"]], TensorType["1"]]

def gaussian_kernel_elem(x: TensorType[" d"], y: TensorType[" d"], sigma_sq: float = 1.0):
    assert x.shape == y.shape and y.ndim == 1
    ret = torch.exp( - (x - y).square_().sum() / (2*sigma_sq))
    return ret

def build_kernel_diffs(kernel_fcn: KernelType):
    def process_kernel_fcn(x,y):
        ret = kernel_fcn(x,y)
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
        grad_log_p: Callable[[TensorType[" d"]], TensorType["1"]],
        kernel_fcn: KernelType,
        is_grad_vectorized: bool = False
) -> Callable[[TensorType[" batch", " d"]], TensorType[" batch", " batch"]]:
    kernel_diffs = build_kernel_diffs(kernel_fcn)
    grad_log_p_v: Callable[[TensorType]] = grad_log_p if is_grad_vectorized else torch.vmap(grad_log_p)
    def stein_kernel(pts: TensorType[" batch", " dim"]):
        grad_log_p_eval = grad_log_p_v(pts)
        trace_kernel, grad1_kernel, eval_kernel = kernel_diffs(pts, pts)
        grad_term_1 = torch.einsum("ijd,jd->ij", grad1_kernel, grad_log_p_eval)
        grad_term_2 = grad_term_1.T
        ev_term = torch.einsum("ij,id,jd->ij", eval_kernel, grad_log_p_eval, grad_log_p_eval)
        return trace_kernel + grad_term_1 + grad_term_2 + ev_term
    return stein_kernel
