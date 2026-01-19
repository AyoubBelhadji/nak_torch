# # import jax
# # import jax.numpy as jnp
# # from typing import Callable, Tuple, Any, Protocol
# # from jaxtyping import Float, Array
# # from nak.kernel import VectorMSKernel
# # from nak.target import KernelizedTarget
# # from nak.util import CentroidT, WeightT, PointT, QuantizationState
# import torch
# from jaxtyping import Float
# from .types import PtType, KernelType, BatchGradLogDensity, GradLogDensity, MatSelfKernelType
# from torch import Tensor
# from typing import Callable
# __all__ = [
#     "build_stein_kernel",
# ]

# def build_kernel_diffs(
#         kernel_fcn: KernelType,
# ) -> Callable[
#     [PtType, PtType, Float], tuple[Float, PtType, PtType, Float]
# ]:
#     kernel_grad = torch.func.grad_and_value(kernel_fcn, has_aux=True, argnums=(0, 1))

#     def process_kernel_grad(x, y, bandwidth):
#         (grad1, grad2), aux = kernel_grad(x, y, bandwidth)
#         return grad1, (grad1, grad2, aux)

#     kernel_jac = torch.func.jacrev(process_kernel_grad, has_aux=True, argnums=1)

#     def process_kernel_jac(x, y, bandwidth):
#         jac, (grad1, grad2, ev) = kernel_jac(x, y, bandwidth)
#         return torch.trace(jac), grad1, grad2, ev

#     return process_kernel_jac

# def build_stein_kernel(
#         grad_log_p: GradLogDensity | BatchGradLogDensity,
#         kernel_fcn: KernelType,
#         is_grad_vectorized: bool = False
# ) -> KernelType:
#     kernel_diffs = build_kernel_diffs(kernel_fcn)
#     def stein_kernel(p1: PtType, p2: PtType, bandwidth: Float) -> Float:
#         grad_log_p_eval1: PtType
#         grad_log_p_eval2: PtType
#         if is_grad_vectorized:
#             grad_log_p_eval1 = grad_log_p(p1.reshape(1,-1)).flatten()
#             grad_log_p_eval2 = grad_log_p(p2.reshape(1,-1)).flatten()
#         else:
#             grad_log_p_eval1 = grad_log_p(p1)
#             grad_log_p_eval2 = grad_log_p(p2)
#         diffs_eval = kernel_diffs(p1, p2, bandwidth)
#         trace_kernel, grad1_kernel, grad2_kernel, eval_kernel = diffs_eval
#         grad_term1 = grad1_kernel.dot(grad_log_p_eval2)
#         grad_term2 = grad2_kernel.dot(grad_log_p_eval1)
#         return trace_kernel + grad_term1 + grad_term2 + eval_kernel

#     return stein_kernel
