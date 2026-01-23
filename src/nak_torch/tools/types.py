import torch
from torch import Tensor
from jaxtyping import Float
from typing import Callable, Optional
from dataclasses import dataclass

BatchType = Float[Tensor, "batch"]
PtType = Float[Tensor, " d"]
BatchPtType = Float[Tensor, "batch d"]
QuadrulePtType = Float[Tensor, "quad d"]
QuadruleWtType = Float[Tensor, "quad"]
BatchQuadrulePtType = Float[Tensor, "batch quad d"]
BatchQuadruleWtType = Float[Tensor, "batch quad"]
KernelMatrixType = Float[Tensor, "batch batch"]

KernelFunction = Callable[[PtType, PtType, float], Float]

MatSelfKernelFunction = Callable[[
    BatchPtType, float, Optional[BatchPtType]
], KernelMatrixType]

LogDensity = Callable[[PtType], Float]

GradLogDensity = Callable[[PtType], PtType]

BatchLogDensity = Callable[[BatchPtType], BatchType]

BatchLogDensityGradVal = Callable[[BatchPtType], tuple[BatchPtType, BatchType]]

MultiBatchLogDensity = Callable[[BatchQuadrulePtType], BatchQuadruleWtType]

BatchGradLogDensity = Callable[[BatchPtType], BatchPtType]

MultiBatchLogDensityGradVal = Callable[[BatchQuadrulePtType], tuple[BatchQuadrulePtType, BatchQuadruleWtType]]

BatchQuadratureRule = Callable[[int], tuple[BatchQuadrulePtType, BatchQuadruleWtType]]

ForwardModel = Callable[
    [Float[Tensor, " dim"]], Float[Tensor, " obs"]
]

BatchForwardModel = Callable[
    [Float[Tensor, "batch dim"]], Float[Tensor, "batch obs"]
]

@dataclass
class GaussianModel:
    forward_model: BatchForwardModel
    likelihood_precision: float | Float[Tensor, "obs obs"]
    prior_precision: float | Float[Tensor, "dim dim"]
    true_obs: Float | Float[Tensor, " obs"]
    prior_mean: float | Float[Tensor, " dim"]
    def __init__(
            self,
            forward_model: ForwardModel | BatchForwardModel,
            likelihood_precision: float | Float[Tensor, "obs obs"] = 1.0,
            prior_precision: float | Float[Tensor, "dim dim"] = 1.0,
            true_obs: Float | Float[Tensor, " obs"] = torch.zeros(()),
            prior_mean: float | Float[Tensor, " dim"] = 0.0,
            is_vectorized: bool = False
    ):
        if is_vectorized:
            self.forward_model = forward_model
        else:
            self.forward_model = torch.vmap(forward_model)
        if prior_mean != 0.0:
            raise ValueError("Only support zero prior mean for now")
        self.likelihood_precision = likelihood_precision
        self.prior_precision = prior_precision
        self.true_obs = true_obs
        self.prior_mean = prior_mean
