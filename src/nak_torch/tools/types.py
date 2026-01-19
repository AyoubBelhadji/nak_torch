
from torch import Tensor
from jaxtyping import Float
from typing import Callable, Optional

BatchType = Float[Tensor, "batch"]
PtType = Float[Tensor, " d"]
BatchPtType = Float[Tensor, "batch d"]
KernelMatrixType = Float[Tensor, "batch batch"]

KernelFunction = Callable[[PtType, PtType, float], Float]

MatSelfKernelFunction = Callable[[
    BatchPtType, float, Optional[BatchPtType]
], KernelMatrixType]

LogDensity = Callable[[PtType], Float]

GradLogDensity = Callable[[PtType], PtType]

BatchLogDensity = Callable[[BatchPtType], BatchType]

BatchGradLogDensity = Callable[[BatchPtType], BatchPtType]
