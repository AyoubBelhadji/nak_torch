from dataclasses import dataclass
from typing import Optional
from nak_torch.tools import GaussianModel
from nak_torch.tools.types import BatchPtType, BatchLogDensity
from .aristoff_bangerth import build_aristoff_bangerth

@dataclass
class Problem:
    model: GaussianModel | BatchLogDensity
    reference_samples: Optional[BatchPtType]

aristoff_bangerth = Problem(build_aristoff_bangerth(), None)
