from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass
class LightCurveData:
  """Container for a Light Curve"""
  time: np.ndarray
  flux: np.ndarray
  flux_error: np.ndarray
  quality: np.ndarray

  target_id: str
  mission: str
  quarter: int | None

  file_path: Path
  