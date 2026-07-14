from common.data_models import LightCurveData
from pathlib import Path
import numpy as np

def load_refined_curve(path: Path):
  
  data = np.load(path, allow_pickle=True)
  
  return LightCurveData(
    time=data["time"],
    flux=data["flux"],
    flux_error=data["flux_error"],
    quality=data["quality"],

    target_id=str(data["target_id"]),
    mission=str(data["mission"]),
    quarter=int(data["quarter"]),
    file_path=Path(str(data["file_path"]))
  )
  

