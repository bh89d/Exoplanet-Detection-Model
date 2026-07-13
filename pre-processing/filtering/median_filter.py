from scipy.signal import medfilt
from preparing_data.data_models import LightCurveData
import numpy as np

def median_filter(data: LightCurveData, kernel_size: int=7):
  
  filtered_flux = medfilt(data.flux.astype(np.float64), kernel_size)
  
  return LightCurveData(
    time=data.time,
    flux=filtered_flux,
    flux_error=data.flux_error,
    quality=data.quality,

    target_id=data.target_id,
    mission=data.mission,
    quarter=data.quarter,

    file_path=data.file_path,
  )