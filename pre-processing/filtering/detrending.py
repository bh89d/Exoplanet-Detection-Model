import numpy as np
from scipy.signal import savgol_filter
from common.data_models import LightCurveData

def detrend_flux(data: LightCurveData, window_length: int=401, polyorder: int = 2):
  
  window_length = min(
      window_length,
      len(data.flux)-1
  )

  if (
      window_length % 2 == 0
  ):
      window_length -= 1

  trend = savgol_filter(
    data.flux,
    window_length,
    polyorder
  )
  
  trend[trend == 0] = 1
  
  detrended_flux = (data.flux/trend)
  
  return LightCurveData(
        time=data.time,
        flux=detrended_flux,
        flux_error=(
            data.flux_error / trend
        ),
        quality=data.quality,

        target_id=data.target_id,
        mission=data.mission,
        quarter=data.quarter,

        file_path=data.file_path,
    )

