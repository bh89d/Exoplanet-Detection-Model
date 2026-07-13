from scipy.signal import savgol_filter
from preparing_data.data_models import LightCurveData

def apply_savgol_filter(data: LightCurveData, window_length: int = 11, polyorder: int = 2):
  
  window_length = min(window_length, len(data.flux) - 1)
  
  if window_length % 2 == 0:\
    window_length -= 1
    
  if window_length <= polyorder:
    return data
  
  filtered_flux = savgol_filter(
    data.flux,
    window_length,
    polyorder
  )
  
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