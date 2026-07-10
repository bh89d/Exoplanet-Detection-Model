import numpy as np
from data_models import LightCurveData

def normalize_flux(data: LightCurveData) -> LightCurveData:
  
  if median_flux == 0:
    raise ValueError(
        f"Median flux is zero for {data.target_id}"
    )

    
  median_flux = np.median(data.flux)
  
  normalised_flux = data.flux / median_flux
  
  normalised_flux_error = data.flux_error / median_flux
  
  return LightCurveData(
    time=data.time,
    flux=normalised_flux,
    flux_error=normalised_flux_error,
    quality=data.quality,
    
    target_id=data.target_id,
    mission=data.mission,
    quarter=data.quarter,
    
    file_path=data.file_path,
  )