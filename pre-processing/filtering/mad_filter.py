import numpy as np
from common.data_models import LightCurveData

def mad_filter(data: LightCurveData, threshold: float = 7.0, replacement_window: int = 5):
  flux = data.flux.copy()
  
  median_flux = np.median(flux)
  
  mad = np.median(np.abs(flux - median_flux))
  
  if mad == 0:
    stats = {
      "outliers_found" : 0,
      "mad" : 0
    }
    
    return data, stats
  
  modified_z = (
    0.6745 * (flux - median_flux) / mad
  )
  
  outlier_mask = (np.abs(modified_z) > threshold)
  
  outlier_indices = np.where(
    outlier_mask
  )[0]
  
  for idx in outlier_indices:
    left = max(0, idx-replacement_window)
    
    right = min(len(flux),
            idx + replacement_window + 1)
    
    local_flux = flux[left:right]
    
    local_flux = local_flux[~outlier_mask[left:right]]
    
    if len(local_flux) == 0:
      continue
    
    flux[idx] = np.median(
      local_flux
    )
    
  stats = {
    "mad" : mad,
    "threshold" : threshold,
    "outliers_found" : len(outlier_indices),
    "outlier_indices" : outlier_indices.tolist()
  }
  
  filtered_data = LightCurveData(
          time=data.time,
          flux=flux,
          flux_error=data.flux_error,
          quality=data.quality,

          target_id=data.target_id,
          mission=data.mission,
          quarter=data.quarter,

          file_path=data.file_path,
      )
  
  return filtered_data, stats  
