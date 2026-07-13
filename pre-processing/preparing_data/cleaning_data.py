from common.data_models import LightCurveData
import numpy as np

def apply_mask(data: LightCurveData, mask: np.ndarray) -> LightCurveData:
  return LightCurveData(
    time=data.time[mask],
    flux=data.flux[mask],
    flux_error=data.flux_error[mask],
    quality=data.quality[mask],
    
    target_id=data.target_id,
    mission=data.mission,
    quarter=data.quarter,
    file_path=data.file_path
  )

def remove_nan_rows(data: LightCurveData) -> LightCurveData:
  mask = (
    ~np.isnan(data.time)
    &
    ~np.isnan(data.flux)
    &
    ~np.isnan(data.flux_error)
    &
    ~np.isnan(data.quality)
  )
  
  return apply_mask(data=data, mask=mask)

def remove_inf_values(data: LightCurveData) -> LightCurveData:
  mask = (
    ~np.isinf(data.time)
    &
    ~np.isinf(data.flux)
    &
    ~np.isinf(data.flux_error)
    &
    ~np.isinf(data.quality)
  )
  
  return apply_mask(data=data, mask=mask)

def remove_bad_flags(data: LightCurveData) -> LightCurveData:
  mask = data.quality == 0
  
  return apply_mask(data=data, mask=mask)

def remove_duplicates(data: LightCurveData) -> LightCurveData:
  _, indices = np.unique(
    data.time,
    return_index=True
  )
  
  mask = np.zeros(len(data.time), dtype=bool)
  
  mask[indices] = True
  
  return apply_mask(data=data, mask=mask)

def clean_data(data: LightCurveData):
  
  original_len = len(data.time)
  
  current_len = len(data.time)
  
  data = remove_nan_rows(data=data)
  nan_removed = current_len - len(data.time)
  
  current_len = len(data.time)
  data = remove_inf_values(data=data)
  inf_removed = current_len - len(data.time)
  
  current_len = len(data.time)
  data = remove_bad_flags(data=data)
  bad_flags_removed = current_len - len(data.time)
  
  current_len = len(data.time)
  data = remove_duplicates(data=data)
  duplicates_removed = current_len - len(data.time)
  
  stats = {
    "original_points" : original_len,
    "final_points" : current_len,
    "nan_removed" : nan_removed,
    "inf_removed" : inf_removed,
    "bad_flags_removed" : bad_flags_removed,
    "duplicates_removed" : duplicates_removed
  }
  
  return data, stats