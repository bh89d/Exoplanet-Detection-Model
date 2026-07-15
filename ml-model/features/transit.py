from common.data_models import LightCurveData
import numpy as np

def extract_transit_features(data: LightCurveData) -> dict:
  
  flux = data.flux
  
  deepest_dip = 1 - np.min(flux)
  
  threshold = np.median(flux) - np.std(flux) * 2
  
  significant_dips = np.sum(
      flux < threshold
  )
  
  dip_mask = flux < threshold
  
  dip_values = flux[dip_mask]
  
  dip_depth = 1 - dip_values
    
  if len(dip_depth) == 0:
    mean_dip_depth = 0
  else:
    mean_dip_depth = np.mean(dip_depth)
  
  num_dip_events = 0
  
  for i in range(len(dip_mask)-1):
    if (
    not dip_mask[i] and dip_mask[i+1]):
      num_dip_events += 1
  
  dip_fraction = significant_dips / len(flux)
  
  max_dip_duration = 0
  current_dip_duration = 0
  
  for i in range(len(dip_mask)):
    
    if dip_mask[i]:
      current_dip_duration += 1
    
    else:
      max_dip_duration = max(current_dip_duration, max_dip_duration)
      current_dip_duration = 0
      
  max_dip_duration = max(current_dip_duration, max_dip_duration)
      
  return {
    "deepest_dip" : deepest_dip,
    "num_dip_points" : significant_dips,
    "num_dip_events" : num_dip_events,
    "mean_dip_depth" : mean_dip_depth,
    "max_dip_duration" : max_dip_duration,
    "dip_fraction" : dip_fraction
  }