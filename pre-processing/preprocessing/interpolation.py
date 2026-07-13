import numpy as np
from preprocessing.data_models import LightCurveData

def interpolate_gaps(data: LightCurveData, 
                     max_missing_points: int = 3) -> LightCurveData:
  
  cadence = np.median(np.diff(data.time))
  
  gaps_interpolated = 0
  points_added = 0
  largest_gap = 0
  remaining_large_gaps = 0
  
  new_time = []
  new_flux = []
  new_flux_error = []
  new_quality = []
  
  for i in range(len(data.time) - 1):
    
    t1 = data.time[i]
    t2 = data.time[i+1]
    
    f1 = data.flux[i]
    f2 = data.flux[i+1]
    
    e1 = data.flux_error[i]
    e2 = data.flux_error[i+1]
    
    new_time.append(t1)
    new_flux.append(f1)
    new_flux_error.append(e1)
    new_quality.append(0)
    
    dt = t2 - t1
    
    missing_obs = int(round(dt / cadence) - 1)
    
    largest_gap = max(largest_gap, missing_obs)
    
    if (missing_obs > 0 and missing_obs <= max_missing_points):
      
      gaps_interpolated += 1
      points_added += missing_obs
      
      for j in range(1, missing_obs + 1):
        alpha = j / (missing_obs + 1)
        
        new_time.append(t1 + alpha*dt)
        
        new_flux.append(f1 + alpha*(f2-f1))
        
        new_flux_error.append(e1 + alpha*(e2-e1))
        
        new_quality.append(0)
    
    elif missing_obs > max_missing_points:
      remaining_large_gaps += 1
    
  
  new_time.append(data.time[-1])
  new_flux.append(data.flux[-1])
  new_flux_error.append(data.flux_error[-1])
  new_quality.append(0)
  
  interpolation_percentage = (
    points_added
    / len(data.time)
  ) * 100
  
  stats = {
    "points_added": points_added,
    "gaps_interpolated": gaps_interpolated,
    "largest_gap": largest_gap,
    "remaining_large_gaps":
        remaining_large_gaps,
    "interpolation_percent":
        interpolation_percentage,
  }
  
  interpolate_data = LightCurveData(
    time=np.array(new_time),
    flux=np.array(new_flux),
    flux_error=np.array(new_flux_error),
    quality=np.array(new_quality),
    
    target_id=data.target_id,
    mission=data.mission,
    quarter=data.quarter,
    file_path=data.file_path,
  )
  
  return interpolate_data, stats