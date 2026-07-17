from common.data_models import LightCurveData
import numpy as np

def extract_transit_features(data: LightCurveData) -> dict:
  
  flux = data.flux
  time = data.time
  
  dip_segments = []
  
  current = []
  
  dip_depths = []
  
  dip_durations = []
  
  num_dip_points = 0
  
  threshold = np.median(flux) - np.std(flux) * 2
  
  dip_mask = flux < threshold
  
  for i in range(len(flux)):
    
    if dip_mask[i]:
      current.append(flux[i])
      
    else:
      if len(current) > 0:
        dip_segments.append(current)
        current = []
        
  if len(current) > 0:
    dip_segments.append(current)
    
  symmetries = []
  
  for dip in dip_segments:
    if len(dip) < 4:
      continue
    
    mid = len(dip)//2
    
    left = np.array(dip[:mid])
    
    right = np.array(dip[mid:][::-1])
    
    n=min(len(left), len(right))
    
    symmetry = np.mean(np.abs(left[:n] - right[:n]))
    
    symmetries.append(symmetry)
    
  if len(symmetries) == 0:
      mean_symmetry = 0
      std_symmetry = 0
      min_symmetry = 0
  else:
      mean_symmetry = np.mean(symmetries)
      std_symmetry = np.std(symmetries)
      min_symmetry = np.min(symmetries)
  
  for dip in dip_segments:
    
    depth = np.median(flux) - np.min(dip)
    
    duration = len(dip)
    
    num_dip_points += duration
    
    dip_depths.append(depth)
    dip_durations.append(duration)
    
  depth_std = np.std(dip_depths)
  
  depth_mean = np.mean(dip_depths)

  depth_cv = (
      depth_std / depth_mean
      if depth_mean > 0
      else 0
  )
  
  duration_std = np.std(dip_durations)
  
  duration_mean = np.mean(dip_durations)

  duration_cv = (
    duration_std / duration_mean
    if duration_mean > 0
    else 0
  )
  
  deepest_dip = 1 - np.min(flux)
  
  noise = np.median(np.abs(flux - np.median(flux)))
  
  if noise > 0:
    simple_transit_snr = deepest_dip / noise
    enhanced_transit_snr = (
        deepest_dip
        * np.sqrt(num_dip_points)
        / noise
    )
  else:
    simple_transit_snr = 0
    enhanced_transit_snr = 0
  
  significant_dips = np.sum(
      flux < threshold
  )
  
  cadence = np.median(np.diff(time))
  
  dip_values = flux[dip_mask]
  
  dip_depth = 1 - dip_values
    
  if len(dip_depth) == 0:
    mean_dip_depth = 0
  else:
    mean_dip_depth = np.mean(dip_depth)
  
  num_dip_events = 0
  
  num_dip_events = len(dip_segments)
  
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
  
  duration_days = max_dip_duration * cadence
  
  auto_correlation = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode="full")
  
  center = len(auto_correlation)//2

  if auto_correlation[center] != 0:
    auto_correlation /= auto_correlation[center]
      
  acf = auto_correlation
  
  if len(acf) > 1:
    acf_max = np.max(acf[1:])
    acf_peak_lag = np.argmax(acf[1:]) + 1
  else:
    acf_max = 0
    acf_peak_lag = 0
  
  acf_mean = np.mean(acf)
  
  acf_std = np.std(acf)
      
  return {
    "deepest_dip" : deepest_dip,
    "num_dip_points" : num_dip_points,
    "num_dip_events" : num_dip_events,
    "mean_dip_depth" : mean_dip_depth,
    "max_dip_duration" : max_dip_duration,
    "dip_fraction" : dip_fraction,
    "transit_depth_ppm" : deepest_dip * 1e6,
    "mean_symmetry" : mean_symmetry,
    "min_symmetry" : min_symmetry,
    "std_symmetry" : std_symmetry,
    "depth_std" : depth_std,
    "depth_cv" : depth_cv,
    "duration_std" : duration_std,
    "duration_cv" : duration_cv,
    "simple_transit_snr" : simple_transit_snr,
    "enhanced_transit_snr" : enhanced_transit_snr,
    "duration_days" : duration_days,
    "acf_max" : acf_max,
    "acf_peak_lag" : acf_peak_lag,
    "acf_mean" : acf_mean,
    "acf_std" : acf_std
  }