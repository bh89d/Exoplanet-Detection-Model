from common.data_models import LightCurveData
import numpy as np

def extract_transit_features(data: LightCurveData) -> dict:
  
  flux = data.flux
  time = data.time
  
  dip_segments = []
  
  dip_time_segments = []
  
  dip_centers = []
  
  current_flux = []
  
  current_time = []
  
  dip_depths = []
  
  dip_durations = []
  
  flatness_scores = []
  
  ingress_egress_ratios = []
  
  dip_areas = []
  
  num_dip_points = 0
  
  threshold = np.median(flux) - np.std(flux) * 2
  
  dip_mask = flux < threshold
  
  for i in range(len(flux)):
    
    if dip_mask[i]:
      current_flux.append(flux[i])
      current_time.append(time[i])
      
    else:
      if len(current_flux) > 0:
        dip_segments.append(current_flux)
        dip_time_segments.append(current_time)
        current_flux = []
        current_time = []
        
  if len(current_flux) > 0:
    dip_segments.append(current_flux)
  
  if len(current_time) > 0:
    dip_time_segments.append(current_time)
    
  for dip_times in dip_time_segments:
    dip_centers.append(np.mean(dip_times))
    
  num_dip_events = len(dip_segments)
  
  if len(dip_centers) >= 2:

      dip_gap = np.diff(dip_centers)

      mean_dip_gap = np.mean(dip_gap)
      std_dip_gap = np.std(dip_gap)

      dip_gap_cv = (
          std_dip_gap / mean_dip_gap
          if mean_dip_gap > 0
          else 0
      )

  else:
      mean_dip_gap = 0
      std_dip_gap = 0
      dip_gap_cv = 0
  
  if num_dip_events < 2:
    std_dip_gap = 0
    dip_gap_cv = 0
  
  
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

  dip_fraction = significant_dips / len(flux)
  
  max_dip_duration = 0
  min_dip_duration = np.inf
  current_dip_duration = 0
  
  for i in range(len(dip_mask)):
    
    if dip_mask[i]:
      current_dip_duration += 1
    
    else:
      max_dip_duration = max(current_dip_duration, max_dip_duration)
      min_dip_duration = min(current_dip_duration, min_dip_duration)
      current_dip_duration = 0
      
  max_dip_duration = max(current_dip_duration, max_dip_duration)
  
  min_dip_duration = min(current_dip_duration, min_dip_duration)
  
  if min_dip_duration == np.inf:
    min_dip_duration = 0
  
  mean_dip_duration = np.mean(dip_durations)

  std_dip_duration = np.std(dip_durations)
  
  duration_days = max_dip_duration * cadence
  
  auto_correlation = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode="full")
  
  center = len(auto_correlation)//2

  if auto_correlation[center] != 0:
    auto_correlation /= auto_correlation[center]
      
  acf = auto_correlation[len(auto_correlation)//2:]
  
  if len(acf) > 1:
    acf_max = np.max(acf[1:])
    acf_peak_lag = np.argmax(acf[1:]) + 1
  else:
    acf_max = 0
    acf_peak_lag = 0
  
  acf_mean = np.mean(acf)
  
  acf_std = np.std(acf)
  
  v_shape_score = deepest_dip / max_dip_duration
  
  for dip in dip_segments:
    
    dip_area = np.sum(1-np.array(dip))
    
    dip_areas.append(dip_area)
    
    
    
    center = np.argmin(dip)
    
    left = dip[:center]
    
    right = dip[center + 1:]
    
    ingress_duration = len(left)
    
    egress_duration = len(right)
    
    ingress_egress_ratio = min(ingress_duration, egress_duration) / max(egress_duration, ingress_duration)
    
    ingress_egress_ratios.append(ingress_egress_ratio)
    
    
    
    dip = np.array(dip)
    
    depth = 1 - np.min(dip)
    
    threshold = np.min(dip) + 0.1*depth
    
    flat_region = dip[dip<=threshold]
    
    if len(flat_region) > 1 :
      flatness = np.std(flat_region)
      
      flatness_scores.append(flatness)
      
  mean_bottom_flatness = np.mean(flatness_scores)
  
  mean_ingress_egress_ratio = np.mean(ingress_egress_ratios)
  
  min_ingress_egress_ratio = np.min(ingress_egress_ratios)
  
  std_ingress_egress_ratio = np.std(ingress_egress_ratios)
  
  mean_dip_area = np.mean(dip_areas)
  
  max_dip_area = np.max(dip_areas)
  
  std_dip_area = np.std(dip_areas)
      
  return {
    "deepest_dip" : deepest_dip,
    "num_dip_points" : num_dip_points,
    "num_dip_events" : num_dip_events,
    "mean_dip_depth" : mean_dip_depth,
    "max_dip_duration" : max_dip_duration,
    "min_dip_duration" : min_dip_duration,
    "mean_dip_duration" : mean_dip_duration,
    "std_dip_duration" : std_dip_duration,
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
    "acf_std" : acf_std,
    "mean_dip_gap" : mean_dip_gap,
    "std_dip_gap" : std_dip_gap,
    "dip_gap_cv" : dip_gap_cv,
    "mean_bottom_flatness" : mean_bottom_flatness,
    "mean_ingress_egress_ratio" : mean_ingress_egress_ratio,
    "min_ingress_egress_ratio" : min_ingress_egress_ratio,
    "std_ingress_egress_ratio" : std_ingress_egress_ratio,
    "v_shape_score" : v_shape_score,
    "mean_dip_area" : mean_dip_area,
    "max_dip_area" : max_dip_area,
    "std_dip_area" : std_dip_area
  }