from astropy.timeseries import LombScargle
from common.data_models import LightCurveData
import numpy as np

def extract_periodicity_features(data: LightCurveData) -> dict:
  
  time = data.time
  flux = data.flux
   
  frequency, power = LombScargle(time, flux).autopower()
  
  max_power = max(power)
  
  max_power_idx = np.argmax(power)
  
  best_frequency = frequency[max_power_idx]

  if best_frequency == 0:
      best_period = np.inf
  else:
      best_period = 1 / best_frequency
      
  phase = (time % best_period) / best_period
  
  idx = np.argsort(phase)
  
  phase = phase[idx]
  phase_flux = flux[idx]
  
  phase_deepest_dip = 1 - np.min(phase_flux)
  
  phase_threshold = np.median(phase_flux) - 2*np.std(phase_flux)
  
  phase_dip_fraction = np.sum(phase_flux < phase_threshold) / len(phase_flux)
  
  phase_mask = phase_flux < phase_threshold
  
  max_duration = 0
  current_duration = 0

  for i in range(len(phase_mask)):

      if phase_mask[i]:
          current_duration += 1

      else:
          max_duration = max(
              max_duration,
              current_duration
          )

          current_duration = 0

  phase_max_duration = max(
      max_duration,
      current_duration
  )
  
  phase_duration = phase_max_duration * best_period / len(flux)
  
  dip = phase_flux[phase_mask]
  
  mid = len(dip)//2

  left = dip[:mid]

  right = dip[mid:][::-1]

  n = min(len(left),len(right))

  phase_symmetry = np.mean(np.abs(left[:n]-right[:n]))
  
  phase_noise = np.median(np.abs(phase_flux - np.median(phase_flux)))
  
  phase_transit_snr = phase_deepest_dip / phase_noise

    
  return {
    "best_period" : best_period,
    "max_power" : max_power,
    "periodicity_strength" : (max_power - np.mean(power)) / np.std(power),
    "phase_deepest_dip" : phase_deepest_dip,
    "phase_dip_fraction" : phase_dip_fraction,
    "phase_duration" : phase_duration,
    "phase_symmetry" : phase_symmetry,
    "phase_transit_snr" : phase_transit_snr
  }
  
  