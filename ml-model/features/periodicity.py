from astropy.timeseries import LombScargle
from common.data_models import LightCurveData
import numpy as np

def extract_periodicity_features(data: LightCurveData) -> dict:
  
  time = data.time
  flux = data.flux
  
  frequency, power = LombScargle(time, flux).autopower()
  
  max_power = max(power)
  
  max_power_idx = power.index(max_power)
  
  best_period = 1/frequency[max_power_idx]
  
  return {
    "best_period" : best_period,
    "max_power" : max_power,
    "periodicity_strength" : max_power / np.mean(power) 
  }
  
  