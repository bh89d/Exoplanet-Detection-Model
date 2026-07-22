from common.data_models import LightCurveData
import numpy as np

def extract_frequency_features(data: LightCurveData) -> dict:
  
  flux = data.flux
  
  dt = np.median(np.diff(data.time))
  
  flux = flux - np.mean(flux)
  
  fft = np.fft.rfft(flux)
  
  power = np.abs(fft)
  
  frequency = np.fft.rfftfreq(len(flux), d=dt)
  
  dominant_power_idx = np.argmax(power[1:]) + 1
  
  dominant_freq = frequency[dominant_power_idx]
  
  dominant_power = power[dominant_power_idx]
  
  signal_energy = np.sum(power**2)
  
  power_sum = np.sum(power)

  if power_sum == 0:
      entropy = 0
      spectral_centroid = 0
      spectral_spread = 0

  else:
      p = power / power_sum
      p = p[p > 0]

      entropy = -np.sum(
          p * np.log(p)
      )

      spectral_centroid = (
          np.sum(frequency * power)
          / power_sum
      )

      variance = (
          np.sum(
              power *
              (
                  frequency
                  - spectral_centroid
              )**2
          )
          / power_sum
      )

      spectral_spread = np.sqrt(
          variance
      )
  
  return {
    "dominant_frequency" : dominant_freq,
    "dominant_power" : dominant_power,
    "signal_energy" : signal_energy,
    "spectral_entropy" : entropy,
    "spectral_centroid" : spectral_centroid,
    "spectral_spread" : spectral_spread
  }