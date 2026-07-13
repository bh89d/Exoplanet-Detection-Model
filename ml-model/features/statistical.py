from common.data_models import LightCurveData
import numpy as np
from scipy.stats import (skew, kurtosis)

def extract_statistical_features(data: LightCurveData) -> dict:
  
  flux = data.flux
  
  mean_flux = np.mean(flux)
  
  median_flux = np.median(flux)
  
  std = np.std(flux)
  
  var_flux = np.var(flux)
  
  min_flux = np.min(flux)
  
  max_flux = np.max(flux)
  
  range_flux = max_flux - min_flux
  
  rms_flux = np.sqrt(np.mean(flux**2))
  
  mad_flux = np.median(np.abs(flux - median_flux))
  
  skew_flux = skew(flux)
  
  kurtosis_flux = kurtosis(flux)
  
  flux_p5 = np.percentile(flux, 5)
  
  flux_25 = np.percentile(flux, 25)
  
  flux_75 = np.percentile(flux, 75)
  
  flux_95 = np.percentile(flux, 95)
  
  return {
    "mean" : mean_flux,
    "median" : median_flux,
    "standard_deviation" : std,
    "min" : min_flux,
    "max" : max_flux,
    "variance" : var_flux,
    "range" : range_flux,
    "rms" : rms_flux,
    "median_abs_deviation" : mad_flux,
    "skew" : skew_flux,
    "kurtosis" : kurtosis_flux,
    "p5" : flux_p5,
    "p25" : flux_25,
    "p75" : flux_75,
    "p95" : flux_95
  }
  