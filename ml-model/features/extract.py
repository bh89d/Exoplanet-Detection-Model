from features.statistical import extract_statistical_features
from features.transit import extract_transit_features
from features.periodicity import extract_periodicity_features
from features.frequency import extract_frequency_features
from common.data_models import LightCurveData

def extract_features(data: LightCurveData) -> dict:
  
  features = {}
  
  features.update(extract_statistical_features(data))
  
  features.update(extract_transit_features(data))
  
  features.update(extract_periodicity_features(data))
  
  features.update(extract_frequency_features(data))
  
  return features