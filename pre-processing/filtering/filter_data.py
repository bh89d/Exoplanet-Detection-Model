from filtering.mad_filter import mad_filter
from filtering.detrending import detrend_flux
from filtering.median_filter import median_filter
from filtering.savgol_filter import apply_savgol_filter
from common.data_models import LightCurveData


def apply_all_filters(data: LightCurveData):
  
  mad_filtered_data, stat = mad_filter(data)

  detrended = detrend_flux(mad_filtered_data)

  medianed = median_filter(detrended)

  smoothed = apply_savgol_filter(data=medianed)

  return smoothed
  
  
def filter_data_user(data: LightCurveData):
  print("""
        Apply Filters :\n
        1. Apply MAD Filter\n
        2. Apply Detrending\n
        3. Apply Median Filter\n
        4. Apply Savitzky-Golay filter\n
        5. Apply all\n
        6. Exit\n\n
        """)
  
  filter_choice = int(input("Input : "))
  
  while True:
    if filter_choice == 1:
      mad_filtered_data, stat = mad_filter(data)
      return mad_filtered_data
    
    elif filter_choice == 2:
      return detrend_flux(data)
    
    elif filter_choice == 3:
      return median_filter(data)
    
    elif filter_choice == 4:
      return apply_savgol_filter(data)
    
    elif filter_choice == 5:
      mad_filtered_data, stat = mad_filter(data)

      detrended = detrend_flux(mad_filtered_data)

      medianed = median_filter(detrended)

      smoothed = apply_savgol_filter(data=medianed)

      return smoothed
    
    elif filter_choice == 6:
      break
    
    else: 
      print("Invalid Input")