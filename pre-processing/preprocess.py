from pathlib import Path

import numpy as np

from preparing_data.read_fits import read_fits
from preparing_data.cleaning_data import clean_data
from common.data_models import LightCurveData

from filtering.filter_data import apply_all_filters

def preprocess_file(file_path: Path):
  
  data = read_fits(file_path)
  
  data, stat = clean_data(data)
  
  data = apply_all_filters(data)
  
  return data, stat

def save_refined_curve(data: LightCurveData, save_path: Path):
  
  save_path.parent.mkdir(parents=True, exist_ok=True)
  
  np.savez_compressed(
    save_path,
    
    time=data.time,
    flux=data.flux,
    flux_error=data.flux_error,
    quality=data.quality,
    
    target_id=data.target_id,
    mission=data.mission,
    quarter=data.quarter,
    file_path=str(data.file_path),
    
    original_star = data.target_id,
    pipeline_version = 1,
    processed = True
    
    if data.file_path is not None
    else ""
  )
  
def preprocess_data(raw_root: Path, refined_root: Path):
  
  total_stars = 0
  total_files = 0
  failed_files = 0
  successful_files = 0
  original_len = 0
  current_len = 0
  nan_removed = 0
  inf_removed = 0
  bad_flags_removed = 0
  duplicates_removed = 0
  
  star_folders = [
    folder
    for folder in raw_root.iterdir()
    if folder.is_dir()
  ]
  
  for star_folder in star_folders:
    
    total_stars += 1
    
    print(f"\nProcessing"
          f"{star_folder.name}")
    
    
    refined_star_folder = (
      refined_root / f"P_{star_folder.name}"
    )
    
    refined_star_folder.mkdir(
      parents=True,
      exist_ok=True
    )
    
    fits_files = sorted(star_folder.glob("*fits"))
    
    for fits_file in fits_files:
      
      total_files += 1
      
      try:
        print(f" "
              f"{fits_file.name}")
        
        processed, stat = preprocess_file(fits_file)
        
        original_len += stat["original_points"]
        current_len += stat["final_points"]
        nan_removed += stat["nan_removed"]
        inf_removed += stat["inf_removed"]
        bad_flags_removed += stat["bad_flags_removed"]
        duplicates_removed += stat["duplicates_removed"]
        
        output_file = (
          refined_star_folder / f"P_{fits_file.stem}.npz"
        )
        
        save_refined_curve(
          processed, output_file
        )
        
        successful_files += 1
        
      except Exception as e:
        
        failed_files += 1
        
        print(f"Failed : "
              f"{fits_file.name}")
        
        print(e)

  print(f"""
  Cleaning Report
  -------------------------
  Original Points : {original_len}
  Final Points    : {current_len}

  NaN Removed         : {nan_removed}
  Inf Removed         : {inf_removed}
  Bad Flags Removed   : {bad_flags_removed}
  Duplicates Removed  : {duplicates_removed}
  """)
        
  print("\n========== SUMMARY ==========")
  print(f"Total Stars      : {total_stars}")
  print(f"Total Files      : {total_files}")
  print(f"Successful Files : {successful_files}")
  print(f"Failed Files     : {failed_files}")
  
preprocess_data(Path("./data/raw"), Path("./data/refined"))