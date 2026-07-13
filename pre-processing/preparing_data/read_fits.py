from astropy.io import fits
from common.data_models import LightCurveData
from pathlib import Path

def read_fits(path: Path) -> LightCurveData:
  with fits.open(path) as hdu:

    header = hdu[0].header
    table = hdu[1].data
    
    time=table["TIME"]
    flux=table["FLUX"]
    flux_error=table["FLUX_ERR"]
    quality=table["SAP_QUALITY"]

    return LightCurveData(
      time=time,
      flux=flux,
      flux_error=flux_error,
      quality=quality,
      
      target_id=header.get("OBJECT", "Unknown"),
      mission=header.get("MISSION", "Unknown"),
      quarter=header.get("QUARTER"),
      
      file_path=path,
    )