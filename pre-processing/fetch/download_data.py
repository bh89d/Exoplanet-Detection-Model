import lightkurve as lk
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
import warnings
from astropy.time import TimeDeltaMissingUnitWarning

warnings.filterwarnings(
    "ignore",
    category=TimeDeltaMissingUnitWarning
)


def run_query(query: str) -> pd.DataFrame:

    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    response = requests.get(
        url,
        params={
            "query": query,
            "format": "csv"
        }
    )

    response.raise_for_status()

    return pd.read_csv(StringIO(response.text))

def get_positive_star_list(limit: int):

    query = f"""
        SELECT DISTINCT hostname
            hostname
        FROM ps
        WHERE disc_facility = 'Kepler'
    """

    df = run_query(query)

    return sorted(df["hostname"].unique().tolist())

def get_negative_star_list(limit: int):

    query = f"""
        SELECT TOP {limit}
            kepid
        FROM keplerstellar
        WHERE nkoi = 0
    """

    df = run_query(query)

    return (
        "KIC "
        + df["kepid"].astype(str)
    ).tolist()

def download_lightcurves(stars: list[str], mission: str, author: str, exptime: int, savepath: str):
    
    for star in stars:

      print(f"\nDownloading {star}...")   

      lc_search = lk.search_lightcurve(target=star, mission=mission, author=author, exptime= exptime)

      star_folder = Path(savepath) / (star.replace(" ", "_"))
      star_folder.mkdir(parents=True, exist_ok=True)

      fits_files = list(star_folder.glob("*.fits"))

      if len(fits_files) == len(lc_search):
          print(f"{star} already downloaded. Skipping...")
          continue


      for product in lc_search:
        
        try:
          lc = product.download()

          filename = (
              f"{star.replace(" ", "_")}_"
              f"{product.mission[0].replace(' ', '_')}.fits"
          )

          filepath = star_folder / filename
          lc.to_fits(filepath) 

        except Exception as e:
          print(f"Failed to download {product}: {e}")
        
      print(f"Finished {star}")