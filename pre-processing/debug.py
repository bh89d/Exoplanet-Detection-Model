from preprocessing.read_fits import read_fits
from pathlib import Path

print(read_fits(Path("./data/raw/Kepler-135/Kepler-135_Kepler_Quarter_00.fits")))