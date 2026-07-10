from preprocessing.read_fits import read_fits
from preprocessing.cleaning_data import clean_data
from pathlib import Path

file = read_fits(Path("./data/raw/Kepler-135/Kepler-135_Kepler_Quarter_00.fits"))

clean_data(data=file)