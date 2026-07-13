from preparing_data.read_fits import read_fits
from preparing_data.cleaning_data import clean_data
from filtering.data_filters import mad_filter
from pathlib import Path

file = read_fits(Path("./data/raw/Kepler-135/Kepler-135_Kepler_Quarter_02.fits"))

file, _ = clean_data(file)

data, stat = mad_filter(file, threshold=3)

print(stat)