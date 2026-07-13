from fetch.download_data import (get_positive_star_list, get_negative_star_list, download_lightcurves)
import pandas as pd

N_POSITIVE = 1000
N_NEGATIVE = 1000

positive = get_positive_star_list(N_POSITIVE)
negative = get_negative_star_list(N_NEGATIVE)

pd.DataFrame(
  {"target_id" : positive}
).to_csv("./data/metadata/positive_stars.csv", index=False)

pd.DataFrame(
  {"target_id" : negative}
).to_csv("./data/metadata/negative_stars.csv", index=False)

metadata = pd.DataFrame(
  {
    "target_id" : positive + negative,
    "label" : [1] * len(positive) + [0] * len(negative),
    "class_names" : ["planet"] * len(positive) + ["non_planet"] * len(negative)
  }
)

metadata.to_csv(
  "./data/metadata/labels.csv",
  index= False
)

positive_to_download = pd.read_csv("./data/metadata/positive_stars.csv")["target_id"].tolist()

negative_to_download = pd.read_csv("./data/metadata/negative_stars.csv")["target_id"].tolist()

download_lightcurves(positive_to_download[1:101], "Kepler", "Kepler", 1800, "./data/raw/positive")

download_lightcurves(negative_to_download[:100], "Kepler", "Kepler", 1800, "./data/raw/negative")