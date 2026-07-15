import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.data_models import LightCurveData
from features.extract import extract_features
import pandas as pd

import numpy as np

def get_refined_files(path: Path):
  
  files = []
  
  for star_folder in path.iterdir():
    
    if not star_folder.is_dir():
      continue
    
    npz_files = sorted(
      star_folder.glob("*.npz")
    )
    
    files.extend(npz_files)
    
  return files
  
def load_refined_curve(path: Path):
  
  data = np.load(path, allow_pickle=True)
  
  return LightCurveData(
    time=data["time"],
    flux=data["flux"],
    flux_error=data["flux_error"],
    quality=data["quality"],

    target_id=str(data["target_id"]),
    mission=str(data["mission"]),
    quarter=int(data["quarter"]),
    file_path=Path(str(data["file_path"]))
  )
  
  
def build_dataset(
    path: Path,
    label: int
):
    rows = []
    metadata_rows = []

    curve_files = (
        get_refined_files(path)
    )

    for file_path in curve_files:

        try:
            curve = (
                load_refined_curve(
                    file_path
                )
            )

            features = (
                extract_features(
                    curve
                )
            )

            features["label"] = label

            rows.append(features)

            metadata_rows.append(
                {
                    "target_id":
                        curve.target_id,

                    "quarter":
                        curve.quarter,

                    "file_path":
                        str(file_path),

                    "label":
                        label
                }
            )

        except Exception as e:

            print(
                f"Failed:"
                f" {file_path.name}"
            )

            print(e)

    return rows, metadata_rows

def build_complete_dataset(
    positive_path: Path,
    negative_path: Path
  ):
    
    print(
        "Processing positives..."
    )

    positive_rows, positive_meta = (
        build_dataset(
            positive_path,
            label=1
        )
    )

    print(
        "Processing negatives..."
    )

    negative_rows, negative_meta = (
        build_dataset(
            negative_path,
            label=0
        )
    )

    rows = (
        positive_rows
        + negative_rows
    )

    metadata_rows = (
        positive_meta
        + negative_meta
    )

    features_df = pd.DataFrame(
        rows
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    return (
        features_df,
        metadata_df
    )
    
def save_dataset(
    features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    save_dir: Path
  ):

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    features_df.to_csv(
        save_dir
        / "features.csv",

        index=False
    )

    metadata_df.to_csv(
        save_dir
        / "metadata.csv",

        index=False
    )
  
features_df, metadata_df = (
    build_complete_dataset(

        Path(
            "pre-processing/"
            "data/refined/positive"
        ),

        Path(
            "pre-processing/"
            "data/refined/negative"
        )
    )
)

print(features_df.shape)
print(metadata_df.shape)

print(
    features_df[
        "label"
    ].value_counts()
)

print(
    features_df
    .isna()
    .sum()
)

save_dataset(
    features_df,
    metadata_df,

    Path(
        "data/ml/features"
    )
)



