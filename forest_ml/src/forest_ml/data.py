from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float, use_tsr: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    if use_tsr is not False:
        features_train, features_val, target_train, target_val = train_test_split(
            features, target, test_size=test_split_ratio, random_state=random_state
        )
        return features_train, features_val, target_train, target_val
    else:
        return features, target
