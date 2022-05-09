from pathlib import Path
from joblib import dump

import click
import numpy as np
# import mlflow
# import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

from .data import get_dataset
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)

@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)

@click.option(
    "--use-tsr",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=500,
    type=int,
    show_default=True,
)
@click.option(
    "--foldcnt",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    use_tsr: bool,
    max_iter: int,
    foldcnt: int,
    logreg_c: float,
) -> None:
    pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
    if use_tsr is not False:
        features_train, features_val, target_train, target_val = get_dataset(
            dataset_path,
            random_state,
            test_split_ratio,
            use_tsr
        )
        click.echo(f"Features_train shape: {features_train.shape}.")
        pipeline.fit(features_train, target_train)
        accuracy = accuracy_score(target_val, pipeline.predict(features_val))
        click.echo(f"Accuracy: {accuracy}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")

    else:
        features_train, target_train = get_dataset(
            dataset_path,
            random_state,
            test_split_ratio,
            use_tsr
        )
        scoring = ['accuracy', 'f1_macro', 'roc_auc_ovr']
        scores  = cross_validate(pipeline, features_train, target_train, cv=foldcnt, scoring=scoring)
        for i in scoring:
            click.echo('{} = {}'.format(i, np.mean(scores['test_' + i])))
    # with mlflow.start_run():
    
    # print(pipeline)
    # 
    # print(cv_results)
    
    
    # 
    # mlflow.log_param("use_scaler", use_scaler)
    # mlflow.log_param("max_iter", max_iter)
    # mlflow.log_param("logreg_c", logreg_c)
    # mlflow.log_metric("accuracy", accuracy)
    # 
