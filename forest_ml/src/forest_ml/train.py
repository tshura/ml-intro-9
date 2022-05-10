from pathlib import Path
from joblib import dump

import click
import numpy as np
import mlflow
# import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold, GridSearchCV
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


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
    "--m",
    default='logreg',
    type=str,
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
    "--fs",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--s",
    default='ss',
    type=str,
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
@click.option(
    "--criterion",
    default= 'entropy',
    type=str,
    show_default=True,
)
@click.option(
    "--n",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=15,
    type=int,
    show_default=True,
)
@click.option(
    "--use-ncv",
    default=False,
    type=bool,
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
    m: str,
    s: str,
    n:int, 
    criterion: str,
    max_depth: int,
    fs: bool,
    use_ncv: bool

) -> None:
    with mlflow.start_run():
        if criterion not in ['gini', 'entropy']:
            raise click.BadParameter("criterion takes values 'gini' or 'entropy'")
        if m not in ['logreg', 'rf']:
            raise click.BadParameter("model takes values 'logreg' for LogisticRegression or 'rf' for RandomForestClassifier")
        if s not in ['ss', 'mm']:
            raise click.BadParameter("scaler takes values 'ss' for StandardScaler or 'mm' for MinMaxScaler")
        simplefilter("ignore", category=ConvergenceWarning)
        pipeline = create_pipeline(use_scaler, s, m, max_iter, logreg_c, random_state, n, criterion, max_depth, fs)
        if use_ncv is False:
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
                mlflow.log_param("model", pipeline['classifier'])
                mlflow.log_param("N of folds", foldcnt)
                mlflow.log_param("use_scaler", use_scaler)
                mlflow.log_param("scaler", pipeline['scaler'])
                if fs:
                    mlflow.log_param("feature_selection", pipeline['feature_selection'])
                if m == 'logreg':  
                    mlflow.log_param("max_iter", max_iter)
                    mlflow.log_param("logreg_c", logreg_c)
                elif m == 'rf':   
                    mlflow.log_param("max_depth", max_depth)
                    mlflow.log_param("n_estimators", n)
                    mlflow.log_param("criterion", criterion)
                scores  = cross_validate(pipeline, features_train, target_train, cv=foldcnt, scoring=scoring)
                for i in scoring:
                    click.echo('{} = {}'.format(i, np.mean(scores['test_' + i])))
                    mlflow.log_metric(i, np.mean(scores['test_' + i]))
                        
                pipeline.fit(features_train, target_train)
                dump(pipeline, save_model_path)
                click.echo(f"Model is saved to {save_model_path}.")
        else:
            features_train, target_train = get_dataset(
                    dataset_path,
                    random_state,
                    test_split_ratio,
                    use_tsr
                )      
            cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
            outer_results_acc = list()
            outer_results_ras = list()
            outer_results_f1 = list()
            mlflow.log_param("nested_cv", use_ncv)
            mlflow.log_param("model", pipeline['classifier'])
            for train_ix, test_ix in cv_outer.split(features_train):
                # split data
                X_train, X_test = features_train.iloc[train_ix, :], features_train.iloc[test_ix, :]
                y_train, y_test = target_train.iloc[train_ix], target_train.iloc[test_ix]
                # configure the cross-validation procedure
                cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
                # define search space
                space = dict()
                if m == 'rf':
                    space['classifier__n_estimators'] = [10, 100, 500]
                    space['classifier__max_depth'] = [2, 4, 6]
                elif m == 'logreg': 
                    space['classifier__max_iter'] = [300, 500, 1000]
                    space['classifier__C'] = [0.1, 1, 10]
                # define search
                search = GridSearchCV(pipeline, space, scoring='accuracy', cv=cv_inner, refit=True)
                # execute search
                result = search.fit(X_train, y_train)
                # get the best performing model fit on the whole training set
                best_model = result.best_estimator_
                # evaluate model on the hold out dataset
                yhat = best_model.predict(X_test)
                y_pred = best_model.predict_proba(X_test)
                # evaluate the model
                acc = accuracy_score(y_test, yhat)
                ras = roc_auc_score(y_test, y_pred, multi_class='ovr')
                f1 = f1_score(y_test, yhat, average='macro')
                # store the result
                outer_results_acc.append(acc)
                outer_results_ras.append(ras)
                outer_results_f1.append(f1)
            click.echo('Accuracy: {}'.format(np.mean(outer_results_acc)))
            click.echo('Roc Auc: {}'.format(np.mean(outer_results_ras)))
            click.echo('F1 score: {}'.format(np.mean(outer_results_f1)))     
            mlflow.log_metric("accuracy", np.mean(outer_results_acc))
            mlflow.log_metric("f1_macro", np.mean(outer_results_f1))
            mlflow.log_metric("roc_auc_ovr", np.mean(outer_results_ras))
            

                
    
