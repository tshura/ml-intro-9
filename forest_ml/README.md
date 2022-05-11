# ml-intro-9

This demo uses [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.
This package allows you to train model for Forest Cover Type Prediction.

## Usage

1. Clone this repository to your machine.
2. Download [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset, save csv locally (default path is *data/heart.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Make sure to run all of the commands from the forest_ml folder, so the path for the active directory should look something like this C:\Users\...\forest_ml

To change directory run the following command:
```sh
cd <your path>\forest_ml
```
5. Install the project dependencies:
```sh
poetry install --no-dev
```
6. Run train with the following command (you can skip -d and -s if you are ok with the default path):
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
``` 
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
Let's take a closer look at some of the available options

You can choose one of the two models, LogisticRegression or RandomForestClassifier, by running the following command:
```sh
poetry run train --m logreg
```
or
```sh
poetry run train --m rf
```
LogisticRegression is the defualt model, so to get this model you can simply run:
```sh
poetry run train
```
By running the defualt script above you are getting the metrics evaluated on 5 K-fold CV. You can change the number of folds with 
```sh
foldcnt --3
```

Available parameters for LogisticRegression:
```sh
poetry run train --logreg-c 50 --max-iter 10000
```
Available parameters for RandomForestClassifier:
```sh
poetry run train --m rf --n 500 --max-depth 20 --criterion gini
```
where n - n_estimator, criterion could take values 'gini' or 'entropy', 'entropy' is the defualt

For each model you can remove scaler, change scaler or apply feature selection with SelectFromModel(LogisticRegression(penalty="l2")):
```sh
poetry run train --use_scaler False
```
```sh
poetry run train --s mm --fs True
```
where s - scaler - takes values 'ss' for StandardScaler or 'mm' for MinMaxScaler

To estimate quality with nested cross-validation and get the best model run the following command (works with either model, to choose the model use parameter --m):
```sh
poetry run train --use-ncv True
```

7. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
MLflow runs completed for Task 8
![image](https://user-images.githubusercontent.com/99091756/167840084-b5d62681-7400-4026-be22-a998f05d7675.png)


## Development

The code in this repository has been tested, formatted with black and linted with flake8.

To check it yourself you need to install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
Format code with black/check that the code has already been formatted:
```
poetry run black src
```
![image](https://user-images.githubusercontent.com/99091756/167720715-5f797a0a-2f75-4775-9122-0aaeba969267.png)

Lint code with flake8/check that the code has already been linted:
```
poetry run flake8 src --ignore=E501
```
![image](https://user-images.githubusercontent.com/99091756/167723211-1c36de3f-a19e-4e8c-9938-8f3c2a4e4e63.png)

I'm using --ignore=E501 because after formating with black there will be lines longer than 79 characters



