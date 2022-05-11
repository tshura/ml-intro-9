# ml-intro-9

This demo uses [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset.
This package allows you to train model for detecting the presence of heart disease in the patient.

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
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```

You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox [-r]
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
nox -[r]s black
poetry run black src tests noxfile.py
```

To get the result for the 6th task run
`poetry run train --use-tsr True`

![image](https://user-images.githubusercontent.com/99091756/167840084-b5d62681-7400-4026-be22-a998f05d7675.png)

![image](https://user-images.githubusercontent.com/99091756/167720715-5f797a0a-2f75-4775-9122-0aaeba969267.png)

![image](https://user-images.githubusercontent.com/99091756/167723211-1c36de3f-a19e-4e8c-9938-8f3c2a4e4e63.png)

