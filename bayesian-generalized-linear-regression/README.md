# Bayesian Generalized Linear Regression

## Pre-requisites
* Numpy

## Run Command

For 10-fold Cross-Validated accuracy:

Set temporary environment variables
```
export PROJECT_DIRECTORY=<PROJECT_DIRECTORY_PATH>
export OUTPUT_FOLDER = <OUTPUT_FOLDER>
```

```
PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/bg_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --max_degree 4 --metrics_file $OUTPUT_FOLDER/results_bayesian_linear_regression.json
```
