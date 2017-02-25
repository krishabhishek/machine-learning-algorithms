# Generalized Linear Regression

## Pre-requisites
* Numpy

## Run Command

For 10-fold Cross-Validated accuracy:

```
export PROJECT_DIRECTORY=<PROJECT_DIRECTORY_PATH>

PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/rg_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --max_degree 1 --metrics_file $OUTPUT_FOLDER/results_linear_reg.json
```
