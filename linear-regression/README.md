# Linear Regression

## Pre-requisites
* Numpy

## Run Command

For 10-fold Cross-Validated accuracy:

```
PYTHONPATH=<PROJECT_DIRECTORY> /usr/bin/python <PROJECT_DIRECTORY>/linear_regression_runner.py --input_data_folder <PROJECT_DIRECTORY>/data --min_lambda 0 --max_lambda 4 --lambda_increment 0.1 --metrics_file  <OUTPUT_FOLDER>/results_linear_reg.json
```
