# K Nearest Neighbours

## Pre-requisites
* Numpy
* Scipy
* Scikit-Learn


## Run Command

For 10-fold Cross-Validated accuracy:

```
PYTHONPATH=<PROJECT_DIRECTORY> /usr/bin/python2.7 <PROJECT_DIRECTORY>/knn_runner.py --crossval True --input_data_folder <PROJECT_DIRECTORY>/data/ --min_k 1 --max_k 30 --metrics_file <OUTPUT_FOLDER>/results_crossval.json 
```

For full training dataset accuracy:

```
PYTHONPATH=<PROJECT_DIRECTORY> /usr/bin/python2.7 <PROJECT_DIRECTORY>/knn_runner.py --full True --input_data_folder <PROJECT_DIRECTORY>/data/ --min_k 1 --max_k 30 --metrics_file <OUTPUT_FOLDER>/results_full.json
```
