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

### Identity Kernel
```
PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/gp_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --kernel_type IdentityKernel --metrics_file $OUTPUT_FOLDER/results_gaussian_process_reg_identity.json
```

### Gaussian Kernel
```
PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/gp_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --kernel_type GaussianKernel --gaussian_stddev 6 --metrics_file $OUTPUT_FOLDER/results_gaussian_process_reg_gaussian.json
```

### Polynomial Kernel
```
PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/gp_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --kernel_type PolynomialKernel --polynomial_degree 4 --metrics_file $OUTPUT_FOLDER/results_gaussian_process_reg_polynomial.json
```
