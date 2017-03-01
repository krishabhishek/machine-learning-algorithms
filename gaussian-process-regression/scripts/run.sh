PROJECT_DIRECTORY="/home/v2john/Projects/machine-learning-algorithms/gaussian-process-regression/"
OUTPUT_FOLDER="/tmp/"

PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/gp_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --kernel_type IdentityKernel --metrics_file $OUTPUT_FOLDER/results_bayesian_linear_reg.json

PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/gp_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --kernel_type GaussianKernel --gaussian_variance 1 --metrics_file $OUTPUT_FOLDER/results_bayesian_linear_reg.json

PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/gp_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --kernel_type PolynomialKernel --polynomial_degree 1 --metrics_file $OUTPUT_FOLDER/results_bayesian_linear_reg.json
