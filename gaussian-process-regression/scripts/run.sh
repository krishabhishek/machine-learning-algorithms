PROJECT_DIRECTORY="/home/v2john/Projects/machine-learning-algorithms/gaussian-process-regression/"
OUTPUT_FOLDER="/tmp/"

PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/gp_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --kernel_type IdentityKernel --metrics_file $OUTPUT_FOLDER/results_gaussian_process_reg_identity.json

PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/gp_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --kernel_type GaussianKernel --gaussian_stddev 6 --metrics_file $OUTPUT_FOLDER/results_gaussian_process_reg_gaussian.json

PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/gp_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --kernel_type PolynomialKernel --polynomial_degree 4 --metrics_file $OUTPUT_FOLDER/results_gaussian_process_reg_polynomial.json
