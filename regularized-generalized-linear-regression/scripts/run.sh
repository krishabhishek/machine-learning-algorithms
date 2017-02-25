PROJECT_DIRECTORY="/home/v2john/Projects/machine-learning-algorithms/regularized-generalized-linear-regression/"
OUTPUT_FOLDER="/home/v2john/Projects/machine-learning-algorithms/regularized-generalized-linear-regression/"

PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python /home/v2john/Projects/machine-learning-algorithms/regularized-generalized-linear-regression/rg_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --max_degree 1 --metrics_file $OUTPUT_FOLDER/results_linear_reg.json
