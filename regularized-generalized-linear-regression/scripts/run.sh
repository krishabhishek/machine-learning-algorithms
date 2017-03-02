PROJECT_DIRECTORY="/home/v2john/Projects/machine-learning-algorithms/regularized-generalized-linear-regression/"
OUTPUT_FOLDER="/tmp/"

PYTHONPATH=$PROJECT_DIRECTORY /usr/bin/python $PROJECT_DIRECTORY/rg_linear_regression_runner.py --input_data_folder $PROJECT_DIRECTORY/data/ --max_degree 4 --metrics_file $OUTPUT_FOLDER/results_generalized_linear_regression.json
