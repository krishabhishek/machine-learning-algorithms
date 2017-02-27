from utils import file_helper
from utils import metrics_helper
from utils.bayesian_linear_regressor import BayesianLinearRegression

input_file = \
    "/home/v2john/Projects/machine-learning-algorithms/bayesian-generalized-linear-regression/data/fData1.csv"
label_file = \
    "/home/v2john/Projects/machine-learning-algorithms/bayesian-generalized-linear-regression/data/fLabels1.csv"

input_data = file_helper.read_data_file(input_file)
labels = file_helper.read_label_file(label_file)

lr = BayesianLinearRegression(input_data, labels, 0.1)
predictions1 = lr.predict(input_data)

lr = BayesianLinearRegression(input_data, labels, 4)
predictions2 = lr.predict(input_data)


print(predictions1)
print(metrics_helper.calculate_euclidean_loss(predictions1, labels))

print(predictions2)
print(metrics_helper.calculate_euclidean_loss(predictions2, labels))
