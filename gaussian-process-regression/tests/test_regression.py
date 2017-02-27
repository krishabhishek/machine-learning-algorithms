from utils import file_helper
from utils.gaussian_process_regressor import GaussianProcessRegression

input_file = "/home/v2john/CourseProjects/machine-learning-algorithms/gaussian-process-regression/data/fData1.csv"
label_file = "/home/v2john/CourseProjects/machine-learning-algorithms/gaussian-process-regression/fLabels1.csv"

input_data = file_helper.read_data_file(input_file)
labels = file_helper.read_label_file(label_file)

lr = GaussianProcessRegression(input_data, labels, 0)
predictions1 = lr.predict(input_data)

lr = GaussianProcessRegression(input_data, labels, 4)
predictions2 = lr.predict(input_data)

print(predictions1)
print(predictions2)
