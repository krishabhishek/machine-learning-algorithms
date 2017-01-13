from utils import file_helper
from utils.linear_regressor import LinearRegression

input_file = "/home/v2john/cs698_code/cs698-assignments/linear-regression/data/fData1.csv"
label_file = "/home/v2john/cs698_code/cs698-assignments/linear-regression/data/fLabels1.csv"

input_data = file_helper.read_data_file(input_file)
labels = file_helper.read_label_file(label_file)

lr = LinearRegression(input_data, labels)
predictions = lr.predict(input_data)

print(predictions)
