from utils import file_helper
from utils.knn import Knn

data_file_path = "/home/v2john/CourseProjects/machine-learning-algorithms/k-nearest-neighbours/data/data1.csv"
label_file_path = "/home/v2john/CourseProjects/machine-learning-algorithms/k-nearest-neighbours/data/labels1.csv"


data_vectors = list()
labels = list()

with open(data_file_path) as data_file:
    for line in data_file:
        vector = line.split(",")
        result = list(map(int, vector))
        data_vectors.append(result)

with open(label_file_path) as label_file:
    for line in label_file:
        label = line.strip()
        labels.append(label)

total = 0
correct = 0

x_test = file_helper.read_data_file(data_file_path)
y_test = file_helper.read_label_file(label_file_path)
x_train = file_helper.read_data_file(data_file_path)
y_train = file_helper.read_label_file(label_file_path)

knn = Knn()
knn.fit(x_train, y_train)
predicted_y = knn.predict(x_test, 4)

for k in range(len(predicted_y)):
    if predicted_y[k] == y_test[k]:
        correct += 1
    total += 1

print("accuracy: " + str(correct / total))
