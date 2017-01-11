from sklearn.model_selection import KFold
import numpy as np

from utils.knn import Knn

data_file_path = "/home/v2john/cs698_code/cs698-assignments/k-nearest-neighbours/data/data1.csv"
label_file_path = "/home/v2john/cs698_code/cs698-assignments/k-nearest-neighbours/data/labels1.csv"


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


# X_train, X_test, y_train, y_test = train_test_split(data_vectors, labels, test_size=0.1, random_state=42)
#
# knn = Knn(X_train, X_test, y_train, y_test, 3)
# test_set_predicted_y = knn.get_test_set_predictions()
# print test_set_predicted_y
# print y_test

kf = KFold(n_splits=10)

total = 0
correct = 0
for train_index, test_index in kf.split(data_vectors):
    X_train, X_test = np.array(data_vectors)[train_index], np.array(data_vectors)[test_index]
    y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

    knn = Knn()
    knn.fit(X_train, y_train)
    predicted_y = knn.predict(X_test, 4)

    for i in xrange(len(predicted_y)):
        if predicted_y[i] == y_test[i]:
            correct += 1
        total += 1

print "accuracy: " + str(correct * 1.0/total)
