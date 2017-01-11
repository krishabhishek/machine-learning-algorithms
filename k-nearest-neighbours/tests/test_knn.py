from sklearn.model_selection import train_test_split

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

print("There are " + str(len(data_vectors)) + " data vectors")
print("There are " + str(len(labels)) + " labels")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)