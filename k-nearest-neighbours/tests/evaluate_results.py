import json

import matplotlib.pyplot as plt

results_file_path = "/home/v2john/results_crossval.json"

with open(results_file_path) as results_file:
    results = json.load(results_file)


x_values = list()
y_values = list()

y_max = -1
best_k = None
for k in results.keys():
    x_values.append(k)
    y_values.append(results[k])

    if results[k] > y_max:
        y_max = results[k]
        best_k = k


plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy Graph - Full Training Set\nBest k = ' + str(best_k))
plt.plot(x_values, y_values)
plt.axis([0, 30, 0, 1])

plt.show()
