import json

import matplotlib.pyplot as plt

results_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" \
    "Winter2017/cs698/assignments/A1/results/results_linear_reg.json"

with open(results_file_path) as results_file:
    results = json.load(results_file)


x_values = list()
y_values = list()

y_max = -1
best_lambda = None
for k in results.keys():
    x_values.append(k)
    y_values.append(results[k])

    if results[k] > y_max:
        y_max = results[k]
        best_lambda = k


plt.xlabel('lambda')
plt.ylabel('Accuracy - r2 score')
plt.title('Linear Regression Accuracy Graph\n Best Lambda = ' + str(best_lambda))
plt.plot(x_values, y_values, 'ro')
plt.axis([0, 4, 0.9, 1])

plt.show()
