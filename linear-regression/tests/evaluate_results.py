import json

import matplotlib.pyplot as plt

results_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/" \
    "Academics/Winter2017/cs698/assignments/A1/results_linear_reg.json"

with open(results_file_path) as results_file:
    results = json.load(results_file)


x_values = list()
y_values = list()

for k in results.keys():
    x_values.append(k)
    y_values.append(results[k])

plt.xlabel('lambda')
plt.ylabel('Accuracy - r2 score')
plt.title('Linear Regression Accuracy Graph')
plt.plot(x_values, y_values, 'ro')
# plt.axis([0, 4, 0, 1])

plt.show()
