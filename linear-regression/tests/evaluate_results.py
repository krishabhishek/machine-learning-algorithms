import json

import matplotlib.pyplot as plt

results_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" \
    "Winter2017/CS698/Assignments/A1/results/results_linear_reg.json"

with open(results_file_path) as results_file:
    results = json.load(results_file)


x_values = list()
y_values = list()

y_min = 10000
best_lambda = None
for k in results.keys():
    x_values.append(k)
    y_values.append(results[k])

    if results[k] < y_min:
        y_min = results[k]
        best_lambda = k


plt.xlabel('Lambda')
plt.ylabel('Error - Euclidean loss')
plt.title('Linear Regression Error Graph\n Best Lambda = ' + str(best_lambda))
plt.plot(x_values, y_values, 'ro')

# plt.axis([0, 40, 0, 4])

plt.show()
