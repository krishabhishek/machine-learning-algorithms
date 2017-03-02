import json

import matplotlib.pyplot as plt

results_file_path = \
    "/home/v2john/Projects/machine-learning-algorithms/assignments/a3/results/" \
    "results_gaussian_process_reg_identity.json"

with open(results_file_path) as results_file:
    results = json.load(results_file)

print(results)

x_values = list()
y_values = list()

y_min = 10000
best_lambda = None

for k in results.keys():
    x_values.append(k)
    y_values.append(results[k][''])

    if results[k] < y_min:
        y_min = results[k]
        best_lambda = k


plt.xlabel('Gaussian - Standard Deviation')
plt.ylabel('Error - Euclidean loss')
plt.title('Gaussian Process Regression Error Graph\n Best Kernel Standard Deviation = ' + str(best_lambda))
plt.plot(x_values, y_values)

# plt.axis([0, 40, 0, 4])

plt.show()
