import json

import matplotlib.pyplot as plt

results_file_path = \
    "/home/v2john/Projects/machine-learning-algorithms/assignments/a3/" \
    "results/results_generalized_linear_regression.json"

with open(results_file_path) as results_file:
    results = json.load(results_file)

print(results)

x_values = list()
y_values = list()

y_min = 10000
best_lambda = None
for result in results:
    x_values.append(result['degree'])
    y_values.append(result['elapsed_time'])

plt.xlabel('Basis Function Degree')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Generalized Regularized Linear Regression Computation Time')
plt.plot(x_values, y_values, linestyle='-', marker='o', color='b')

# plt.axis([0, 40, 0, 4])

plt.show()
