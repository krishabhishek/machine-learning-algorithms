import json

import matplotlib.pyplot as plt

results_file_path = \
    "/home/v2john/Projects/machine-learning-algorithms/assignments/a3/results/" \
    "results_gaussian_process_reg_polynomial.json"

with open(results_file_path) as results_file:
    results = json.load(results_file)

print(results)

x_values = list()
y_values = list()

y_min = 10000
best_stddev = None

for k in results.keys():
    x_values.append(k)
    y_values.append(results[k]['error'])

    if results[k]['error'] < y_min:
        y_min = results[k]['error']
        best_stddev = k


plt.xlabel('Polynomial - Degree')
plt.ylabel('Error - Euclidean loss')
plt.title('Gaussian Process Regression Error Graph\n Best Kernel Standard Deviation = ' + str(best_stddev))
plt.plot(x_values, y_values, 'bx:')

plt.axis([None, None, 0, 10])

plt.show()
