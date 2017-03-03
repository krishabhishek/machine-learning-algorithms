import numpy as np
import json

import matplotlib.pyplot as plt

results_file_path = \
    "/home/v2john/Projects/machine-learning-algorithms/assignments/a3/results/" \
    "results_gaussian_process_reg_gaussian.json"

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

plt.figure(0)
plt.xlabel('Gaussian - Standard Deviation')
plt.ylabel('Mean Squared Error')
plt.title('Gaussian Process Regression Error Graph (Gaussian Kernel) \n Best Kernel Standard Deviation = ' + str(best_stddev))
plt.plot(x_values, y_values, linestyle='-', marker='o', color='b')

plt.axis([None, None, 0, None])

plt.show()

############################


x_values = list()
y_values = list()
y_min = 10000

for k in results.keys():
    x_values.append(k)
    y_values.append(results[k]['elapsed_time'])

    if results[k]['elapsed_time'] < y_min:
        y_min = results[k]['elapsed_time']
        best_stddev = k

plt.figure(1)
plt.xlabel('Gaussian - Standard Deviation')
plt.ylabel('Computational Time (seconds)')
plt.title('Gaussian Process Regression - Computational Time (Gaussian Kernel)')
plt.plot(x_values, y_values, linestyle='-', marker='o', color='b')

times = np.asarray(y_values)
print("Mean: " + str(np.mean(times)))
print("Std: " + str(np.std(times)))

plt.axis([None, None, 0, 10])

plt.show()
