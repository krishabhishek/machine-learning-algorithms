import numpy as np


class GaussianProcessRegression(object):

    def __init__(self, x_train, y_train, kernel):
        self.x_train = x_train
        self.y_train = np.asarray(y_train)
        self.kernel = kernel
        self.gram_matrix = self.calculate_gram_matrix(x_train)
        self.model_variance = 1
        self.matrix_term = self.gram_matrix + self.model_variance * np.identity(len(x_train))

    def calculate_gram_matrix(self, x_train):
        sample_size = len(x_train)
        x_train = np.asarray(x_train)
        gram_matrix = np.empty([sample_size, sample_size])

        for i in range(len(x_train)):
            for j in range(len(x_train)):
                gram_matrix.itemset((i, j), self.kernel.compute_kernel_function(x_train.item(i), x_train.item(j)))

        return gram_matrix

    def predict(self, x_test):
        predictions = list()
        x_test = np.asarray(x_test)

        for x_vector in x_test:
            vector_term = self.kernel.compute_kernel_function(np.transpose(x_vector), np.transpose(self.x_train))
            temp = np.matmul(vector_term, np.linalg.inv(self.matrix_term))
            prediction = np.matmul(temp, np.transpose(self.y_train))
            predictions.append(prediction)

        return predictions
