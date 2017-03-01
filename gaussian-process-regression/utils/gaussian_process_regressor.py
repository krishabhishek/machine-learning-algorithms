import sys
import numpy as np


class GaussianProcessRegression(object):

    def __init__(self, x_train, y_train, kernel):
        self.x_train = np.asmatrix(x_train)
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
                gram_matrix.itemset((i, j), self.kernel.compute_kernel_function(x_train[i], x_train[j]))

        return gram_matrix

    def predict(self, x_test):
        predictions = list()
        x_test = np.matrix(x_test)

        for x_vector in x_test:
            vector_term = list()
            for x_train_vector in self.x_train:
                vector_term_item = self.kernel.compute_kernel_function(x_vector, x_train_vector)
                vector_term.append(vector_term_item)

            prediction = np.matmul(np.matmul(np.asmatrix(vector_term), np.linalg.inv(self.matrix_term)), self.y_train)
            predictions.append(prediction.item(0))

        return predictions
