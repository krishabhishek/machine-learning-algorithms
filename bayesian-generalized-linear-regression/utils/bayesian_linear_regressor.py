import numpy as np
import numpy.linalg.linalg as lg

from utils import basis_exponent_helper


class BayesianLinearRegression(object):

    def __init__(self, x_train, y_train, lambda_value, degree):
        self.x_train = x_train
        self.y_train = y_train
        self.degree = degree
        self.variance = 1

    def predict(self, x_test):
        predictions = list()

        self.x_train = self.modify_dimensions(self.x_train, self.degree)
        x_test = self.modify_dimensions(x_test, self.degree)
        dimensions = len(self.x_train[0])

        X = np.asarray(self.x_train)
        identity_matrix = np.matrix(np.identity(dimensions))

        A = (self.variance * np.matmul(np.transpose(X), X)) + lg.inv(identity_matrix)

        for test_vector in x_test:
            y_prediction = np.dot(np.transpose(test_vector) * lg.inv(A) * np.transpose(X), np.transpose(self.y_train))
            predictions.append(y_prediction.item((0, 0)))

        return predictions

    def modify_dimensions(self, x_train, degree):
        exponent_combo_list = basis_exponent_helper.get_exponent_combos(degree)
        x_train_new = list(map(lambda x: self.get_transformed_row(x, exponent_combo_list), x_train))

        return x_train_new

    def get_transformed_row(self, x, exponent_combo_list):

        transformed_x = list()
        for combo in exponent_combo_list:
            (f, s) = combo
            transformed_x.append(pow(x[0], f) * pow(x[1], s))

        return transformed_x
