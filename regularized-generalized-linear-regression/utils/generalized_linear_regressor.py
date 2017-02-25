import numpy

from utils import basis_exponent_helper


class GeneralizedLinearRegression(object):

    def __init__(self, x_train, y_train, lambda_value, degree):
        self.weights = None
        self.degree = degree

        x_train = self.modify_dimensions(x_train, degree)
        dimensions = len(x_train[0])

        identity_matrix = numpy.matrix(numpy.identity(dimensions))
        lambda_matrix = identity_matrix * lambda_value
        A = numpy.matrix(numpy.zeros((dimensions, dimensions)))
        b = numpy.matrix(numpy.zeros((dimensions, 1)))
        for i in range(len(x_train)):
            nmat = numpy.matrix(x_train[i])
            temp_a = nmat.T * nmat
            A += temp_a
            temp_b = nmat.T * numpy.matrix(y_train[i])
            b += temp_b

        A += 2 * lambda_matrix
        self.weights = A.I * b

    def predict(self, x_test):
        predictions = list()
        x_test  = self.modify_dimensions(x_test, self.degree)

        for i in range(len(x_test)):
            nmat = numpy.matrix(x_test[i])
            res = nmat * self.weights
            predictions.append(res.getA()[0][0])

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
