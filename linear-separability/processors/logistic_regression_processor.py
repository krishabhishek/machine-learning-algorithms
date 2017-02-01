import numpy
import math

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("LogisticRegressionProcessor")
file_range = range(1, 11)


class LogisticRegressionProcessor(Processor):

    def process(self):
        log.info("LogisticRegressionProcessor begun")

        train_set_vectors, train_set_labels, test_set_vectors, test_set_labels = \
            file_helper.get_datasets(self.options.input_data_folder, file_range)

        accuracy = self.run_classifier(train_set_vectors, train_set_labels, test_set_vectors, test_set_labels)

        log.info("Accuracy: " + str(accuracy))
        log.info("LogisticRegressionProcessor concluded")

    @staticmethod
    def run_classifier(train_set_vectors, train_set_labels, test_set_vectors, test_set_labels):

        train_set_size = len(train_set_labels)
        log.debug("Train set size: " + str(train_set_size))
        dimensions = len(train_set_vectors[0]) + 1

        weights = numpy.zeros(dimensions)
        log.debug("Weight dimensions: " + str(weights.shape))

        constant_weight_term = numpy.ones(train_set_size)

        x_matrix = \
            numpy.vstack(
                (numpy.transpose(constant_weight_term), numpy.transpose(numpy.array(train_set_vectors))),
            )
        x_matrix_transpose = numpy.transpose(x_matrix)
        log.debug("Input dimensions: " + str(x_matrix.shape))

        for count in range(1, 101):
            weights_transpose = numpy.transpose(weights)
            r_matrix = numpy.zeros((train_set_size, train_set_size))
            gradient = numpy.zeros(dimensions)

            for i in range(len(x_matrix_transpose)):
                vector = numpy.array(x_matrix_transpose[i])
                exponent = -1 * numpy.dot(weights_transpose, vector)
                sigma = 1 / (1 + math.exp(exponent))

                r_matrix[i][i] = sigma * (1 - sigma)
                if train_set_labels[i] == 5:
                    y = 1
                else:
                    y = 0

                gradient = numpy.add(gradient, (sigma - y) * vector)

            log.debug("R matrix dimensions: " + str(r_matrix.shape))

            hessian = numpy.dot(numpy.dot(x_matrix, r_matrix), x_matrix_transpose)  # + (1 * numpy.identity(dimensions))
            hessian_inverse = numpy.linalg.inv(hessian)
            log.debug("Hessian dimensions: " + str(hessian.shape))
            log.debug("Gradient dimensions: " + str(gradient.shape))

            step = numpy.dot(hessian_inverse, gradient)
            log.debug("Step dimensions: " + str(step.shape))
            weights -= step

        log.info("weights: " + str(weights))
        test_vectors_matrix = \
            numpy.vstack(
                (numpy.transpose(numpy.ones(len(test_set_vectors))),
                 numpy.transpose(numpy.array(test_set_vectors))),
            )

        accuracy_score = 0
        for i in range(len(numpy.transpose(test_vectors_matrix))):
            sigma = \
                1 / (
                    1 + math.exp(-1 * numpy.dot(numpy.transpose(weights), numpy.transpose(test_vectors_matrix)[i]))
                )

            if sigma > 0.5 and test_set_labels[i] == 5:
                accuracy_score += 1
            elif sigma <= 0.5 and test_set_labels[i] == 6:
                accuracy_score += 1

        return round((accuracy_score/len(test_set_labels)), 3)
