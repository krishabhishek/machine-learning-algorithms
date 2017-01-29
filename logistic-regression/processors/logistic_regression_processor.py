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

        for test_set_identifier in file_range:
            train_set_vectors, train_set_labels, test_set_vectors, test_set_labels = \
                file_helper.get_datasets(self.options.input_data_folder, test_set_identifier, file_range)

            self.run_classifier(train_set_vectors, train_set_labels, test_set_vectors, test_set_labels)
            break

        log.info("LogisticRegressionProcessor concluded")

    @staticmethod
    def run_classifier(train_set_vectors, train_set_labels, test_set_vectors, test_set_labels):

        train_set_size = len(train_set_labels)
        log.info("Train set size: " + str(train_set_size))
        dimensions = len(train_set_vectors[0]) + 1

        weights = numpy.zeros(dimensions)
        weights_transpose = numpy.transpose(weights)
        log.info("Weight dimensions: " + str(weights.shape))

        constant_weight_term = numpy.ones(train_set_size)

        x_matrix = \
            numpy.vstack(
                (numpy.transpose(constant_weight_term), numpy.transpose(numpy.array(train_set_vectors))),
            )
        x_matrix_transpose = numpy.transpose(x_matrix)
        log.info("Input dimensions: " + str(x_matrix.shape))

        sigma_list = list()
        r_matrix = numpy.zeros((train_set_size, train_set_size))
        for i in range(len(x_matrix_transpose)):
            vector = numpy.array(x_matrix_transpose[i])
            sigma = 1 / (1 + math.exp(-1 * numpy.dot(weights_transpose, vector)))
            sigma_list.append(sigma)
            r_matrix[i][i] = sigma * (1 - sigma)

        log.info("Sigma list length: " + str(len(sigma_list)))
        log.info("R matrix dimensions: " + str(r_matrix.shape))

        hessian = numpy.dot(numpy.dot(x_matrix, r_matrix), x_matrix_transpose)
        log.info("Hessian dimensions: " + str(hessian.shape))
