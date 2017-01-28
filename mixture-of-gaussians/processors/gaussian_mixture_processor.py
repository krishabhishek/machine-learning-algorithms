import numpy

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("GaussianMixtureProcessor")
file_range = range(1, 11)


class GaussianMixtureProcessor(Processor):

    def process(self):
        log.info("GaussianMixtureProcessor begun")

        for test_set_identifier in file_range:
            train_set_vectors, train_set_labels, test_set_vectors, test_set_labels = \
                file_helper.get_datasets(self.options.input_data_folder, test_set_identifier, file_range)

            self.run_classifier(train_set_vectors, train_set_labels, test_set_vectors, test_set_labels)
            break

        log.info("GaussianMixtureProcessor concluded")

    def run_classifier(self, train_set_vectors, train_set_labels, test_set_vectors, test_set_labels):

        distinct_labels = set(train_set_labels)

        for label in distinct_labels:
            class_count = train_set_labels.count(label)
            prior = train_set_labels.count(label)/class_count
            mean = numpy.mean(numpy.array(train_set_vectors), axis=0)
            class_covariance = \
                self.get_class_covariance(label, train_set_vectors, train_set_labels, mean, class_count)


    def get_class_covariance(self, label, train_set_vectors, train_set_labels, mean, class_count):

        dimensions = len(train_set_vectors[0])
        final_matrix = numpy.zeros((dimensions, dimensions))

        for i in range(len(train_set_vectors)):
            if label == train_set_labels[i]:
                vector = numpy.subtract(numpy.array(train_set_vectors[i]), mean)
                matrix = numpy.multiply(vector, numpy.transpose(vector))
                final_matrix = numpy.add(final_matrix, matrix)

        return final_matrix/class_count
