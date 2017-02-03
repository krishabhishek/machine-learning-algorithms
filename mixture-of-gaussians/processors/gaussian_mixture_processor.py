import numpy
import math

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("GaussianMixtureProcessor")
file_range = range(1, 11)


class GaussianMixtureProcessor(Processor):

    def process(self):
        log.info("GaussianMixtureProcessor begun")

        if self.options.full:
            train_set_vectors, train_set_labels, test_set_vectors, test_set_labels = \
                file_helper.get_full_datasets(self.options.input_data_folder, file_range)

            accuracy = self.run_classifier(train_set_vectors, train_set_labels, test_set_vectors, test_set_labels)

            log.info("Accuracy: " + str(accuracy))
        else:
            accuracy_list = list()
            for test_set_identifier in file_range:
                train_set_vectors, train_set_labels, test_set_vectors, test_set_labels = \
                    file_helper.get_datasets(self.options.input_data_folder, test_set_identifier, file_range)

                accuracy = self.run_classifier(train_set_vectors, train_set_labels, test_set_vectors, test_set_labels)
                accuracy_list.append(accuracy)

            log.info(accuracy_list)
            log.info("Average Accuracy: " + str(numpy.mean(accuracy_list)))

        log.info("GaussianMixtureProcessor concluded")

    def run_classifier(self, train_set_vectors, train_set_labels, test_set_vectors, test_set_labels):

        distinct_labels = set(train_set_labels)
        train_set_size = len(train_set_vectors)
        dimensions = len(train_set_vectors[0])
        covariance_matrix = numpy.zeros((dimensions, dimensions))
        class_properties = dict()

        label_train_vectors = dict()
        for i in range(len(train_set_vectors)):
            key = train_set_labels[i]
            if key in label_train_vectors:
                vectors = label_train_vectors[key]
            else:
                vectors = list()
            vectors.append(train_set_vectors[i])
            label_train_vectors[key] = vectors

        for label in distinct_labels:
            class_count = len(label_train_vectors[label])
            prior = class_count/len(train_set_labels)
            mean = numpy.mean(numpy.array(label_train_vectors[label]), axis=0)

            property = dict()
            property['mean'] = mean
            property['prior'] = prior
            class_properties[label] = property

            class_covariance_matrix = \
                self.get_class_covariance(
                    label_train_vectors[label], mean, dimensions, train_set_size
                )
            log.info("Variance for Label " + str(label) + ": \n" + str(numpy.diagonal(class_covariance_matrix)))
            covariance_matrix = numpy.add(covariance_matrix, class_covariance_matrix)

        inv_covariance_matrix = numpy.linalg.inv(covariance_matrix)
        mean_diff = class_properties.get('5').get('mean') - class_properties.get('6').get('mean')

        w = \
            numpy.dot(
                inv_covariance_matrix,
                mean_diff
            )

        w0 = \
            -0.5 * (
                numpy.dot(
                    numpy.dot(
                        numpy.transpose(class_properties.get('5').get('mean')),
                        inv_covariance_matrix
                    ),
                    class_properties.get('5').get('mean')
                )
            ) + \
            0.5 * (
                numpy.dot(
                    numpy.dot(
                        numpy.transpose(class_properties.get('6').get('mean')),
                        inv_covariance_matrix
                    ),
                    class_properties.get('6').get('mean')
                )
            ) + \
            math.log(class_properties.get('5').get('prior')/class_properties.get('6').get('prior'))

        accuracy_score = 0
        log.info("w = " + str(w))
        log.info("w0 = " + str(w0))
        for i in range(len(test_set_vectors)):
            wx_term = numpy.dot(numpy.transpose(w), numpy.array(test_set_vectors[i]))
            expt_term = wx_term + w0
            prob_5 = 1 / (1 + math.exp(-1 * expt_term))
            if prob_5 > 0.5 and test_set_labels[i] == '5':
                accuracy_score += 1
            elif 1 - prob_5 > 0.5 and test_set_labels[i] == '6':
                accuracy_score += 1

        accuracy = round(accuracy_score/len(test_set_vectors), 3)
        return accuracy

    def get_class_covariance(self, class_vectors, mean, dimensions, train_set_size):

        final_matrix = numpy.zeros((dimensions, dimensions))

        for i in range(len(class_vectors)):
            vector = numpy.subtract(numpy.array(class_vectors[i]), mean)
            matrix = numpy.outer(vector, numpy.transpose(vector))
            final_matrix = numpy.add(final_matrix, matrix)

        return final_matrix/train_set_size
