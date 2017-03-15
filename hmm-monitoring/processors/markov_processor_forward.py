import math
import numpy as np
from scipy import stats
from copy import deepcopy

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("MarkovProcessorForward")

file_range = range(1, 6)


class MarkovProcessorForward(Processor):

    def process(self):
        log.info("MarkovProcessorForward begun")

        train_set_vectors, train_set_labels, test_set_vectors, test_set_labels, vector_sequences, label_sequences = \
            file_helper.get_datasets(self.options.input_data_folder, file_range)

        train_set_vectors.extend(test_set_vectors)
        train_set_labels.extend(test_set_labels)
        accuracy = self.run_classifier(train_set_vectors, train_set_labels, label_sequences)
        log.info("Accuracy: " + str(accuracy))

        log.info("MarkovProcessorForward concluded")

    def run_classifier(self, train_set_vectors, train_set_labels, label_sequences):

        distinct_labels = set(train_set_labels)
        log.debug("distinct labels: " + str(distinct_labels))
        log.debug("train set size: " + str(len(train_set_labels)))

        initial_state_distribution = self.get_initial_state_distribution(distinct_labels, label_sequences)
        log.info("initial_state_distribution: " + str(initial_state_distribution))

        label_train_vectors = dict()
        for i in range(len(train_set_vectors)):

            key = train_set_labels[i]
            if key in label_train_vectors:
                vectors = label_train_vectors[key]
            else:
                vectors = list()
            vectors.append(train_set_vectors[i])
            label_train_vectors[key] = vectors

        transition_probabilities_matrix = dict()
        class_count_dict = dict()

        for i in range(len(train_set_vectors) - 1):
            if not train_set_labels[i] in transition_probabilities_matrix.keys():
                transition_probabilities_matrix[train_set_labels[i]] = dict()

            if not train_set_labels[i + 1] in transition_probabilities_matrix[train_set_labels[i]].keys():
                transition_probabilities_matrix[train_set_labels[i]][train_set_labels[i + 1]] = 0

            transition_probabilities_matrix[train_set_labels[i]][train_set_labels[i + 1]] += 1

            if not train_set_labels[i] in class_count_dict.keys():
                class_count_dict[train_set_labels[i]] = 1
            else:
                class_count_dict[train_set_labels[i]] += 1

        for class_key in transition_probabilities_matrix.keys():
            for succeeding_class_count in transition_probabilities_matrix[class_key].keys():
                transition_probabilities_matrix[class_key][succeeding_class_count] /= class_count_dict[class_key]

        log.info("transition_probabilities_matrix: " + str(transition_probabilities_matrix))

        class_properties = dict()
        for key in label_train_vectors.keys():
            class_properties[key] = dict()
            class_properties[key]['mean'] = np.mean(label_train_vectors[key], axis=0)
            class_properties[key]['std'] = np.std(label_train_vectors[key], axis=0)
        log.debug("class_properties: " + str(class_properties))

        class_properties = dict()
        covariance_matrix = np.zeros((len(train_set_vectors[0]), len(train_set_vectors[0])))
        for label in distinct_labels:
            mean = np.mean(np.array(label_train_vectors[label]), axis=0)

            property = dict()
            property['mean'] = mean
            class_properties[label] = property

            class_covariance_matrix = \
                self.get_class_covariance(
                    label_train_vectors[label], mean, len(train_set_vectors[0]), len(train_set_vectors)
                )
            covariance_matrix = np.add(covariance_matrix, class_covariance_matrix)
        log.info("covariance matrix:\n" + str(covariance_matrix))

        transition_probability = deepcopy(initial_state_distribution)
        historical_probability = dict()
        for label in distinct_labels:
            historical_probability[label] = 1

        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        accuracy_counter = 0
        for i in range(len(train_set_vectors)):
            log.debug("input vector = " + str(train_set_vectors[i]))
            best_score = 0
            best_label = None
            log.debug("transition_probability: " + str(transition_probability))
            historical_probability = self.normalize_probability_weights(historical_probability)
            log.debug("historical_probability: " + str(historical_probability))

            second_term = 0
            for label in distinct_labels:
                second_term += transition_probability[label] * historical_probability[label]
                log.debug("second_term: " + str(second_term))

            for label in distinct_labels:
                log.debug("for label=" + str(label))
                vector_deviation = (train_set_vectors[i] - class_properties[label]['mean'])
                expt_term = np.matmul(np.matmul(vector_deviation, inv_covariance_matrix), vector_deviation)
                emission_pdf = math.exp(-0.5 * expt_term)
                log.debug("emission_pdf: " + str(emission_pdf))
                op = emission_pdf * second_term
                log.debug("output: " + str(op))
                historical_probability[label] = op
                if op > best_score:
                    best_score = op
                    best_label = label

            log.debug("best_label: " + str(best_label))
            log.debug("actual_label: " + str(train_set_labels[i]))

            # counting accuracy for only the test examples
            if train_set_labels[i] == best_label and i >= 500:
                accuracy_counter += 1

            transition_probability = transition_probabilities_matrix[train_set_labels[i]]

        return accuracy_counter/ (len(train_set_vectors)/2)

    def get_initial_state_distribution(self, distinct_labels, label_sequences):
        initial_state_distribution = dict()
        for label in distinct_labels:
            initial_state_distribution[label] = 0

        for sequence in label_sequences:
            initial_state_distribution[sequence[0]] += 1

        for label in initial_state_distribution.keys():
            initial_state_distribution[label] /= len(label_sequences)

        return initial_state_distribution

    def normalize_probability_weights(self, current_probability_weights):
        sum_of_probabilities = 0
        for label in current_probability_weights.keys():
            sum_of_probabilities += current_probability_weights[label]

        if sum_of_probabilities != 1:
            factor = 1 / sum_of_probabilities
            for label in current_probability_weights.keys():
                current_probability_weights[label] *= factor

        return current_probability_weights

    def get_class_covariance(self, class_vectors, mean, dimensions, train_set_size):

        final_matrix = np.zeros((dimensions, dimensions))

        for i in range(len(class_vectors)):
            vector = np.subtract(np.array(class_vectors[i]), mean)
            matrix = np.outer(vector, np.transpose(vector))
            final_matrix = np.add(final_matrix, matrix)

        return final_matrix/train_set_size