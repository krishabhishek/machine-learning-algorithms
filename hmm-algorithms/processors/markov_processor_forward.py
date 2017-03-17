import time
import math

import numpy as np

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("MarkovProcessorForward")

file_range = range(1, 6)


def get_initial_state_distribution(distinct_labels, label_sequences):
    initial_state_distribution = dict()
    for label in distinct_labels:
        initial_state_distribution[label] = 0

    for sequence in label_sequences:
        initial_state_distribution[sequence[0]] += 1

    for label in initial_state_distribution.keys():
        initial_state_distribution[label] /= len(label_sequences)

    return initial_state_distribution


def normalize_probability_weights(current_probability_weights):
    sum_of_probabilities = 0
    for label in current_probability_weights.keys():
        sum_of_probabilities += current_probability_weights[label]

    if sum_of_probabilities != 1:
        factor = 1 / sum_of_probabilities
        for label in current_probability_weights.keys():
            current_probability_weights[label] *= factor

    return current_probability_weights


def get_class_covariance(class_vectors, mean, dimensions, train_set_size):

    final_matrix = np.zeros((dimensions, dimensions))

    for i in range(len(class_vectors)):
        vector = np.subtract(np.array(class_vectors[i]), mean)
        matrix = np.outer(vector, np.transpose(vector))
        final_matrix = np.add(final_matrix, matrix)

    return final_matrix/train_set_size


class MarkovProcessorForward(Processor):

    def process(self):
        log.info("MarkovProcessorForward begun")

        train_set_vectors, train_set_labels, test_set_vectors, test_set_labels, train_vector_sequences, \
            test_vector_sequences, train_label_sequences, test_label_sequences = \
            file_helper.get_datasets(self.options.input_data_folder, file_range)

        start_time = time.time()
        accuracy = self.run_classifier(train_set_vectors, train_set_labels, test_set_vectors,
                                       test_set_labels, train_vector_sequences, test_vector_sequences,
                                       train_label_sequences, test_label_sequences)
        elapsed_time = time.time() - start_time
        log.info("elapsed time: " + str(elapsed_time) + " seconds")
        log.info("Accuracy: " + str(accuracy))

        log.info("MarkovProcessorForward concluded")

    def run_classifier(self, train_set_vectors, train_set_labels, test_set_vectors, test_set_labels,
                       train_vector_sequences, test_vector_sequences, train_label_sequences, test_label_sequences):

        distinct_labels = set(train_set_labels)
        log.debug("distinct labels: " + str(distinct_labels))
        log.debug("train set size: " + str(len(train_set_labels)))

        initial_state_distribution = get_initial_state_distribution(distinct_labels, train_label_sequences)
        log.info("initial_state_distribution: " + str(initial_state_distribution))

        # partition the training vectors by label
        label_train_vectors = dict()
        for i in range(len(train_set_vectors)):
            key = train_set_labels[i]
            if key in label_train_vectors:
                vectors = label_train_vectors[key]
            else:
                vectors = list()
            vectors.append(train_set_vectors[i])
            label_train_vectors[key] = vectors


        # calculate the transition probability matrix
        transition_probabilities_matrix = dict()
        class_count_dict = dict()

        for j in range(len(train_vector_sequences)):
            for i in range(len(train_vector_sequences[j]) - 1):
                if not train_label_sequences[j][i] in transition_probabilities_matrix.keys():
                    transition_probabilities_matrix[train_label_sequences[j][i]] = dict()

                if not train_label_sequences[j][i + 1] in transition_probabilities_matrix[train_label_sequences[j][i]].keys():
                    transition_probabilities_matrix[train_label_sequences[j][i]][train_label_sequences[j][i + 1]] = 0

                transition_probabilities_matrix[train_label_sequences[j][i]][train_label_sequences[j][i + 1]] += 1

                if not train_label_sequences[j][i] in class_count_dict.keys():
                    class_count_dict[train_label_sequences[j][i]] = 1
                else:
                    class_count_dict[train_label_sequences[j][i]] += 1

        for class_key in transition_probabilities_matrix.keys():
            for succeeding_class_count in transition_probabilities_matrix[class_key].keys():
                transition_probabilities_matrix[class_key][succeeding_class_count] /= class_count_dict[class_key]

        log.info("transition_probabilities_matrix: " + str(transition_probabilities_matrix))

        # compute label mean and covariance matrix
        class_properties = dict()
        covariance_matrix = np.zeros((len(train_set_vectors[0]), len(train_set_vectors[0])))
        for label in distinct_labels:
            mean = np.mean(np.array(label_train_vectors[label]), axis=0)
            properties = dict()
            properties['mean'] = mean
            class_properties[label] = properties
            class_covariance_matrix = \
                get_class_covariance(
                    label_train_vectors[label], mean, len(train_set_vectors[0]), len(train_set_vectors)
                )
            covariance_matrix = np.add(covariance_matrix, class_covariance_matrix)
        log.info("class_properties:" + str(class_properties))
        log.info("covariance_matrix:\n" + str(covariance_matrix))

        # perform HMM monitoring
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        accuracy_counter = 0
        for j in range(len(test_vector_sequences)):

            historical_probability = dict()
            for label in distinct_labels:
                historical_probability[label] = 1

            for i in range(len(test_vector_sequences[j])):
                log.debug("input_vector_" + str(i) + " = " + str(test_vector_sequences[j][i]))
                best_score = 0
                best_label = None
                log.debug("historical_probability: " + str(historical_probability))

                for label in sorted(distinct_labels):

                    log.debug("current_label=" + str(label))
                    vector_deviation = (test_vector_sequences[j][i] - class_properties[label]['mean'])
                    expt_term = np.matmul(np.matmul(vector_deviation, inv_covariance_matrix), vector_deviation)

                    emission_probability = math.exp(-0.5 * expt_term)
                    log.debug("emission_probability: " + str(emission_probability))

                    belief_term = 0
                    for previous_label in sorted(distinct_labels):
                        belief_term += \
                            transition_probabilities_matrix[previous_label][label] * \
                            historical_probability[previous_label]

                    if i == 0:
                        belief_term = initial_state_distribution[label]
                    log.debug("belief_term: " + str(belief_term))

                    label_probability = emission_probability * belief_term
                    log.debug("label_probability: " + str(label_probability))
                    historical_probability[label] = label_probability

                    if label_probability > best_score:
                        best_score = label_probability
                        best_label = label

                log.debug("best_label: " + str(best_label))
                log.debug("actual_label: " + str(test_label_sequences[j][i]))

                if test_label_sequences[j][i] == best_label:
                    accuracy_counter += 1

                historical_probability = normalize_probability_weights(historical_probability)

        log.info("Corrected classified examples: " + str(accuracy_counter))
        return accuracy_counter/len(test_set_vectors)
