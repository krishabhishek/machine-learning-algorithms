import time
import math

import numpy as np

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("MarkovProcessorViterbi")

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


class MarkovProcessorViterbi(Processor):

    def process(self):
        log.info("MarkovProcessorViterbi begun")

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

        log.info("MarkovProcessorViterbi concluded")

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
            prediction_list = list()
            belief_dict = initial_state_distribution

            for i in range(len(test_vector_sequences[j]) - 1):
                log.debug("input_vector_" + str(i) + " = " + str(test_vector_sequences[j][i]))

                next_label_probabilities = dict()
                for current_label in sorted(distinct_labels):

                    log.debug("current_label=" + str(current_label))
                    vector_deviation = (test_vector_sequences[j][i] - class_properties[current_label]['mean'])
                    expt_term = np.matmul(np.matmul(vector_deviation, inv_covariance_matrix), vector_deviation)
                    emission_probability = math.exp(-0.5 * expt_term)
                    log.debug("emission_probability: " + str(emission_probability))

                    for next_label in sorted(distinct_labels):
                        transition_probability = transition_probabilities_matrix[current_label][next_label]
                        belief_term = belief_dict[current_label]

                        final_probability = transition_probability * emission_probability * belief_term

                        if next_label in next_label_probabilities.keys():
                            previous_probability, _ = next_label_probabilities[next_label]
                            if final_probability > previous_probability:
                                next_label_probabilities[next_label] = (final_probability, current_label)
                        else:
                            next_label_probabilities[next_label] = (final_probability, current_label)

                log.debug("next_label_probabilities: " + str(next_label_probabilities))

                # copy next label probs to belief and normalize them
                # construct future basis prediction dict at the same time
                prediction_dict = dict()
                for label in next_label_probabilities.keys():
                    probability, prev_label = next_label_probabilities[label]
                    belief_dict[label] = probability
                    prediction_dict[label] = prev_label
                belief_dict = normalize_probability_weights(belief_dict)
                log.debug("belief_dict: " + str(belief_dict))
                log.debug("prediction_dict: " + str(prediction_dict))

                prediction_list.append(prediction_dict)

            log.debug("final_belief_dict: " + str(belief_dict))
            sequence_list = list()
            max_prob = 0
            for label in belief_dict:
                if belief_dict[label] > max_prob:
                    max_prob = belief_dict[label]
                    final_label = label
            log.debug("final_label: " + str(final_label))
            sequence_list.append(final_label)

            for i in range(len(test_vector_sequences[j]) - 1):
                index = len(test_vector_sequences[j]) - i - 2
                final_label = prediction_list[index][final_label]
                sequence_list.append(final_label)

            predictions = list(reversed(sequence_list))

            for i in range(len(predictions)):
                if predictions[i] == test_label_sequences[j][i]:
                    accuracy_counter += 1

        log.info("Corrected classified examples: " + str(accuracy_counter))
        return accuracy_counter/len(test_set_vectors)
