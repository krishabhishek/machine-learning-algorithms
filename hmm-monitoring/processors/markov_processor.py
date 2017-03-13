import numpy as np
from scipy import stats
from copy import deepcopy

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("MarkovProcessor")

file_range = range(1, 6)


class MarkovProcessor(Processor):

    def process(self):
        log.info("MarkovProcessor begun")

        train_set_vectors, train_set_labels, test_set_vectors, test_set_labels, vector_sequences, label_sequences = \
            file_helper.get_datasets(self.options.input_data_folder, file_range)

        train_set_vectors.extend(test_set_vectors)
        train_set_labels.extend(test_set_labels)
        accuracy = self.run_classifier(train_set_vectors, train_set_labels, label_sequences)
        log.info("Accuracy: " + str(accuracy))

        log.info("MarkovProcessor concluded")

    def get_initial_state_distribution(self, distinct_labels, label_sequences):
        initial_state_distribution = dict()
        for label in distinct_labels:
            initial_state_distribution[label] = 0

        for sequence in label_sequences:
            initial_state_distribution[sequence[0]] += 1

        for label in initial_state_distribution.keys():
            initial_state_distribution[label] /= len(label_sequences)

        return initial_state_distribution

    def run_classifier(self, train_set_vectors, train_set_labels, label_sequences):

        distinct_labels = set(train_set_labels)
        log.info("distinct labels: " + str(distinct_labels))
        log.info("train set size: " + str(len(train_set_labels)))

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

        emission_probabilities = dict()
        for key in label_train_vectors.keys():
            emission_probabilities[key] = dict()
            emission_probabilities[key]['mean'] = np.mean(label_train_vectors[key], axis=0)
            emission_probabilities[key]['std'] = np.std(label_train_vectors[key], axis=0)
            emission_probabilities[key]['nd'] = \
                stats.norm(emission_probabilities[key]['mean'], emission_probabilities[key]['std'])
        log.debug("emission_probabilities: " + str(emission_probabilities))

        log.info("covariance matrix: " + str(np.cov(train_set_vectors).shape))

        transition_probability = deepcopy(initial_state_distribution)
        historical_probability = dict()
        for label in distinct_labels:
            historical_probability[label] = 1

        accuracy_counter = 0
        for i in range(len(train_set_vectors)):
            log.info("input vector = " + str(train_set_vectors[i]))
            best_score = 0
            best_label = None
            log.info("historical_probability: " + str(historical_probability))
            log.info("transition_probability: " + str(transition_probability))
            for label in distinct_labels:
                log.info("for label=" + str(label))
                emission_pdf = emission_probabilities[label]['nd'].pdf(train_set_vectors[i])
                log.info("emission_pdf: " + str(emission_pdf))
                op = emission_pdf * (transition_probability[label] * historical_probability[label])
                log.info("output: " + str(op))
                historical_probability[label] = np.linalg.norm(op)

                if np.linalg.norm(op) > best_score:
                    best_score = np.linalg.norm(op)
                    best_label = label

                log.info("final prob = " + str(np.linalg.norm(op)))

            historical_probability = self.normalize_probability_weights(historical_probability)
            log.info("best label: " + str(best_label))
            if train_set_labels[i] == best_label:
                accuracy_counter += 1

            transition_probability = transition_probabilities_matrix[best_label]

            if i == 100:
                break

        return accuracy_counter/len(train_set_vectors)

    def normalize_probability_weights(self, current_probability_weights):
        sum_of_probabilities = 0
        for label in current_probability_weights.keys():
            sum_of_probabilities += current_probability_weights[label]

        if sum_of_probabilities != 1:
            factor = 1 / sum_of_probabilities
            for label in current_probability_weights.keys():
                current_probability_weights[label] *= factor

        return current_probability_weights
