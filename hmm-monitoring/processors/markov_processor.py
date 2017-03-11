import numpy as np
import math

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("MarkovProcessor")

file_range = range(1, 6)


class MarkovProcessor(Processor):

    def process(self):
        log.info("MarkovProcessor begun")

        train_set_vectors, train_set_labels, test_set_vectors, test_set_labels = \
            file_helper.get_datasets(self.options.input_data_folder, file_range)

        accuracy = self.run_classifier(train_set_vectors, train_set_labels, test_set_vectors, test_set_labels)
        log.info("Accuracy: " + str(accuracy))

        log.info("MarkovProcessor concluded")

    def run_classifier(self, train_set_vectors, train_set_labels, test_set_vectors, test_set_labels):

        distinct_labels = set(train_set_labels)
        log.info("distinct labels: " + str(distinct_labels))

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

        for i in range(len(train_set_vectors) - 1):
            if not train_set_labels[i] in transition_probabilities_matrix.keys():
                transition_probabilities_matrix[train_set_labels[i]] = dict()

            if not train_set_labels[i + 1] in transition_probabilities_matrix[train_set_labels[i]].keys():
                transition_probabilities_matrix[train_set_labels[i]][train_set_labels[i + 1]] = 0

            transition_probabilities_matrix[train_set_labels[i]][train_set_labels[i + 1]] += 1

        print(transition_probabilities_matrix)

        return 0
