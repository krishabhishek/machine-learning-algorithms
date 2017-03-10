import numpy as np
import math

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("MarkovProcessor")


class MarkovProcessor(Processor):

    def process(self):
        log.info("MarkovProcessor begun")

        train_set_vectors, train_set_labels, test_set_vectors, test_set_labels = \
            file_helper.get_datasets(self.options.input_data_folder)

        accuracy = self.run_classifier(train_set_vectors, train_set_labels, test_set_vectors, test_set_labels)
        log.info("Accuracy: " + str(accuracy))

        log.info("MarkovProcessor concluded")

    def run_classifier(self, train_set_vectors, train_set_labels, test_set_vectors, test_set_labels):
        return None
