from processors.processor import Processor
from utils import file_helper
from utils import log_helper
from utils import options

log = log_helper.get_logger("GaussianMixtureProcessor")
file_range = range(1, 11)


class GaussianMixtureProcessor(Processor):

    def process(self):
        log.info("GaussianMixtureProcessor begun")

        for test_set_identifier in file_range:
            train_set_vectors, train_set_labels, test_set_vectors, test_set_labels = \
                file_helper.get_datasets(self.options.input_data_folder, test_set_identifier, file_range)

            log.info(len(train_set_vectors))
            log.info(len(train_set_labels))
            log.info(len(test_set_vectors))
            log.info(len(test_set_labels))

        log.info("GaussianMixtureProcessor concluded")
