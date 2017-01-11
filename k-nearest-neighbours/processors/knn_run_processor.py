from processors.processor import Processor
from utils import log_helper

log = log_helper.get_logger("KnnRunProcessor")


class KnnRunProcessor(Processor):

    def process(self):
        log.info("KnnRunProcessor begun")

        log.info(self.options.input_data_folder)
        log.info(self.options.min_k)
        log.info(self.options.max_k)

        log.info("KnnRunProcessor concluded")
