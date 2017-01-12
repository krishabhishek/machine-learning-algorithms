from processors.processor import Processor
from utils import log_helper

log = log_helper.get_logger("LinearRegressionProcessor")


class LinearRegressionProcessor(Processor):

    def process(self):
        log.info("LinearRegressionProcessor begun")

        log.info("LinearRegressionProcessor concluded")
