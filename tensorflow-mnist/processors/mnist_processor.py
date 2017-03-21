from processors.processor import Processor
from utils import log_helper

log = log_helper.get_logger("MnistProcessor")


class MnistProcessor(Processor):

    def process(self):
        log.info("MnistProcessor begun")

        log.info("MnistProcessor concluded")

