from argparse import ArgumentParser

import sys

from processors.cnn_processor import CNNProcessor
from processors.softmax_regression_processor import SoftmaxRegressionProcessor
from utils.options import Options


def main(argv):
    processor = None
    options = parse_args(argv)

    if options.mode == "softmax":
        processor = SoftmaxRegressionProcessor(options)
    elif options.mode == "cnn":
        processor = CNNProcessor(options)

    if processor:
        processor.process()


def parse_args(argv):
    parser = ArgumentParser(prog="tensorflow_mnist")
    parser.add_argument('--mode', metavar='Run mode', type=str, required=True)

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
