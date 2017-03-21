from argparse import ArgumentParser

import sys

from processors.mnist_processor import MnistProcessor
from utils.options import Options


def main(argv):
    options = parse_args(argv)
    processor = MnistProcessor(options)
    processor.process()


def parse_args(argv):
    parser = ArgumentParser(prog="tensorflow_mnist")
    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
