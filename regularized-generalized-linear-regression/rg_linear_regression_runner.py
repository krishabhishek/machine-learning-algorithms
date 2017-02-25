from argparse import ArgumentParser

import sys

from processors.generalized_linear_regression_processor import GeneralizedLinearRegressionProcessor
from utils.options import Options


def main(argv):
    """
    Main function to kick start execution
    :param argv:
    :return: null
    """
    options = parse_args(argv)
    processor = GeneralizedLinearRegressionProcessor(options)
    processor.process()


def parse_args(argv):
    """
    Parses command line arguments form an options object
    :param argv:
    :return:
    """
    parser = ArgumentParser(prog="linear_regression")
    parser.add_argument('--input_data_folder', metavar='Data folder',
                        type=str, required=True)
    parser.add_argument('--max_degree', metavar='Maximum Basis Function Degree',
                        type=float, required=True)
    parser.add_argument('--metrics_file', metavar='Metrics File',
                        type=str, required=True)

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
