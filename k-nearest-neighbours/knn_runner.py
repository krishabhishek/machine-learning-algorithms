from argparse import ArgumentParser

import sys

from processors.knn_run_processor import KnnRunProcessor
from utils.options import Options


def main(argv):
    """
    Main function to kick start execution
    :param argv:
    :return: null
    """
    options = parse_args(argv)
    processor = KnnRunProcessor(options)
    processor.process()


def parse_args(argv):
    """
    Parses command line arguments form an options object
    :param argv:
    :return:
    """
    parser = ArgumentParser(prog="semeval2015-task5")
    parser.add_argument('--input_data_folder', metavar='Data folder',
                        type=str, required=True)
    parser.add_argument('--min_k', metavar='Minimum value of k',
                        type=int, required=True)
    parser.add_argument('--max_k', metavar='Maximum value of k',
                        type=int, required=True)

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
