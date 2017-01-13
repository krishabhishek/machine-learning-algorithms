from argparse import ArgumentParser

import sys

from processors.knn_crossval_run_processor import KnnCrossvalRunProcessor
from processors.knn_full_run_processor import KnnFullRunProcessor
from utils.options import Options


def main(argv):
    """
    Main function to kick start execution
    :param argv:
    :return: null
    """
    options = parse_args(argv)
    processor = None

    if options.crossval:
        processor = KnnCrossvalRunProcessor(options)
    elif options.full:
        processor = KnnFullRunProcessor(options)
    processor.process()


def parse_args(argv):
    """
    Parses command line arguments form an options object
    :param argv:
    :return:
    """
    parser = ArgumentParser(prog="k_nearest_neighbours")
    parser.add_argument('--crossval', metavar='Cross-validated run',
                        type=bool, required=False)
    parser.add_argument('--full', metavar='Full training data run',
                        type=bool, required=False)
    parser.add_argument('--input_data_folder', metavar='Data folder',
                        type=str, required=True)
    parser.add_argument('--min_k', metavar='Minimum value of k',
                        type=int, required=True)
    parser.add_argument('--max_k', metavar='Maximum value of k',
                        type=int, required=True)
    parser.add_argument('--metrics_file', metavar='Metrics File',
                        type=str, required=True)

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
