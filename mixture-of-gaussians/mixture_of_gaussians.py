from argparse import ArgumentParser

import sys

from processors.gaussian_mixture_processor import GaussianMixtureProcessor
from utils.options import Options


def main(argv):
    options = parse_args(argv)
    processor = GaussianMixtureProcessor(options)
    processor.process()


def parse_args(argv):
    parser = ArgumentParser(prog="mixture_of_gaussians_classifier")
    parser.add_argument('--input_data_folder', metavar='Data folder', type=str, required=True)
    parser.add_argument('--full', dest='full', action='store_true')

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
