from argparse import ArgumentParser

import sys

from processors.markov_processor_forward import MarkovProcessorForward
from processors.markov_processor_viterbi import MarkovProcessorViterbi
from utils.options import Options


def main(argv):
    options = parse_args(argv)

    if options.viterbi:
        processor = MarkovProcessorViterbi(options)
    else:
        processor = MarkovProcessorForward(options)

    processor.process()


def parse_args(argv):
    parser = ArgumentParser(prog="hmm_monitoring_app")
    parser.add_argument('--input_data_folder', metavar='Data folder', type=str, required=True)
    parser.add_argument('--viterbi', dest='viterbi', action='store_true')

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
