import sys
from argparse import ArgumentParser

from kernels.gaussian_kernel import GaussianKernel
from kernels.identity_kernel import IdentityKernel
from kernels.polynomial_kernel import PolynomialKernel
from processors.gaussian_process_regression_processor import GaussianProcessRegressionProcessor
from utils.log_helper import get_logger
from utils.options import Options

log = get_logger("main")


def validate_options(options):
    """
    Validates the options object
    :param options: options object
    :return: null
    """

    log.info("Options read successfully")


def main(argv):
    """
    Main function to kick start execution
    :param argv: command line args list
    :return: null
    """
    options = parse_args(argv)
    validate_options(options)
    processor = GaussianProcessRegressionProcessor(options)
    processor.process()


def parse_args(argv):
    """
    Parses command line arguments form an options object
    :param argv: command line args list
    :return: null
    """
    parser = ArgumentParser(prog="linear_regression")
    parser.add_argument('--input_data_folder', metavar='Data folder',
                        type=str, required=True)
    parser.add_argument('--kernel_type', metavar='Type of Kernel',
                        type=str, required=True)
    parser.add_argument('--metrics_file', metavar='Metrics File',
                        type=str, required=True)
    parser.add_argument('--gaussian_stddev', metavar='Standard Deviation for Gaussian Kernel',
                        type=int, required=False)
    parser.add_argument('--polynomial_degree', metavar='Degree for Polynomial Kernel',
                        type=int, required=False)

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
