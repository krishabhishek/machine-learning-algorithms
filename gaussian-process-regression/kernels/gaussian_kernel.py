from math import exp, pow

import numpy.linalg as la

from kernels.generic_kernel import Kernel


class GaussianKernel(Kernel):

    def __init__(self, variance):

        super().__init__()
        self.variance = variance

    def compute_kernel_function(self, x, x_prime):\

        exponent_term = (-1 * pow(la.norm(x - x_prime), 2)) / (2 * pow(self.variance, 2))

        return exp(exponent_term)
