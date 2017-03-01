from math import exp, pow

import numpy.linalg as la

from kernels.generic_kernel import Kernel


class GaussianKernel(Kernel):

    def __init__(self, std_dev):

        super().__init__()
        self.std_dev = std_dev

    def compute_kernel_function(self, x, x_prime):

        exponent_term = (-1 * pow(la.norm(x - x_prime), 2)) / (2 * pow(self.std_dev, 2))

        return exp(exponent_term)

    def random_yo(self):
        print(self.std_dev)
