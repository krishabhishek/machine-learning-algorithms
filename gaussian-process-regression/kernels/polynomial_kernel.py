from math import pow

import numpy as np

from kernels.generic_kernel import Kernel


class PolynomialKernel(Kernel):

    def __init__(self, degree):

        super().__init__()
        self.degree = degree

    def compute_kernel_function(self, x, x_prime):
        return pow(np.dot(np.transpose(x), x_prime) + 1, self.degree)
