import numpy as np
from kernels.generic_kernel import Kernel


class IdentityKernel(Kernel):

    def __init__(self):
        super().__init__()

    def compute_kernel_function(self, x, x_prime):
        return np.matmul(x, np.transpose(x_prime)).item(0)