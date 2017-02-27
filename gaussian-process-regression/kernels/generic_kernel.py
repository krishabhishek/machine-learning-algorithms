import abc


class Kernel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return

    @abc.abstractmethod
    def compute_kernel_function(self, x, x_prime):
        return
