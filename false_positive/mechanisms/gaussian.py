import numpy as np
from mechanisms.abstract import Mechanism
import time
import math

def _identity(x):
    return x.item(0)

class GaussianMechanism(Mechanism):

    def __init__(self, fun=_identity, var: float = 0, eps: float = 0.1):
        """
        Create a Gaussian mechanism.

        Args:
            fun: The function performed before adding noise. The function must accept a 1d array and produce a scalar.
            eps: target epsilon
        """
        self.fun = fun
        # self.scale = 1.0 / eps

        # to get the same utility of the laplace mechanism, we directly give the variance
        self.var = var

    # original laplace
    def m(self, a, n_samples: int = 1):
        loc = self.fun(a)
        return np.random.normal(loc=loc, scale=math.sqrt(self.var), size=n_samples)

    def _test_utility(self, n_samples):
        result = np.random.normal(scale=math.sqrt(self.var), size=n_samples)
        # noise between [-0.1, 0.1]
        new_result = result[np.where(abs(result) <= 0.1)]
        # the ratio of noise between [-0.1, 0.1]
        ratio = len(new_result)/n_samples
        print("original ratio: ",ratio)
        # variance of noise
        var = np.var(result)
        print("original utility: ",var)

if __name__ == '__main__':
    mechanism = GaussianMechanism()
    # n_samples = 1000000
    # mechanism._test_utility(n_samples)