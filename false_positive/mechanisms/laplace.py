import numpy as np
from mechanisms.abstract import Mechanism
import time
from sample.sample_laplace import *

def _identity(x):
    return x.item(0)


class LaplaceMechanism(Mechanism):

    def __init__(self, original=0, fun=_identity, eps: float = 5.0, r=2.0, beta=10, c=0.05 ):
        """
        Create a Laplace mechanism.

        Args:
            fun: The function performed before adding noise. The function must accept a 1d array and produce a scalar.
            eps: target epsilon
        """
        self.fun = fun
        self.scale = 1.0 / eps
        self.eps = eps
        self.beta = beta
        # use original laplace or practical laplace
        self.original = original
        # r will change according to c and eps
        self.r = r

    # original laplace
    def m(self, a, n_samples: int = 1):
        if self.original == 0:
            # original laplace noise
            loc = self.fun(a)
            return np.random.laplace(loc=loc, scale=self.scale, size=n_samples)
        elif self.original == 1:
            # pdp noise, laplace shaped
            loc = self.fun(a)
            result = practical_continuous_laplace(b=self.scale, r=self.r, size = n_samples)
            result += loc
            return result
        else:
            # pdp noise, laplace shape with flat ends
            # print("flat")
            loc = self.fun(a)
            result = practical_continuous_laplace_flat(beta=self.beta, eps0=self.eps/2, size=n_samples)
            result += loc
            return result


    # uniform distribution
    # def m(self, a, n_samples: int =1):
    #     loc = self.fun(a)
    #     result = np.random.uniform(low=-self.scale, high=self.scale, size=n_samples)
    #     return result+loc

    def _test_original_utility(self, n_samples):
        result = np.random.laplace(scale=self.scale, size=n_samples)
        # noise between [-0.1, 0.1]
        new_result = result[np.where(abs(result) <= 0.1)]
        # the ratio of noise between [-0.1, 0.1]
        ratio = len(new_result)/n_samples
        print("original ratio: ",ratio)
        # variance of noise
        var = np.var(result)
        print("original utility: ",var)

    def _test_practical_utility(self, n_samples):
        result = practical_continuous_laplace(b=self.scale, r=self.r, size = n_samples)
        # result = np.random.uniform(low=-self.scale, high=self.scale, size=n_samples)
        new_result = result[np.where(abs(result) <= 0.1)]
        ratio = len(new_result) / n_samples
        print("practical ratio: ", ratio)
        var = np.var(result)
        print("practical utility: ",var)


if __name__ == '__main__':
    mechanism = LaplaceMechanism()
    # n_samples = 1000000
    # mechanism._test_original_utility(n_samples)
    # mechanism._test_practical_utility(n_samples)