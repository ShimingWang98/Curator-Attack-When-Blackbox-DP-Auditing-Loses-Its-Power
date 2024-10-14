import numpy as np
from mechanisms.abstract import Mechanism
import math

def practical_continuous_laplace(b, r, n_samples, vector_dimension):
    ratio = math.ceil(1.0/(b * r) * 1.1)
    scale = 1.0/r
    B = -scale * np.log(1 - (b * r ))
    result = np.random.laplace(scale=scale, size=(ratio*n_samples, vector_dimension))
    filtered = result[abs(result) <= B][:n_samples*vector_dimension]
    noise = filtered.reshape(-1, vector_dimension)
    assert (len(noise) == n_samples)

    return noise

class ReportNoisyMax1(Mechanism):
    """
    Alg. 5 from:
        Zeyu Ding, YuxinWang, GuanhongWang, Danfeng Zhang, and Daniel Kifer. 2018.
        Detecting Violations of Differential Privacy. CCS 2018.
    """

    def __init__(self, original=True, eps: float = 0.1, r = 2.0):
        self.eps = eps
        self.r = r
        self.original = original

    def m(self, a, n_samples: int = 1):
        v = np.atleast_2d(a) # v.shape is (1, vector_dimension)
        # each row in m is one sample
        if self.original:
            # original lapalce
            noise = np.random.laplace(scale=1/self.eps, size=(n_samples, a.shape[0]))
        else:
            #practical laplace
            noise = practical_continuous_laplace(b=1.0/self.eps, r = self.r, n_samples=n_samples, vector_dimension=a.shape[0])
        m = v + noise
        return np.argmax(m, axis=1)