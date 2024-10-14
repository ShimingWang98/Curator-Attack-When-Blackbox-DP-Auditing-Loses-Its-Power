import numpy as np
from mechanisms.abstract import Mechanism

class SparseVectorTechnique1(Mechanism):
    """
    practical svt, rho from uniform distribution, nu from rescaled laplace distribution.
    """

    def __init__(self, eps: float = 0.1, beta: float=2.1, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        # self.eps2 = eps * 1.5
        self.c = 1 #c  # maximum number of queries answered with 1
        self.t = 1
        self.beta = beta

    def m(self, a, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
                -1 = ABORTED;
        """

        # columns: queries
        # rows: samples
        # print(f'a is {a}, c is {self.c}')
        x = np.atleast_2d(a)

        # rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))
        # nu = np.random.laplace(scale=2*self.c / self.eps2, size=(n_samples, a.shape[0]))

        epsc = 0.05
        theta1, theta2 =30, 5 * epsc
        rho = np.random.uniform(low=-2 * theta1, high=-theta1, size=(n_samples, 1))
        nu = np.random.laplace(scale=1 / theta2, size=(n_samples, a.shape[0]))
        # print(f'theta1 is {theta1}')
        # print(self.t, self.c)
        # rho = np.random.uniform(low=-2 * self.beta, high=-self.beta, size=(n_samples, 1))
        # nu = np.random.laplace(scale= 1 / self.eps2, size=(n_samples, a.shape[0]))

        m = nu + x  # broadcasts x vertically
        cmp = m >= (rho + self.t)   # broadcasts rho horizontally
        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)
        res = cmp.astype(int)

        col_idx = 0
        # print(f'epsilon1 {self.eps1}')
        for column in cmp.T:
            # print('hi')
            # print(f'column is {column}, aborted is {aborted}')
            res[aborted, col_idx] = -1
            count = count + column
            # print(f'count is {count}')
            aborted = np.logical_or(aborted, count == self.c)
            col_idx = col_idx + 1
        # print(f'res is {res}')
        # print('==============\n')
        return res
