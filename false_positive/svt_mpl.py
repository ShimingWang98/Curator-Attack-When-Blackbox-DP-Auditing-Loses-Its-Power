import numpy as np
import scipy.stats as stats
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.stats import laplace, norm
from mechanisms.abstract import Mechanism
from mechanisms.sparse_vector_technique import SparseVectorTechnique1
from input.pattern_generator import PatternGenerator
from input.input_domain import InputDomain, InputBaseType
from input.input_pair_generator import InputPairGenerator



def hyp_distance_max(scal_par):
    m_1 = 2
    m_2 = 1
    out = scal_par * (max(m_1, m_2) - min(m_1, m_2)) + np.log(2 - np.exp(
        -scal_par * max(m_1, m_2))) - np.log(2 -
                                             np.exp(-scal_par * min(m_1, m_2)))
    return out



def inverse(f, lower=-100, upper=100):

    def func(y):
        return root_scalar(lambda x: f(x) - y, bracket=[lower, upper]).root

    return func
 

hyp_inverse = inverse(hyp_distance_max, 0, 10)

class SparseVectorTechnique2():
    """
    Alg. 2 from:
        M. Lyu, D. Su, and N. Li. 2017.
        Understanding the Sparse Vector Technique for Differential Privacy.
        Proceedings of the VLDB Endowment.
    """

    def __init__(self, eps: float = 0.1, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t

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
        x = np.atleast_2d(a)
        n_queries = a.shape[0]

        # rho = np.random.laplace(scale=self.c / self.eps1, size=(n_samples,))
        # nu = np.random.laplace(scale=2*self.c / self.eps2, size=(n_samples, n_queries))

        theta1, theta2 = 1.1, 5 * eps
        # rho = np.random.uniform(low=-2 * theta1, high=-theta1, size=(n_samples,))
        rho = np.random.laplace(scale=self.c / self.eps1, size=(n_samples,))
        nu = np.random.laplace(scale=1 / theta2, size=(n_samples, n_queries))
        # nu = np.random.laplace(scale=2 * self.c / self.eps2, size=(n_samples, n_queries))

        m = nu + x  # broadcasts x vertically

        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)
        res = np.empty(shape=m.shape, dtype=int)
        for col_idx in range(0, n_queries):
            cmp = m[:, col_idx] >= (rho + self.t)
            res[:, col_idx] = cmp.astype(int)
            res[aborted, col_idx] = -1
            count = count + cmp

            # update rho whenever we answer TRUE
            # new_rho = np.random.laplace(scale=self.c / self.eps1, size=(n_samples,))
            # rho[cmp] = new_rho[cmp]

            aborted = np.logical_or(aborted, count == self.c)
        return res


class SparseVectorTechnique1_sniper(Mechanism):
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
        # #print(f'a is {a}, c is {self.c}')
        x = np.atleast_2d(a)

        # rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))
        # nu = np.random.laplace(scale=2*self.c / self.eps2, size=(n_samples, a.shape[0]))

        epsc = 5
        theta1, theta2 =0.9, 5 * epsc
        rho = np.random.uniform(low=-2 * theta1, high=-theta1, size=(n_samples, 1))
        nu = np.random.laplace(scale=1 / theta2, size=(n_samples, a.shape[0]))
        # #print(f'theta1 is {theta1}')
        # #print(self.t, self.c)
        # rho = np.random.uniform(low=-2 * self.beta, high=-self.beta, size=(n_samples, 1))
        # nu = np.random.laplace(scale= 1 / self.eps2, size=(n_samples, a.shape[0]))

        m = nu + x  # broadcasts x vertically
        cmp = m >= (rho + self.t)   # broadcasts rho horizontally
        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)
        res = cmp.astype(int)

        col_idx = 0
        # #print(f'epsilon1 {self.eps1}')
        for column in cmp.T:
            # #print('hi')
            # #print(f'column is {column}, aborted is {aborted}')
            res[aborted, col_idx] = -1
            count = count + column
            # #print(f'count is {count}')
            aborted = np.logical_or(aborted, count == self.c)
            col_idx = col_idx + 1
        # #print(f'res is {res}')
        # #print('==============\n')
        return res


# #print(data_bases.shape[1] // 2 - 1)
v1 = np.array([0, 0, 0, 1, 1, 1])
v2 = np.array([1, 1, 1, 1, 1, 1])

data_bases = np.column_stack((v1, v2))

for i in range(3):
    v2[i + 3] = 0
    data_bases = np.column_stack((data_bases, v1, v2))

for i in range(2, 5):
    v2[7 - i] = 2
    data_bases = np.column_stack((data_bases, v1, v2))

v1 = np.array([1, 1, 1, 1, 1, 1])
v2 = np.array([0, 0, 0, 2, 2, 2])

data_bases = np.column_stack((data_bases, v1, v2))

for i in range(2, 4):
    v2[4 - i] = 1
    data_bases = np.column_stack((data_bases, v1, v2))


def MPL(data_set, sens, n, N, alpha, delta,privacy_mechanism):
    epsilon_max = -1
    t_max = np.array([])
    x1_max = np.array([])
    x2_max = np.array([])

    #print(f'database is {data_set.shape}')
    # database shape: pattern (6), querynum*2 (20)
    #print(f'database is {data_set}')

    for i in range(data_set.shape[1] // 2):
       
        # Sample output data and create density vector for the first query
        qu_1 = data_set[:, (2 * i)]
        result_1 = privacy_mechanism.m(qu_1, n)
        #print(f'check the input shape: {qu_1.shape}')
        #print(f'check the input: {qu_1}')
        # #print(f'check the shape of mechanism output: {result_1.shape}')
        # Create an array to store the frequency of each unique vector
        unique_data, counts = np.unique(result_1, axis=0, return_counts=True)
        dens_1 = np.zeros(unique_data.shape)
        dens_1 = counts / n
        

        # Sample output data and create density vector for the neighboring query
        qu_2 = data_set[:, 2 * i + 1]
        # #print(qu_2)
        result_2 = privacy_mechanism.m(qu_2, n)
        unique_data, counts = np.unique(result_2, axis=0, return_counts=True)
        dens_2 = np.zeros(unique_data.shape)

        # 使用 NumPy 向量化操作计算 dens_2
        dens_2 = counts / n

        # #print(f'check dens_1 and dens_2 shape {dens_2.shape, dens_1.shape}')
     
        # Compute loss function with floor and determine maximum loss
        loss_hat = np.abs(
            np.log(np.maximum(delta, dens_1)) -
            np.log(np.maximum(delta, dens_2)))

        epsilon_hat = max(loss_hat)
        t_hat = np.argmax(loss_hat)

        # Record data bases and location associated with the current maximum loss
        if epsilon_hat > epsilon_max:
            epsilon_max = epsilon_hat
            t_max = t_hat
            x1_max = qu_1
            x2_max = qu_2

    # Recompute Maximum Loss and Statistic on larger sample sizes
    result_3 = privacy_mechanism.m(x1_max, N)
    
    # Create an array to store the frequency of each unique vector
    unique_data, counts = np.unique(result_3, axis=0, return_counts=True)

    
    dens_3 = np.zeros(unique_data.shape)

    
    dens_3 = (counts / N)
    f_star_1 = max(dens_3[t_max], delta)

    # Recompute Maximum Loss and Statistic on larger sample sizes
    result_4 = privacy_mechanism.m(x2_max, N)
    
    # Create an array to store the frequency of each unique vector
    unique_data, counts = np.unique(result_4, axis=0, return_counts=True)

   
    dens_4 = np.zeros(unique_data.shape)

    
    dens_4 = (counts / N)
    f_star_2 = max(dens_4[t_max], delta)

    sigma_hat = (1 / f_star_1 + 1 / f_star_2) - 2
    loss_star = np.abs(np.log(f_star_1) - np.log(f_star_2))

    LB = loss_star + (stats.norm.ppf(alpha) * np.sqrt(sigma_hat)) / np.sqrt(N)

    return LB


def return_density(union, unique_data, counts, n_samples):
    indices = np.nonzero((union[:, None] == unique_data).all(-1))[0]
    new_count = np.zeros(len(union), dtype=int)
    new_count[indices] = counts
    dens_1 = np.zeros(new_count.shape)
    dens_1 = new_count / n_samples
    return dens_1



def MPL_wsm(input_generator: InputPairGenerator, sens, n, N, alpha, delta, privacy_mechanism):
    epsilon_max = -1
    t_max = np.array([])
    x1_max = np.array([])
    x2_max = np.array([])

    # for i in range(data_set.shape[1] // 2):
    for (qu_1, qu_2) in input_generator.get_input_pairs():
        result_1 = privacy_mechanism.m(qu_1, n)
        unique_data_1, counts_1 = np.unique(result_1, axis=0, return_counts=True)

        result_2 = privacy_mechanism.m(qu_2, n)
        unique_data_2, counts_2 = np.unique(result_2, axis=0, return_counts=True)


        union, union_indices = np.unique(np.vstack((unique_data_1, unique_data_2)), axis=0, return_inverse=True)
        dens_1 = return_density(union, unique_data_1, counts_1, n)
        dens_2 = return_density(union, unique_data_2, counts_2, n)

        #print(f'check dens_1 and dens_2 shape, and check dens_1 and dens_2 {dens_1, dens_2}')

        # Compute loss function with floor and determine maximum loss
        loss_hat = np.abs(
            np.log(np.maximum(delta, dens_1)) -
            np.log(np.maximum(delta, dens_2)))

        epsilon_hat = max(loss_hat)
        t_hat = np.argmax(loss_hat)
        print(epsilon_hat)
        # Record data bases and location associated with the current maximum loss
        if epsilon_hat > epsilon_max:
            epsilon_max = epsilon_hat
            t_max = t_hat
            x1_max = qu_1
            x2_max = qu_2

    print('---------------------')
    # Recompute Maximum Loss and Statistic on larger sample sizes
    result_3 = privacy_mechanism.m(x1_max, N)
    unique_data_3, counts_3 = np.unique(result_3, axis=0, return_counts=True)
    result_4 = privacy_mechanism.m(x2_max, N)
    unique_data_4, counts_4 = np.unique(result_4, axis=0, return_counts=True)

    union, _ = np.unique(np.vstack((unique_data_3, unique_data_4)), axis=0, return_inverse=True)
    dens_3 = return_density(union, unique_data_3, counts_3, N)
    dens_4 = return_density(union, unique_data_4, counts_4, N)

    #print(f'unique 3 4 are {unique_data_3, counts_3, dens_3}')
    #print(f'unique 3 4 are {unique_data_4, counts_4, dens_4}')

    f_star_1 = max(dens_3[t_max], delta)
    f_star_2 = max(dens_4[t_max], delta)

    sigma_hat = (1 / f_star_1 + 1 / f_star_2) - 2
    loss_star = np.abs(np.log(f_star_1) - np.log(f_star_2))

    LB = loss_star + (stats.norm.ppf(alpha) * np.sqrt(sigma_hat)) / np.sqrt(N)

    # return LB
    return loss_star


eps = 0.7
sens = 1
n = 100000
N = 500000
alpha = 0.05
delta = 0.0001
epsilon = hyp_inverse(eps)
# privacy_mechanism = SparseVectorTechnique2()
# privacy_mechanism = SparseVectorTechnique1_sniper()
privacy_mechanism = SparseVectorTechnique1()
input_generator = PatternGenerator(InputDomain(10, InputBaseType.FLOAT, [-10, 10]), False)

reps = 5
# A = Parallel(n_jobs=3)(delayed(MPL_wsm)(data_bases, sens, n, N, alpha, delta, privacy_mechanism)for _ in range(reps))
A = Parallel(n_jobs=1)(delayed(MPL_wsm)(input_generator, sens, n, N, alpha, delta, privacy_mechanism)for _ in range(reps))
final_array = np.array(A)
print(final_array)


